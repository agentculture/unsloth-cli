"""Tests for sloth.tune._exporter — the lazy in-container export seam.

Everything here runs **without a GPU, without docker, and without the ML stack**:
the single ``_load_backend`` seam is monkeypatched with fakes, exactly as
``tests/test_tune_trainer.py`` does for the trainer. Covered:

  1. torch/unsloth/llmcompressor are never imported at module top level (AST
     guard + subprocess import guard).
  2. A missing stack surfaces as ``CliError(code=2)`` with the NGC hint; a CUDA
     OOM surfaces as ``CliError(code=2)`` with the memory hint.
  3. merged-16bit / merged-4bit call ``save_pretrained_merged`` with the right
     ``save_method``; gguf calls ``save_pretrained_gguf`` and normalises Unsloth's
     ``<dir>_gguf`` suffix directory (dropping the F16 intermediate unless
     ``keep_intermediate``).
  4. AWQ/NVFP4 build the right recipe, use ``pipeline="basic"``, and save with
     ``save_compressed=True``; LFM2 gets per-layer mappings with **no**
     ``v_proj -> out_proj`` pair.
  5. Calibration defaults to the run dataset, honours ``--calib`` /
     ``--calib-samples``, warns once below 64 samples, and is recorded.
  6. The ``torch.accelerator.get_memory_info`` shim is applied/skipped correctly.
  7. ``export.json`` is written next to the artifacts and appended to
     ``<adapter>/exports.json``.
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sloth.cli._errors import CliError
from sloth.tune import _exporter
from sloth.tune._exporter import DEFAULT_GGUF_QUANT, _append_index, _find_gguf, run_export
from tests.test_tune_trainer import _FakeEvalModel as _EvalModelStub
from tests.test_tune_trainer import _FakeTokenizer as _TokenizerStub

_REPO_ROOT = str(Path(__file__).parent.parent)
_HEAVY = {"torch", "unsloth", "llmcompressor", "compressed_tensors", "transformers", "peft"}


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


def _write_task_dataset(path: Path, count: int) -> Path:
    lines = [
        json.dumps({"task": "echo", "input": f"in-{i}", "expected_output": f"out-{i}"})
        for i in range(count)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _adapter_dir(tmp_path: Path) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "LiquidAI/LFM2.5-1.2B-Base"}), encoding="utf-8"
    )
    return adapter


def _plan(tmp_path: Path, fmt: str, **overrides) -> dict:
    plan = {
        "format": fmt,
        "quant": [],
        "base": "LiquidAI/LFM2.5-1.2B-Base",
        "adapter": str(_adapter_dir(tmp_path)),
        "output": str(tmp_path / "out.partial"),
        "calib": None,
        "calib_samples": None,
        "keep_intermediate": False,
        "dataset": None,
    }
    plan.update(overrides)
    return plan


class _FakeTokenizer:
    eos_token = "</s>"

    def __init__(self, events: dict) -> None:
        self._events = events

    def save_pretrained(self, path):
        self._events["tokenizer_saved"].append(str(path))


class _FakeAdapterModel:
    """Stands in for the Unsloth-loaded adapter (merged / gguf paths)."""

    def __init__(self, events: dict) -> None:
        self._events = events

    def save_pretrained_merged(self, path, tokenizer, **kwargs):
        self._events["merged"].append({"path": str(path), "tokenizer": tokenizer, **kwargs})
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.safetensors").write_bytes(b"x" * 16)
        (out / "config.json").write_text(json.dumps({"model_type": "lfm2"}), encoding="utf-8")

    def save_pretrained_gguf(self, path, tokenizer, **kwargs):
        self._events["gguf"].append({"path": str(path), "tokenizer": tokenizer, **kwargs})
        # Live-measured behaviour: Unsloth writes into a *sibling* "<dir>_gguf"
        # directory, plus an F16 intermediate it converted from.
        staged = Path(str(path) + "_gguf")
        staged.mkdir(parents=True, exist_ok=True)
        for method in kwargs.get("quantization_method", []):
            (staged / f"LFM2.5-1.2B-Base.{method.upper()}.gguf").write_bytes(b"q" * 8)
        (staged / "LFM2.5-1.2B-Base.F16.gguf").write_bytes(b"f" * 32)


class _FakeMergedModel:
    """Stands in for the transformers-reloaded merged model (awq/nvfp4 path)."""

    def __init__(self, events: dict, config) -> None:
        self._events = events
        self.config = config

    def save_pretrained(self, path, **kwargs):
        self._events["compressed_saved"].append({"path": str(path), **kwargs})
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.safetensors").write_bytes(b"c" * 24)


def _fake_backend(
    events: dict | None = None,
    *,
    model_type: str = "lfm2",
    layer_types: list[str] | None = None,
    compressed_tensors_version: str = "0.15.0",
) -> tuple[_exporter._Backend, dict]:
    events = events if events is not None else {}
    for key in (
        "merged",
        "gguf",
        "tokenizer_saved",
        "compressed_saved",
        "oneshot",
        "loaded",
        "awq_modifier",
        "quant_modifier",
    ):
        events.setdefault(key, [])

    tokenizer = _FakeTokenizer(events)
    adapter_model = _FakeAdapterModel(events)

    def from_pretrained(**kwargs):
        events["loaded"].append(kwargs)
        return adapter_model, tokenizer

    config = SimpleNamespace(
        model_type=model_type,
        layer_types=layer_types if layer_types is not None else ["full_attention", "conv"],
    )
    merged_model = _FakeMergedModel(events, config)

    def awq_modifier(**kwargs):
        events["awq_modifier"].append(kwargs)
        return SimpleNamespace(kind="awq", **kwargs)

    def quantization_modifier(**kwargs):
        events["quant_modifier"].append(kwargs)
        return SimpleNamespace(kind="quant", **kwargs)

    def awq_mapping(smooth_layer, balance_layers):
        return SimpleNamespace(smooth_layer=smooth_layer, balance_layers=balance_layers)

    def oneshot(**kwargs):
        events["oneshot"].append(kwargs)

    torch = SimpleNamespace(
        bfloat16="bfloat16",
        cuda=SimpleNamespace(mem_get_info=lambda device=None: (1, 2)),
        accelerator=SimpleNamespace(),
    )
    backend = _exporter._Backend(
        torch=torch,
        fast_model=SimpleNamespace(from_pretrained=from_pretrained),
        auto_model_for_causal_lm=SimpleNamespace(
            from_pretrained=lambda *a, **k: merged_model,
        ),
        auto_tokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
        oneshot=oneshot,
        awq_modifier=awq_modifier,
        quantization_modifier=quantization_modifier,
        awq_mapping=awq_mapping,
        compressed_tensors=SimpleNamespace(__version__=compressed_tensors_version),
    )
    return backend, events


@pytest.fixture
def install_backend(monkeypatch):
    """Install a fake backend and return (backend, events)."""

    def _install(**kwargs):
        backend, events = _fake_backend(**kwargs)
        monkeypatch.setattr(_exporter, "_load_backend", lambda **_: backend)
        return backend, events

    return _install


# ---------------------------------------------------------------------------
# 1. Lazy-import discipline
# ---------------------------------------------------------------------------


def test_no_heavy_imports_at_module_top_level():
    """The AST of _exporter.py must carry no module-level heavy import."""
    source = Path(_exporter.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in _HEAVY, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in _HEAVY, node.module


def test_importing_exporter_does_not_load_torch_or_unsloth():
    """Importing the module in a fresh interpreter must not pull in the ML stack."""
    code = (
        "import sloth.tune._exporter, sys; "
        "assert 'torch' not in sys.modules; "
        "assert 'unsloth' not in sys.modules; "
        "assert 'llmcompressor' not in sys.modules; "
        "print('PASS')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO_ROOT
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# 2. Error policy
# ---------------------------------------------------------------------------


def test_missing_stack_raises_cli_error_with_ngc_hint(tmp_path, monkeypatch):
    def _boom(**_):
        raise ImportError("No module named 'llmcompressor'")

    monkeypatch.setattr(_exporter, "_load_backend", _boom)
    plan = _plan(tmp_path, "merged-16bit")
    with pytest.raises(CliError) as excinfo:
        run_export(plan)
    assert excinfo.value.code == 2
    assert "nvcr.io/nvidia/pytorch" in excinfo.value.remediation


def test_backend_import_oom_raises_cli_error_with_memory_hint(tmp_path, monkeypatch):
    class OutOfMemoryError(RuntimeError):
        pass

    def _boom(**_):
        raise OutOfMemoryError("CUDA error: out of memory")

    monkeypatch.setattr(_exporter, "_load_backend", _boom)
    plan = _plan(tmp_path, "merged-16bit")
    with pytest.raises(CliError) as excinfo:
        run_export(plan)
    assert excinfo.value.code == 2
    assert "out of memory" in excinfo.value.message.lower()
    assert "drop_caches" in excinfo.value.remediation


def test_export_oom_during_run_raises_cli_error(tmp_path, install_backend):
    backend, _ = install_backend()

    def _boom(**kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")

    backend.fast_model.from_pretrained = _boom
    plan = _plan(tmp_path, "merged-16bit")
    with pytest.raises(CliError) as excinfo:
        run_export(plan)
    assert excinfo.value.code == 2
    assert "drop_caches" in excinfo.value.remediation


def test_unknown_format_is_a_user_error(tmp_path):
    plan = _plan(tmp_path, "ggml")
    with pytest.raises(CliError) as excinfo:
        run_export(plan)
    assert excinfo.value.code == 1
    assert "merged-16bit" in excinfo.value.remediation


# ---------------------------------------------------------------------------
# 3. Merged + GGUF (Unsloth)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "save_method"),
    [("merged-16bit", "merged_16bit"), ("merged-4bit", "merged_4bit_forced")],
)
def test_merged_export_calls_save_pretrained_merged(tmp_path, install_backend, fmt, save_method):
    _, events = install_backend()
    plan = _plan(tmp_path, fmt)
    result = run_export(plan)

    assert len(events["merged"]) == 1
    call = events["merged"][0]
    assert call["path"] == plan["output"]
    assert call["save_method"] == save_method
    # merged_4bit needs a quantised base: only the 4-bit format loads in 4-bit.
    if events.get("from_pretrained"):
        assert events["from_pretrained"][0].get("load_in_4bit") is (fmt == "merged-4bit")
    assert isinstance(call["tokenizer"], _FakeTokenizer)

    assert result["format"] == fmt
    assert result["files"]["model.safetensors"] == 16
    assert Path(result["export_json"]).is_file()
    assert result["calibration"] is None


def test_gguf_export_moves_quants_and_drops_intermediate(tmp_path, install_backend):
    _, events = install_backend()
    plan = _plan(tmp_path, "gguf", quant=["q4_k_m"])
    result = run_export(plan)

    call = events["gguf"][0]
    assert call["path"] == plan["output"]
    assert call["quantization_method"] == ["q4_k_m"]

    output = Path(plan["output"])
    names = sorted(p.name for p in output.glob("*.gguf"))
    assert names == ["LFM2.5-1.2B-Base.Q4_K_M.gguf"]
    # The F16 intermediate is deleted and Unsloth's "_gguf" staging dir removed.
    assert not Path(str(output) + "_gguf").exists()
    assert "LFM2.5-1.2B-Base.Q4_K_M.gguf" in result["files"]


def test_gguf_export_defaults_to_q4_k_m(tmp_path, install_backend):
    _, events = install_backend()
    run_export(_plan(tmp_path, "gguf"))
    assert events["gguf"][0]["quantization_method"] == ["q4_k_m"]


def test_gguf_keep_intermediate_retains_f16(tmp_path, install_backend):
    install_backend()
    plan = _plan(tmp_path, "gguf", quant=["q4_k_m"], keep_intermediate=True)
    result = run_export(plan)
    assert "LFM2.5-1.2B-Base.F16.gguf" in result["files"]
    assert "LFM2.5-1.2B-Base.Q4_K_M.gguf" in result["files"]
    assert not Path(str(Path(plan["output"])) + "_gguf").exists()


# ---------------------------------------------------------------------------
# 4. AWQ / NVFP4 (llm-compressor)
# ---------------------------------------------------------------------------


def _awq_plan(tmp_path: Path, **overrides) -> dict:
    dataset = _write_task_dataset(tmp_path / "calib.jsonl", 80)
    return _plan(tmp_path, "awq", dataset=str(dataset), **overrides)


def test_awq_merges_first_then_oneshots_with_basic_pipeline(tmp_path, install_backend):
    _, events = install_backend()
    plan = _awq_plan(tmp_path)
    result = run_export(plan)

    # merged-16bit first, into an intermediate dir that is cleaned up afterwards.
    assert events["merged"][0]["save_method"] == "merged_16bit"
    assert not (Path(plan["output"]) / "_merged-16bit").exists()

    call = events["oneshot"][0]
    assert call["max_seq_length"] == 512
    # MANDATORY: the default sequential pipeline dies under torch.fx for LFM2.
    assert call["pipeline"] == "basic"
    assert call["num_calibration_samples"] == 80
    recipe = call["recipe"]
    assert [m.kind for m in recipe] == ["awq", "quant"]
    assert recipe[0].duo_scaling == "both"
    assert recipe[1].scheme == "W4A16_ASYM"
    assert recipe[1].targets == ["Linear"]
    assert recipe[1].ignore == ["lm_head"]

    saved = events["compressed_saved"][0]
    assert saved["save_compressed"] is True
    assert saved["path"] == plan["output"]
    assert result["files"]["model.safetensors"] == 24


def test_awq_lfm2_mappings_are_per_layer_and_drop_v_to_out_proj(tmp_path, install_backend):
    _, events = install_backend(layer_types=["full_attention", "conv"])
    run_export(_awq_plan(tmp_path))

    mappings = events["oneshot"][0]["recipe"][0].mappings
    pairs = [(m.smooth_layer, tuple(m.balance_layers)) for m in mappings]
    flat = " ".join(m.smooth_layer + " " + " ".join(m.balance_layers) for m in mappings)

    # 3 mappings per layer (norm->attn/conv, ffn_norm->w1/w3, w3->w2).
    assert len(mappings) == 6
    # Layer 0 is full_attention: operator_norm smooths q/k/v.
    assert any(
        "layers\\.0\\.operator_norm$" in smooth
        and all(f"self_attn\\.{p}_proj$" in " ".join(bal) for p in ("q", "k", "v"))
        for smooth, bal in pairs
    )
    # Layer 1 is a conv layer: operator_norm smooths conv.in_proj.
    assert any(
        "layers\\.1\\.operator_norm$" in smooth and "conv\\.in_proj$" in " ".join(bal)
        for smooth, bal in pairs
    )
    assert any("ffn_norm$" in s and "feed_forward\\.w1$" in " ".join(b) for s, b in pairs)
    assert any("feed_forward\\.w3$" in s and "feed_forward\\.w2$" in " ".join(b) for s, b in pairs)
    # GQA 32/8: a v_proj -> out_proj pair fails with "size of tensor a (512) must
    # match the size of tensor b (2048)" — it must never be emitted.
    assert "out_proj" not in flat


def test_awq_non_lfm2_uses_llmcompressor_defaults(tmp_path, install_backend):
    _, events = install_backend(model_type="qwen3")
    run_export(_awq_plan(tmp_path))
    awq_kwargs = events["awq_modifier"][0]
    assert "mappings" not in awq_kwargs
    assert awq_kwargs["duo_scaling"] == "both"


def test_nvfp4_recipe_is_quantization_only(tmp_path, install_backend):
    _, events = install_backend()
    dataset = _write_task_dataset(tmp_path / "calib.jsonl", 70)
    run_export(_plan(tmp_path, "nvfp4", dataset=str(dataset)))

    call = events["oneshot"][0]
    assert call["pipeline"] == "basic"
    recipe = call["recipe"]
    assert [m.kind for m in recipe] == ["quant"]
    assert recipe[0].scheme == "NVFP4"
    assert recipe[0].targets == ["Linear"]
    assert recipe[0].ignore == ["lm_head"]
    assert not events["awq_modifier"]
    assert events["compressed_saved"][0]["save_compressed"] is True


def test_compressed_keep_intermediate_retains_merged_dir(tmp_path, install_backend):
    install_backend()
    plan = _awq_plan(tmp_path, keep_intermediate=True)
    run_export(plan)
    assert (Path(plan["output"]) / "_merged-16bit" / "model.safetensors").is_file()


# ---------------------------------------------------------------------------
# 5. Calibration
# ---------------------------------------------------------------------------


def test_calibration_defaults_to_run_dataset_and_uses_trainer_renderer(tmp_path, install_backend):
    _, events = install_backend()
    plan = _awq_plan(tmp_path)
    result = run_export(plan)

    rows = events["oneshot"][0]["dataset"]
    assert len(rows) == 80
    # Rendered exactly as _trainer._format_records renders a task record.
    assert rows[0] == {"text": "Task: echo\nInput: in-0\nOutput: out-0</s>"}
    assert result["calibration"] == {"source": plan["dataset"], "count": 80}


def test_calib_flag_overrides_the_run_dataset(tmp_path, install_backend):
    _, events = install_backend()
    override = _write_task_dataset(tmp_path / "override.jsonl", 90)
    plan = _awq_plan(tmp_path, calib=str(override))
    result = run_export(plan)

    assert result["calibration"]["source"] == str(override)
    assert result["calibration"]["count"] == 90
    assert len(events["oneshot"][0]["dataset"]) == 90


def test_calib_samples_caps_the_record_count(tmp_path, install_backend):
    _, events = install_backend()
    plan = _awq_plan(tmp_path, calib_samples=10)
    result = run_export(plan)
    assert len(events["oneshot"][0]["dataset"]) == 10
    assert result["calibration"]["count"] == 10


def test_few_calibration_samples_emits_one_stderr_diagnostic(tmp_path, install_backend, capsys):
    install_backend()
    dataset = _write_task_dataset(tmp_path / "small.jsonl", 8)
    run_export(_plan(tmp_path, "awq", dataset=str(dataset)))
    captured = capsys.readouterr()
    assert captured.out == ""
    warnings = [ln for ln in captured.err.splitlines() if "calibration samples" in ln]
    assert len(warnings) == 1
    assert "8" in warnings[0]


def test_enough_calibration_samples_emits_no_diagnostic(tmp_path, install_backend, capsys):
    install_backend()
    run_export(_awq_plan(tmp_path))
    assert "calibration samples" not in capsys.readouterr().err


def test_quantised_export_without_calibration_is_a_user_error(tmp_path):
    plan = _plan(tmp_path, "awq")
    with pytest.raises(CliError) as excinfo:
        run_export(plan)
    assert excinfo.value.code == 1
    assert "--calib" in excinfo.value.remediation


# ---------------------------------------------------------------------------
# 6. The torch.accelerator.get_memory_info shim
# ---------------------------------------------------------------------------


def test_memory_info_shim_is_applied_on_compressed_tensors_0_16_0():
    torch = SimpleNamespace(
        accelerator=SimpleNamespace(),
        cuda=SimpleNamespace(mem_get_info=lambda device=None: (11, 22)),
    )
    applied = _exporter._apply_memory_info_shim(torch, SimpleNamespace(__version__="0.16.0"))
    assert applied is True
    assert torch.accelerator.get_memory_info() == (11, 22)


@pytest.mark.parametrize(
    "torch_mod, ct_version",
    [
        # Already present (torch >= 2.11) — nothing to shim.
        (
            SimpleNamespace(
                accelerator=SimpleNamespace(get_memory_info=lambda device=None: (1, 2)),
                cuda=SimpleNamespace(mem_get_info=lambda device=None: (9, 9)),
            ),
            "0.16.0",
        ),
        # A different compressed-tensors version — out of the shim's narrow scope.
        (
            SimpleNamespace(
                accelerator=SimpleNamespace(),
                cuda=SimpleNamespace(mem_get_info=lambda device=None: (9, 9)),
            ),
            "0.14.0.1",
        ),
    ],
)
def test_memory_info_shim_is_skipped(torch_mod, ct_version):
    applied = _exporter._apply_memory_info_shim(torch_mod, SimpleNamespace(__version__=ct_version))
    assert applied is False
    if not hasattr(torch_mod.accelerator, "get_memory_info"):
        return
    # When torch already had it, the original (not mem_get_info) is still in place.
    assert torch_mod.accelerator.get_memory_info() == (1, 2)


# ---------------------------------------------------------------------------
# 7. export.json + the adapter-level exports.json index
# ---------------------------------------------------------------------------


def test_export_json_records_every_contract_field(tmp_path, install_backend, monkeypatch):
    install_backend()
    monkeypatch.setenv("UNSLOTH_LLAMA_TAG", "b10909")
    plan = _plan(tmp_path, "gguf", quant=["q4_k_m"])
    result = run_export(plan)

    record = json.loads(Path(result["export_json"]).read_text(encoding="utf-8"))
    assert set(record) == {
        "format",
        "output",
        "quant",
        "base",
        "adapter",
        "files",
        "calibration",
        "versions",
        "timestamp",
    }
    assert record["format"] == "gguf"
    assert record["quant"] == ["q4_k_m"]
    assert record["base"] == plan["base"]
    assert record["adapter"] == plan["adapter"]
    assert record["files"]["LFM2.5-1.2B-Base.Q4_K_M.gguf"] == 8
    assert "export.json" not in record["files"]
    assert record["calibration"] is None
    assert set(record["versions"]) == {
        "unsloth",
        "unsloth_zoo",
        "transformers",
        "peft",
        "llmcompressor",
        "compressed_tensors",
        "llama_cpp_tag",
    }
    assert record["versions"]["llama_cpp_tag"] == "b10909"
    assert record["timestamp"].endswith("+00:00")


def test_exports_index_is_a_list_appended_across_runs(tmp_path, install_backend):
    install_backend()
    adapter = str(_adapter_dir(tmp_path))
    first = _plan(tmp_path, "merged-16bit", adapter=adapter, output=str(tmp_path / "a.partial"))
    second = _plan(tmp_path, "merged-4bit", adapter=adapter, output=str(tmp_path / "b.partial"))
    run_export(first)
    run_export(second)

    index = json.loads((Path(adapter) / "exports.json").read_text(encoding="utf-8"))
    assert isinstance(index, list)
    assert [entry["format"] for entry in index] == ["merged-16bit", "merged-4bit"]


def test_llama_cpp_tag_falls_back_to_the_prebuilt_info_file(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSLOTH_LLAMA_TAG", raising=False)
    home = tmp_path / "home"
    info_dir = home / ".unsloth" / "llama.cpp"
    info_dir.mkdir(parents=True)
    (info_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"id": "b10909-arm64-cpu"}), encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(home))
    assert _exporter._llama_cpp_tag() == "b10909-arm64-cpu"


def test_llama_cpp_tag_is_none_without_env_or_info_file(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSLOTH_LLAMA_TAG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    assert _exporter._llama_cpp_tag() is None


# ---------------------------------------------------------------------------
# PR #20 review fixes (Qodo threads 1, 4, 10, 13)
# ---------------------------------------------------------------------------


def test_base_override_stages_adapter_with_rewritten_config(tmp_path, install_backend):
    """--base that differs from adapter_config's base is what Unsloth actually loads."""
    _, events = install_backend()
    plan = _plan(tmp_path, "merged-16bit")
    plan["base"] = "org/other-base"
    run_export(plan)
    loaded = events["loaded"][0]["model_name"]
    assert loaded.endswith("_adapter-override")
    # Staging dir is cleaned up after the export unless keep_intermediate.
    assert not Path(loaded).exists()


def test_base_matching_adapter_config_loads_adapter_directly(tmp_path, install_backend):
    _, events = install_backend()
    plan = _plan(tmp_path, "merged-16bit")
    run_export(plan)
    assert events["loaded"][0]["model_name"] == plan["adapter"]


def test_find_gguf_rejects_ambiguous_directories(tmp_path):
    (tmp_path / "a.Q4_K_M.gguf").write_bytes(b"GGUF")
    (tmp_path / "a.F16.gguf").write_bytes(b"GGUF")
    with pytest.raises(CliError) as exc_info:
        _find_gguf(tmp_path)
    assert exc_info.value.code == 1
    assert "--model" in (exc_info.value.remediation or "")


def test_find_gguf_returns_the_single_file(tmp_path):
    only = tmp_path / "a.Q4_K_M.gguf"
    only.write_bytes(b"GGUF")
    assert _find_gguf(tmp_path) == only


def test_gguf_export_json_records_the_resolved_default_quant(tmp_path, install_backend):
    install_backend()
    plan = _plan(tmp_path, "gguf")
    plan["quant"] = []
    result = run_export(plan)
    record = json.loads(Path(result["export_json"]).read_text())
    assert record["quant"] == list(DEFAULT_GGUF_QUANT)


def test_append_index_takes_a_lock_file(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    _append_index(adapter, {"format": "gguf"})
    _append_index(adapter, {"format": "awq"})
    assert (adapter / ".exports.json.lock").exists()
    assert [r["format"] for r in json.loads((adapter / "exports.json").read_text())] == [
        "gguf",
        "awq",
    ]


# ---------------------------------------------------------------------------
# 8. run_eval_model — quant selection, batched generation, per-file scoring
# ---------------------------------------------------------------------------


def _eval_suite(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    """Write a task-schema JSONL eval suite from ``(task, input, expected)`` rows."""
    path.write_text(
        "".join(
            json.dumps({"task": t, "input": i, "expected_output": e}) + "\n" for t, i, e in rows
        ),
        encoding="utf-8",
    )
    return path


def _model_dir(tmp_path: Path, name: str = "exported", *, quantized: bool = True) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    config: dict = {"model_type": "lfm2"}
    if quantized:
        config["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
        }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def _eval_backend(tokenizer: _TokenizerStub, model: _EvalModelStub):
    """An ``_EvalBackend`` wired to the shared fake tokenizer/model kit."""

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    class _Loader:
        @staticmethod
        def from_pretrained(*a, **kw):
            return model

    class _TokLoader:
        @staticmethod
        def from_pretrained(*a, **kw):
            return tokenizer

    return _exporter._EvalBackend(
        torch=SimpleNamespace(no_grad=_NoGrad, bfloat16="bfloat16"),
        auto_model_for_causal_lm=_Loader(),
        auto_tokenizer=_TokLoader(),
    )


class TestFindGgufQuant:
    """``--quant`` disambiguates a multi-GGUF export directory."""

    def _two_quants(self, tmp_path: Path) -> tuple[Path, Path]:
        q4 = tmp_path / "Model-Q4_K_M.gguf"
        q8 = tmp_path / "Model-Q8_0.gguf"
        q4.write_bytes(b"GGUF")
        q8.write_bytes(b"GGUF")
        return q4, q8

    def test_quant_selects_the_matching_file_case_insensitively(self, tmp_path: Path) -> None:
        q4, _q8 = self._two_quants(tmp_path)
        assert _find_gguf(tmp_path, "q4_k_m") == q4
        assert _find_gguf(tmp_path, "Q4_K_M") == q4

    def test_quant_can_select_the_other_file(self, tmp_path: Path) -> None:
        _q4, q8 = self._two_quants(tmp_path)
        assert _find_gguf(tmp_path, "q8_0") == q8

    def test_missing_quant_is_a_user_error_listing_the_present_names(self, tmp_path: Path) -> None:
        self._two_quants(tmp_path)
        with pytest.raises(CliError) as exc_info:
            _find_gguf(tmp_path, "q5_k_s")
        err = exc_info.value
        assert err.code == 1
        assert "q5_k_s" in err.message
        assert "Model-Q4_K_M.gguf" in err.remediation
        assert "Model-Q8_0.gguf" in err.remediation

    def test_no_quant_with_several_files_still_raises_the_existing_error(
        self, tmp_path: Path
    ) -> None:
        self._two_quants(tmp_path)
        with pytest.raises(CliError) as exc_info:
            _find_gguf(tmp_path)
        assert exc_info.value.code == 1
        assert "several GGUF files" in exc_info.value.message
        assert "--model" in exc_info.value.remediation

    def test_quant_is_ignored_for_a_single_gguf_directory(self, tmp_path: Path) -> None:
        only = tmp_path / "Model-Q8_0.gguf"
        only.write_bytes(b"GGUF")
        assert _find_gguf(tmp_path, "q4_k_m") == only


class TestRunEvalModel:
    """The ``--model`` seam reports the same per-file + aggregate shape as ``--adapter``."""

    def test_signature_accepts_the_cli_keywords(self) -> None:
        params = inspect.signature(_exporter.run_eval_model).parameters
        assert "suite_paths" in params
        assert "quant" in params
        assert "batch_size" in params
        assert list(params)[0] == "model_dir"

    def test_per_file_entries_aggregate_and_eval_json(self, tmp_path: Path, monkeypatch) -> None:
        directory = _model_dir(tmp_path)
        good = _eval_suite(tmp_path / "good.jsonl", [("reverse", "abc", "cba")])
        bad = _eval_suite(tmp_path / "bad.jsonl", [("reverse", "xyz", "nope")])
        tokenizer = _TokenizerStub(pad_token="<pad>")
        model = _EvalModelStub(tokenizer, {"abc": "cba", "xyz": "zyx"})
        monkeypatch.setattr(
            _exporter, "_load_eval_backend", lambda: _eval_backend(tokenizer, model)
        )

        summary = _exporter.run_eval_model(str(directory), suite_paths=[good, bad])

        assert [f["path"] for f in summary["files"]] == [str(good), str(bad)]
        assert summary["files"][0]["exact_match"] == 1
        assert summary["files"][1]["exact_match"] == 0
        assert summary["total"] == 2
        assert summary["exact_match_pct"] == 50.0
        assert summary["f1"] == 0.5
        assert summary["model_dir"] == str(directory)
        assert summary["quant_method"] == "compressed-tensors"
        assert summary["quant_format"] == "pack-quantized"

        written = json.loads((directory / "eval.json").read_text(encoding="utf-8"))
        assert written["target"] == "model"
        assert written["suite_paths"] == [str(good), str(bad)]
        assert written["written_at"].startswith("20")
        assert written["quant_method"] == "compressed-tensors"
        for key, value in summary.items():
            assert written[key] == value

    def test_eval_json_is_overwritten_on_re_run(self, tmp_path: Path, monkeypatch) -> None:
        directory = _model_dir(tmp_path)
        suite = _eval_suite(tmp_path / "s.jsonl", [("reverse", "abc", "cba")])
        tokenizer = _TokenizerStub(pad_token="<pad>")
        model = _EvalModelStub(tokenizer, {"abc": "cba"})
        monkeypatch.setattr(
            _exporter, "_load_eval_backend", lambda: _eval_backend(tokenizer, model)
        )
        _exporter.run_eval_model(str(directory), suite_paths=[suite])
        bigger = _eval_suite(
            tmp_path / "two.jsonl", [("reverse", "abc", "cba"), ("reverse", "abc", "cba")]
        )
        _exporter.run_eval_model(str(directory), suite_paths=[bigger])

        written = json.loads((directory / "eval.json").read_text(encoding="utf-8"))
        assert written["suite_paths"] == [str(bigger)]
        assert written["total"] == 2

    def test_batched_predictions_exclude_their_own_prompt(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Three prompts of different lengths in one left-padded batch."""
        directory = _model_dir(tmp_path)
        suite = _eval_suite(
            tmp_path / "s.jsonl",
            [
                ("reverse", "abc", "cba"),
                ("reverse", "a much longer input here", "erehtupni"),
                ("reverse", "mid length input", "tupnidim"),
            ],
        )
        tokenizer = _TokenizerStub(pad_token="<pad>")
        model = _EvalModelStub(
            tokenizer,
            {
                "abc": "cba",
                "a much longer input here": "erehtupni",
                "mid length input": "tupnidim",
            },
        )
        monkeypatch.setattr(
            _exporter, "_load_eval_backend", lambda: _eval_backend(tokenizer, model)
        )

        summary = _exporter.run_eval_model(str(directory), suite_paths=[suite], batch_size=8)

        assert model.batch_widths == [3]
        assert tokenizer.padding_side == "left"
        assert [r["prediction"] for r in summary["results"]] == [
            "cba",
            "erehtupni",
            "tupnidim",
        ]
        for entry in summary["results"]:
            prompt = f"Task: {entry['task']}\nInput: {entry['input']}\nOutput:"
            assert not entry["prediction"].startswith(prompt)
        assert summary["exact_match"] == 3

    def test_gguf_quant_selection_picks_the_scored_file(self, tmp_path: Path, monkeypatch) -> None:
        directory = _model_dir(tmp_path, "gguf-out", quantized=False)
        (directory / "Model-Q4_K_M.gguf").write_bytes(b"GGUF")
        (directory / "Model-Q8_0.gguf").write_bytes(b"GGUF")
        suite = _eval_suite(tmp_path / "s.jsonl", [("reverse", "abc", "cba")])
        seen: list[str] = []

        def _fake_completion(gguf: Path, prompt: str, max_tokens: int) -> str:
            seen.append(str(gguf))
            return "cba"

        monkeypatch.setattr(_exporter, "_run_llama_completion", _fake_completion)
        summary = _exporter.run_eval_model(str(directory), suite_paths=[suite], quant="q8_0")

        assert [Path(s).name for s in seen] == ["Model-Q8_0.gguf"]
        assert summary["exact_match"] == 1
        assert summary["f1"] == 1.0
