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
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sloth.cli._errors import CliError
from sloth.tune import _exporter
from sloth.tune._exporter import run_export

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


@pytest.fixture()
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
    with pytest.raises(CliError) as excinfo:
        run_export(_plan(tmp_path, "merged-16bit"))
    assert excinfo.value.code == 2
    assert "nvcr.io/nvidia/pytorch" in excinfo.value.remediation


def test_backend_import_oom_raises_cli_error_with_memory_hint(tmp_path, monkeypatch):
    class OutOfMemoryError(RuntimeError):
        pass

    def _boom(**_):
        raise OutOfMemoryError("CUDA error: out of memory")

    monkeypatch.setattr(_exporter, "_load_backend", _boom)
    with pytest.raises(CliError) as excinfo:
        run_export(_plan(tmp_path, "merged-16bit"))
    assert excinfo.value.code == 2
    assert "out of memory" in excinfo.value.message.lower()
    assert "drop_caches" in excinfo.value.remediation


def test_export_oom_during_run_raises_cli_error(tmp_path, install_backend):
    backend, _ = install_backend()

    def _boom(**kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")

    backend.fast_model.from_pretrained = _boom
    with pytest.raises(CliError) as excinfo:
        run_export(_plan(tmp_path, "merged-16bit"))
    assert excinfo.value.code == 2
    assert "drop_caches" in excinfo.value.remediation


def test_unknown_format_is_a_user_error(tmp_path):
    with pytest.raises(CliError) as excinfo:
        run_export(_plan(tmp_path, "ggml"))
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
    with pytest.raises(CliError) as excinfo:
        run_export(_plan(tmp_path, "awq"))
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
