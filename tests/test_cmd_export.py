"""Tests for ``sloth export`` — adapter → PEFT/safetensors and container-backed formats.

Two lanes, locked in by these tests:

* ``--format safetensors`` stays **pure stdlib** — it reorganises and validates
  filesystem artefacts without loading or converting weights, so
  ``sloth.tune.container.launch`` is **never called** (risk r4 decision).  The
  container-not-called tests below are deliberately scoped to that format.
* Every other format (``merged-16bit``, ``merged-4bit``, ``gguf``, ``awq``,
  ``nvfp4``) routes host→container exactly like ``eval.py``: validation first,
  then an atomic ``<output>.partial`` write inside the NGC container.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from sloth.cli._commands import export as export_mod
from sloth.cli._commands.export import (
    GGML_QUANTS,
    OPTIONAL_TOKENIZER_FILES,
    SUPPORTED_FORMATS,
    cmd_export,
    register,
)
from sloth.cli._errors import CliError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PEFT_FILES = ["adapter_config.json", "adapter_model.safetensors"]

BASE_ID = "LiquidAI/LFM2.5-1.2B-Base"


def _make_adapter(tmp_path: Path, name: str = "adapter", base: str | None = BASE_ID) -> Path:
    """Create a fake adapter directory with standard PEFT files."""
    adapter = tmp_path / name
    adapter.mkdir(parents=True)
    config: dict[str, object] = {"peft_type": "LORA"}
    if base is not None:
        config["base_model_name_or_path"] = base
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00\x01\x02safetensors_magic")
    return adapter


def _make_args(
    *,
    adapter: str,
    format: str = "safetensors",
    output: str | None = None,
    json_mode: bool = False,
    quant: str | None = None,
    calib: str | None = None,
    calib_samples: int | None = None,
    base: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    keep_intermediate: bool = False,
    in_container: bool = False,
) -> argparse.Namespace:
    """Build a Namespace as argparse would produce."""
    return argparse.Namespace(
        adapter=adapter,
        format=format,
        output=output,
        json=json_mode,
        quant=quant,
        calib=calib,
        calib_samples=calib_samples,
        base=base,
        force=force,
        dry_run=dry_run,
        keep_intermediate=keep_intermediate,
        in_container=in_container,
    )


class _RecordingLaunch:
    """Fake ``container.launch`` that records its call and returns *rc*."""

    def __init__(self, rc: int = 0) -> None:
        self.rc = rc
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, sloth_args, **kwargs):  # noqa: D102 - test double
        self.calls.append((tuple(sloth_args), kwargs))
        return self.rc


def _fake_launch_env(monkeypatch: pytest.MonkeyPatch, rc: int = 0) -> _RecordingLaunch:
    """Install a recording fake launch and return it."""
    fake = _RecordingLaunch(rc=rc)
    monkeypatch.setattr("sloth.tune.container.launch", fake)
    return fake


def _fake_statvfs(monkeypatch: pytest.MonkeyPatch, free_bytes: int) -> None:
    """Make every ``os.statvfs`` report *free_bytes* of available space."""

    class _Res:
        f_frsize = 1
        f_bsize = 1

        def __init__(self, avail: int) -> None:
            self.f_bavail = avail

    monkeypatch.setattr(os, "statvfs", lambda _p: _Res(free_bytes))


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


def test_missing_adapter_dir(tmp_path: Path) -> None:
    """Non-existent adapter directory must raise CliError with code=1."""
    args = _make_args(adapter=str(tmp_path / "does_not_exist"))
    with pytest.raises(CliError) as exc_info:
        cmd_export(args)
    err = exc_info.value
    assert err.code == 1
    assert "adapter" in err.message.lower() or "not found" in err.message.lower()
    assert err.remediation  # a hint must be present


def test_unsupported_format(tmp_path: Path) -> None:
    """Unknown --format must raise CliError code=1 with a hint listing the formats."""
    adapter = _make_adapter(tmp_path)
    args = _make_args(adapter=str(adapter), format="onnx")
    with pytest.raises(CliError) as exc_info:
        cmd_export(args)
    err = exc_info.value
    assert err.code == 1
    assert "onnx" in err.message or "unsupported" in err.message.lower()
    # The hint must list every supported format so the agent knows what to use.
    for fmt in SUPPORTED_FORMATS:
        assert fmt in err.remediation


def test_adapter_dir_missing_peft_files_raises(tmp_path: Path) -> None:
    """An adapter dir lacking the canonical PEFT files must fail, not silently succeed."""
    empty = tmp_path / "empty_adapter"
    empty.mkdir()
    args = _make_args(adapter=str(empty))
    with pytest.raises(CliError) as exc_info:
        cmd_export(args)
    err = exc_info.value
    assert err.code == 1
    assert "adapter_config.json" in err.message
    assert err.remediation


def test_adapter_dir_missing_one_peft_file_raises(tmp_path: Path) -> None:
    """A partial adapter (config present, weights missing) must also fail."""
    partial = tmp_path / "partial_adapter"
    partial.mkdir()
    (partial / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
    args = _make_args(adapter=str(partial))
    with pytest.raises(CliError) as exc_info:
        cmd_export(args)
    err = exc_info.value
    assert err.code == 1
    assert "adapter_model.safetensors" in err.message


# ---------------------------------------------------------------------------
# Happy-path tests (safetensors — unchanged, pure stdlib)
# ---------------------------------------------------------------------------


def test_happy_path_safetensors(tmp_path: Path) -> None:
    """Standard PEFT files in adapter dir are written to a separate output dir."""
    adapter = _make_adapter(tmp_path)
    output_dir = tmp_path / "exported"
    args = _make_args(adapter=str(adapter), output=str(output_dir))

    rc = cmd_export(args)

    assert rc == 0
    assert output_dir.is_dir()
    for fname in PEFT_FILES:
        assert (output_dir / fname).exists(), f"expected {fname} in output dir"


def test_happy_path_output_defaults_to_adapter_dir(tmp_path: Path) -> None:
    """When --output is omitted, files remain in the adapter dir (normalise in place)."""
    adapter = _make_adapter(tmp_path)
    args = _make_args(adapter=str(adapter))  # no --output

    rc = cmd_export(args)

    assert rc == 0
    for fname in PEFT_FILES:
        assert (adapter / fname).exists(), f"expected {fname} still in adapter dir"


def test_happy_path_creates_output_dir(tmp_path: Path) -> None:
    """Output dir is created if it does not exist yet."""
    adapter = _make_adapter(tmp_path)
    deep_output = tmp_path / "deep" / "nested" / "out"
    args = _make_args(adapter=str(adapter), output=str(deep_output))

    cmd_export(args)

    assert deep_output.is_dir()


def test_safetensors_into_non_empty_output_without_force(tmp_path: Path) -> None:
    """safetensors keeps today's behaviour byte-for-byte: no no-clobber guard."""
    adapter = _make_adapter(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("previous", encoding="utf-8")

    rc = cmd_export(_make_args(adapter=str(adapter), output=str(output_dir)))

    assert rc == 0
    for fname in PEFT_FILES:
        assert (output_dir / fname).exists()


# ---------------------------------------------------------------------------
# Tokenizer file tests
# ---------------------------------------------------------------------------


def test_tokenizer_files_copied_when_present(tmp_path: Path) -> None:
    """Optional tokenizer files present in the adapter dir are copied to the output dir."""
    adapter = _make_adapter(tmp_path)
    # Add a subset of tokenizer files that the adapter might bundle.
    tokenizer_subset = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    for fname in tokenizer_subset:
        (adapter / fname).write_text('{"model_type": "qwen2"}', encoding="utf-8")

    output_dir = tmp_path / "out"
    args = _make_args(adapter=str(adapter), output=str(output_dir))

    rc = cmd_export(args)

    assert rc == 0
    for fname in tokenizer_subset:
        assert (output_dir / fname).exists(), f"expected tokenizer file {fname} in output"


def test_tokenizer_files_skipped_when_absent(tmp_path: Path) -> None:
    """Export succeeds with no tokenizer files — they are optional, not required."""
    adapter = _make_adapter(tmp_path)
    # Confirm none of the optional files exist.
    for fname in OPTIONAL_TOKENIZER_FILES:
        assert not (adapter / fname).exists()

    output_dir = tmp_path / "out"
    args = _make_args(adapter=str(adapter), output=str(output_dir))

    rc = cmd_export(args)

    assert rc == 0
    # Required PEFT files must be there; optional tokenizer files must NOT be created.
    for fname in PEFT_FILES:
        assert (output_dir / fname).exists()
    for fname in OPTIONAL_TOKENIZER_FILES:
        assert not (output_dir / fname).exists(), f"{fname} should not appear when absent in src"


def test_sentencepiece_tokenizer_model_copied(tmp_path: Path) -> None:
    """tokenizer.model (SentencePiece) is copied when present alongside PEFT files."""
    adapter = _make_adapter(tmp_path)
    (adapter / "tokenizer.model").write_bytes(b"FAKE_SENTENCEPIECE_BLOB")

    output_dir = tmp_path / "out"
    rc = cmd_export(_make_args(adapter=str(adapter), output=str(output_dir)))

    assert rc == 0
    assert (output_dir / "tokenizer.model").exists()
    assert (output_dir / "tokenizer.model").read_bytes() == b"FAKE_SENTENCEPIECE_BLOB"


def test_tokenizer_files_in_json_files_list(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """tokenizer files present in the adapter appear in the JSON `files` list."""
    adapter = _make_adapter(tmp_path)
    (adapter / "tokenizer.json").write_text("{}", encoding="utf-8")
    (adapter / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    output_dir = tmp_path / "out"
    args = _make_args(adapter=str(adapter), output=str(output_dir), json_mode=True)

    rc = cmd_export(args)
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    fnames_in_output = [Path(f).name for f in payload["files"]]
    assert "tokenizer.json" in fnames_in_output
    assert "tokenizer_config.json" in fnames_in_output


# ---------------------------------------------------------------------------
# Container / ML-stack decision tests (resolves risk r4) — safetensors only
# ---------------------------------------------------------------------------


def test_export_does_not_launch_container(tmp_path: Path) -> None:
    """``--format safetensors`` is pure-stdlib — container.launch must NEVER be called.

    This test locks in the risk-r4 decision for the safetensors lane: the files
    are already in safetensors format after training, so no NGC container is
    launched.  Other formats DO launch the container (see the routing tests).
    """
    adapter = _make_adapter(tmp_path)
    output_dir = tmp_path / "out"
    args = _make_args(adapter=str(adapter), output=str(output_dir), format="safetensors")

    with patch("sloth.tune.container.launch") as mock_launch:
        rc = cmd_export(args)

    assert rc == 0
    mock_launch.assert_not_called()


def test_export_does_not_launch_container_on_error(tmp_path: Path) -> None:
    """Even on a validation error (bad adapter dir), no container is launched."""
    args = _make_args(adapter=str(tmp_path / "nonexistent"), format="safetensors")

    with patch("sloth.tune.container.launch") as mock_launch:
        with pytest.raises(CliError):
            cmd_export(args)

    mock_launch.assert_not_called()


def test_module_never_imports_container_at_module_level() -> None:
    """``sloth.tune.container`` must not be imported at export.py module level."""
    source = Path(export_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module or "")
    assert not any("container" in name for name in top_level_imports), top_level_imports
    assert not any("_exporter" in name for name in top_level_imports), top_level_imports


# ---------------------------------------------------------------------------
# JSON output tests
# ---------------------------------------------------------------------------


def test_json_output(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """--json emits a structured payload to stdout with output path, format, files."""
    adapter = _make_adapter(tmp_path)
    output_dir = tmp_path / "out"
    args = _make_args(adapter=str(adapter), output=str(output_dir), json_mode=True)

    rc = cmd_export(args)

    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["format"] == "safetensors"
    assert "output" in payload
    assert isinstance(payload["files"], list)
    assert len(payload["files"]) > 0
    # Each file path in the list must be a string pointing inside the output dir.
    for fpath in payload["files"]:
        assert str(output_dir) in fpath


def test_json_error_on_missing_adapter(tmp_path: Path) -> None:
    """CliError is still raised (not swallowed) in json mode; caller handles rendering."""
    args = _make_args(adapter=str(tmp_path / "missing"), json_mode=True)
    with pytest.raises(CliError) as exc_info:
        cmd_export(args)
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Format set + flag validation
# ---------------------------------------------------------------------------


def test_supported_formats_set() -> None:
    """The six shipped formats."""
    assert set(SUPPORTED_FORMATS) == {
        "safetensors",
        "merged-16bit",
        "merged-4bit",
        "gguf",
        "awq",
        "nvfp4",
    }


def test_ggml_quant_allowlist() -> None:
    """The ggml quant allowlist is the documented Unsloth list."""
    assert set(GGML_QUANTS) == {
        "q4_k_m",
        "q5_k_m",
        "q8_0",
        "f16",
        "q2_k",
        "q3_k_m",
        "q4_0",
        "q4_1",
        "q5_0",
        "q6_k",
    }


def test_invalid_quant_rejected_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An off-allowlist --quant value exits 1 with a hint and never launches a container."""
    adapter = _make_adapter(tmp_path)
    fake = _fake_launch_env(monkeypatch)
    args = _make_args(
        adapter=str(adapter),
        format="gguf",
        output=str(tmp_path / "out"),
        quant="q4_k_m,q9_wrong",
    )
    with pytest.raises(CliError) as exc_info:
        cmd_export(args)
    err = exc_info.value
    assert err.code == 1
    assert "q9_wrong" in err.message
    assert "q4_k_m" in err.remediation
    assert fake.calls == []


def test_quant_list_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--quant is a comma list, normalised to lowercase and forwarded."""
    adapter = _make_adapter(tmp_path)
    fake = _fake_launch_env(monkeypatch)
    _fake_statvfs(monkeypatch, 10**12)
    args = _make_args(
        adapter=str(adapter),
        format="gguf",
        output=str(tmp_path / "out"),
        quant=" Q4_K_M , q8_0 ",
    )
    assert cmd_export(args) == 0
    sloth_args = fake.calls[0][0]
    assert "--quant" in sloth_args
    assert sloth_args[sloth_args.index("--quant") + 1] == "q4_k_m,q8_0"


def test_missing_calib_file_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A --calib path that does not exist exits 1 with a hint, before any launch."""
    adapter = _make_adapter(tmp_path)
    fake = _fake_launch_env(monkeypatch)
    args = _make_args(
        adapter=str(adapter),
        format="awq",
        output=str(tmp_path / "out"),
        calib=str(tmp_path / "nope.jsonl"),
    )
    with pytest.raises(CliError) as exc_info:
        cmd_export(args)
    assert exc_info.value.code == 1
    assert exc_info.value.remediation
    assert fake.calls == []


def test_non_positive_calib_samples_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--calib-samples must be a positive integer."""
    adapter = _make_adapter(tmp_path)
    fake = _fake_launch_env(monkeypatch)
    args = _make_args(
        adapter=str(adapter),
        format="nvfp4",
        output=str(tmp_path / "out"),
        calib_samples=0,
    )
    with pytest.raises(CliError) as exc_info:
        cmd_export(args)
    assert exc_info.value.code == 1
    assert exc_info.value.remediation
    assert fake.calls == []


def test_base_defaults_to_adapter_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--base defaults to adapter_config.json's base_model_name_or_path."""
    adapter = _make_adapter(tmp_path)
    fake = _fake_launch_env(monkeypatch)
    _fake_statvfs(monkeypatch, 10**12)
    args = _make_args(adapter=str(adapter), format="merged-16bit", output=str(tmp_path / "out"))
    assert cmd_export(args) == 0
    sloth_args = fake.calls[0][0]
    assert sloth_args[sloth_args.index("--base") + 1] == BASE_ID


def test_base_flag_overrides_adapter_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit --base wins over adapter_config.json."""
    adapter = _make_adapter(tmp_path)
    fake = _fake_launch_env(monkeypatch)
    _fake_statvfs(monkeypatch, 10**12)
    args = _make_args(
        adapter=str(adapter),
        format="merged-16bit",
        output=str(tmp_path / "out"),
        base="unsloth/Qwen3-4B",
    )
    assert cmd_export(args) == 0
    sloth_args = fake.calls[0][0]
    assert sloth_args[sloth_args.index("--base") + 1] == "unsloth/Qwen3-4B"


def test_unresolvable_base_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No --base and no base_model_name_or_path exits 1 with a hint, before any launch."""
    adapter = _make_adapter(tmp_path, base=None)
    fake = _fake_launch_env(monkeypatch)
    args = _make_args(adapter=str(adapter), format="gguf", output=str(tmp_path / "out"))
    with pytest.raises(CliError) as exc_info:
        cmd_export(args)
    assert exc_info.value.code == 1
    assert "--base" in exc_info.value.remediation
    assert fake.calls == []


def test_container_format_requires_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A container-backed format needs an explicit --output (never in-place)."""
    adapter = _make_adapter(tmp_path)
    fake = _fake_launch_env(monkeypatch)
    args = _make_args(adapter=str(adapter), format="merged-16bit")
    with pytest.raises(CliError) as exc_info:
        cmd_export(args)
    assert exc_info.value.code == 1
    assert "--output" in exc_info.value.remediation
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

_PLAN_KEYS = [
    "format",
    "quant",
    "base",
    "output",
    "estimated_bytes",
    "free_bytes",
    "docker_command",
]


@pytest.mark.parametrize(
    "fmt", ["safetensors", "merged-16bit", "merged-4bit", "gguf", "awq", "nvfp4"]
)
def test_dry_run_every_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, fmt: str
) -> None:
    """--dry-run exits 0 for every format, launches nothing, and carries the plan keys."""
    adapter = _make_adapter(tmp_path)
    fake = _fake_launch_env(monkeypatch)
    _fake_statvfs(monkeypatch, 10**12)
    output_dir = tmp_path / "out"
    args = _make_args(
        adapter=str(adapter),
        format=fmt,
        output=str(output_dir),
        dry_run=True,
        json_mode=True,
        quant="q4_k_m" if fmt == "gguf" else None,
    )

    rc = cmd_export(args)

    assert rc == 0
    assert fake.calls == []
    payload = json.loads(capsys.readouterr().out)
    for key in _PLAN_KEYS:
        assert key in payload, f"{key} missing from the dry-run plan"
    assert payload["format"] == fmt
    assert payload["dry_run"] is True
    # Nothing is written by a dry-run.
    assert not output_dir.exists()


def test_dry_run_docker_command_for_container_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A container-backed dry-run carries a real, shell-quoted docker command."""
    adapter = _make_adapter(tmp_path)
    _fake_launch_env(monkeypatch)
    _fake_statvfs(monkeypatch, 10**12)
    args = _make_args(
        adapter=str(adapter),
        format="gguf",
        output=str(tmp_path / "out"),
        quant="q4_k_m",
        dry_run=True,
        json_mode=True,
    )
    assert cmd_export(args) == 0
    payload = json.loads(capsys.readouterr().out)
    cmd = payload["docker_command"]
    assert isinstance(cmd, str)
    assert cmd.startswith("docker run")
    assert "--in-container" in cmd
    assert payload["quant"] == ["q4_k_m"]
    assert isinstance(payload["estimated_bytes"], int)
    assert payload["free_bytes"] == 10**12


def test_dry_run_safetensors_has_no_docker_command(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """safetensors runs on the host, so its plan's docker_command is null."""
    adapter = _make_adapter(tmp_path)
    args = _make_args(
        adapter=str(adapter),
        format="safetensors",
        output=str(tmp_path / "o"),
        dry_run=True,
        json_mode=True,
    )
    assert cmd_export(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["docker_command"] is None


def test_dry_run_text_mode(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Text-mode dry-run renders the plan to stdout."""
    adapter = _make_adapter(tmp_path)
    args = _make_args(
        adapter=str(adapter), format="safetensors", output=str(tmp_path / "o"), dry_run=True
    )
    assert cmd_export(args) == 0
    out = capsys.readouterr().out
    assert "plan: dry-run" in out
    assert "format:" in out
    assert "estimated-bytes:" in out


def test_estimated_bytes_scales_with_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gguf (F16 intermediate) estimates more than merged-16bit, which beats nvfp4."""
    adapter = _make_adapter(tmp_path, base="unsloth/Qwen3-4B")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-cache"))  # force the id heuristic
    estimates = {}
    for fmt in ("merged-16bit", "gguf", "nvfp4"):
        estimates[fmt] = export_mod._estimate_bytes("unsloth/Qwen3-4B", fmt, adapter)
    assert estimates["gguf"] > estimates["merged-16bit"] > estimates["nvfp4"]
    # 4B params at 2 bytes/param for the merged bf16 artifact.
    assert estimates["merged-16bit"] == pytest.approx(8e9, rel=0.01)


def test_estimate_prefers_cached_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached config.json parameter count wins over the model-id heuristic."""
    snapshot = tmp_path / "hub" / "models--acme--tiny-9000b" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "intermediate_size": 128,
                "vocab_size": 1000,
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    est = export_mod._estimate_bytes("acme/tiny-9000b", "merged-16bit", tmp_path)
    # The id heuristic would say 9000e9 params; the config says ~0.2M.
    assert est is not None and est < 10**8


# ---------------------------------------------------------------------------
# Disk gate
# ---------------------------------------------------------------------------


def test_real_run_fails_closed_on_insufficient_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """free_bytes < estimated_bytes exits 2 with a hint and never launches a container."""
    adapter = _make_adapter(tmp_path, base="unsloth/Qwen3-4B")
    fake = _fake_launch_env(monkeypatch)
    _fake_statvfs(monkeypatch, 1024)  # 1 KiB free, a 4B model needs GBs
    args = _make_args(adapter=str(adapter), format="merged-16bit", output=str(tmp_path / "out"))

    with pytest.raises(CliError) as exc_info:
        cmd_export(args)

    err = exc_info.value
    assert err.code == 2
    assert err.remediation
    assert fake.calls == []


def test_dry_run_does_not_fail_on_insufficient_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A dry-run reports the shortfall instead of failing."""
    adapter = _make_adapter(tmp_path, base="unsloth/Qwen3-4B")
    _fake_statvfs(monkeypatch, 1024)
    args = _make_args(
        adapter=str(adapter),
        format="merged-16bit",
        output=str(tmp_path / "out"),
        dry_run=True,
        json_mode=True,
    )
    assert cmd_export(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["free_bytes"] == 1024
    assert payload["estimated_bytes"] > 1024


# ---------------------------------------------------------------------------
# No-clobber + atomic output
# ---------------------------------------------------------------------------


def test_non_empty_output_without_force_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-empty --output without --force exits 1 with a hint, before any launch."""
    adapter = _make_adapter(tmp_path)
    fake = _fake_launch_env(monkeypatch)
    _fake_statvfs(monkeypatch, 10**12)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "model.safetensors").write_bytes(b"old")

    args = _make_args(adapter=str(adapter), format="merged-16bit", output=str(output_dir))
    with pytest.raises(CliError) as exc_info:
        cmd_export(args)

    assert exc_info.value.code == 1
    assert "--force" in exc_info.value.remediation
    assert fake.calls == []


def test_non_empty_output_with_force_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--force allows overwriting a non-empty output directory."""
    adapter = _make_adapter(tmp_path)
    fake = _fake_launch_env(monkeypatch)
    _fake_statvfs(monkeypatch, 10**12)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "model.safetensors").write_bytes(b"old")

    def _launch(sloth_args, **kwargs):
        fake.calls.append((tuple(sloth_args), kwargs))
        partial = Path(sloth_args[sloth_args.index("--output") + 1])
        (partial / "model.safetensors").write_bytes(b"new")
        return 0

    monkeypatch.setattr("sloth.tune.container.launch", _launch)

    args = _make_args(
        adapter=str(adapter), format="merged-16bit", output=str(output_dir), force=True
    )
    assert cmd_export(args) == 0
    assert (output_dir / "model.safetensors").read_bytes() == b"new"
    assert not Path(str(output_dir) + ".partial").exists()


def test_container_writes_partial_and_host_renames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The container writes <output>.partial; the host renames it on exit 0."""
    adapter = _make_adapter(tmp_path)
    _fake_statvfs(monkeypatch, 10**12)
    output_dir = tmp_path / "out"
    seen: dict[str, str] = {}

    def _launch(sloth_args, **kwargs):
        partial = Path(sloth_args[sloth_args.index("--output") + 1])
        seen["output_arg"] = str(partial)
        assert partial.is_dir(), "the host must create <output>.partial before launching"
        (partial / "model.safetensors").write_bytes(b"merged")
        return 0

    monkeypatch.setattr("sloth.tune.container.launch", _launch)

    args = _make_args(adapter=str(adapter), format="merged-16bit", output=str(output_dir))
    assert cmd_export(args) == 0
    assert seen["output_arg"] == str(output_dir) + ".partial"
    assert (output_dir / "model.safetensors").read_bytes() == b"merged"
    assert not Path(str(output_dir) + ".partial").exists()


def test_killed_container_leaves_partial_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launcher reporting 137 (SIGKILL/OOM) leaves <output> absent and .partial present."""
    adapter = _make_adapter(tmp_path)
    _fake_statvfs(monkeypatch, 10**12)
    output_dir = tmp_path / "out"

    def _launch(sloth_args, **kwargs):
        partial = Path(sloth_args[sloth_args.index("--output") + 1])
        (partial / "half-written.safetensors").write_bytes(b"partial")
        return 137

    monkeypatch.setattr("sloth.tune.container.launch", _launch)

    args = _make_args(adapter=str(adapter), format="merged-16bit", output=str(output_dir))
    with pytest.raises(CliError) as exc_info:
        cmd_export(args)

    assert exc_info.value.code == 2
    assert not output_dir.exists()
    partial = Path(str(output_dir) + ".partial")
    assert partial.is_dir()
    assert (partial / "half-written.safetensors").exists()


def test_launch_error_leaves_partial_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A CliError from launch propagates and leaves <output> absent, .partial present."""
    adapter = _make_adapter(tmp_path)
    _fake_statvfs(monkeypatch, 10**12)
    output_dir = tmp_path / "out"

    def _launch(sloth_args, **kwargs):
        partial = Path(sloth_args[sloth_args.index("--output") + 1])
        (partial / "half.bin").write_bytes(b"x")
        raise CliError(code=2, message="Container was killed (exit 137)", remediation="free memory")

    monkeypatch.setattr("sloth.tune.container.launch", _launch)

    args = _make_args(adapter=str(adapter), format="merged-16bit", output=str(output_dir))
    with pytest.raises(CliError) as exc_info:
        cmd_export(args)

    assert exc_info.value.code == 2
    assert not output_dir.exists()
    assert Path(str(output_dir) + ".partial", "half.bin").exists()


def test_stale_partial_is_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A leftover .partial from a previous killed run is cleared before relaunching."""
    adapter = _make_adapter(tmp_path)
    _fake_statvfs(monkeypatch, 10**12)
    output_dir = tmp_path / "out"
    stale = Path(str(output_dir) + ".partial")
    stale.mkdir()
    (stale / "stale.bin").write_bytes(b"old")

    def _launch(sloth_args, **kwargs):
        partial = Path(sloth_args[sloth_args.index("--output") + 1])
        assert list(partial.iterdir()) == []
        (partial / "fresh.bin").write_bytes(b"new")
        return 0

    monkeypatch.setattr("sloth.tune.container.launch", _launch)

    args = _make_args(adapter=str(adapter), format="merged-16bit", output=str(output_dir))
    assert cmd_export(args) == 0
    assert (output_dir / "fresh.bin").exists()
    assert not (output_dir / "stale.bin").exists()


# ---------------------------------------------------------------------------
# Host → container routing
# ---------------------------------------------------------------------------


def test_host_routing_mirrors_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Identity mounts of the adapter/output/calib parents, repo checkout, recursion guard."""
    adapter = _make_adapter(tmp_path / "adapters", name="a")
    calib = tmp_path / "calib" / "c.jsonl"
    calib.parent.mkdir(parents=True)
    calib.write_text('{"messages": []}\n', encoding="utf-8")
    fake = _fake_launch_env(monkeypatch)
    _fake_statvfs(monkeypatch, 10**12)
    output_dir = tmp_path / "exports" / "awq"

    args = _make_args(
        adapter=str(adapter),
        format="awq",
        output=str(output_dir),
        calib=str(calib),
        calib_samples=128,
    )
    assert cmd_export(args) == 0

    sloth_args, kwargs = fake.calls[0]
    assert sloth_args[0] == "export"
    assert "--in-container" in sloth_args
    assert sloth_args[sloth_args.index("--adapter") + 1] == str(adapter.resolve())
    assert sloth_args[sloth_args.index("--calib") + 1] == str(calib.resolve())
    assert sloth_args[sloth_args.index("--calib-samples") + 1] == "128"
    assert kwargs["checkout"] == str(export_mod._repo_root())
    mounts = kwargs["extra_mounts"]
    for parent in (adapter.resolve().parent, output_dir.resolve().parent, calib.resolve().parent):
        assert (str(parent), str(parent)) in mounts


def test_json_flag_forwarded_to_container(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--json is forwarded so the in-container run emits JSON on stdout."""
    adapter = _make_adapter(tmp_path)
    fake = _fake_launch_env(monkeypatch)
    _fake_statvfs(monkeypatch, 10**12)
    args = _make_args(
        adapter=str(adapter), format="merged-4bit", output=str(tmp_path / "o"), json_mode=True
    )
    assert cmd_export(args) == 0
    assert "--json" in fake.calls[0][0]


def test_keep_intermediate_forwarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--keep-intermediate is forwarded to the in-container run."""
    adapter = _make_adapter(tmp_path)
    fake = _fake_launch_env(monkeypatch)
    _fake_statvfs(monkeypatch, 10**12)
    args = _make_args(
        adapter=str(adapter),
        format="gguf",
        output=str(tmp_path / "o"),
        quant="q4_k_m",
        keep_intermediate=True,
    )
    assert cmd_export(args) == 0
    assert "--keep-intermediate" in fake.calls[0][0]


def test_export_launch_kwargs_env_and_mounts_passed_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """container.export_launch_kwargs() env + extra_mounts reach launch, mounts deduped."""
    adapter = _make_adapter(tmp_path)
    fake = _fake_launch_env(monkeypatch)
    _fake_statvfs(monkeypatch, 10**12)
    cache = tmp_path / "llama-cache"
    adapter_parent = str(adapter.resolve().parent)

    import sloth.tune.container as container_mod

    monkeypatch.setattr(
        container_mod,
        "export_launch_kwargs",
        lambda: {
            "env": [("UNSLOTH_LLAMA_TAG", "b10909"), ("HOME", "/root")],
            # One new mount plus a duplicate of one this command already adds.
            "extra_mounts": [(str(cache), str(cache)), (adapter_parent, adapter_parent)],
        },
        raising=False,
    )

    args = _make_args(
        adapter=str(adapter), format="gguf", output=str(tmp_path / "o"), quant="q4_k_m"
    )
    assert cmd_export(args) == 0

    _, kwargs = fake.calls[0]
    assert kwargs["env"] == [("UNSLOTH_LLAMA_TAG", "b10909"), ("HOME", "/root")]
    mounts = kwargs["extra_mounts"]
    assert (str(cache), str(cache)) in mounts
    targets = [target for _host, target in mounts]
    assert len(targets) == len(set(targets)), f"duplicate mount targets: {mounts}"


def test_missing_export_launch_kwargs_is_tolerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command works against a container module without export_launch_kwargs (t8 pending)."""
    adapter = _make_adapter(tmp_path)
    fake = _fake_launch_env(monkeypatch)
    _fake_statvfs(monkeypatch, 10**12)

    import sloth.tune.container as container_mod

    monkeypatch.delattr(container_mod, "export_launch_kwargs", raising=False)

    args = _make_args(adapter=str(adapter), format="nvfp4", output=str(tmp_path / "o"))
    assert cmd_export(args) == 0
    assert "env" not in fake.calls[0][1]


# ---------------------------------------------------------------------------
# In-container branch (t9 seam)
# ---------------------------------------------------------------------------


def _install_fake_exporter(monkeypatch: pytest.MonkeyPatch, summary: dict) -> list[dict]:
    """Install a fake ``sloth.tune._exporter`` module; return the recorded plans."""
    seen: list[dict] = []
    module = types.ModuleType("sloth.tune._exporter")

    def run_export(plan):
        seen.append(plan)
        return summary

    module.run_export = run_export  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sloth.tune._exporter", module)
    return seen


def test_in_container_calls_run_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """--in-container delegates to sloth.tune._exporter.run_export with the plan dict."""
    adapter = _make_adapter(tmp_path)
    (adapter / "training_metadata.json").write_text(
        json.dumps({"dataset": {"path": str(tmp_path / "train.jsonl")}}), encoding="utf-8"
    )
    calib = tmp_path / "calib.jsonl"
    calib.write_text('{"messages": []}\n', encoding="utf-8")
    summary = {"files": {"model.safetensors": 10}, "bytes": 10, "export_json": "/x/export.json"}
    seen = _install_fake_exporter(monkeypatch, summary)
    fake = _fake_launch_env(monkeypatch)

    output_dir = tmp_path / "out.partial"
    args = _make_args(
        adapter=str(adapter),
        format="gguf",
        output=str(output_dir),
        quant="q4_k_m",
        calib=str(calib),
        calib_samples=64,
        keep_intermediate=True,
        json_mode=True,
        in_container=True,
    )

    assert cmd_export(args) == 0
    assert fake.calls == [], "the in-container branch must never launch another container"

    plan = seen[0]
    assert plan["format"] == "gguf"
    assert plan["quant"] == ["q4_k_m"]
    assert plan["base"] == BASE_ID
    assert plan["adapter"] == str(adapter.resolve())
    assert plan["output"] == str(output_dir.resolve())
    assert plan["calib"] == str(calib.resolve())
    assert plan["calib_samples"] == 64
    assert plan["keep_intermediate"] is True
    assert plan["dataset"] == str(tmp_path / "train.jsonl")

    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "gguf"
    assert payload["files"] == {"model.safetensors": 10}


def test_in_container_dataset_none_without_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No training_metadata.json → plan["dataset"] is None, not an error."""
    adapter = _make_adapter(tmp_path)
    seen = _install_fake_exporter(monkeypatch, {"files": {}, "bytes": 0})
    args = _make_args(
        adapter=str(adapter), format="awq", output=str(tmp_path / "o"), in_container=True
    )
    assert cmd_export(args) == 0
    assert seen[0]["dataset"] is None
    assert seen[0]["quant"] == []


# ---------------------------------------------------------------------------
# Subparser registration tests
# ---------------------------------------------------------------------------


def test_register_subparser(tmp_path: Path) -> None:
    """register() wires the export subparser with expected defaults."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)

    adapter_path = str(tmp_path / "my_adapter")
    args = parser.parse_args(["export", "--adapter", adapter_path])

    assert args.command == "export"
    assert args.adapter == adapter_path
    assert args.format == "safetensors"  # default
    assert args.output is None  # optional, defaults to None
    assert args.json is False  # default
    assert args.quant is None
    assert args.calib is None
    assert args.calib_samples is None
    assert args.base is None
    assert args.force is False
    assert args.dry_run is False
    assert args.keep_intermediate is False
    assert args.in_container is False
    assert callable(args.func)


def test_register_subparser_custom_format(tmp_path: Path) -> None:
    """--format flag is parsed through the subparser."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)

    args = parser.parse_args(["export", "--adapter", str(tmp_path), "--format", "safetensors"])
    assert args.format == "safetensors"


def test_register_subparser_json_flag(tmp_path: Path) -> None:
    """--json flag is parsed correctly by the subparser."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)

    args = parser.parse_args(["export", "--adapter", str(tmp_path), "--json"])
    assert args.json is True


def test_register_subparser_new_flags(tmp_path: Path) -> None:
    """The new export flags are all parsed."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)

    args = parser.parse_args(
        [
            "export",
            "--adapter",
            str(tmp_path),
            "--format",
            "gguf",
            "--quant",
            "q4_k_m,q8_0",
            "--calib",
            "calib.jsonl",
            "--calib-samples",
            "256",
            "--base",
            "unsloth/Qwen3-4B",
            "--force",
            "--dry-run",
            "--keep-intermediate",
            "--in-container",
        ]
    )
    assert args.format == "gguf"
    assert args.quant == "q4_k_m,q8_0"
    assert args.calib == "calib.jsonl"
    assert args.calib_samples == 256
    assert args.base == "unsloth/Qwen3-4B"
    assert args.force is True
    assert args.dry_run is True
    assert args.keep_intermediate is True
    assert args.in_container is True
