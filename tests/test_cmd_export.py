"""Tests for ``sloth export`` — adapter → safetensors PEFT layout export.

Container / ML-stack decision (risk r4)
----------------------------------------
``export`` is pure-stdlib — it reorganises and validates filesystem artefacts
without loading or converting weights.  The tests below assert that
``sloth.tune.container.launch`` is **never called** during export, documenting
and locking in that decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sloth.cli._commands.export import OPTIONAL_TOKENIZER_FILES, cmd_export, register
from sloth.cli._errors import CliError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PEFT_FILES = ["adapter_config.json", "adapter_model.safetensors"]


def _make_adapter(tmp_path: Path, name: str = "adapter") -> Path:
    """Create a fake adapter directory with standard PEFT files."""
    adapter = tmp_path / name
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00\x01\x02safetensors_magic")
    return adapter


def _make_args(
    *,
    adapter: str,
    format: str = "safetensors",
    output: str | None = None,
    json_mode: bool = False,
) -> argparse.Namespace:
    """Build a minimal Namespace as argparse would produce."""
    return argparse.Namespace(
        adapter=adapter,
        format=format,
        output=output,
        json=json_mode,
    )


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
    """Unknown --format must raise CliError code=1 with hint listing supported formats."""
    adapter = _make_adapter(tmp_path)
    args = _make_args(adapter=str(adapter), format="gguf")
    with pytest.raises(CliError) as exc_info:
        cmd_export(args)
    err = exc_info.value
    assert err.code == 1
    assert "gguf" in err.message or "unsupported" in err.message.lower()
    # The hint must mention the supported format(s) so the agent knows what to use.
    assert "safetensors" in err.remediation.lower()


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
# Happy-path tests
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
# Container / ML-stack decision tests (resolves risk r4)
# ---------------------------------------------------------------------------


def test_export_does_not_launch_container(tmp_path: Path) -> None:
    """export is pure-stdlib — container.launch must NEVER be called.

    This test locks in the risk-r4 decision: export reorganises filesystem
    artefacts (files are already in safetensors format after training) and
    needs no torch/peft/ML-stack, therefore no NGC container is launched.
    """
    adapter = _make_adapter(tmp_path)
    output_dir = tmp_path / "out"
    args = _make_args(adapter=str(adapter), output=str(output_dir))

    with patch("sloth.tune.container.launch") as mock_launch:
        rc = cmd_export(args)

    assert rc == 0
    mock_launch.assert_not_called()


def test_export_does_not_launch_container_on_error(tmp_path: Path) -> None:
    """Even on a validation error (bad adapter dir), no container is launched."""
    args = _make_args(adapter=str(tmp_path / "nonexistent"))

    with patch("sloth.tune.container.launch") as mock_launch:
        with pytest.raises(CliError):
            cmd_export(args)

    mock_launch.assert_not_called()


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


def test_json_error_on_missing_adapter(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """CliError is still raised (not swallowed) in json mode; caller handles rendering."""
    args = _make_args(adapter=str(tmp_path / "missing"), json_mode=True)
    with pytest.raises(CliError) as exc_info:
        cmd_export(args)
    assert exc_info.value.code == 1


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
