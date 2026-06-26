"""Tests for ``sloth export`` — adapter → safetensors PEFT layout export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from sloth.cli._commands.export import cmd_export, register
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
