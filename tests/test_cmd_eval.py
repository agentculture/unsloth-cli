"""Tests for ``sloth eval`` command.

Covers:
* missing --adapter dir  → CliError(code=1) with ``error:`` / ``hint:`` two-line contract
* missing --suite file   → CliError(code=1) with ``error:`` / ``hint:`` two-line contract
* valid adapter + suite  → results emitted in text mode and --json mode (mocked backend)
* no-network assertion   → monkeypatching ``socket.socket`` to raise proves no
  network call escapes the mocked code path
"""

from __future__ import annotations

import argparse
import io
import json
import socket
from pathlib import Path
from typing import Any

import pytest

import sloth.cli._commands.eval as eval_mod
from sloth.cli._commands.eval import cmd_eval, register
from sloth.cli._errors import CliError
from sloth.cli._output import emit_error

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs: Any) -> argparse.Namespace:
    """Build a minimal Namespace, defaulting ``json`` to False."""
    defaults: dict[str, Any] = {"json": False}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _mock_backend(adapter_path: str, record: dict[str, str]) -> str:  # noqa: ARG001
    """Always returns the expected output — perfect score for testing."""
    return record["expected_output"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_adapter(tmp_path: Path) -> Path:
    """A minimal adapter directory (just needs to exist as a directory)."""
    d = tmp_path / "adapter"
    d.mkdir()
    return d


@pytest.fixture()
def tmp_suite(tmp_path: Path) -> Path:
    """A two-record task-schema JSONL eval suite."""
    f = tmp_path / "suite.jsonl"
    f.write_text(
        '{"task": "reverse", "input": "abc", "expected_output": "cba"}\n'
        '{"task": "upper", "input": "hello", "expected_output": "HELLO"}\n',
        encoding="utf-8",
    )
    return f


# ---------------------------------------------------------------------------
# Error path: missing adapter directory
# ---------------------------------------------------------------------------


def test_missing_adapter_raises_cli_error(tmp_suite: Path, tmp_path: Path) -> None:
    """Missing --adapter dir raises CliError with code=1."""
    args = _make_args(adapter=str(tmp_path / "no_such_dir"), suite=str(tmp_suite))
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    err = exc_info.value
    assert err.code == 1
    assert "adapter" in err.message.lower()
    assert err.remediation


def test_missing_adapter_emits_error_and_hint(tmp_suite: Path, tmp_path: Path) -> None:
    """Missing --adapter dir renders as ``error: …`` / ``hint: …`` lines."""
    args = _make_args(adapter=str(tmp_path / "no_such_dir"), suite=str(tmp_suite))
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    buf = io.StringIO()
    emit_error(exc_info.value, json_mode=False, stream=buf)
    text = buf.getvalue()
    assert text.startswith("error:")
    assert "hint:" in text


def test_missing_adapter_json_error(tmp_suite: Path, tmp_path: Path) -> None:
    """Missing --adapter dir renders as structured JSON when json_mode=True."""
    args = _make_args(adapter=str(tmp_path / "no_such_dir"), suite=str(tmp_suite))
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    buf = io.StringIO()
    emit_error(exc_info.value, json_mode=True, stream=buf)
    payload = json.loads(buf.getvalue())
    assert payload["code"] == 1
    assert "message" in payload
    assert "remediation" in payload


# ---------------------------------------------------------------------------
# Error path: missing suite file
# ---------------------------------------------------------------------------


def test_missing_suite_raises_cli_error(tmp_adapter: Path, tmp_path: Path) -> None:
    """Missing --suite file raises CliError with code=1."""
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_path / "no_such.jsonl"))
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    err = exc_info.value
    assert err.code == 1
    assert "suite" in err.message.lower()
    assert err.remediation


def test_missing_suite_emits_error_and_hint(tmp_adapter: Path, tmp_path: Path) -> None:
    """Missing --suite file renders as ``error: …`` / ``hint: …`` lines."""
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_path / "no_such.jsonl"))
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    buf = io.StringIO()
    emit_error(exc_info.value, json_mode=False, stream=buf)
    text = buf.getvalue()
    assert text.startswith("error:")
    assert "hint:" in text


# ---------------------------------------------------------------------------
# Happy path: text output (mocked backend)
# ---------------------------------------------------------------------------


def test_eval_text_output_contains_summary(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mocked perfect-score backend → text output includes total / exact / score."""
    monkeypatch.setattr(eval_mod, "_inference_backend", _mock_backend)
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite))
    rc = cmd_eval(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "total" in out
    assert "exact" in out
    assert "score" in out


def test_eval_text_shows_per_item_results(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Text mode shows one line per eval record."""
    monkeypatch.setattr(eval_mod, "_inference_backend", _mock_backend)
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite))
    cmd_eval(args)
    out = capsys.readouterr().out
    # Two records → two per-item lines (each starts with "  [ok]" or "  [fail]")
    item_lines = [ln for ln in out.splitlines() if "[ok]" in ln or "[fail]" in ln]
    assert len(item_lines) == 2


# ---------------------------------------------------------------------------
# Happy path: JSON output (mocked backend)
# ---------------------------------------------------------------------------


def test_eval_json_output_structure(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mocked backend with --json → well-formed JSON with summary fields."""
    monkeypatch.setattr(eval_mod, "_inference_backend", _mock_backend)
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), json=True)
    rc = cmd_eval(args)
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["total"] == 2
    assert data["exact_match"] == 2
    assert data["exact_match_pct"] == 100.0
    assert len(data["results"]) == 2


def test_eval_json_result_items(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each result item has the expected fields and correct exact_match flag."""
    monkeypatch.setattr(eval_mod, "_inference_backend", _mock_backend)
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), json=True)
    cmd_eval(args)
    data = json.loads(capsys.readouterr().out)
    for item in data["results"]:
        assert "index" in item
        assert "task" in item
        assert "input" in item
        assert "expected_output" in item
        assert "prediction" in item
        assert item["exact_match"] is True


def test_eval_json_partial_score(
    tmp_adapter: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A backend that always returns 'WRONG' yields exact_match=0, score=0.0."""
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        '{"task": "t", "input": "x", "expected_output": "y"}\n',
        encoding="utf-8",
    )

    def _always_wrong(adapter_path: str, record: dict[str, str]) -> str:  # noqa: ARG001
        return "WRONG"

    monkeypatch.setattr(eval_mod, "_inference_backend", _always_wrong)
    args = _make_args(adapter=str(tmp_adapter), suite=str(suite), json=True)
    cmd_eval(args)
    data = json.loads(capsys.readouterr().out)
    assert data["exact_match"] == 0
    assert data["exact_match_pct"] == 0.0
    assert data["results"][0]["exact_match"] is False


# ---------------------------------------------------------------------------
# No-network assertion
# ---------------------------------------------------------------------------


def test_no_network_access_with_mocked_backend(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the inference backend is mocked, no socket is opened.

    Replaces ``socket.socket`` with a callable that raises AssertionError, then
    runs the full eval code path and asserts it completes without triggering the
    replacement — proving the code path never touches the network.
    """
    monkeypatch.setattr(eval_mod, "_inference_backend", _mock_backend)

    class _NoSocket:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("network access attempted during eval — local_files_only violated")

    monkeypatch.setattr(socket, "socket", _NoSocket)

    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite))
    rc = cmd_eval(args)
    assert rc == 0  # completed without touching the network


# ---------------------------------------------------------------------------
# register() wires the subparser correctly
# ---------------------------------------------------------------------------


def test_register_adds_eval_subparser() -> None:
    """``register`` adds an ``eval`` subparser with --adapter, --suite, --json."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["eval", "--adapter", "/some/dir", "--suite", "/some/file.jsonl"])
    assert args.adapter == "/some/dir"
    assert args.suite == "/some/file.jsonl"
    assert args.json is False
    assert callable(args.func)


def test_register_json_flag() -> None:
    """--json flag is parsed correctly by the eval subparser."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["eval", "--adapter", "/a", "--suite", "/b.jsonl", "--json"])
    assert args.json is True
