"""Tests for ``sloth eval`` command.

Covers:
* missing --adapter dir  → CliError(code=1) with ``error:`` / ``hint:`` two-line contract
* missing --suite file   → CliError(code=1) with ``error:`` / ``hint:`` two-line contract
* eval.py imports no ML modules (torch / peft / transformers must NOT be in sys.modules
  after importing sloth.cli._commands.eval)
* valid adapter + suite  → results emitted in text mode and --json mode (run_eval mocked)
* no-network assertion   → monkeypatching ``socket.socket`` to raise proves no
  network call escapes the mocked code path
* PeftModel load sequence → now tested via run_eval in test_tune_trainer.py
* container routing      → host path calls container.launch with forwarded args +
  ``--in-container`` and correct identity extra_mounts; handler returns 0 on success
  and propagates CliError raised by launch()
* recursion guard        → ``--in-container`` flag prevents docker recursion
"""

from __future__ import annotations

import argparse
import io
import json
import socket
import sys
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
    """Build a minimal Namespace, defaulting ``json`` and ``in_container`` to False."""
    defaults: dict[str, Any] = {"json": False, "in_container": False}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _fake_run_eval_perfect(adapter_path: str, suite_path: str) -> dict[str, Any]:
    """Return a perfect-score eval summary without touching torch/peft."""
    return {
        "total": 2,
        "exact_match": 2,
        "exact_match_pct": 100.0,
        "results": [
            {
                "index": 0,
                "task": "reverse",
                "input": "abc",
                "expected_output": "cba",
                "prediction": "cba",
                "exact_match": True,
            },
            {
                "index": 1,
                "task": "upper",
                "input": "hello",
                "expected_output": "HELLO",
                "prediction": "HELLO",
                "exact_match": True,
            },
        ],
    }


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
def tmp_adapter_with_config(tmp_path: Path) -> tuple[Path, str]:
    """An adapter directory with a valid adapter_config.json.

    Returns (adapter_dir, base_model_name).
    """
    d = tmp_path / "adapter"
    d.mkdir()
    base_model_name = "unsloth/Qwen3-4B"
    config = {"base_model_name_or_path": base_model_name, "peft_type": "LORA"}
    (d / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return d, base_model_name


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
# ML-free import assertion
# ---------------------------------------------------------------------------


def test_eval_module_imports_no_ml_modules() -> None:
    """eval.py must not cause torch / peft / transformers to enter sys.modules.

    FIX 3: the CLI command module must be ML-free so the introspection verbs
    keep working on machines without the ML stack installed.
    """
    # The module is already imported (import at top of this file), so we just
    # confirm the heavy packages are absent.
    ml_packages = {"torch", "peft", "transformers"}
    leaked = ml_packages & set(sys.modules)
    assert not leaked, f"eval.py caused ML modules to be imported: {sorted(leaked)}"


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
# Happy path: text output (mocked run_eval, in-container)
# ---------------------------------------------------------------------------


def test_eval_text_output_contains_summary(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mocked run_eval → text output includes total / exact / score."""
    monkeypatch.setattr(eval_mod, "run_eval", _fake_run_eval_perfect)
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=True)
    rc = cmd_eval(args)
    assert rc in (None, 0)
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
    monkeypatch.setattr(eval_mod, "run_eval", _fake_run_eval_perfect)
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=True)
    cmd_eval(args)
    out = capsys.readouterr().out
    # Two records → two per-item lines (each starts with "  [ok]" or "  [fail]")
    item_lines = [ln for ln in out.splitlines() if "[ok]" in ln or "[fail]" in ln]
    assert len(item_lines) == 2


# ---------------------------------------------------------------------------
# Happy path: JSON output (mocked run_eval, in-container)
# ---------------------------------------------------------------------------


def test_eval_json_output_structure(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mocked run_eval with --json → well-formed JSON with summary fields."""
    monkeypatch.setattr(eval_mod, "run_eval", _fake_run_eval_perfect)
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), json=True, in_container=True)
    rc = cmd_eval(args)
    assert rc in (None, 0)
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
    monkeypatch.setattr(eval_mod, "run_eval", _fake_run_eval_perfect)
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), json=True, in_container=True)
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
    """A run_eval result with no exact matches → score=0.0."""
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        '{"task": "t", "input": "x", "expected_output": "y"}\n',
        encoding="utf-8",
    )

    def _zero_score(adapter_path: str, suite_path: str) -> dict[str, Any]:
        return {
            "total": 1,
            "exact_match": 0,
            "exact_match_pct": 0.0,
            "results": [
                {
                    "index": 0,
                    "task": "t",
                    "input": "x",
                    "expected_output": "y",
                    "prediction": "WRONG",
                    "exact_match": False,
                }
            ],
        }

    monkeypatch.setattr(eval_mod, "run_eval", _zero_score)
    args = _make_args(adapter=str(tmp_adapter), suite=str(suite), json=True, in_container=True)
    cmd_eval(args)
    data = json.loads(capsys.readouterr().out)
    assert data["exact_match"] == 0
    assert data["exact_match_pct"] == 0.0
    assert data["results"][0]["exact_match"] is False


# ---------------------------------------------------------------------------
# No-network assertion
# ---------------------------------------------------------------------------


def test_no_network_access_with_mocked_run_eval(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When run_eval is mocked, no socket is opened.

    Replaces ``socket.socket`` with a callable that raises AssertionError, then
    runs the full eval code path and asserts it completes without triggering the
    replacement — proving the code path never touches the network.
    """
    monkeypatch.setattr(eval_mod, "run_eval", _fake_run_eval_perfect)

    class _NoSocket:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("network access attempted during eval — local_files_only violated")

    monkeypatch.setattr(socket, "socket", _NoSocket)

    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=True)
    rc = cmd_eval(args)
    assert rc in (None, 0)  # completed without touching the network


# ---------------------------------------------------------------------------
# Acceptance — in-container branch calls run_eval (the ML seam)
# ---------------------------------------------------------------------------


def test_in_container_calls_run_eval(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With --in-container, cmd_eval delegates to run_eval (the tune ML seam).

    FIX 3: the real eval logic (model loading, PeftModel wrapping, scoring) lives
    in sloth.tune._trainer.run_eval, not in this CLI module.
    """
    calls: list[tuple[str, str]] = []

    def _capture_run_eval(adapter_path: str, suite_path: str) -> dict[str, Any]:
        calls.append((adapter_path, suite_path))
        return _fake_run_eval_perfect(adapter_path, suite_path)

    monkeypatch.setattr(eval_mod, "run_eval", _capture_run_eval)

    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=True)
    rc = cmd_eval(args)
    assert rc in (None, 0)
    assert len(calls) == 1, "run_eval must be called exactly once"
    assert calls[0][0] == str(tmp_adapter)
    assert calls[0][1] == str(tmp_suite)
    out = capsys.readouterr().out
    assert "total" in out


# ---------------------------------------------------------------------------
# Acceptance — container routing: host path calls container.launch
# ---------------------------------------------------------------------------


def test_host_routes_to_container_launch(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cmd_eval on the host (no --in-container) calls container.launch.

    Asserts that:
    - container.launch IS called
    - the sloth_args list starts with 'eval' and ends with '--in-container'
    - '--adapter' and '--suite' are forwarded with absolute paths
    - the function returns 0 (launch() returns 0 on success)
    """
    launch_calls: list[dict[str, Any]] = []

    def _fake_launch(
        sloth_args: list[str],
        *,
        workdir: str | None = None,
        checkout: str | None = None,
        **kwargs: Any,
    ) -> int:
        launch_calls.append(
            {"sloth_args": list(sloth_args), "workdir": workdir, "checkout": checkout, **kwargs}
        )
        return 0

    monkeypatch.setattr(eval_mod.container, "launch", _fake_launch)

    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=False)
    rc = cmd_eval(args)

    assert rc in (None, 0)
    assert len(launch_calls) == 1, "container.launch must be called exactly once"

    forwarded = launch_calls[0]["sloth_args"]
    assert forwarded[0] == "eval", "first sloth_arg must be 'eval'"
    assert "--in-container" in forwarded, "recursion guard must be forwarded"
    assert "--adapter" in forwarded
    assert "--suite" in forwarded
    # workdir must be set (points to the adapter's parent)
    assert launch_calls[0]["workdir"] is not None
    # checkout must be set (the repo root)
    assert launch_calls[0]["checkout"] is not None


def test_host_routes_json_flag_when_set(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--json is forwarded to the container when set on the host."""
    forwarded_args: list[list[str]] = []

    def _fake_launch(sloth_args: list[str], **kwargs: Any) -> int:
        forwarded_args.append(list(sloth_args))
        return 0

    monkeypatch.setattr(eval_mod.container, "launch", _fake_launch)

    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), json=True, in_container=False)
    cmd_eval(args)

    assert forwarded_args, "launch must be called"
    assert "--json" in forwarded_args[0], "--json must be forwarded to the container"
    assert "--in-container" in forwarded_args[0]


def test_host_returns_0_on_launch_success(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cmd_eval returns None when container.launch succeeds.

    FIX 1+2: launch() raises on failure; on success the handler falls through
    (implicit None return) — no explicit return value.
    """
    monkeypatch.setattr(eval_mod.container, "launch", lambda *a, **kw: 0)
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=False)
    rc = cmd_eval(args)
    assert rc in (None, 0)


def test_host_propagates_launch_cli_error(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CliError raised by container.launch propagates out of cmd_eval.

    FIX 1+2: launch() raises CliError (code 1 or 2) on container failure.
    The handler must not swallow it — the CliError must surface to the caller.
    """

    def _raise_cli_error(*args: Any, **kwargs: Any) -> int:
        raise CliError(
            code=2,
            message="Container exited with status 2",
            remediation="Check the in-container error output.",
        )

    monkeypatch.setattr(eval_mod.container, "launch", _raise_cli_error)
    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=False)
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    assert exc_info.value.code == 2


def test_host_extra_mounts_cover_adapter_and_suite_parents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity mounts cover the parent dirs of adapter and suite.

    FIX 4 (path visibility): host-absolute paths forwarded in sloth_args must
    resolve unchanged inside the container; extra_mounts achieves this by
    mounting each parent dir at the same path (identity mount).
    """
    adapter_dir = tmp_path / "adapters" / "my-lora"
    adapter_dir.mkdir(parents=True)
    suite_file = tmp_path / "data" / "eval.jsonl"
    suite_file.parent.mkdir(parents=True)
    suite_file.write_text(
        '{"task": "t", "input": "x", "expected_output": "y"}\n',
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def _capture_launch(sloth_args: list[str], **kwargs: Any) -> int:
        captured["sloth_args"] = list(sloth_args)
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(eval_mod.container, "launch", _capture_launch)

    args = _make_args(adapter=str(adapter_dir), suite=str(suite_file), in_container=False)
    rc = cmd_eval(args)
    assert rc in (None, 0)

    extra_mounts = captured.get("extra_mounts") or []
    mounted_container_paths = {ct for _, ct in extra_mounts}

    assert (
        str(adapter_dir.resolve().parent) in mounted_container_paths
    ), f"adapter parent not in extra_mounts: {extra_mounts}"
    assert (
        str(suite_file.resolve().parent) in mounted_container_paths
    ), f"suite parent not in extra_mounts: {extra_mounts}"


# ---------------------------------------------------------------------------
# Acceptance — recursion guard: --in-container path does NOT call launch
# ---------------------------------------------------------------------------


def test_in_container_does_not_call_launch(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With --in-container set, cmd_eval runs run_eval, never calls launch.

    Stubs container.launch to raise AssertionError; the in-container path must
    complete without triggering it.
    """

    def _must_not_launch(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("container.launch called inside container — recursion guard broken")

    monkeypatch.setattr(eval_mod.container, "launch", _must_not_launch)
    monkeypatch.setattr(eval_mod, "run_eval", _fake_run_eval_perfect)

    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=True)
    rc = cmd_eval(args)
    assert rc in (None, 0)


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


def test_register_in_container_flag() -> None:
    """``--in-container`` is a hidden flag that defaults to False."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    # Default: in_container is False.
    args_default = parser.parse_args(["eval", "--adapter", "/a", "--suite", "/b.jsonl"])
    assert args_default.in_container is False

    # When passed explicitly it is True.
    args_set = parser.parse_args(
        ["eval", "--adapter", "/a", "--suite", "/b.jsonl", "--in-container"]
    )
    assert args_set.in_container is True


def test_register_in_container_not_in_help() -> None:
    """--in-container must not appear in the public help text (it is SUPPRESS'd)."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    help_text = sub.choices["eval"].format_help()
    assert "--in-container" not in help_text
