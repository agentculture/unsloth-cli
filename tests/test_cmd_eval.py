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
import inspect
import io
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import sloth.cli._commands.eval as eval_mod
import sloth.tune._exporter as exporter_mod
import sloth.tune._trainer as trainer_mod
import sloth.tune.container as container_mod
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


#: A minimal, well-formed eval summary standing in for the dict
#: ``sloth.tune.container.launch`` now returns (t1/t2: launch never returns an
#: int). Host-path launch fakes below return this so the host's own text-mode
#: rendering (``_render_text``, which indexes ``total``/``exact_match``/
#: ``exact_match_pct``) has real keys to render, exactly mirroring what the
#: in-container run would have printed.
_FAKE_EVAL_SUMMARY: dict[str, Any] = {
    "total": 1,
    "exact_match": 1,
    "exact_match_pct": 100.0,
    "results": [],
}


def _fake_run_eval_perfect(
    adapter_path: str, suite_path: str | None = None, *, suite_paths=None, quant=None, batch_size=8
) -> dict[str, Any]:
    if suite_path is None:
        suite_path = str(suite_paths[0])
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


@pytest.fixture
def tmp_adapter(tmp_path: Path) -> Path:
    """A minimal adapter directory (just needs to exist as a directory)."""
    d = tmp_path / "adapter"
    d.mkdir()
    return d


@pytest.fixture
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


@pytest.fixture
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
    payload = data["suites"]["suite"]
    assert payload["total"] == 2
    assert payload["exact_match"] == 2
    assert payload["exact_match_pct"] == 100.0
    assert len(payload["results"]) == 2


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
    for item in data["suites"]["suite"]["results"]:
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

    def _zero_score(
        adapter_path: str,
        suite_path: str | None = None,
        *,
        suite_paths=None,
        quant=None,
        batch_size=8,
    ) -> dict[str, Any]:
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
    payload = data["suites"]["suite"]
    assert payload["exact_match"] == 0
    assert payload["exact_match_pct"] == 0.0
    assert payload["results"][0]["exact_match"] is False


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

    def _capture_run_eval(
        adapter_path: str,
        suite_path: str | None = None,
        *,
        suite_paths=None,
        quant=None,
        batch_size=8,
    ) -> dict[str, Any]:
        calls.append((adapter_path, suite_path or str(suite_paths[0])))
        return _fake_run_eval_perfect(adapter_path, suite_path, suite_paths=suite_paths)

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
        return dict(_FAKE_EVAL_SUMMARY)

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
        return dict(_FAKE_EVAL_SUMMARY)

    monkeypatch.setattr(eval_mod.container, "launch", _fake_launch)

    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), json=True, in_container=False)
    cmd_eval(args)

    assert forwarded_args, "launch must be called"
    assert "--json" in forwarded_args[0], "--json must be forwarded to the container"
    assert "--in-container" in forwarded_args[0]


# ---------------------------------------------------------------------------
# t2 acceptance — --json always forwarded into the container; the host emits
# the dict container.launch() returns via emit_result, honouring the HOST's
# own --json flag (independent of what was forwarded into the container).
# ---------------------------------------------------------------------------


def test_host_json_forwarded_even_without_host_json(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--json is forwarded into the container argv UNCONDITIONALLY, even when
    the host itself was not invoked with --json."""
    forwarded_args: list[list[str]] = []

    def _fake_launch(sloth_args: list[str], **kwargs: Any) -> dict[str, Any]:
        forwarded_args.append(list(sloth_args))
        return dict(_FAKE_EVAL_SUMMARY)

    monkeypatch.setattr(eval_mod.container, "launch", _fake_launch)

    args = _make_args(
        adapter=str(tmp_adapter), suite=str(tmp_suite), json=False, in_container=False
    )
    cmd_eval(args)

    assert forwarded_args, "launch must be called"
    assert "--json" in forwarded_args[0], "--json must be forwarded to the container"


def test_host_emits_launch_result_as_json(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The dict container.launch() returns is emitted (already suite-keyed, since
    the container's own cmd_eval always runs --json) as JSON on stdout when the
    host's own --json flag is set."""
    suite_payload = {
        "total": 2,
        "exact_match": 2,
        "exact_match_pct": 100.0,
        "results": [{"index": 0, "task": "reverse", "exact_match": True}],
    }
    fake_result = {"suites": {"suite": suite_payload}}
    monkeypatch.setattr(eval_mod.container, "launch", lambda *a, **kw: dict(fake_result))

    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), json=True, in_container=False)
    cmd_eval(args)

    out = capsys.readouterr().out
    assert json.loads(out) == fake_result


def test_host_emits_launch_result_as_text(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """In text mode, the launch() result (already suite-keyed) is rendered with
    the same one-block-per-suite renderer the in-container path uses
    (_render_named_report_text) — host stdout is exactly that."""
    suite_payload = {
        "total": 2,
        "exact_match": 2,
        "exact_match_pct": 100.0,
        "results": [{"index": 0, "task": "reverse", "exact_match": True}],
    }
    fake_result = {"suites": {"suite": suite_payload}}
    monkeypatch.setattr(eval_mod.container, "launch", lambda *a, **kw: dict(fake_result))

    args = _make_args(
        adapter=str(tmp_adapter), suite=str(tmp_suite), json=False, in_container=False
    )
    cmd_eval(args)

    out = capsys.readouterr().out
    target = eval_mod._resolve_target(args)
    named_suites = {"suite": [tmp_suite]}
    expected = eval_mod._render_named_report_text(target, named_suites, {"suite": suite_payload})
    assert out == expected + "\n"


def test_host_returns_0_on_launch_success(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cmd_eval returns None when container.launch succeeds.

    FIX 1+2: launch() raises on failure; on success the handler falls through
    (implicit None return) — no explicit return value.
    """
    monkeypatch.setattr(eval_mod.container, "launch", lambda *a, **kw: dict(_FAKE_EVAL_SUMMARY))
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
        return dict(_FAKE_EVAL_SUMMARY)

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
    assert args.suite == ["/some/file.jsonl"]  # --suite is repeatable (action="append")
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


# ---------------------------------------------------------------------------
# t11 — ``--model DIR`` (merged / quantized outputs)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_model(tmp_path: Path) -> Path:
    """A merged-model directory carrying an AWQ compressed-tensors config.json."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_type": "lfm2",
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "format": "pack-quantized",
                },
            }
        ),
        encoding="utf-8",
    )
    return d


def _fake_run_eval_model(
    model_dir: str, suite_path: str | None = None, *, suite_paths=None, quant=None, batch_size=8
) -> dict[str, Any]:
    """A run_eval_model summary (adapter score fields + the model-specific ones)."""
    summary = _fake_run_eval_perfect(model_dir, suite_path, suite_paths=suite_paths)
    summary["model_dir"] = model_dir
    summary["quant_method"] = "compressed-tensors"
    summary["quant_format"] = "pack-quantized"
    return summary


def test_adapter_and_model_are_mutually_exclusive(
    tmp_adapter: Path, tmp_model: Path, tmp_suite: Path
) -> None:
    """Passing both --adapter and --model is a user error (exit 1 + hint)."""
    args = _make_args(adapter=str(tmp_adapter), model=str(tmp_model), suite=str(tmp_suite))
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    err = exc_info.value
    assert err.code == 1
    assert "--adapter" in err.message
    assert "--model" in err.message
    assert err.remediation


def test_neither_adapter_nor_model_is_an_error(tmp_suite: Path) -> None:
    """Passing neither --adapter nor --model is a user error (exit 1 + hint)."""
    args = _make_args(adapter=None, model=None, suite=str(tmp_suite))
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    err = exc_info.value
    assert err.code == 1
    assert err.remediation
    buf = io.StringIO()
    emit_error(err, json_mode=False, stream=buf)
    text = buf.getvalue()
    assert text.startswith("error:")
    assert "hint:" in text


def test_missing_model_dir_raises_cli_error(tmp_suite: Path, tmp_path: Path) -> None:
    """A non-existent --model directory raises CliError(code=1)."""
    args = _make_args(adapter=None, model=str(tmp_path / "nope"), suite=str(tmp_suite))
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    assert exc_info.value.code == 1
    assert "model" in exc_info.value.message.lower()


def test_host_routes_model_to_container(
    tmp_model: Path, tmp_suite: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--model routes to the container like --adapter, with the llama.cpp cache mounted."""
    captured: dict[str, Any] = {}

    def _capture_launch(sloth_args: list[str], **kwargs: Any) -> int:
        captured["sloth_args"] = list(sloth_args)
        captured.update(kwargs)
        return dict(_FAKE_EVAL_SUMMARY)

    monkeypatch.setattr(eval_mod.container, "launch", _capture_launch)

    # A GGUF target: the export home mount (llama.cpp cache) must ride along.
    (tmp_model / "model.gguf").write_bytes(b"GGUF")
    args = _make_args(adapter=None, model=str(tmp_model), suite=str(tmp_suite))
    rc = cmd_eval(args)
    assert rc in (None, 0)

    forwarded = captured["sloth_args"]
    assert forwarded[0] == "eval"
    assert "--model" in forwarded
    assert "--adapter" not in forwarded
    assert "--in-container" in forwarded
    assert str(tmp_model.resolve()) in forwarded

    mounts = captured.get("extra_mounts") or []
    targets = {target for _, target in mounts}
    assert str(tmp_model.resolve().parent) in targets
    assert str(tmp_suite.resolve().parent) in targets
    # The export home mount (export_launch_kwargs) must be present: it carries the
    # llama.cpp cache at <EXPORT_HOME>/.unsloth/llama.cpp needed for GGUF scoring.
    assert container_mod.EXPORT_HOME in targets
    env = dict(captured.get("env") or [])
    assert env.get("HOME") == container_mod.EXPORT_HOME

    # A transformers-loadable dir (bf16 / awq / nvfp4) needs no llama.cpp cache and
    # therefore no writable export home.
    (tmp_model / "model.gguf").unlink()
    captured.clear()
    cmd_eval(_make_args(adapter=None, model=str(tmp_model), suite=str(tmp_suite)))
    targets = {target for _, target in (captured.get("extra_mounts") or [])}
    assert container_mod.EXPORT_HOME not in targets
    assert "HOME" not in dict(captured.get("env") or [])


def test_in_container_model_calls_run_eval_model(
    tmp_model: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--model --in-container delegates to run_eval_model and emits the extra fields."""
    calls: list[tuple[str, str]] = []

    def _capture(
        model_dir: str, suite_path: str | None = None, *, suite_paths=None, quant=None, batch_size=8
    ) -> dict[str, Any]:
        calls.append((model_dir, suite_path or str(suite_paths[0])))
        return _fake_run_eval_model(model_dir, suite_path, suite_paths=suite_paths)

    monkeypatch.setattr(eval_mod, "run_eval_model", _capture)
    monkeypatch.setattr(
        eval_mod.container,
        "launch",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("recursion guard broken")),
    )

    args = _make_args(
        adapter=None, model=str(tmp_model), suite=str(tmp_suite), json=True, in_container=True
    )
    rc = cmd_eval(args)
    assert rc in (None, 0)
    assert calls == [(str(tmp_model), str(tmp_suite))]

    data = json.loads(capsys.readouterr().out)
    payload = data["suites"]["suite"]
    assert payload["total"] == 2
    assert payload["exact_match"] == 2
    assert payload["exact_match_pct"] == 100.0
    assert payload["model_dir"] == str(tmp_model)
    assert payload["quant_method"] == "compressed-tensors"
    assert payload["quant_format"] == "pack-quantized"


def test_in_container_model_text_output(
    tmp_model: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Text mode names the model dir and its quantisation."""
    monkeypatch.setattr(eval_mod, "run_eval_model", _fake_run_eval_model)
    args = _make_args(adapter=None, model=str(tmp_model), suite=str(tmp_suite), in_container=True)
    cmd_eval(args)
    out = capsys.readouterr().out
    assert str(tmp_model) in out
    assert "compressed-tensors" in out
    assert "pack-quantized" in out
    assert "score" in out


def test_register_model_flag() -> None:
    """``--model`` parses and defaults to None; --adapter is no longer required."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["eval", "--model", "/some/model", "--suite", "/b.jsonl"])
    assert args.model == "/some/model"
    assert args.adapter is None
    args2 = parser.parse_args(["eval", "--adapter", "/a", "--suite", "/b.jsonl"])
    assert args2.model is None
    args3 = parser.parse_args(
        ["eval", "--model", "/m", "--suite", "/b.jsonl", "--in-container", "--json"]
    )
    assert args3.in_container is True
    assert args3.json is True


# ---------------------------------------------------------------------------
# t11 — run_eval_model (the in-container seam, fake backend only)
# ---------------------------------------------------------------------------


class _FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return False


class _FakeTorch:
    bfloat16 = "bfloat16"

    @staticmethod
    def no_grad() -> _FakeNoGrad:
        return _FakeNoGrad()


class _FakeTokenizer:
    """Echoes the prompt through generate() so decode() can map it to an answer."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers

    def __call__(self, prompt: str, return_tensors: str = "pt") -> Any:
        class _Inputs(dict):
            def to(self, _device: Any) -> dict:
                return dict(self)

        return _Inputs(input_ids=prompt)

    def decode(self, token: str, skip_special_tokens: bool = True) -> str:
        return self.answers.get(token, "UNKNOWN")


class _FakeParam:
    device = "cpu"


class _FakeModel:
    def __init__(self) -> None:
        self.eval_called = False
        self.forward_calls: list[str] = []

    def parameters(self) -> Any:
        return iter([_FakeParam()])

    def eval(self) -> None:
        self.eval_called = True

    def __call__(
        self, input_ids: Any = None, attention_mask: Any = None, labels: Any = None
    ) -> Any:
        """A labelled forward pass — the only shape run_perplexity may use."""
        assert labels is not None, "run_perplexity must pass labels= for a scored pass"
        self.forward_calls.append(input_ids)
        return SimpleNamespace(loss=0.0)

    def generate(self, input_ids: str, max_new_tokens: int = 0) -> list[str]:
        # prompt + continuation, like a real generate(); the continuation echoes the
        # prompt so decode() can map it to the configured answer.
        return [input_ids + input_ids]


def _fake_backend(answers: dict[str, str]) -> Any:
    tokenizer = _FakeTokenizer(answers)
    model = _FakeModel()

    class _Loader:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            return model

    class _TokLoader:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            return tokenizer

    return exporter_mod._EvalBackend(
        torch=_FakeTorch(),
        auto_model_for_causal_lm=_Loader(),
        auto_tokenizer=_TokLoader(),
    )


def test_run_eval_model_transformers_path(
    tmp_model: Path, tmp_suite: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fake transformers backend scores the suite and reports the quant fields."""
    answers = {
        "Task: reverse\nInput: abc\nOutput:": "cba",
        "Task: upper\nInput: hello\nOutput:": "WRONG",
    }
    monkeypatch.setattr(exporter_mod, "_load_eval_backend", lambda: _fake_backend(answers))

    summary = exporter_mod.run_eval_model(str(tmp_model), str(tmp_suite))

    assert summary["total"] == 2
    assert summary["exact_match"] == 1
    assert summary["exact_match_pct"] == 50.0
    assert summary["model_dir"] == str(tmp_model)
    assert summary["quant_method"] == "compressed-tensors"
    assert summary["quant_format"] == "pack-quantized"
    assert summary["results"][0]["exact_match"] is True
    assert summary["results"][1]["prediction"] == "WRONG"


def test_run_eval_model_bf16_has_null_quant(
    tmp_path: Path, tmp_suite: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain bf16 merged dir (no quantization_config) reports null quant fields."""
    model_dir = tmp_path / "merged"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "lfm2"}), encoding="utf-8")
    monkeypatch.setattr(exporter_mod, "_load_eval_backend", lambda: _fake_backend({}))

    summary = exporter_mod.run_eval_model(str(model_dir), str(tmp_suite))
    assert summary["quant_method"] is None
    assert summary["quant_format"] is None
    assert summary["exact_match"] == 0


def test_run_eval_model_gguf_path(
    tmp_path: Path, tmp_suite: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dir holding a .gguf is scored via llama-completion, never via transformers."""
    model_dir = tmp_path / "gguf-out"
    model_dir.mkdir()
    (model_dir / "Model.Q4_K_M.gguf").write_bytes(b"\x00")

    seen: list[tuple[str, str, int]] = []

    def _fake_completion(gguf: Path, prompt: str, max_tokens: int) -> str:
        seen.append((str(gguf), prompt, max_tokens))
        return "cba" if "reverse" in prompt else "HELLO"

    monkeypatch.setattr(exporter_mod, "_run_llama_completion", _fake_completion)
    monkeypatch.setattr(
        exporter_mod,
        "_load_eval_backend",
        lambda: (_ for _ in ()).throw(AssertionError("transformers used for a GGUF dir")),
    )

    summary = exporter_mod.run_eval_model(str(model_dir), str(tmp_suite))
    assert summary["total"] == 2
    assert summary["exact_match"] == 2
    assert summary["quant_method"] is None
    assert summary["quant_format"] is None
    assert len(seen) == 2
    assert seen[0][0].endswith("Model.Q4_K_M.gguf")


def test_run_eval_model_missing_dir(tmp_path: Path, tmp_suite: Path) -> None:
    """A non-existent model dir raises CliError(code=1)."""
    with pytest.raises(CliError) as exc_info:
        exporter_mod.run_eval_model(str(tmp_path / "gone"), str(tmp_suite))
    assert exc_info.value.code == 1


def test_run_eval_model_without_ml_stack(
    tmp_model: Path, tmp_suite: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable ML stack surfaces as CliError(code=2) with the NGC hint."""

    def _boom() -> Any:
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(exporter_mod, "_load_eval_backend", _boom)
    with pytest.raises(CliError) as exc_info:
        exporter_mod.run_eval_model(str(tmp_model), str(tmp_suite))
    assert exc_info.value.code == 2
    assert "NGC" in exc_info.value.remediation or "container" in exc_info.value.remediation


# ---------------------------------------------------------------------------
# t4 — --suite accepts a directory, validated before any container launch
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_suite_dir_valid(tmp_path: Path) -> Path:
    """A suite directory with two valid task-schema *.jsonl files."""
    d = tmp_path / "suite_dir"
    d.mkdir()
    (d / "a.jsonl").write_text(
        '{"task": "reverse", "input": "abc", "expected_output": "cba"}\n',
        encoding="utf-8",
    )
    (d / "b.jsonl").write_text(
        '{"task": "upper", "input": "hello", "expected_output": "HELLO"}\n',
        encoding="utf-8",
    )
    return d


@pytest.fixture
def tmp_suite_dir_malformed(tmp_path: Path) -> Path:
    """A suite directory whose second file has a malformed line 2."""
    d = tmp_path / "suite_dir_bad"
    d.mkdir()
    (d / "a.jsonl").write_text(
        '{"task": "reverse", "input": "abc", "expected_output": "cba"}\n',
        encoding="utf-8",
    )
    (d / "z_bad.jsonl").write_text(
        '{"task": "ok", "input": "x", "expected_output": "y"}\n' "not valid json\n",
        encoding="utf-8",
    )
    return d


def test_directory_suite_validated_before_launch_on_malformed_file(
    tmp_adapter: Path,
    tmp_suite_dir_malformed: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed file inside a --suite directory exits 1 naming file + line,
    and the container launcher is never called (fails fast, before any launch)."""

    def _must_not_launch(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("container.launch called despite a malformed suite file")

    monkeypatch.setattr(eval_mod.container, "launch", _must_not_launch)

    args = _make_args(
        adapter=str(tmp_adapter), suite=[str(tmp_suite_dir_malformed)], in_container=False
    )
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)

    err = exc_info.value
    assert err.code == 1
    assert "z_bad.jsonl" in err.message
    assert "line 2" in err.message


def test_directory_suite_all_valid_expands_and_launches(
    tmp_adapter: Path,
    tmp_suite_dir_valid: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A directory of valid *.jsonl files expands to one --suite flag per file,
    sorted, and launches the container."""
    captured: dict[str, Any] = {}

    def _capture_launch(sloth_args: list[str], **kwargs: Any) -> int:
        captured["sloth_args"] = list(sloth_args)
        return dict(_FAKE_EVAL_SUMMARY)

    monkeypatch.setattr(eval_mod.container, "launch", _capture_launch)

    args = _make_args(
        adapter=str(tmp_adapter), suite=[str(tmp_suite_dir_valid)], in_container=False
    )
    rc = cmd_eval(args)
    assert rc in (None, 0)

    forwarded = captured["sloth_args"]
    suite_flags_idx = [i for i, tok in enumerate(forwarded) if tok == "--suite"]
    assert len(suite_flags_idx) == 2, f"expected 2 --suite flags, forwarded={forwarded}"
    suite_values = [forwarded[i + 1] for i in suite_flags_idx]
    assert suite_values == sorted(suite_values), "directory files must be forwarded sorted"
    assert str((tmp_suite_dir_valid / "a.jsonl").resolve()) in suite_values
    assert str((tmp_suite_dir_valid / "b.jsonl").resolve()) in suite_values


def test_single_jsonl_suite_still_works_unchanged(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single .jsonl --suite path still works exactly as before (one --suite flag)."""
    captured: dict[str, Any] = {}

    def _capture_launch(sloth_args: list[str], **kwargs: Any) -> int:
        captured["sloth_args"] = list(sloth_args)
        return dict(_FAKE_EVAL_SUMMARY)

    monkeypatch.setattr(eval_mod.container, "launch", _capture_launch)

    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=False)
    rc = cmd_eval(args)
    assert rc in (None, 0)

    forwarded = captured["sloth_args"]
    assert forwarded.count("--suite") == 1
    idx = forwarded.index("--suite")
    assert forwarded[idx + 1] == str(tmp_suite.resolve())


def test_missing_suite_directory_raises_before_launch(
    tmp_adapter: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A --suite path that does not exist at all raises CliError(code=1); no launch."""

    def _must_not_launch(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("container.launch called for a missing suite path")

    monkeypatch.setattr(eval_mod.container, "launch", _must_not_launch)
    args = _make_args(
        adapter=str(tmp_adapter), suite=[str(tmp_path / "does_not_exist")], in_container=False
    )
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    assert exc_info.value.code == 1


def test_empty_suite_directory_raises_before_launch(
    tmp_adapter: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A --suite directory holding no *.jsonl files raises CliError(code=1); no launch."""
    empty_dir = tmp_path / "empty_suite"
    empty_dir.mkdir()

    def _must_not_launch(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("container.launch called for an empty suite directory")

    monkeypatch.setattr(eval_mod.container, "launch", _must_not_launch)
    args = _make_args(adapter=str(tmp_adapter), suite=[str(empty_dir)], in_container=False)
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# t4 — --quant / --batch-size flags
# ---------------------------------------------------------------------------


def test_register_quant_and_batch_size_flags() -> None:
    """--quant and --batch-size parse; --batch-size defaults to 8."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["eval", "--adapter", "/a", "--suite", "/b.jsonl"])
    assert args.quant is None
    assert args.batch_size == 8

    args2 = parser.parse_args(
        [
            "eval",
            "--adapter",
            "/a",
            "--suite",
            "/b.jsonl",
            "--quant",
            "q4_k_m",
            "--batch-size",
            "16",
        ]
    )
    assert args2.quant == "q4_k_m"
    assert args2.batch_size == 16


def test_host_forwards_quant_and_batch_size_to_container(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--quant and --batch-size are forwarded into the in-container argv."""
    captured: dict[str, Any] = {}

    def _capture_launch(sloth_args: list[str], **kwargs: Any) -> int:
        captured["sloth_args"] = list(sloth_args)
        return dict(_FAKE_EVAL_SUMMARY)

    monkeypatch.setattr(eval_mod.container, "launch", _capture_launch)

    args = _make_args(
        adapter=str(tmp_adapter),
        suite=str(tmp_suite),
        quant="q4_k_m",
        batch_size=16,
        in_container=False,
    )
    rc = cmd_eval(args)
    assert rc in (None, 0)

    forwarded = captured["sloth_args"]
    assert "--quant" in forwarded
    assert forwarded[forwarded.index("--quant") + 1] == "q4_k_m"
    assert "--batch-size" in forwarded
    assert forwarded[forwarded.index("--batch-size") + 1] == "16"


def test_host_forwards_default_batch_size_when_not_set(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--batch-size still reaches the container argv with its default (8) and
    --quant is omitted entirely when not passed."""
    captured: dict[str, Any] = {}

    def _capture_launch(sloth_args: list[str], **kwargs: Any) -> int:
        captured["sloth_args"] = list(sloth_args)
        return dict(_FAKE_EVAL_SUMMARY)

    monkeypatch.setattr(eval_mod.container, "launch", _capture_launch)

    args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=False)
    cmd_eval(args)

    forwarded = captured["sloth_args"]
    assert "--quant" not in forwarded
    assert "--batch-size" in forwarded
    assert forwarded[forwarded.index("--batch-size") + 1] == "8"


def test_in_container_forwards_quant_and_batch_size_when_seam_accepts_them(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When run_eval's signature accepts suite_paths/quant/batch_size, cmd_eval
    passes them through (the richer signature t5, a sibling task, is adding)."""
    calls: list[dict[str, Any]] = []

    def _fake_rich_run_eval(
        adapter_path: str,
        *,
        suite_paths: list[Path],
        quant: str | None = None,
        batch_size: int = 8,
    ) -> dict[str, Any]:
        calls.append(
            {
                "adapter_path": adapter_path,
                "suite_paths": list(suite_paths),
                "quant": quant,
                "batch_size": batch_size,
            }
        )
        return _fake_run_eval_perfect(adapter_path, str(suite_paths[0]))

    monkeypatch.setattr(eval_mod, "run_eval", _fake_rich_run_eval)

    args = _make_args(
        adapter=str(tmp_adapter),
        suite=str(tmp_suite),
        quant="q4_k_m",
        batch_size=4,
        in_container=True,
    )
    rc = cmd_eval(args)
    assert rc in (None, 0)
    assert len(calls) == 1
    assert calls[0]["adapter_path"] == str(tmp_adapter)
    assert calls[0]["suite_paths"] == [tmp_suite]
    assert calls[0]["quant"] == "q4_k_m"
    assert calls[0]["batch_size"] == 4


# ---------------------------------------------------------------------------
# qodo — --batch-size must be validated on the host before suite validation
# or container launch; 0 and negative values are user errors, not silent 1s.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_batch_size", [0, -3])
def test_batch_size_below_one_raises_cli_error(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_batch_size: int,
) -> None:
    """--batch-size 0 or negative raises CliError(code=1) before the container
    is ever launched — batch-size validation happens before suite validation
    and before any launch() call."""
    launched: list[Any] = []

    def _fail_if_launched(sloth_args: list[str], **kwargs: Any) -> int:
        launched.append(sloth_args)
        raise AssertionError("container.launch must not be called for a bad --batch-size")

    monkeypatch.setattr(eval_mod.container, "launch", _fail_if_launched)

    args = _make_args(
        adapter=str(tmp_adapter),
        suite=str(tmp_suite),
        batch_size=bad_batch_size,
        in_container=False,
    )
    with pytest.raises(CliError) as exc_info:
        cmd_eval(args)
    assert exc_info.value.code == 1
    assert not launched


def test_batch_size_one_is_accepted(
    tmp_adapter: Path,
    tmp_suite: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--batch-size 1 stays the explicit unbatched mode — it is not rejected."""
    captured: dict[str, Any] = {}

    def _capture_launch(sloth_args: list[str], **kwargs: Any) -> dict[str, Any]:
        captured["sloth_args"] = list(sloth_args)
        return dict(_FAKE_EVAL_SUMMARY)

    monkeypatch.setattr(eval_mod.container, "launch", _capture_launch)

    args = _make_args(
        adapter=str(tmp_adapter),
        suite=str(tmp_suite),
        batch_size=1,
        in_container=False,
    )
    rc = cmd_eval(args)
    assert rc in (None, 0)
    forwarded = captured["sloth_args"]
    assert forwarded[forwarded.index("--batch-size") + 1] == "1"


# ---------------------------------------------------------------------------
# t6 — run_eval_model: latency + token counts, per-schema scoring, perplexity
# ---------------------------------------------------------------------------


class _Clock:
    """A deterministic ``perf_counter`` stand-in advancing *step* seconds per call."""

    def __init__(self, step: float = 0.5) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def _freeze_clock(monkeypatch: pytest.MonkeyPatch, step: float = 0.5) -> None:
    monkeypatch.setattr(trainer_mod.time, "perf_counter", _Clock(step))


def test_run_eval_model_reports_timing_and_base_precision(
    tmp_model: Path, tmp_suite: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every row carries generated_tokens + latency_ms; the suite carries the rollups."""
    answers = {
        "Task: reverse\nInput: abc\nOutput:": "cba",
        "Task: upper\nInput: hello\nOutput:": "HELLO",
    }
    monkeypatch.setattr(exporter_mod, "_load_eval_backend", lambda: _fake_backend(answers))
    _freeze_clock(monkeypatch, step=0.5)

    summary = exporter_mod.run_eval_model(
        str(tmp_model), str(tmp_suite), batch_size=1, base_load_in_4bit=True
    )

    for row in summary["results"]:
        assert row["generated_tokens"] > 0
        assert row["latency_ms"] == 500.0
    assert summary["median_latency_ms"] == 500.0
    assert summary["tokens_per_s"] > 0
    assert summary["batch_size"] == 1
    assert summary["base_load_in_4bit"] is True


def test_run_eval_model_gguf_reports_timing(
    tmp_path: Path, tmp_suite: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The llama.cpp path times each process and approximates its token count."""
    model_dir = tmp_path / "gguf-out"
    model_dir.mkdir()
    (model_dir / "Model.Q4_K_M.gguf").write_bytes(b"\x00")
    monkeypatch.setattr(
        exporter_mod,
        "_run_llama_completion",
        lambda gguf, prompt, max_tokens: "cba" if "reverse" in prompt else "HELLO",
    )
    _freeze_clock(monkeypatch, step=0.25)

    summary = exporter_mod.run_eval_model(str(model_dir), str(tmp_suite))

    assert [r["latency_ms"] for r in summary["results"]] == [250.0, 250.0]
    assert [r["generated_tokens"] for r in summary["results"]] == [1, 1]
    assert summary["median_latency_ms"] == 250.0


def test_run_eval_model_perplexity_on_gguf_is_a_user_error(
    tmp_path: Path, tmp_suite: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Perplexity needs a transformers forward pass — a GGUF export cannot give one."""
    model_dir = tmp_path / "gguf-out"
    model_dir.mkdir()
    (model_dir / "Model.Q4_K_M.gguf").write_bytes(b"\x00")

    with pytest.raises(CliError) as exc_info:
        exporter_mod.run_eval_model(str(model_dir), str(tmp_suite), perplexity=True)
    assert exc_info.value.code == 1
    assert "perplexity" in exc_info.value.message.lower()
    assert exc_info.value.remediation


def test_run_eval_model_perplexity_uses_a_labelled_forward_pass(
    tmp_model: Path, tmp_suite: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _fake_backend({})
    monkeypatch.setattr(exporter_mod, "_load_eval_backend", lambda: backend)

    summary = exporter_mod.run_eval_model(str(tmp_model), str(tmp_suite), perplexity=True)

    assert summary["perplexity"] == pytest.approx(1.0)  # loss 0.0 ⇒ exp(0) == 1
    assert summary["files"][0]["perplexity"] == pytest.approx(1.0)


def test_run_eval_model_writes_a_suite_keyed_eval_json(
    tmp_model: Path, tmp_suite: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exporter_mod, "_load_eval_backend", lambda: _fake_backend({}))

    exporter_mod.run_eval_model(
        str(tmp_model), str(tmp_suite), batch_size=2, base_load_in_4bit=False
    )

    written = json.loads((tmp_model / "eval" / "suite.json").read_text(encoding="utf-8"))
    assert written["suite"] == "suite"
    assert written["target"] == "model"
    assert written["batch_size"] == 2
    assert written["base_load_in_4bit"] is False
    assert (tmp_model / "eval.json").is_file()


def test_run_eval_model_scores_an_instruction_suite(
    tmp_model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = tmp_path / "instr.jsonl"
    suite.write_text(
        json.dumps(
            {
                "task": "short",
                "input": "abc",
                "expected_output": "ok",
                "constraints": [{"max_words": 1}],
            }
        )
        + "\n"
        + json.dumps(
            {
                "task": "short",
                "input": "xyz",
                "expected_output": "ok",
                "constraints": [{"max_words": 1}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    answers = {
        "Task: short\nInput: abc\nOutput:": "ok",
        "Task: short\nInput: xyz\nOutput:": "far too many words",
    }
    monkeypatch.setattr(exporter_mod, "_load_eval_backend", lambda: _fake_backend(answers))

    summary = exporter_mod.run_eval_model(str(tmp_model), str(suite))

    assert [r["constraints_passed"] for r in summary["results"]] == [True, False]
    assert summary["compliance_pct"] == 50.0


def test_run_eval_model_signature_keeps_backward_compatible_keywords() -> None:
    params = inspect.signature(exporter_mod.run_eval_model).parameters
    for name in ("perplexity", "tool_call_family", "base_load_in_4bit"):
        assert params[name].default is None or params[name].default is False


# t8 — repeated --suite by name, per-suite eval/<name>.json, --perplexity,
# --tool-call-family, and the train/eval overlap refusal.
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_suite_b(tmp_path: Path) -> Path:
    """A second, distinctly-named task-schema suite file."""
    f = tmp_path / "b.jsonl"
    f.write_text(
        '{"task": "shout", "input": "hi", "expected_output": "HI"}\n',
        encoding="utf-8",
    )
    return f


def _fake_run_eval_suites_keyed(named: dict[str, dict[str, Any]]) -> Any:
    """Build a fake ``run_eval``/``run_eval_model`` that returns ``{"suites": ...}``.

    Stands in for the richer seam signature (t6) that reports true per-suite
    results, keyed by the suite name this test expects.
    """

    def _fake(
        target_path: str,
        *,
        suite_paths: list[Path],
        quant: str | None = None,
        batch_size: int = 8,
        perplexity: bool = False,
        tool_call_family: str | None = None,
    ) -> dict[str, Any]:
        return {"suites": dict(named)}

    return _fake


class TestNamedSuitesMultiInvocation:
    """``--suite A --suite B`` in one invocation: named suites, one JSON file each."""

    def test_json_output_lists_each_suite_by_name(
        self,
        tmp_adapter: Path,
        tmp_suite: Path,
        tmp_suite_b: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        payload_a = {"total": 2, "exact_match": 2, "exact_match_pct": 100.0, "results": []}
        payload_b = {"total": 1, "exact_match": 0, "exact_match_pct": 0.0, "results": []}
        monkeypatch.setattr(
            eval_mod,
            "run_eval",
            _fake_run_eval_suites_keyed({"suite": payload_a, "b": payload_b}),
        )

        args = _make_args(
            adapter=str(tmp_adapter),
            suite=[str(tmp_suite), str(tmp_suite_b)],
            json=True,
            in_container=True,
        )
        rc = cmd_eval(args)
        assert rc in (None, 0)

        data = json.loads(capsys.readouterr().out)
        assert set(data["suites"]) == {"suite", "b"}
        assert data["suites"]["suite"]["total"] == 2
        assert data["suites"]["b"]["exact_match_pct"] == 0.0

    def test_text_output_prints_one_block_per_suite(
        self,
        tmp_adapter: Path,
        tmp_suite: Path,
        tmp_suite_b: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        payload_a = {"total": 2, "exact_match": 2, "exact_match_pct": 100.0, "results": []}
        payload_b = {"total": 1, "exact_match": 0, "exact_match_pct": 0.0, "results": []}
        monkeypatch.setattr(
            eval_mod,
            "run_eval",
            _fake_run_eval_suites_keyed({"suite": payload_a, "b": payload_b}),
        )

        args = _make_args(
            adapter=str(tmp_adapter),
            suite=[str(tmp_suite), str(tmp_suite_b)],
            in_container=True,
        )
        cmd_eval(args)
        out = capsys.readouterr().out
        assert "suite: suite" in out
        assert "suite: b" in out
        # two separate blocks, blank-line separated
        assert out.count("total:") == 2

    def test_writes_eval_json_per_suite_in_one_invocation(
        self,
        tmp_adapter: Path,
        tmp_suite: Path,
        tmp_suite_b: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        payload_a = {"total": 2, "exact_match": 2, "exact_match_pct": 100.0, "results": []}
        payload_b = {"total": 1, "exact_match": 0, "exact_match_pct": 0.0, "results": []}
        monkeypatch.setattr(
            eval_mod,
            "run_eval",
            _fake_run_eval_suites_keyed({"suite": payload_a, "b": payload_b}),
        )

        args = _make_args(
            adapter=str(tmp_adapter),
            suite=[str(tmp_suite), str(tmp_suite_b)],
            in_container=True,
        )
        cmd_eval(args)

        suite_json = tmp_adapter / "eval" / "suite.json"
        b_json = tmp_adapter / "eval" / "b.json"
        assert suite_json.is_file()
        assert b_json.is_file()
        assert json.loads(suite_json.read_text())["total"] == 2
        assert json.loads(b_json.read_text())["total"] == 1

    def test_old_flat_seam_shape_is_wrapped_under_every_given_suite_name(
        self,
        tmp_adapter: Path,
        tmp_suite: Path,
        tmp_suite_b: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A seam still returning the old single-payload shape gets wrapped under
        every suite name given — a graceful fallback, not a crash."""
        monkeypatch.setattr(eval_mod, "run_eval", _fake_run_eval_perfect)

        args = _make_args(
            adapter=str(tmp_adapter),
            suite=[str(tmp_suite), str(tmp_suite_b)],
            json=True,
            in_container=True,
        )
        cmd_eval(args)
        data = json.loads(capsys.readouterr().out)
        assert set(data["suites"]) == {"suite", "b"}
        assert data["suites"]["suite"] == data["suites"]["b"]


class TestSuiteNaming:
    """Suite names derive from the path stem; collisions exit 1 with a hint."""

    def test_name_derives_from_path_stem(
        self, tmp_adapter: Path, tmp_suite: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        named = eval_mod._resolve_named_suites([str(tmp_suite)])
        assert list(named) == ["suite"]
        assert named["suite"] == [tmp_suite]

    def test_two_entries_with_the_same_stem_collide(
        self, tmp_adapter: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        d1 = tmp_path / "one"
        d2 = tmp_path / "two"
        d1.mkdir()
        d2.mkdir()
        f1 = d1 / "suite.jsonl"
        f2 = d2 / "suite.jsonl"
        for f in (f1, f2):
            f.write_text('{"task": "t", "input": "x", "expected_output": "y"}\n', encoding="utf-8")

        def _must_not_launch(*args: Any, **kwargs: Any) -> int:
            raise AssertionError("container.launch called despite a suite-name collision")

        monkeypatch.setattr(eval_mod.container, "launch", _must_not_launch)

        args = _make_args(adapter=str(tmp_adapter), suite=[str(f1), str(f2)], in_container=False)
        with pytest.raises(CliError) as exc_info:
            cmd_eval(args)
        err = exc_info.value
        assert err.code == 1
        assert "collision" in err.message.lower()
        assert err.remediation


class TestSuiteSchemaDetection:
    """Every suite kind (chat/task/instruction/structured/toolcall) validates via detect_schema."""

    def test_chat_suite_is_accepted(
        self, tmp_adapter: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        suite = tmp_path / "chatsuite.jsonl"
        suite.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(eval_mod, "run_eval", _fake_run_eval_perfect)
        args = _make_args(adapter=str(tmp_adapter), suite=str(suite), in_container=True)
        rc = cmd_eval(args)
        assert rc in (None, 0)

    def test_instruction_suite_is_accepted(
        self, tmp_adapter: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        suite = tmp_path / "instr.jsonl"
        suite.write_text(
            json.dumps(
                {
                    "task": "t",
                    "input": "x",
                    "expected_output": "y",
                    "constraints": [{"max_words": 5}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(eval_mod, "run_eval", _fake_run_eval_perfect)
        args = _make_args(adapter=str(tmp_adapter), suite=str(suite), in_container=True)
        rc = cmd_eval(args)
        assert rc in (None, 0)

    def test_structured_suite_is_accepted(
        self, tmp_adapter: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        suite = tmp_path / "structured.jsonl"
        suite.write_text(
            json.dumps({"task": "t", "input": "x", "json_schema": {"type": "object"}}) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(eval_mod, "run_eval", _fake_run_eval_perfect)
        args = _make_args(adapter=str(tmp_adapter), suite=str(suite), in_container=True)
        rc = cmd_eval(args)
        assert rc in (None, 0)

    def test_toolcall_suite_is_accepted(
        self, tmp_adapter: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        suite = tmp_path / "toolcall.jsonl"
        suite.write_text(
            json.dumps(
                {
                    "task": "t",
                    "input": "x",
                    "expected_tool_call": {"name": "search", "arguments": {"q": "x"}},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(eval_mod, "run_eval", _fake_run_eval_perfect)
        args = _make_args(adapter=str(tmp_adapter), suite=str(suite), in_container=True)
        rc = cmd_eval(args)
        assert rc in (None, 0)

    def test_unrecognisable_suite_raises_before_launch(
        self, tmp_adapter: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        suite = tmp_path / "mystery.jsonl"
        suite.write_text(json.dumps({"foo": "bar"}) + "\n", encoding="utf-8")

        def _must_not_launch(*args: Any, **kwargs: Any) -> int:
            raise AssertionError("container.launch called for an undetectable suite schema")

        monkeypatch.setattr(eval_mod.container, "launch", _must_not_launch)
        args = _make_args(adapter=str(tmp_adapter), suite=str(suite), in_container=False)
        with pytest.raises(CliError) as exc_info:
            cmd_eval(args)
        assert exc_info.value.code == 1


class TestTrainEvalOverlapRefusal:
    """--train-dataset (or training_metadata.json's dataset.path) gates the launch."""

    def test_explicit_train_dataset_overlap_exits_1_before_launch(
        self,
        tmp_adapter: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        train = tmp_path / "train.jsonl"
        train.write_text(
            '{"task": "reverse", "input": "abc", "expected_output": "cba"}\n',
            encoding="utf-8",
        )
        suite = tmp_path / "eval.jsonl"
        suite.write_text(
            '{"task": "reverse", "input": "abc", "expected_output": "cba"}\n',
            encoding="utf-8",
        )

        launched: list[Any] = []

        def _must_not_launch(*args: Any, **kwargs: Any) -> int:
            launched.append(args)
            raise AssertionError("container.launch called despite a train/eval overlap")

        monkeypatch.setattr(eval_mod.container, "launch", _must_not_launch)

        args = _make_args(
            adapter=str(tmp_adapter),
            suite=str(suite),
            train_dataset=str(train),
            in_container=False,
        )
        with pytest.raises(CliError) as exc_info:
            cmd_eval(args)
        err = exc_info.value
        assert err.code == 1
        assert not launched
        assert "train.jsonl:1" in err.remediation
        assert "eval.jsonl:1" in err.remediation

    def test_no_overlap_proceeds_to_launch(
        self,
        tmp_adapter: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        train = tmp_path / "train.jsonl"
        train.write_text(
            '{"task": "reverse", "input": "abc", "expected_output": "cba"}\n',
            encoding="utf-8",
        )
        suite = tmp_path / "eval.jsonl"
        suite.write_text(
            '{"task": "upper", "input": "hello", "expected_output": "HELLO"}\n',
            encoding="utf-8",
        )

        monkeypatch.setattr(eval_mod.container, "launch", lambda *a, **kw: dict(_FAKE_EVAL_SUMMARY))

        args = _make_args(
            adapter=str(tmp_adapter),
            suite=str(suite),
            train_dataset=str(train),
            in_container=False,
        )
        rc = cmd_eval(args)
        assert rc in (None, 0)

    def test_training_metadata_dataset_used_when_no_train_dataset_flag(
        self,
        tmp_adapter: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import sloth.tune.metadata as metadata_mod

        train = tmp_path / "train.jsonl"
        train.write_text(
            '{"task": "reverse", "input": "abc", "expected_output": "cba"}\n',
            encoding="utf-8",
        )
        metadata_mod.write_metadata(
            tmp_adapter,
            model="unsloth/Qwen3-4B",
            method="lora",
            dataset_path=train,
            hyperparameters={},
            timestamp="2026-01-01T00:00:00+00:00",
        )
        suite = tmp_path / "eval.jsonl"
        suite.write_text(
            '{"task": "reverse", "input": "abc", "expected_output": "cba"}\n',
            encoding="utf-8",
        )

        def _must_not_launch(*args: Any, **kwargs: Any) -> int:
            raise AssertionError("container.launch called despite a train/eval overlap")

        monkeypatch.setattr(eval_mod.container, "launch", _must_not_launch)

        args = _make_args(adapter=str(tmp_adapter), suite=str(suite), in_container=False)
        with pytest.raises(CliError) as exc_info:
            cmd_eval(args)
        assert exc_info.value.code == 1

    def test_train_dataset_flag_overrides_training_metadata(
        self,
        tmp_adapter: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit --train-dataset wins even when training_metadata.json points
        elsewhere (and that elsewhere would have overlapped)."""
        import sloth.tune.metadata as metadata_mod

        metadata_train = tmp_path / "metadata_train.jsonl"
        metadata_train.write_text(
            '{"task": "reverse", "input": "abc", "expected_output": "cba"}\n',
            encoding="utf-8",
        )
        metadata_mod.write_metadata(
            tmp_adapter,
            model="unsloth/Qwen3-4B",
            method="lora",
            dataset_path=metadata_train,
            hyperparameters={},
            timestamp="2026-01-01T00:00:00+00:00",
        )

        explicit_train = tmp_path / "explicit_train.jsonl"
        explicit_train.write_text(
            '{"task": "unrelated", "input": "zz", "expected_output": "qq"}\n',
            encoding="utf-8",
        )
        suite = tmp_path / "eval.jsonl"
        suite.write_text(
            '{"task": "reverse", "input": "abc", "expected_output": "cba"}\n',
            encoding="utf-8",
        )

        monkeypatch.setattr(eval_mod.container, "launch", lambda *a, **kw: dict(_FAKE_EVAL_SUMMARY))

        args = _make_args(
            adapter=str(tmp_adapter),
            suite=str(suite),
            train_dataset=str(explicit_train),
            in_container=False,
        )
        rc = cmd_eval(args)
        assert rc in (None, 0)  # no overlap against the *explicit* train dataset

    def test_neither_train_dataset_nor_metadata_skips_with_diagnostic(
        self,
        tmp_adapter: Path,
        tmp_suite: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(eval_mod.container, "launch", lambda *a, **kw: dict(_FAKE_EVAL_SUMMARY))
        args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=False)
        rc = cmd_eval(args)
        assert rc in (None, 0)
        err = capsys.readouterr().err
        assert "skipping" in err.lower()


class TestPerplexityAndToolCallFamilyFlags:
    """--perplexity / --tool-call-family register, forward in-container, and reach the seam."""

    def test_flags_parse_and_default(self) -> None:
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        register(sub)
        args = parser.parse_args(["eval", "--adapter", "/a", "--suite", "/b.jsonl"])
        assert args.perplexity is False
        assert args.tool_call_family is None

        args2 = parser.parse_args(
            [
                "eval",
                "--adapter",
                "/a",
                "--suite",
                "/b.jsonl",
                "--perplexity",
                "--tool-call-family",
                "qwen",
            ]
        )
        assert args2.perplexity is True
        assert args2.tool_call_family == "qwen"

    def test_forwarded_into_container_argv(
        self,
        tmp_adapter: Path,
        tmp_suite: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def _capture_launch(sloth_args: list[str], **kwargs: Any) -> dict[str, Any]:
            captured["sloth_args"] = list(sloth_args)
            return dict(_FAKE_EVAL_SUMMARY)

        monkeypatch.setattr(eval_mod.container, "launch", _capture_launch)

        args = _make_args(
            adapter=str(tmp_adapter),
            suite=str(tmp_suite),
            perplexity=True,
            tool_call_family="qwen",
            in_container=False,
        )
        cmd_eval(args)

        forwarded = captured["sloth_args"]
        assert "--perplexity" in forwarded
        assert "--tool-call-family" in forwarded
        assert forwarded[forwarded.index("--tool-call-family") + 1] == "qwen"

    def test_omitted_when_not_set(
        self,
        tmp_adapter: Path,
        tmp_suite: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def _capture_launch(sloth_args: list[str], **kwargs: Any) -> dict[str, Any]:
            captured["sloth_args"] = list(sloth_args)
            return dict(_FAKE_EVAL_SUMMARY)

        monkeypatch.setattr(eval_mod.container, "launch", _capture_launch)

        args = _make_args(adapter=str(tmp_adapter), suite=str(tmp_suite), in_container=False)
        cmd_eval(args)

        forwarded = captured["sloth_args"]
        assert "--perplexity" not in forwarded
        assert "--tool-call-family" not in forwarded

    def test_forwarded_to_seam_when_signature_accepts_them(
        self,
        tmp_adapter: Path,
        tmp_suite: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[dict[str, Any]] = []

        def _rich_run_eval(
            adapter_path: str,
            *,
            suite_paths: list[Path],
            quant: str | None = None,
            batch_size: int = 8,
            perplexity: bool = False,
            tool_call_family: str | None = None,
        ) -> dict[str, Any]:
            calls.append({"perplexity": perplexity, "tool_call_family": tool_call_family})
            return _fake_run_eval_perfect(adapter_path, suite_paths=suite_paths)

        monkeypatch.setattr(eval_mod, "run_eval", _rich_run_eval)

        args = _make_args(
            adapter=str(tmp_adapter),
            suite=str(tmp_suite),
            perplexity=True,
            tool_call_family="qwen",
            in_container=True,
        )
        rc = cmd_eval(args)
        assert rc in (None, 0)
        assert calls == [{"perplexity": True, "tool_call_family": "qwen"}]

    def test_dropped_when_seam_signature_lacks_them(
        self,
        tmp_adapter: Path,
        tmp_suite: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A seam with the older, narrower signature (no perplexity/tool_call_family
        params, no **kwargs) is called without those keywords — no TypeError."""

        def _narrow_run_eval(
            adapter_path: str,
            *,
            suite_paths: list[Path],
            quant: str | None = None,
            batch_size: int = 8,
        ) -> dict[str, Any]:
            return _fake_run_eval_perfect(adapter_path, suite_paths=suite_paths)

        monkeypatch.setattr(eval_mod, "run_eval", _narrow_run_eval)

        args = _make_args(
            adapter=str(tmp_adapter),
            suite=str(tmp_suite),
            perplexity=True,
            tool_call_family="qwen",
            in_container=True,
        )
        rc = cmd_eval(args)
        assert rc in (None, 0)  # no TypeError: unexpected keyword argument


class TestBatchSizeRecordedInResult:
    """--batch-size is recorded in every written eval/<name>.json result."""

    def test_batch_size_recorded_in_every_suite_file(
        self,
        tmp_adapter: Path,
        tmp_suite: Path,
        tmp_suite_b: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        payload_a = {"total": 2, "exact_match": 2, "exact_match_pct": 100.0, "results": []}
        payload_b = {"total": 1, "exact_match": 0, "exact_match_pct": 0.0, "results": []}
        monkeypatch.setattr(
            eval_mod,
            "run_eval",
            _fake_run_eval_suites_keyed({"suite": payload_a, "b": payload_b}),
        )

        args = _make_args(
            adapter=str(tmp_adapter),
            suite=[str(tmp_suite), str(tmp_suite_b)],
            batch_size=4,
            in_container=True,
        )
        cmd_eval(args)

        for name in ("suite", "b"):
            record = json.loads((tmp_adapter / "eval" / f"{name}.json").read_text())
            assert record["batch_size"] == 4


# ---------------------------------------------------------------------------
# Integration of t6 (per-file seam results) with t8 (named suites): the CLI must
# split the seam's ``files`` entries by suite instead of duplicating the aggregate
# under every name (which would clobber the trainer's own eval/<stem>.json).
# ---------------------------------------------------------------------------


class TestNamedResultsSplitPerSuite:
    @staticmethod
    def _seam_payload(paths: list[Path]) -> dict[str, Any]:
        from sloth.tune import metrics

        files = []
        for i, p in enumerate(paths):
            rows = [
                {
                    "index": i,
                    "task": "t",
                    "input": "i",
                    "expected_output": "a",
                    "prediction": "a" if i == 0 else "b",
                    "exact_match": i == 0,
                    "f1": 1.0 if i == 0 else 0.0,
                    "file": str(p),
                }
            ]
            files.append(metrics.file_entry(p, rows))
        payload = metrics.aggregate(files)
        payload["batch_size"] = 4
        payload["base_load_in_4bit"] = True
        return payload

    def test_single_file_suites_get_their_own_file_entry(self, tmp_path: Path) -> None:
        from sloth.cli._commands.eval import _normalize_named_results

        a, b = tmp_path / "alpha.jsonl", tmp_path / "beta.jsonl"
        raw = self._seam_payload([a, b])
        named = _normalize_named_results(raw, ["alpha", "beta"], {"alpha": [a], "beta": [b]})
        assert named["alpha"]["exact_match"] == 1 and named["beta"]["exact_match"] == 0
        assert named["alpha"]["total"] == 1 and named["beta"]["total"] == 1
        # run-level fields are carried onto every suite payload
        assert named["alpha"]["batch_size"] == 4 and named["beta"]["base_load_in_4bit"] is True
        # and the aggregate is NOT duplicated under both names
        assert named["alpha"] is not named["beta"]

    def test_directory_suite_aggregates_its_files(self, tmp_path: Path) -> None:
        from sloth.cli._commands.eval import _normalize_named_results

        d = tmp_path / "suite-dir"
        d.mkdir()
        a, b = d / "one.jsonl", d / "two.jsonl"
        raw = self._seam_payload([a, b])
        named = _normalize_named_results(raw, ["suite-dir"], {"suite-dir": [a, b]})
        assert named["suite-dir"]["total"] == 2
        assert named["suite-dir"]["exact_match"] == 1
        assert [f["path"] for f in named["suite-dir"]["files"]] == [str(a), str(b)]

    def test_flat_payload_without_files_is_duplicated(self) -> None:
        from sloth.cli._commands.eval import _normalize_named_results

        raw = {"total": 1, "exact_match": 1, "exact_match_pct": 100.0, "f1": 1.0}
        named = _normalize_named_results(raw, ["x", "y"], {"x": [], "y": []})
        assert named == {"x": raw, "y": raw}
