"""Smoke tests for the unsloth-cli CLI entry point and its verbs."""

from __future__ import annotations

import json

import pytest

from sloth import __version__
from sloth.cli import main
from sloth.cli._output import emit_result
from sloth.explain import known_paths


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    assert rc == 0
    assert "usage: unsloth-cli" in capsys.readouterr().out


def test_unknown_command_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["bogus"])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "hint:" in err


# --- whoami ---------------------------------------------------------------


def test_whoami_text(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["whoami"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "nick: unsloth-cli" in out
    assert "backend: claude" in out
    assert "model:" in out


def test_whoami_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["whoami", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["nick"] == "unsloth-cli"
    assert payload["version"] == __version__
    assert payload["backend"] == "claude"


# --- learn ----------------------------------------------------------------


def test_learn_text(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["learn"])
    assert rc == 0
    out = capsys.readouterr().out
    assert len(out) >= 200
    assert "unsloth-cli" in out
    assert "Exit-code policy" in out
    assert "--json" in out
    assert "explain" in out


def test_learn_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["learn", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "unsloth-cli"
    assert payload["version"] == __version__
    assert payload["json_support"] is True


# --- explain --------------------------------------------------------------


def test_explain_root(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["explain"])
    assert rc == 0
    assert "# unsloth-cli" in capsys.readouterr().out


def test_explain_self(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["explain", "unsloth-cli"])
    assert rc == 0
    assert capsys.readouterr().out.startswith("#")


def test_explain_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["explain", "whoami", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["path"] == ["whoami"]
    assert "unsloth-cli whoami" in payload["markdown"]


def test_explain_script_name_resolves(capsys: pytest.CaptureFixture[str]) -> None:
    # The console-script name is `sloth` (dist name is `unsloth-cli`). The
    # rubric's explain_self check runs `explain sloth`, so it must resolve.
    rc = main(["explain", "sloth"])
    assert rc == 0
    assert "# unsloth-cli" in capsys.readouterr().out


def test_explain_unknown_path_errors(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["explain", "nonexistent"])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "hint:" in captured.err


def test_every_catalog_path_resolves(capsys: pytest.CaptureFixture[str]) -> None:
    for path in known_paths():
        rc = main(["explain", *path])
        assert rc == 0, f"explain {' '.join(path)} failed"
        capsys.readouterr()


# --- output contract ------------------------------------------------------


def test_emit_result_json_is_single_line(capsys: pytest.CaptureFixture[str]) -> None:
    # Agents read stdout line by line: a nested payload must stay on ONE line
    # (no `indent=` in sloth/cli/_output.py), so the last stdout line of a
    # container run is a complete JSON document.
    payload = {"ok": True, "nested": {"a": [1, 2, {"b": "c"}]}, "n": 3}
    emit_result(payload, json_mode=True)
    out = capsys.readouterr().out
    assert out.endswith("\n")
    assert out.count("\n") == 1
    assert json.loads(out) == payload
