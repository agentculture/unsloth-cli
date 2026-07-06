"""Tests for ``sloth validate`` — the standalone dataset validator.

``sloth validate`` calls the *same* :func:`sloth.tune.datasets.validate_dataset`
function that ``sloth train`` uses internally (see ``tests/test_tune_datasets.py``
and ``tests/test_cmd_train.py`` for the shared rule set), so no validation rule
is duplicated here — these tests exercise the CLI wiring around that shared
function: schema auto-detection, the error/hint contract, --json shapes, and
exit codes.

Covers:
* a valid chat/task dataset is accepted (exit 0, ``{valid, schema, line_count}``)
* an invalid dataset is rejected with the SAME rules ``sloth train`` applies
  (exit 1, ``error:``/``hint:`` two-line contract)
* a missing dataset file exits 1 (a friendlier user error than the validator's
  own env-error path, which is reserved for an existing-but-unreadable file)
* --schema auto-detection (chat vs task) and an explicit --schema override
* register() wires the subparser correctly
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

import sloth.cli._commands.validate as validate_mod
from sloth.cli._commands.validate import cmd_validate, register
from sloth.cli._errors import CliError
from sloth.cli._output import emit_error
from sloth.tune.datasets import validate_dataset

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_VALID_CHAT = (
    '{"messages": [{"role": "user", "content": "hi"}, '
    '{"role": "assistant", "content": "hello"}]}\n'
)

_VALID_TASK = '{"task": "reverse", "input": "abc", "expected_output": "cba"}\n'


def _make_args(
    dataset: Path,
    *,
    schema: str | None = None,
    json_mode: bool = False,
) -> argparse.Namespace:
    """Build the Namespace argparse would produce for ``sloth validate``."""
    return argparse.Namespace(
        dataset=str(dataset),
        schema=schema,
        json=json_mode,
    )


def _write_dataset(tmp_path: Path, body: str, name: str = "data.jsonl") -> Path:
    f = tmp_path / name
    f.write_text(body, encoding="utf-8")
    return f


@pytest.fixture()
def valid_chat_dataset(tmp_path: Path) -> Path:
    return _write_dataset(tmp_path, _VALID_CHAT, name="chat.jsonl")


@pytest.fixture()
def valid_task_dataset(tmp_path: Path) -> Path:
    return _write_dataset(tmp_path, _VALID_TASK, name="task.jsonl")


# ---------------------------------------------------------------------------
# Valid dataset — accepted, exit 0
# ---------------------------------------------------------------------------


def test_valid_chat_dataset_auto_detected(
    valid_chat_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A valid chat-schema dataset is accepted with the schema auto-detected."""
    rc = cmd_validate(_make_args(valid_chat_dataset))
    assert rc in (None, 0)
    out = capsys.readouterr().out
    assert "valid" in out.lower()
    assert "chat" in out


def test_valid_task_dataset_explicit_schema(
    valid_task_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A valid task-schema dataset is accepted when --schema is passed explicitly."""
    rc = cmd_validate(_make_args(valid_task_dataset, schema="task"))
    assert rc in (None, 0)
    out = capsys.readouterr().out
    assert "task" in out


def test_valid_dataset_json_shape(
    valid_chat_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--json emits exactly {valid, schema, line_count} on success."""
    rc = cmd_validate(_make_args(valid_chat_dataset, json_mode=True))
    assert rc in (None, 0)
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"valid": True, "schema": "chat", "line_count": 1}


def test_valid_task_dataset_json_line_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """line_count reflects the number of parsed records."""
    body = (
        '{"task": "a", "input": "1", "expected_output": "1"}\n'
        '{"task": "b", "input": "2", "expected_output": "2"}\n'
        '{"task": "c", "input": "3", "expected_output": "3"}\n'
    )
    dataset = _write_dataset(tmp_path, body, name="multi.jsonl")
    rc = cmd_validate(_make_args(dataset, schema="task", json_mode=True))
    assert rc in (None, 0)
    payload = json.loads(capsys.readouterr().out)
    assert payload["line_count"] == 3


def test_explicit_schema_skips_auto_detect_diagnostic(
    valid_task_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An explicit --schema does not emit the auto-detected-schema diagnostic."""
    rc = cmd_validate(_make_args(valid_task_dataset, schema="task"))
    assert rc in (None, 0)
    err = capsys.readouterr().err
    assert "auto-detected" not in err


def test_auto_detect_emits_diagnostic_to_stderr(
    valid_chat_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Schema auto-detection is echoed as a diagnostic on stderr (never stdout)."""
    rc = cmd_validate(_make_args(valid_chat_dataset))
    assert rc in (None, 0)
    captured = capsys.readouterr()
    assert "auto-detected schema" in captured.err
    assert "auto-detected schema" not in captured.out


# ---------------------------------------------------------------------------
# Invalid dataset — the SAME rules train applies, exit 1
# ---------------------------------------------------------------------------


def test_invalid_role_rejected_same_as_train(
    tmp_path: Path,
) -> None:
    """An invalid chat role is rejected — the same rule sloth train enforces."""
    bad = _write_dataset(
        tmp_path, '{"messages": [{"role": "wizard", "content": "x"}]}\n', name="bad.jsonl"
    )
    with pytest.raises(CliError) as exc_info:
        cmd_validate(_make_args(bad, schema="chat"))
    assert exc_info.value.code == 1
    assert "wizard" in exc_info.value.message
    assert exc_info.value.remediation


def test_invalid_dataset_matches_validate_dataset_directly(tmp_path: Path) -> None:
    """The CliError raised by the verb is identical in shape to calling
    validate_dataset() directly — proving no rules are duplicated."""
    bad = _write_dataset(tmp_path, '{"messages": []}\n', name="bad.jsonl")

    with pytest.raises(CliError) as from_verb:
        cmd_validate(_make_args(bad, schema="chat"))

    with pytest.raises(CliError) as from_lib:
        validate_dataset(bad, "chat")

    assert from_verb.value.code == from_lib.value.code
    assert from_verb.value.message == from_lib.value.message


def test_invalid_json_line_rejected(tmp_path: Path) -> None:
    """Malformed JSON on a line raises CliError(code=1)."""
    bad = _write_dataset(tmp_path, "not valid json\n", name="bad.jsonl")
    with pytest.raises(CliError) as exc_info:
        cmd_validate(_make_args(bad, schema="chat"))
    assert exc_info.value.code == 1


def test_empty_dataset_rejected(tmp_path: Path) -> None:
    """A dataset with no records (empty file) raises CliError(code=1)."""
    empty = _write_dataset(tmp_path, "", name="empty.jsonl")
    with pytest.raises(CliError) as exc_info:
        cmd_validate(_make_args(empty, schema="chat"))
    assert exc_info.value.code == 1


def test_invalid_dataset_error_hint_contract(tmp_path: Path) -> None:
    """The invalid-dataset CliError renders as ``error:``/``hint:`` lines."""
    bad = _write_dataset(tmp_path, "not valid json\n", name="bad.jsonl")
    with pytest.raises(CliError) as exc_info:
        cmd_validate(_make_args(bad, schema="chat"))
    buf = io.StringIO()
    emit_error(exc_info.value, json_mode=False, stream=buf)
    text = buf.getvalue()
    assert text.startswith("error:")
    assert "hint:" in text


def test_invalid_dataset_json_error(tmp_path: Path) -> None:
    """The invalid-dataset CliError renders as structured JSON when requested."""
    bad = _write_dataset(tmp_path, "not valid json\n", name="bad.jsonl")
    with pytest.raises(CliError) as exc_info:
        cmd_validate(_make_args(bad, schema="chat", json_mode=True))
    buf = io.StringIO()
    emit_error(exc_info.value, json_mode=True, stream=buf)
    payload = json.loads(buf.getvalue())
    assert payload["code"] == 1
    assert "message" in payload
    assert "remediation" in payload


def test_task_schema_mismatch_rejected(tmp_path: Path) -> None:
    """A record missing a required task key is rejected against --schema task."""
    bad = _write_dataset(tmp_path, '{"task": "x", "input": "y"}\n', name="bad_task.jsonl")
    with pytest.raises(CliError) as exc_info:
        cmd_validate(_make_args(bad, schema="task"))
    assert exc_info.value.code == 1
    assert "expected_output" in exc_info.value.message


# ---------------------------------------------------------------------------
# Missing file — exit 1 (a friendlier user error than validate_dataset's own
# env-error path, which is reserved for an existing-but-unreadable file)
# ---------------------------------------------------------------------------


def test_missing_file_raises_cli_error_1(tmp_path: Path) -> None:
    """A dataset path that does not exist raises CliError(code=1)."""
    with pytest.raises(CliError) as exc_info:
        cmd_validate(_make_args(tmp_path / "nope.jsonl"))
    assert exc_info.value.code == 1
    assert "not found" in exc_info.value.message
    assert exc_info.value.remediation


def test_missing_file_does_not_call_validate_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing file is caught before validate_dataset is ever invoked."""
    mock_validate = Mock()
    monkeypatch.setattr(validate_mod, "validate_dataset", mock_validate)
    with pytest.raises(CliError):
        cmd_validate(_make_args(tmp_path / "nope.jsonl"))
    mock_validate.assert_not_called()


def test_missing_file_error_hint_contract(tmp_path: Path) -> None:
    """The missing-file CliError renders as ``error:``/``hint:`` lines."""
    with pytest.raises(CliError) as exc_info:
        cmd_validate(_make_args(tmp_path / "nope.jsonl"))
    buf = io.StringIO()
    emit_error(exc_info.value, json_mode=False, stream=buf)
    text = buf.getvalue()
    assert text.startswith("error:")
    assert "hint:" in text


# ---------------------------------------------------------------------------
# Shared code path — validate.py calls the SAME validate_dataset as train
# ---------------------------------------------------------------------------


def test_calls_shared_validate_dataset(
    valid_chat_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cmd_validate delegates to sloth.tune.datasets.validate_dataset (the same
    function sloth train calls) rather than reimplementing the rules."""
    calls: list[tuple[Any, str]] = []

    def _spy(path: Any, schema: str) -> list[dict]:
        calls.append((path, schema))
        return validate_dataset(path, schema)

    monkeypatch.setattr(validate_mod, "validate_dataset", _spy)
    rc = cmd_validate(_make_args(valid_chat_dataset, schema="chat"))
    assert rc in (None, 0)
    assert len(calls) == 1
    assert calls[0][1] == "chat"


# ---------------------------------------------------------------------------
# register() wires the subparser correctly
# ---------------------------------------------------------------------------


def test_register_adds_validate_subparser(tmp_path: Path) -> None:
    """register() adds a ``validate`` subparser with --dataset/--schema/--json."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    ds = str(tmp_path / "data.jsonl")
    args = parser.parse_args(["validate", "--dataset", ds])
    assert args.command == "validate"
    assert args.dataset == ds
    assert args.schema is None
    assert args.json is False
    assert callable(args.func)


def test_register_schema_choices(tmp_path: Path) -> None:
    """--schema only accepts 'chat' or 'task'."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(
        ["validate", "--dataset", str(tmp_path / "d.jsonl"), "--schema", "task"]
    )
    assert args.schema == "task"

    with pytest.raises(SystemExit):
        parser.parse_args(["validate", "--dataset", str(tmp_path / "d.jsonl"), "--schema", "bogus"])


def test_register_json_flag(tmp_path: Path) -> None:
    """--json flag is parsed correctly by the validate subparser."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["validate", "--dataset", str(tmp_path / "d.jsonl"), "--json"])
    assert args.json is True


def test_register_dataset_required() -> None:
    """--dataset is a required flag."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    with pytest.raises(SystemExit):
        parser.parse_args(["validate"])


# ---------------------------------------------------------------------------
# End-to-end through main() — proves the verb is actually wired into the CLI
# ---------------------------------------------------------------------------


def test_main_validate_end_to_end(
    valid_chat_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``sloth validate`` resolves through sloth.cli.main (regression: the verb
    must be both imported AND registered in _build_parser)."""
    from sloth.cli import main

    rc = main(["validate", "--dataset", str(valid_chat_dataset), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True


def test_main_validate_unknown_command_would_have_caught_missing_registration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Sanity check that an actually-unregistered verb fails the way the bug did,
    so test_main_validate_end_to_end is a meaningful regression guard."""
    from sloth.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["totally-bogus-verb"])
    assert exc.value.code == 1
