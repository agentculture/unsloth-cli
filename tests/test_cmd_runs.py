"""Tests for ``sloth runs`` — the run-registry noun (list/show/overview).

Covers:
* ``runs list``: missing/empty registry -> honest empty list (exit 0);
  newest-first ordering; --json shape; a corrupt line is skipped with a
  stderr diagnostic (never a crash).
* ``runs show <run_id>``: full record + output_dir_exists; unknown run_id ->
  CliError(code=1).
* ``runs overview``: non-empty text/JSON.
* register() wires list/show/overview correctly.
* main()-level end-to-end wiring (regression: a verb must be both imported
  AND registered in _build_parser — see test_cmd_validate.py's
  test_main_validate_end_to_end for the pattern this follows).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from sloth.cli._commands.runs import cmd_runs_list, cmd_runs_overview, cmd_runs_show, register
from sloth.cli._errors import CliError
from sloth.tune.config import RunConfig
from sloth.tune.registry import STATUS_OK, finish_run, registry_path, start_run

_VALID_CHAT = (
    '{"messages": [{"role": "user", "content": "hi"}, '
    '{"role": "assistant", "content": "hello"}]}\n'
)


def _write_dataset(tmp_path: Path, name: str = "train.jsonl") -> Path:
    p = tmp_path / name
    p.write_text(_VALID_CHAT, encoding="utf-8")
    return p


def _make_config(tmp_path: Path, output: Path) -> RunConfig:
    return RunConfig(
        model="unsloth/Qwen3-4B",
        dataset=str(_write_dataset(tmp_path)),
        output=str(output),
        method="qlora",
    )


def _args(*, runs_root: str | None = None, run_id: str | None = None, json_mode: bool = False):
    return argparse.Namespace(runs_root=runs_root, run_id=run_id, json=json_mode)


# ---------------------------------------------------------------------------
# runs list
# ---------------------------------------------------------------------------


def test_list_missing_registry_is_honest_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cmd_runs_list(_args(runs_root=str(tmp_path)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "no runs registered" in out


def test_list_json_empty_is_empty_array(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cmd_runs_list(_args(runs_root=str(tmp_path), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == []


def test_list_newest_first(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _make_config(tmp_path, tmp_path / "adapters" / "out")
    r1 = start_run(cfg, timestamp="2026-07-01T00:00:00+00:00")
    finish_run(r1, STATUS_OK, finished="2026-07-01T00:05:00+00:00")
    r2 = start_run(cfg, timestamp="2026-07-02T00:00:00+00:00")
    finish_run(r2, STATUS_OK, finished="2026-07-02T00:05:00+00:00")

    rc = cmd_runs_list(_args(runs_root=str(tmp_path / "adapters"), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["run_id"] for r in payload] == [r2.run_id, r1.run_id]


def test_list_text_table_contains_run_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _make_config(tmp_path, tmp_path / "adapters" / "out")
    record = start_run(cfg)
    finish_run(record, STATUS_OK)

    rc = cmd_runs_list(_args(runs_root=str(tmp_path / "adapters")))
    assert rc == 0
    out = capsys.readouterr().out
    assert record.run_id in out
    assert "ok" in out


def test_list_corrupt_line_emits_stderr_diagnostic_not_crash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "adapters"
    root.mkdir()
    registry_path(root).write_text("not valid json\n", encoding="utf-8")

    rc = cmd_runs_list(_args(runs_root=str(root)))
    assert rc == 0
    captured = capsys.readouterr()
    assert "invalid JSON" in captured.err
    assert "invalid JSON" not in captured.out


# ---------------------------------------------------------------------------
# runs show
# ---------------------------------------------------------------------------


def test_show_found_includes_output_dir_exists_true(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "adapters" / "out"
    out_dir.mkdir(parents=True)
    cfg = _make_config(tmp_path, out_dir)
    record = start_run(cfg)
    finish_run(record, STATUS_OK)

    rc = cmd_runs_show(_args(runs_root=str(tmp_path / "adapters"), run_id=record.run_id))
    assert rc == 0
    out = capsys.readouterr().out
    assert record.run_id in out
    assert "output_dir_exists: True" in out


def test_show_output_dir_exists_false_when_removed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _make_config(tmp_path, tmp_path / "adapters" / "out")  # never created on disk
    record = start_run(cfg)

    rc = cmd_runs_show(
        _args(runs_root=str(tmp_path / "adapters"), run_id=record.run_id, json_mode=True)
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_dir_exists"] is False


def test_show_unknown_run_id_raises_cli_error_1(tmp_path: Path) -> None:
    with pytest.raises(CliError) as exc_info:
        cmd_runs_show(_args(runs_root=str(tmp_path), run_id="does-not-exist"))
    assert exc_info.value.code == 1
    assert exc_info.value.remediation


def test_show_json_shape_includes_registry_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _make_config(tmp_path, tmp_path / "adapters" / "out")
    record = start_run(cfg)
    finish_run(record, STATUS_OK)

    rc = cmd_runs_show(
        _args(runs_root=str(tmp_path / "adapters"), run_id=record.run_id, json_mode=True)
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    for key in (
        "run_id",
        "config_hash",
        "output_dir",
        "model",
        "method",
        "dataset",
        "started",
        "finished",
        "status",
        "output_dir_exists",
    ):
        assert key in payload


# ---------------------------------------------------------------------------
# runs overview
# ---------------------------------------------------------------------------


def test_overview_text(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cmd_runs_overview(argparse.Namespace(json=False))
    assert rc == 0
    out = capsys.readouterr().out
    assert "runs" in out.lower()
    assert "runs.jsonl" in out


def test_overview_json_shape(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cmd_runs_overview(argparse.Namespace(json=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "subject" in payload
    assert "sections" in payload


# ---------------------------------------------------------------------------
# register() wires the subparser correctly
# ---------------------------------------------------------------------------


def test_register_adds_runs_list(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["runs", "list", "--runs-root", str(tmp_path), "--json"])
    assert args.command == "runs"
    assert args.runs_root == str(tmp_path)
    assert args.json is True
    assert callable(args.func)


def test_register_adds_runs_show_requires_run_id(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["runs", "show", "some-run-id"])
    assert args.run_id == "some-run-id"
    assert callable(args.func)

    with pytest.raises(SystemExit):
        parser.parse_args(["runs", "show"])


def test_register_adds_runs_overview(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["runs", "overview"])
    assert callable(args.func)


def test_bare_runs_prints_help_and_exits_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["runs"])
    rc = args.func(args)
    assert rc == 0
    assert capsys.readouterr().out.strip()


# ---------------------------------------------------------------------------
# End-to-end through main() — proves the verb is actually wired into the CLI
# ---------------------------------------------------------------------------


def test_main_runs_list_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``sloth runs list`` resolves through sloth.cli.main (regression: the verb
    must be both imported AND registered in _build_parser)."""
    from sloth.cli import main

    rc = main(["runs", "list", "--runs-root", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == []


def test_main_runs_show_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from sloth.cli import main

    cfg = _make_config(tmp_path, tmp_path / "adapters" / "out")
    record = start_run(cfg)
    finish_run(record, STATUS_OK)

    rc = main(["runs", "show", record.run_id, "--runs-root", str(tmp_path / "adapters"), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == record.run_id


def test_main_runs_overview_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    from sloth.cli import main

    rc = main(["runs", "overview"])
    assert rc == 0
    assert capsys.readouterr().out.strip()
