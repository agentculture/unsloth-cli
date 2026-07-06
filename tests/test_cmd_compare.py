"""Tests for ``sloth compare <a> <b>``.

Covers:
* config/hyperparameter deltas between two runs' training_metadata.json
* identical configs -> empty deltas
* both full summaries are included in the report
* --json shape; unresolvable target -> CliError(code=1)
* register() wiring + main()-level end-to-end (regression pattern from
  test_cmd_validate.py's test_main_validate_end_to_end)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from sloth.cli._commands.compare import cmd_compare, register
from sloth.cli._errors import CliError
from sloth.tune.metadata import write_metadata

_VALID_CHAT = (
    '{"messages": [{"role": "user", "content": "hi"}, '
    '{"role": "assistant", "content": "hello"}]}\n'
)


def _write_dataset(tmp_path: Path, name: str = "train.jsonl", body: str = _VALID_CHAT) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def _args(a: str, b: str, *, runs_root: str | None = None, json_mode: bool = False):
    return argparse.Namespace(a=a, b=b, runs_root=runs_root, json=json_mode)


# ---------------------------------------------------------------------------
# Config deltas
# ---------------------------------------------------------------------------


def test_compare_reports_hyperparameter_delta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(
        dir_a,
        model="unsloth/Qwen3-4B",
        method="qlora",
        dataset_path=dataset,
        hyperparameters={"lora_r": 16, "learning_rate": 2e-4},
    )
    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(
        dir_b,
        model="unsloth/Qwen3-4B",
        method="qlora",
        dataset_path=dataset,
        hyperparameters={"lora_r": 32, "learning_rate": 2e-4},
    )

    rc = cmd_compare(_args(str(dir_a), str(dir_b), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deltas"]["lora_r"] == {"a": 16, "b": 32}
    assert "learning_rate" not in payload["deltas"]


def test_compare_reports_model_delta(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(
        dir_a,
        model="unsloth/Qwen3-4B",
        method="qlora",
        dataset_path=dataset,
        hyperparameters={},
    )
    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(
        dir_b,
        model="unsloth/Qwen3-9B",
        method="qlora",
        dataset_path=dataset,
        hyperparameters={},
    )

    rc = cmd_compare(_args(str(dir_a), str(dir_b), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deltas"]["model"] == {"a": "unsloth/Qwen3-4B", "b": "unsloth/Qwen3-9B"}


def test_compare_reports_dataset_delta_on_sha_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_a = _write_dataset(tmp_path, name="a.jsonl", body=_VALID_CHAT)
    dataset_b = _write_dataset(
        tmp_path,
        name="b.jsonl",
        body='{"messages": [{"role": "user", "content": "different"}]}\n',
    )
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(dir_a, model="m", method="lora", dataset_path=dataset_a, hyperparameters={})
    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(dir_b, model="m", method="lora", dataset_path=dataset_b, hyperparameters={})

    rc = cmd_compare(_args(str(dir_a), str(dir_b), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "dataset" in payload["deltas"]


def test_compare_identical_configs_empty_deltas(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(
        dir_a, model="m", method="lora", dataset_path=dataset, hyperparameters={"lora_r": 16}
    )
    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(
        dir_b, model="m", method="lora", dataset_path=dataset, hyperparameters={"lora_r": 16}
    )

    rc = cmd_compare(_args(str(dir_a), str(dir_b), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deltas"] == {}


def test_compare_text_mode_no_deltas_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(dir_a, model="m", method="lora", dataset_path=dataset, hyperparameters={})
    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(dir_b, model="m", method="lora", dataset_path=dataset, hyperparameters={})

    rc = cmd_compare(_args(str(dir_a), str(dir_b)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "configs match" in out


# ---------------------------------------------------------------------------
# Both full summaries are included
# ---------------------------------------------------------------------------


def test_compare_includes_both_summaries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(dir_a, model="m-a", method="lora", dataset_path=dataset, hyperparameters={})
    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(dir_b, model="m-b", method="lora", dataset_path=dataset, hyperparameters={})

    rc = cmd_compare(_args(str(dir_a), str(dir_b), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["a"]["metadata"]["model"] == "m-a"
    assert payload["b"]["metadata"]["model"] == "m-b"


# ---------------------------------------------------------------------------
# Unresolvable target
# ---------------------------------------------------------------------------


def test_compare_unresolvable_a_raises_cli_error_1(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(dir_b, model="m", method="lora", dataset_path=dataset, hyperparameters={})

    with pytest.raises(CliError) as exc_info:
        cmd_compare(_args("totally-bogus", str(dir_b), runs_root=str(tmp_path)))
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# register() wires the subparser correctly
# ---------------------------------------------------------------------------


def test_register_adds_compare_subparser(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["compare", "a-target", "b-target", "--json"])
    assert args.command == "compare"
    assert args.a == "a-target"
    assert args.b == "b-target"
    assert args.json is True
    assert callable(args.func)


def test_register_requires_both_targets() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    with pytest.raises(SystemExit):
        parser.parse_args(["compare", "only-one"])


# ---------------------------------------------------------------------------
# End-to-end through main()
# ---------------------------------------------------------------------------


def test_main_compare_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``sloth compare`` resolves through sloth.cli.main (regression: the verb
    must be both imported AND registered in _build_parser)."""
    from sloth.cli import main

    dataset = _write_dataset(tmp_path)
    dir_a = tmp_path / "exp-a"
    dir_a.mkdir()
    write_metadata(
        dir_a, model="m", method="lora", dataset_path=dataset, hyperparameters={"lora_r": 8}
    )
    dir_b = tmp_path / "exp-b"
    dir_b.mkdir()
    write_metadata(
        dir_b, model="m", method="lora", dataset_path=dataset, hyperparameters={"lora_r": 16}
    )

    rc = main(["compare", str(dir_a), str(dir_b), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deltas"]["lora_r"] == {"a": 8, "b": 16}
