"""Tests for ``sloth summarize <run_id|output_dir>``.

Covers:
* joins training_metadata.json + the newest checkpoint's trainer_state.json
* tolerates either half being absent (metadata-only / training-only / neither)
* resolves <target> as a run_id (via the registry) or a literal directory
* --json shape; unresolvable target -> CliError(code=1)
* register() wiring + main()-level end-to-end (regression pattern from
  test_cmd_validate.py's test_main_validate_end_to_end)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from sloth.cli._commands.summarize import cmd_summarize, register
from sloth.cli._errors import CliError
from sloth.tune.config import RunConfig
from sloth.tune.metadata import write_metadata
from sloth.tune.registry import start_run

_VALID_CHAT = (
    '{"messages": [{"role": "user", "content": "hi"}, '
    '{"role": "assistant", "content": "hello"}]}\n'
)


def _write_dataset(tmp_path: Path, name: str = "train.jsonl") -> Path:
    p = tmp_path / name
    p.write_text(_VALID_CHAT, encoding="utf-8")
    return p


def _write_checkpoint(
    output_dir: Path, step: int, *, loss: float = 0.5, best_metric: float | None = None
) -> Path:
    ckpt = output_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True)
    state = {
        "global_step": step,
        "log_history": [{"step": step // 2, "loss": 1.0}, {"step": step, "loss": loss}],
        "best_metric": best_metric,
        "best_model_checkpoint": str(ckpt) if best_metric is not None else None,
    }
    (ckpt / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    return ckpt


def _args(target: str, *, runs_root: str | None = None, json_mode: bool = False):
    return argparse.Namespace(target=target, runs_root=runs_root, json=json_mode)


# ---------------------------------------------------------------------------
# Full summary — metadata + trainer_state both present
# ---------------------------------------------------------------------------


def test_summary_joins_metadata_and_trainer_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    dataset = _write_dataset(tmp_path)
    write_metadata(
        output_dir,
        model="unsloth/Qwen3-4B",
        method="qlora",
        dataset_path=dataset,
        hyperparameters={"lora_r": 16},
        timestamp="2026-07-06T00:00:00+00:00",
    )
    _write_checkpoint(output_dir, 30, loss=0.75, best_metric=0.7)
    _write_checkpoint(output_dir, 60, loss=0.5, best_metric=0.6)  # newest -> should win

    rc = cmd_summarize(_args(str(output_dir), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["metadata"]["model"] == "unsloth/Qwen3-4B"
    assert payload["training"]["checkpoint"] == "checkpoint-60"
    assert payload["training"]["final_step"] == 60
    assert payload["training"]["final_loss"] == 0.5
    assert payload["notes"] == []


def test_summary_text_mode_includes_key_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    dataset = _write_dataset(tmp_path)
    write_metadata(
        output_dir,
        model="unsloth/Qwen3-4B",
        method="lora",
        dataset_path=dataset,
        hyperparameters={"lora_r": 8},
    )
    _write_checkpoint(output_dir, 10, loss=0.9)

    rc = cmd_summarize(_args(str(output_dir)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "unsloth/Qwen3-4B" in out
    assert "checkpoint-10" in out
    assert "0.9" in out


# ---------------------------------------------------------------------------
# Tolerates absence of either half
# ---------------------------------------------------------------------------


def test_summary_metadata_only_when_no_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    dataset = _write_dataset(tmp_path)
    write_metadata(
        output_dir,
        model="m",
        method="lora",
        dataset_path=dataset,
        hyperparameters={},
    )

    rc = cmd_summarize(_args(str(output_dir), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"] is not None
    assert payload["training"] is None
    assert any("checkpoint" in note for note in payload["notes"])


def test_summary_training_only_when_no_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    _write_checkpoint(output_dir, 5, loss=1.2)

    rc = cmd_summarize(_args(str(output_dir), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"] is None
    assert payload["training"]["final_step"] == 5
    assert any("training_metadata.json" in note for note in payload["notes"])


def test_summary_neither_present_still_returns_gracefully(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "empty-adapter"
    output_dir.mkdir()

    rc = cmd_summarize(_args(str(output_dir), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"] is None
    assert payload["training"] is None
    assert len(payload["notes"]) == 2


# ---------------------------------------------------------------------------
# Target resolution — run_id vs literal directory
# ---------------------------------------------------------------------------


def test_summarize_by_run_id(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "adapters" / "out"
    output_dir.mkdir(parents=True)
    dataset = _write_dataset(tmp_path)
    write_metadata(output_dir, model="m", method="lora", dataset_path=dataset, hyperparameters={})

    cfg = RunConfig(model="m", dataset=str(dataset), output=str(output_dir), method="lora")
    record = start_run(cfg)

    rc = cmd_summarize(_args(record.run_id, runs_root=str(tmp_path / "adapters"), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_dir"] == str(output_dir)


def test_summarize_unresolvable_target_raises_cli_error_1(tmp_path: Path) -> None:
    with pytest.raises(CliError) as exc_info:
        cmd_summarize(_args("totally-bogus", runs_root=str(tmp_path)))
    assert exc_info.value.code == 1
    assert exc_info.value.remediation


# ---------------------------------------------------------------------------
# register() wires the subparser correctly
# ---------------------------------------------------------------------------


def test_register_adds_summarize_subparser(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(["summarize", str(tmp_path), "--json"])
    assert args.command == "summarize"
    assert args.target == str(tmp_path)
    assert args.json is True
    assert callable(args.func)


def test_register_target_required() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    with pytest.raises(SystemExit):
        parser.parse_args(["summarize"])


# ---------------------------------------------------------------------------
# End-to-end through main()
# ---------------------------------------------------------------------------


def test_main_summarize_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``sloth summarize`` resolves through sloth.cli.main (regression: the verb
    must be both imported AND registered in _build_parser)."""
    from sloth.cli import main

    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    dataset = _write_dataset(tmp_path)
    write_metadata(output_dir, model="m", method="lora", dataset_path=dataset, hyperparameters={})

    rc = main(["summarize", str(output_dir), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["model"] == "m"
