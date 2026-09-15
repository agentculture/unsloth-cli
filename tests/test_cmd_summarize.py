"""Tests for ``sloth summarize <run_id|output_dir>``.

Covers:
* joins training_metadata.json + the newest checkpoint's trainer_state.json
* tolerates either half being absent (metadata-only / training-only / neither)
* resolves <target> as a run_id (via the registry) or a literal directory
* --json shape; unresolvable target -> CliError(code=1)
* register() wiring + main()-level end-to-end (regression pattern from
  test_cmd_validate.py's test_main_validate_end_to_end)

Also covers (t6): an ``eval`` block (aggregate exact_match_pct/f1 + suite
file count) rendered from ``<run>/eval.json`` when present, omitted silently
when absent — both for the run itself and for each export dir it lists.
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


# ---------------------------------------------------------------------------
# Exports block (t10)
# ---------------------------------------------------------------------------


def test_summary_json_includes_exports_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    (output_dir / "exports.json").write_text(
        json.dumps(
            [
                {
                    "format": "gguf",
                    "quant": ["q4_k_m"],
                    "base": "unsloth/Qwen3-4B",
                    "adapter": str(output_dir),
                    "files": {"model.gguf": 100},
                    "calibration": None,
                    "versions": {},
                    "timestamp": "2026-07-06T00:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )

    rc = cmd_summarize(_args(str(output_dir), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["exports"][0]["format"] == "gguf"


def test_summary_text_mode_prints_one_line_per_export(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    (output_dir / "exports.json").write_text(
        json.dumps(
            [
                {
                    "format": "awq",
                    "quant": ["int4"],
                    "base": "unsloth/Qwen3-4B",
                    "adapter": str(output_dir),
                    "files": {"model.safetensors": 900, "config.json": 100},
                    "calibration": {"source": "train.jsonl", "count": 64},
                    "versions": {},
                    "timestamp": "2026-07-06T00:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )

    rc = cmd_summarize(_args(str(output_dir)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "awq" in out
    assert "int4" in out
    assert "files=2" in out
    assert "bytes=1000" in out


def test_summary_text_mode_no_exports_no_export_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()

    rc = cmd_summarize(_args(str(output_dir)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "export" not in out.lower()


# ---------------------------------------------------------------------------
# Eval block (t6)
# ---------------------------------------------------------------------------

_EVAL_PAYLOAD = {
    "total": 4,
    "exact_match": 3,
    "exact_match_pct": 75.0,
    "f1": 0.82,
    "results": [],
    "files": [
        {"path": "a.jsonl", "total": 2, "exact_match": 2, "exact_match_pct": 100.0, "f1": 1.0},
        {"path": "b.jsonl", "total": 2, "exact_match": 1, "exact_match_pct": 50.0, "f1": 0.64},
    ],
    "suite_paths": ["a.jsonl", "b.jsonl"],
    "target": "adapter",
    "written_at": "2026-07-06T02:00:00+00:00",
}


def _write_eval_json(directory: Path, payload: dict = _EVAL_PAYLOAD) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "eval.json").write_text(json.dumps(payload), encoding="utf-8")


def test_summary_json_includes_eval_block_when_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "adapter"
    _write_eval_json(output_dir)

    rc = cmd_summarize(_args(str(output_dir), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["eval"] == {"exact_match_pct": 75.0, "f1": 0.82, "file_count": 2}


def test_summary_text_mode_shows_eval_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "adapter"
    _write_eval_json(output_dir)

    rc = cmd_summarize(_args(str(output_dir)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "eval:" in out
    assert "75.0" in out
    assert "0.82" in out
    assert "files:" in out


def test_summary_omits_eval_block_silently_when_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()

    rc = cmd_summarize(_args(str(output_dir)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "eval:" not in out
    assert "exact_match_pct" not in out


def test_summary_json_eval_none_when_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()

    rc = cmd_summarize(_args(str(output_dir), json_mode=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["eval"] is None


def test_summary_export_line_shows_eval_when_export_has_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    export_dir = output_dir / "gguf-export"
    export_dir.mkdir()
    export_record = {
        "format": "gguf",
        "quant": ["q4_k_m"],
        "base": "m",
        "adapter": str(output_dir),
        "files": {"model.gguf": 100},
        "calibration": None,
        "versions": {},
        "timestamp": "2026-07-06T00:00:00+00:00",
    }
    (export_dir / "export.json").write_text(json.dumps(export_record), encoding="utf-8")
    _write_eval_json(export_dir)

    rc = cmd_summarize(_args(str(output_dir)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "gguf" in out
    assert "eval(" in out
    assert "75.0" in out


def test_summary_export_line_no_eval_suffix_when_export_lacks_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    export_dir = output_dir / "gguf-export"
    export_dir.mkdir()
    export_record = {
        "format": "gguf",
        "quant": ["q4_k_m"],
        "base": "m",
        "adapter": str(output_dir),
        "files": {"model.gguf": 100},
        "calibration": None,
        "versions": {},
        "timestamp": "2026-07-06T00:00:00+00:00",
    }
    (export_dir / "export.json").write_text(json.dumps(export_record), encoding="utf-8")

    rc = cmd_summarize(_args(str(output_dir)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "gguf" in out
    assert "eval(" not in out


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
