"""Tests for ``sloth config init`` — writes a starting ``run.toml``.

Covers:
* the generated ``run.toml`` loads via :func:`sloth.tune.config.load_config`
  and passes its validation (required keys + hyperparameter bounds)
* --force overwrite behavior and refusal without --force
* --method (lora/qlora) and --path overrides
* text / --json output shapes
* register() wires the subparser correctly
* a regression guard for the bare ``sloth config`` (no sub-verb) crash found
  during review: ``args.func`` was never defaulted on the bare noun, so
  dispatch raised an unhandled AttributeError instead of the structured
  CliError contract.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pytest

from sloth.cli import main
from sloth.cli._commands.config import cmd_config_init, register
from sloth.cli._errors import CliError
from sloth.cli._output import emit_error
from sloth.tune.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRAD_ACCUM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOAD_IN_4BIT,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_MAX_STEPS,
    DEFAULT_SEED,
    load_config,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_args(
    *,
    model: str = "unsloth/Qwen3-4B",
    dataset: str = "data/train.jsonl",
    output: str,
    method: str | None = None,
    force: bool = False,
    path: str | None = None,
    json_mode: bool = False,
) -> argparse.Namespace:
    """Build the Namespace argparse would produce for ``sloth config init``."""
    return argparse.Namespace(
        model=model,
        dataset=dataset,
        output=output,
        method=method,
        force=force,
        path=path,
        json=json_mode,
    )


# ---------------------------------------------------------------------------
# Happy path — writes a run.toml that passes load_config validation
# ---------------------------------------------------------------------------


def test_writes_run_toml_that_loads_via_load_config(tmp_path: Path) -> None:
    """The generated run.toml round-trips through load_config without error."""
    output = tmp_path / "adapters" / "out"
    rc = cmd_config_init(_make_args(output=str(output)))
    assert rc in (None, 0)

    written = output / "run.toml"
    assert written.is_file()

    cfg = load_config(written)
    assert cfg.model == "unsloth/Qwen3-4B"
    assert cfg.dataset == "data/train.jsonl"
    assert cfg.output == str(output)
    assert cfg.method == "qlora"  # DEFAULT_METHOD


def test_special_chars_in_values_round_trip_through_load_config(tmp_path: Path) -> None:
    """model/dataset/output values containing backslashes and double-quotes
    (e.g. Windows-style paths) must produce a run.toml that load_config
    parses without error — the raw f-string interpolation this regression
    guards against emits invalid TOML for such values."""
    output = tmp_path / "out"
    model = 'unsloth/Qwen3-4B "special"'
    dataset = r"C:\Users\ori\train.jsonl"
    rc = cmd_config_init(
        _make_args(model=model, dataset=dataset, output=str(output)),
    )
    assert rc in (None, 0)

    cfg = load_config(output / "run.toml")
    assert cfg.model == model
    assert cfg.dataset == dataset
    assert cfg.output == str(output)


def test_generated_hyperparameters_match_tune_config_defaults(tmp_path: Path) -> None:
    """Every [hyperparameters] value matches the documented default constants."""
    output = tmp_path / "out"
    cmd_config_init(_make_args(output=str(output)))
    cfg = load_config(output / "run.toml")

    assert cfg.lora_r == DEFAULT_LORA_R
    assert cfg.lora_alpha == DEFAULT_LORA_ALPHA
    assert cfg.lora_dropout == DEFAULT_LORA_DROPOUT
    assert cfg.learning_rate == DEFAULT_LEARNING_RATE
    assert cfg.max_seq_len == DEFAULT_MAX_SEQ_LEN
    assert cfg.batch_size == DEFAULT_BATCH_SIZE
    assert cfg.grad_accum == DEFAULT_GRAD_ACCUM
    assert cfg.max_steps == DEFAULT_MAX_STEPS
    assert cfg.seed == DEFAULT_SEED
    assert cfg.load_in_4bit == DEFAULT_LOAD_IN_4BIT


def test_default_path_is_output_slash_run_toml(tmp_path: Path) -> None:
    """With no --path, the file lands at <output>/run.toml."""
    output = tmp_path / "myrun"
    cmd_config_init(_make_args(output=str(output)))
    assert (output / "run.toml").is_file()


def test_path_override_writes_elsewhere(tmp_path: Path) -> None:
    """--path overrides the default <output>/run.toml location."""
    output = tmp_path / "out"
    custom = tmp_path / "somewhere" / "custom.toml"
    rc = cmd_config_init(_make_args(output=str(output), path=str(custom)))
    assert rc in (None, 0)
    assert custom.is_file()
    assert not (output / "run.toml").is_file()
    # Still loads cleanly via load_config.
    cfg = load_config(custom)
    assert cfg.output == str(output)


@pytest.mark.parametrize("method", ["lora", "qlora"])
def test_method_flag_written_and_valid(tmp_path: Path, method: str) -> None:
    """--method lora / --method qlora is written and passes load_config."""
    output = tmp_path / "out"
    cmd_config_init(_make_args(output=str(output), method=method))
    cfg = load_config(output / "run.toml")
    assert cfg.method == method


def test_method_defaults_to_qlora_when_omitted(tmp_path: Path) -> None:
    """Omitting --method resolves to sloth.tune.config.DEFAULT_METHOD ('qlora')."""
    output = tmp_path / "out"
    cmd_config_init(_make_args(output=str(output), method=None))
    cfg = load_config(output / "run.toml")
    assert cfg.method == "qlora"


# ---------------------------------------------------------------------------
# --force / overwrite refusal
# ---------------------------------------------------------------------------


def test_refuses_overwrite_without_force(tmp_path: Path) -> None:
    """A second init at the same path without --force raises CliError(code=1)."""
    output = tmp_path / "out"
    cmd_config_init(_make_args(output=str(output)))

    with pytest.raises(CliError) as exc_info:
        cmd_config_init(_make_args(output=str(output)))
    assert exc_info.value.code == 1
    assert "already exists" in exc_info.value.message
    assert exc_info.value.remediation


def test_refusal_error_hint_contract(tmp_path: Path) -> None:
    """The overwrite-refusal CliError renders as ``error:``/``hint:`` lines."""
    output = tmp_path / "out"
    cmd_config_init(_make_args(output=str(output)))

    with pytest.raises(CliError) as exc_info:
        cmd_config_init(_make_args(output=str(output)))
    buf = io.StringIO()
    emit_error(exc_info.value, json_mode=False, stream=buf)
    text = buf.getvalue()
    assert text.startswith("error:")
    assert "hint:" in text


def test_force_overwrites_existing_file(tmp_path: Path) -> None:
    """--force allows overwriting a pre-existing run.toml with new content."""
    output = tmp_path / "out"
    cmd_config_init(_make_args(output=str(output), model="unsloth/Qwen3-4B"))
    rc = cmd_config_init(_make_args(output=str(output), model="unsloth/Qwen3-8B", force=True))
    assert rc in (None, 0)
    cfg = load_config(output / "run.toml")
    assert cfg.model == "unsloth/Qwen3-8B"


def test_force_not_required_for_a_fresh_path(tmp_path: Path) -> None:
    """A brand-new path needs no --force."""
    output = tmp_path / "brand-new"
    rc = cmd_config_init(_make_args(output=str(output), force=False))
    assert rc in (None, 0)


# ---------------------------------------------------------------------------
# Output shapes — text and --json
# ---------------------------------------------------------------------------


def test_text_output_reports_written_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "out"
    cmd_config_init(_make_args(output=str(output)))
    out = capsys.readouterr().out
    assert "written" in out.lower()
    assert str(output / "run.toml") in out


def test_json_output_shape(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "out"
    rc = cmd_config_init(_make_args(output=str(output), json_mode=True))
    assert rc in (None, 0)
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"] == str(output / "run.toml")
    assert payload["model"] == "unsloth/Qwen3-4B"
    assert payload["method"] == "qlora"
    assert payload["dataset"] == "data/train.jsonl"
    assert payload["output"] == str(output)


def test_diagnostic_written_to_stderr_not_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The 'wrote <path>' diagnostic goes to stderr; stdout stays result-only."""
    output = tmp_path / "out"
    cmd_config_init(_make_args(output=str(output), json_mode=True))
    captured = capsys.readouterr()
    assert "wrote" in captured.err
    # stdout must be valid JSON only — no diagnostic text mixed in.
    json.loads(captured.out)


# ---------------------------------------------------------------------------
# register() wires the subparser correctly
# ---------------------------------------------------------------------------


def test_register_adds_config_init_subparser(tmp_path: Path) -> None:
    """register() adds a nested 'config init' subparser with the right flags."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(
        [
            "config",
            "init",
            "--model",
            "unsloth/Qwen3-4B",
            "--dataset",
            "data/train.jsonl",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert args.command == "config"
    assert args.config_command == "init"
    assert args.model == "unsloth/Qwen3-4B"
    assert args.dataset == "data/train.jsonl"
    assert args.output == str(tmp_path / "out")
    assert args.method is None
    assert args.force is False
    assert args.path is None
    assert args.json is False
    assert callable(args.func)


def test_register_method_choices(tmp_path: Path) -> None:
    """--method only accepts 'lora' or 'qlora'."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(
        [
            "config",
            "init",
            "--model",
            "m",
            "--dataset",
            "d",
            "--output",
            str(tmp_path / "o"),
            "--method",
            "lora",
        ]
    )
    assert args.method == "lora"

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "config",
                "init",
                "--model",
                "m",
                "--dataset",
                "d",
                "--output",
                str(tmp_path / "o"),
                "--method",
                "bogus",
            ]
        )


def test_register_required_flags(tmp_path: Path) -> None:
    """--model, --dataset, and --output are all required."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    with pytest.raises(SystemExit):
        parser.parse_args(["config", "init"])


def test_register_force_and_json_flags(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)
    args = parser.parse_args(
        [
            "config",
            "init",
            "--model",
            "m",
            "--dataset",
            "d",
            "--output",
            str(tmp_path / "o"),
            "--force",
            "--json",
        ]
    )
    assert args.force is True
    assert args.json is True


# ---------------------------------------------------------------------------
# Regression: bare `sloth config` (no sub-verb) must not crash
# ---------------------------------------------------------------------------


def test_bare_config_noun_does_not_crash(capsys: pytest.CaptureFixture[str]) -> None:
    """`sloth config` with no sub-verb prints usage instead of an unhandled
    AttributeError (found during review: `args.func` was never defaulted on
    the bare 'config' parser, so _dispatch's generic exception wrap surfaced
    'unexpected: AttributeError' instead of a clean, structured error).

    Help is diagnostic output, not a result, so it must land on stderr —
    stdout stays result-only per the output contract.
    """
    rc = main(["config"])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "config" in captured.err.lower()


def test_bare_config_noun_json_flag_does_not_crash(capsys: pytest.CaptureFixture[str]) -> None:
    """Bare `sloth config --json` also must not raise (json attr is defaulted)."""
    rc = main(["config"])
    assert rc == 0
    capsys.readouterr()


# ---------------------------------------------------------------------------
# End-to-end through main() — proves the verb is actually wired into the CLI
# ---------------------------------------------------------------------------


def test_main_config_init_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``sloth config init`` resolves through sloth.cli.main (regression: the
    verb must be both imported AND registered in _build_parser)."""
    output = tmp_path / "out"
    rc = main(
        [
            "config",
            "init",
            "--model",
            "unsloth/Qwen3-4B",
            "--dataset",
            "data/train.jsonl",
            "--output",
            str(output),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"] == str(output / "run.toml")
    load_config(output / "run.toml")  # must not raise


def test_main_config_init_refusal_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second `sloth config init` at the same path exits 1 through main()."""
    output = tmp_path / "out"
    args = [
        "config",
        "init",
        "--model",
        "unsloth/Qwen3-4B",
        "--dataset",
        "data/train.jsonl",
        "--output",
        str(output),
    ]
    rc_first = main(args)
    assert rc_first == 0
    capsys.readouterr()

    rc_second = main(args)
    assert rc_second == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "hint:" in err
