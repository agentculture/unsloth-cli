"""Tests for sloth.tune.config — TOML run-config loader (TDD, criteria-first)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sloth.cli._errors import CliError
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
    DEFAULT_METHOD,
    DEFAULT_SEED,
    RunConfig,
    load_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toml(path: Path, content: str) -> Path:
    """Write a TOML file next to the given path (which is a directory)."""
    toml_file = path / "run.toml"
    toml_file.write_text(content, encoding="utf-8")
    return toml_file


# ---------------------------------------------------------------------------
# Criterion 1a — valid full TOML loads into a typed config object
# ---------------------------------------------------------------------------


def test_load_valid_full_config(tmp_path: Path) -> None:
    toml_file = _write_toml(
        tmp_path,
        """
[run]
model = "unsloth/Qwen3-4B"
method = "lora"
dataset = "data/train.jsonl"
output = "adapters/qwen3-4b-lora"

[hyperparameters]
lora_r = 32
lora_alpha = 64
lora_dropout = 0.05
learning_rate = 1e-4
max_seq_len = 4096
batch_size = 4
grad_accum = 8
max_steps = 100
seed = 42
load_in_4bit = false
""",
    )
    cfg = load_config(toml_file)
    assert isinstance(cfg, RunConfig)
    assert cfg.model == "unsloth/Qwen3-4B"
    assert cfg.method == "lora"
    assert cfg.dataset == "data/train.jsonl"
    assert cfg.output == "adapters/qwen3-4b-lora"
    assert cfg.lora_r == 32
    assert cfg.lora_alpha == 64
    assert abs(cfg.lora_dropout - 0.05) < 1e-9
    assert abs(cfg.learning_rate - 1e-4) < 1e-9
    assert cfg.max_seq_len == 4096
    assert cfg.batch_size == 4
    assert cfg.grad_accum == 8
    assert cfg.max_steps == 100
    assert cfg.seed == 42
    assert cfg.load_in_4bit is False


def test_load_valid_qlora(tmp_path: Path) -> None:
    toml_file = _write_toml(
        tmp_path,
        """
[run]
model = "unsloth/Qwen3-9B"
method = "qlora"
dataset = "data/chat.jsonl"
output = "adapters/qwen3-9b-qlora"
""",
    )
    cfg = load_config(toml_file)
    assert cfg.method == "qlora"


# ---------------------------------------------------------------------------
# Criterion 1b — missing required keys raise CliError with a hint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_key", ["model", "dataset", "output"])
def test_missing_required_key_raises_cli_error(tmp_path: Path, missing_key: str) -> None:
    keys = {"model": "unsloth/Qwen3-4B", "dataset": "data/train.jsonl", "output": "out/"}
    del keys[missing_key]
    lines = "\n".join(f'{k} = "{v}"' for k, v in keys.items())
    toml_file = _write_toml(tmp_path, f"[run]\n{lines}\n")

    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)

    err = exc_info.value
    assert err.code == 1
    assert missing_key in err.message
    assert err.remediation  # non-empty hint


# ---------------------------------------------------------------------------
# Criterion 1c — invalid method is rejected with CliError
# ---------------------------------------------------------------------------


def test_invalid_method_raises_cli_error(tmp_path: Path) -> None:
    toml_file = _write_toml(
        tmp_path,
        """
[run]
model = "unsloth/Qwen3-4B"
method = "full"
dataset = "data/train.jsonl"
output = "adapters/"
""",
    )
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)

    err = exc_info.value
    assert err.code == 1
    assert "full" in err.message or "method" in err.message
    assert err.remediation


# ---------------------------------------------------------------------------
# Criterion 2a — omitted optional fields fall back to documented defaults
# ---------------------------------------------------------------------------


def test_defaults_applied_when_fields_omitted(tmp_path: Path) -> None:
    toml_file = _write_toml(
        tmp_path,
        """
[run]
model = "unsloth/Qwen3-4B"
dataset = "data/train.jsonl"
output = "adapters/out"
""",
    )
    cfg = load_config(toml_file)

    assert cfg.method == DEFAULT_METHOD
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


# ---------------------------------------------------------------------------
# Criterion 2b — loading the same file twice yields an identical config
# ---------------------------------------------------------------------------


def test_deterministic_load(tmp_path: Path) -> None:
    toml_file = _write_toml(
        tmp_path,
        """
[run]
model = "unsloth/Qwen3-4B"
dataset = "data/train.jsonl"
output = "adapters/out"

[hyperparameters]
lora_r = 8
""",
    )
    cfg_a = load_config(toml_file)
    cfg_b = load_config(toml_file)
    assert cfg_a == cfg_b


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_unreadable_path_raises_cli_error(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.toml"
    with pytest.raises(CliError) as exc_info:
        load_config(missing)
    assert exc_info.value.code == 2  # env/setup error — file not found
    assert exc_info.value.remediation


def test_malformed_toml_raises_cli_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("this is not valid TOML ===\n", encoding="utf-8")
    with pytest.raises(CliError) as exc_info:
        load_config(bad)
    assert exc_info.value.code == 2
    assert exc_info.value.remediation


def test_run_config_is_a_dataclass(tmp_path: Path) -> None:
    """RunConfig must be a dataclass so it is eq-comparable and introspectable."""
    import dataclasses

    assert dataclasses.is_dataclass(RunConfig)
