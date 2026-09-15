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
from sloth.tune.presets import PRESETS

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


# ---------------------------------------------------------------------------
# Criterion 1f — invalid hyperparameter types/ranges raise CliError(code=1)
# ---------------------------------------------------------------------------

_BASE_RUN = """
[run]
model = "unsloth/Qwen3-4B"
dataset = "data/train.jsonl"
output = "adapters/out"
"""


def test_non_int_hyperparameter_raises_cli_error(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _BASE_RUN + '\n[hyperparameters]\nlora_r = "big"\n')
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1
    assert "lora_r" in str(exc_info.value.message)


def test_bool_for_int_hyperparameter_is_rejected(tmp_path: Path) -> None:
    # bool is an int subclass — must still be rejected for a numeric field.
    toml_file = _write_toml(tmp_path, _BASE_RUN + "\n[hyperparameters]\nmax_steps = true\n")
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1


def test_out_of_range_int_hyperparameter_raises_cli_error(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _BASE_RUN + "\n[hyperparameters]\nlora_r = 0\n")
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1


def test_out_of_range_dropout_raises_cli_error(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _BASE_RUN + "\n[hyperparameters]\nlora_dropout = 1.5\n")
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1


def test_non_bool_load_in_4bit_raises_cli_error(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _BASE_RUN + '\n[hyperparameters]\nload_in_4bit = "yes"\n')
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# target_modules — c2, h28
# ---------------------------------------------------------------------------


def test_target_modules_absent_defaults_to_none(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _BASE_RUN)
    config = load_config(toml_file)
    assert config.target_modules is None


def test_target_modules_explicit_list_round_trips(tmp_path: Path) -> None:
    toml_file = _write_toml(
        tmp_path,
        _BASE_RUN + '\n[hyperparameters]\ntarget_modules = ["q_proj", "k_proj", "v_proj"]\n',
    )
    config = load_config(toml_file)
    assert config.target_modules == ["q_proj", "k_proj", "v_proj"]


def test_target_modules_regex_string_round_trips(tmp_path: Path) -> None:
    regex = r"model\.layers\.\d+\.self_attn\.(q|k|v)_proj"
    # TOML literal strings (single quotes) pass backslashes through verbatim.
    toml_file = _write_toml(
        tmp_path,
        _BASE_RUN + f"\n[hyperparameters]\ntarget_modules = '{regex}'\n",
    )
    config = load_config(toml_file)
    assert config.target_modules == regex


def test_target_modules_preset_lfm2_stored_verbatim(tmp_path: Path) -> None:
    # load_config stores the value as written in TOML — resolution to the
    # regex happens separately via presets.resolve_target_modules() at the
    # call site (the trainer), not inside load_config.
    toml_file = _write_toml(
        tmp_path,
        _BASE_RUN + '\n[hyperparameters]\ntarget_modules = "preset:lfm2"\n',
    )
    config = load_config(toml_file)
    assert config.target_modules == "preset:lfm2"


def test_preset_lfm2_resolves_to_documented_regex() -> None:
    assert PRESETS["lfm2"] == (
        r"model\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|"
        r"conv\.(in|out)_proj|feed_forward\.w[123])"
    )


def test_target_modules_wrong_type_raises_cli_error_with_hint(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _BASE_RUN + "\n[hyperparameters]\ntarget_modules = 42\n")
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1
    remediation = exc_info.value.remediation
    assert "list" in remediation
    assert "regex" in remediation or "string" in remediation
    assert "preset" in remediation


def test_target_modules_empty_list_raises_cli_error(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _BASE_RUN + "\n[hyperparameters]\ntarget_modules = []\n")
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1


def test_target_modules_empty_string_raises_cli_error(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _BASE_RUN + '\n[hyperparameters]\ntarget_modules = ""\n')
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1


def test_target_modules_list_with_non_string_element_raises_cli_error(
    tmp_path: Path,
) -> None:
    toml_file = _write_toml(
        tmp_path, _BASE_RUN + '\n[hyperparameters]\ntarget_modules = ["q_proj", 1]\n'
    )
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1


def test_target_modules_list_with_empty_string_element_raises_cli_error(
    tmp_path: Path,
) -> None:
    toml_file = _write_toml(
        tmp_path, _BASE_RUN + '\n[hyperparameters]\ntarget_modules = ["q_proj", ""]\n'
    )
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1


def test_target_modules_unknown_preset_raises_cli_error(tmp_path: Path) -> None:
    toml_file = _write_toml(
        tmp_path, _BASE_RUN + '\n[hyperparameters]\ntarget_modules = "preset:nope"\n'
    )
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1
    assert "preset:nope" in exc_info.value.message or "nope" in exc_info.value.message
