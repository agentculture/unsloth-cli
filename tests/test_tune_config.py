"""Tests for sloth.tune.config — TOML run-config loader (TDD, criteria-first)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sloth.cli._errors import CliError
from sloth.tune.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COMPLIANCE_MIN_PCT,
    DEFAULT_EVAL_HOLDOUT_FRACTION,
    DEFAULT_EVAL_PERPLEXITY,
    DEFAULT_EVAL_SEED,
    DEFAULT_EVAL_STEPS,
    DEFAULT_EVAL_TOOL_CALL_FAMILY,
    DEFAULT_GRAD_ACCUM,
    DEFAULT_LATENCY_MAX_RATIO,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOAD_IN_4BIT,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_MAX_STEPS,
    DEFAULT_METHOD,
    DEFAULT_MIN_SUITE_ROWS,
    DEFAULT_REGRESSION_DROP_PP,
    DEFAULT_SEED,
    EvalConfig,
    RunConfig,
    ThresholdsConfig,
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


# ---------------------------------------------------------------------------
# Criterion 2 — [eval] section + [eval.thresholds] baseline (c23, c36)
# ---------------------------------------------------------------------------


def test_no_eval_section_leaves_eval_and_thresholds_none(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _BASE_RUN)
    cfg = load_config(toml_file)
    assert cfg.eval is None
    assert cfg.thresholds is None


def test_empty_eval_section_applies_all_defaults(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _BASE_RUN + "\n[eval]\n")
    cfg = load_config(toml_file)
    assert cfg.eval == EvalConfig(
        holdout_fraction=DEFAULT_EVAL_HOLDOUT_FRACTION,
        seed=DEFAULT_EVAL_SEED,
        eval_steps=DEFAULT_EVAL_STEPS,
        perplexity=DEFAULT_EVAL_PERPLEXITY,
        tool_call_family=DEFAULT_EVAL_TOOL_CALL_FAMILY,
    )
    assert cfg.thresholds == ThresholdsConfig(
        regression_drop_pp=DEFAULT_REGRESSION_DROP_PP,
        compliance_min_pct=DEFAULT_COMPLIANCE_MIN_PCT,
        latency_max_ratio=DEFAULT_LATENCY_MAX_RATIO,
        min_suite_rows=DEFAULT_MIN_SUITE_ROWS,
    )


def test_eval_defaults_match_c36_baseline_literals() -> None:
    # The c36 frame decision is the single source of truth for these numbers;
    # this test pins them so a docs/code drift is caught here first.
    assert DEFAULT_EVAL_HOLDOUT_FRACTION == 0.0
    assert DEFAULT_EVAL_STEPS == 0
    assert DEFAULT_EVAL_PERPLEXITY is False
    assert DEFAULT_EVAL_TOOL_CALL_FAMILY == ""
    assert DEFAULT_REGRESSION_DROP_PP == 2.0
    assert DEFAULT_COMPLIANCE_MIN_PCT == 95.0
    assert DEFAULT_LATENCY_MAX_RATIO == 1.10
    assert DEFAULT_MIN_SUITE_ROWS == 100


def test_eval_section_overrides_are_applied(tmp_path: Path) -> None:
    toml_file = _write_toml(
        tmp_path,
        _BASE_RUN + """
[eval]
holdout_fraction = 0.1
seed = 42
eval_steps = 20
perplexity = true
tool_call_family = "qwen3"

[eval.thresholds]
regression_drop_pp = 3.5
compliance_min_pct = 90.0
latency_max_ratio = 1.25
min_suite_rows = 40
""",
    )
    cfg = load_config(toml_file)
    assert cfg.eval == EvalConfig(
        holdout_fraction=0.1,
        seed=42,
        eval_steps=20,
        perplexity=True,
        tool_call_family="qwen3",
    )
    assert cfg.thresholds == ThresholdsConfig(
        regression_drop_pp=3.5,
        compliance_min_pct=90.0,
        latency_max_ratio=1.25,
        min_suite_rows=40,
    )


def test_eval_thresholds_partial_override_keeps_other_defaults(tmp_path: Path) -> None:
    toml_file = _write_toml(
        tmp_path,
        _BASE_RUN + "\n[eval]\n[eval.thresholds]\nmin_suite_rows = 250\n",
    )
    cfg = load_config(toml_file)
    assert cfg.thresholds.min_suite_rows == 250
    assert cfg.thresholds.regression_drop_pp == DEFAULT_REGRESSION_DROP_PP
    assert cfg.thresholds.compliance_min_pct == DEFAULT_COMPLIANCE_MIN_PCT
    assert cfg.thresholds.latency_max_ratio == DEFAULT_LATENCY_MAX_RATIO


def test_eval_config_and_thresholds_config_are_frozen_dataclasses() -> None:
    import dataclasses

    assert dataclasses.is_dataclass(EvalConfig)
    assert dataclasses.fields(EvalConfig)[0].name  # has fields
    assert EvalConfig.__dataclass_params__.frozen is True
    assert dataclasses.is_dataclass(ThresholdsConfig)
    assert ThresholdsConfig.__dataclass_params__.frozen is True

    cfg = EvalConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.holdout_fraction = 0.5  # type: ignore[misc]


def test_eval_unknown_key_raises_cli_error_with_hint(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _BASE_RUN + "\n[eval]\nbogus_key = 1\n")
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1
    assert "bogus_key" in exc_info.value.message
    assert exc_info.value.remediation


def test_eval_thresholds_unknown_key_raises_cli_error_with_hint(tmp_path: Path) -> None:
    toml_file = _write_toml(
        tmp_path, _BASE_RUN + "\n[eval]\n[eval.thresholds]\nbogus_threshold = 1\n"
    )
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1
    assert "bogus_threshold" in exc_info.value.message
    assert exc_info.value.remediation


def test_eval_holdout_fraction_out_of_range_raises_cli_error(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _BASE_RUN + "\n[eval]\nholdout_fraction = 1.5\n")
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1


def test_eval_perplexity_non_bool_raises_cli_error(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _BASE_RUN + '\n[eval]\nperplexity = "yes"\n')
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1


def test_eval_tool_call_family_non_string_raises_cli_error(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _BASE_RUN + "\n[eval]\ntool_call_family = 5\n")
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1


def test_eval_thresholds_compliance_min_pct_out_of_range_raises_cli_error(
    tmp_path: Path,
) -> None:
    toml_file = _write_toml(
        tmp_path,
        _BASE_RUN + "\n[eval]\n[eval.thresholds]\ncompliance_min_pct = 150.0\n",
    )
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1


def test_eval_thresholds_not_a_table_raises_cli_error(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _BASE_RUN + "\n[eval]\nthresholds = 5\n")
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Criterion 3 — existing config hash is unchanged when no [eval] section
# ---------------------------------------------------------------------------


def test_config_hash_unchanged_for_config_without_eval_section() -> None:
    # Recorded from compute_config_hash(load_config("examples/lora-smoke.toml"))
    # BEFORE this task added the eval/thresholds fields to RunConfig. If this
    # literal ever needs to change, every pre-existing run's run_id (which
    # embeds config_hash[:12]) silently changes underneath it — so this test
    # is the single source of truth that adding eval/thresholds as
    # None-by-default fields does not perturb the hash of a config that omits
    # [eval] entirely.
    from sloth.tune.registry import compute_config_hash

    cfg = load_config("examples/lora-smoke.toml")
    assert cfg.eval is None
    assert cfg.thresholds is None
    assert (
        compute_config_hash(cfg)
        == "752c600641c508e00a4720ee7ce503530706e29be51915ee52ba71efd82f39a2"[:64]
    )


# ---------------------------------------------------------------------------
# [run.dataset_map] — external (hf:) dataset column mapping (t11 / c34)
# ---------------------------------------------------------------------------

_HF_RUN = """
[run]
model = "unsloth/Qwen3-4B"
dataset = "hf:my-org/my-dataset:train"
output = "adapters/out"
"""


def test_dataset_map_absent_defaults_to_none(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _BASE_RUN)
    cfg = load_config(toml_file)
    assert cfg.dataset_map is None


def test_dataset_map_parses_chat_mapping(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _HF_RUN + '\n[run.dataset_map]\nmessages = "conversations"\n')
    cfg = load_config(toml_file)
    assert cfg.dataset_map == {"messages": "conversations"}


def test_dataset_map_parses_task_mapping(tmp_path: Path) -> None:
    toml_file = _write_toml(
        tmp_path,
        _HF_RUN + '\n[run.dataset_map]\ntask = "instruction"\ninput = "context"\n'
        'expected_output = "response"\n',
    )
    cfg = load_config(toml_file)
    assert cfg.dataset_map == {
        "task": "instruction",
        "input": "context",
        "expected_output": "response",
    }


def test_dataset_map_unknown_key_raises_cli_error_with_hint(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _HF_RUN + '\n[run.dataset_map]\nbogus = "col"\n')
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1
    assert "bogus" in exc_info.value.message
    assert exc_info.value.remediation


def test_dataset_map_non_string_value_raises_cli_error(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _HF_RUN + "\n[run.dataset_map]\nmessages = 5\n")
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1


def test_dataset_map_empty_string_value_raises_cli_error(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _HF_RUN + '\n[run.dataset_map]\nmessages = ""\n')
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1


def test_dataset_map_not_a_table_raises_cli_error(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _HF_RUN + "\ndataset_map = 5\n")
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# [run.dataset_map] completeness — a partial map must never reach the trainer
# (qodo PR #29 finding 1: a one-key task map inferred the task schema, then
# _render_hf_row indexed all three columns and crashed mid-run)
# ---------------------------------------------------------------------------


def test_dataset_map_partial_task_mapping_raises_naming_the_missing_keys(tmp_path: Path) -> None:
    toml_file = _write_toml(
        tmp_path, _HF_RUN + '\n[run.dataset_map]\ntask = "instruction"\ninput = "context"\n'
    )
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1
    assert "expected_output" in exc_info.value.message
    assert exc_info.value.remediation


def test_dataset_map_single_task_key_raises(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _HF_RUN + '\n[run.dataset_map]\ntask = "instruction"\n')
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1
    assert "input" in exc_info.value.message
    assert "expected_output" in exc_info.value.message


def test_dataset_map_mixing_chat_and_task_keys_raises(tmp_path: Path) -> None:
    toml_file = _write_toml(
        tmp_path,
        _HF_RUN + '\n[run.dataset_map]\nmessages = "conversations"\ntask = "instruction"\n'
        'input = "context"\nexpected_output = "response"\n',
    )
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1
    assert "messages" in exc_info.value.message
    assert exc_info.value.remediation


def test_dataset_map_empty_table_raises(tmp_path: Path) -> None:
    toml_file = _write_toml(tmp_path, _HF_RUN + "\n[run.dataset_map]\n")
    with pytest.raises(CliError) as exc_info:
        load_config(toml_file)
    assert exc_info.value.code == 1
    assert exc_info.value.remediation
