"""TOML run-config loader for unsloth-cli fine-tune verbs.

Loads a ``run.toml`` file into a typed :class:`RunConfig` dataclass using only
stdlib ``tomllib`` (Python 3.11+). No torch or ML imports — the introspection
CLI must keep working on machines without a GPU.

Default values are documented as named module constants so they can be
referenced in ``explain`` catalog text and CLI help strings without duplicating
magic literals.

Typical ``run.toml`` layout::

    [run]
    model   = "unsloth/Qwen3-4B"
    method  = "qlora"          # "lora" or "qlora"  (default: "qlora")
    dataset = "data/train.jsonl"
    output  = "adapters/qwen3-4b-qlora"

    [hyperparameters]
    lora_r         = 16
    lora_alpha     = 16
    lora_dropout   = 0.0
    learning_rate  = 2e-4
    max_seq_len    = 2048
    batch_size     = 2
    grad_accum     = 4
    max_steps      = 60
    seed           = 3407
    load_in_4bit   = true   # always true for qlora; ignored for lora

Required keys: ``model``, ``dataset``, ``output`` (all under ``[run]``).
``method`` is optional — defaults to ``"qlora"``.
All ``[hyperparameters]`` fields are optional and fall back to the defaults
documented below.

External datasets (``[run.dataset_map]``)
------------------------------------------
``dataset`` may also name a Hugging Face Hub dataset instead of a local JSONL
file, using the form ``"hf:<org>/<name>"`` or ``"hf:<org>/<name>:<split>"``
(``split`` defaults to ``"train"``). The actual ``datasets.load_dataset`` call
happens lazily, inside the NGC container (see ``sloth.tune._trainer``) — this
module only parses and stores the mapping; it never imports ``datasets``.

An optional ``[run.dataset_map]`` table documents how the hub dataset's
columns map onto the chat or task schema. Keys are the target schema field,
values are the source column name in the hub dataset::

    [run]
    dataset = "hf:my-org/my-chat-dataset:train"

    [run.dataset_map]
    messages = "conversations"      # chat schema

    # -- or, for the task schema --
    # [run.dataset_map]
    # task            = "instruction"
    # input           = "context"
    # expected_output = "response"

Accepted keys: ``messages`` (chat schema), and ``task`` / ``input`` /
``expected_output`` (task schema). ``dataset_map`` is optional for a local
JSONL ``dataset`` (ignored there) but required to resolve an ``hf:`` dataset's
column names to a known schema.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from sloth.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError

# ---------------------------------------------------------------------------
# Spark-friendly defaults — documented, named, importable
# ---------------------------------------------------------------------------

DEFAULT_METHOD: str = "qlora"
"""Adapter method. ``"qlora"`` (4-bit quantised) is the Spark-friendly default."""

VALID_METHODS: frozenset[str] = frozenset({"lora", "qlora"})
"""Accepted values for ``method``."""

DEFAULT_LORA_R: int = 16
"""LoRA rank. Lower values use less VRAM; 16 balances expressivity and cost."""

DEFAULT_LORA_ALPHA: int = 16
"""LoRA alpha scaling factor. Conventionally set equal to ``lora_r``."""

DEFAULT_LORA_DROPOUT: float = 0.0
"""LoRA dropout. 0.0 is standard for small adapter runs."""

DEFAULT_LEARNING_RATE: float = 2e-4
"""Peak learning rate. 2e-4 is the Unsloth-recommended default for QLoRA."""

DEFAULT_MAX_SEQ_LEN: int = 2048
"""Maximum sequence length in tokens. 2048 fits comfortably on 16 GB VRAM."""

DEFAULT_BATCH_SIZE: int = 2
"""Per-device training batch size. Small default keeps VRAM usage low."""

DEFAULT_GRAD_ACCUM: int = 4
"""Gradient accumulation steps. Effective batch = ``batch_size × grad_accum``."""

DEFAULT_MAX_STEPS: int = 60
"""Training steps. 60 is a quick smoke-run; increase for production adapters."""

DEFAULT_SEED: int = 3407
"""Random seed. 3407 is the Unsloth canonical default for reproducibility."""

DEFAULT_LOAD_IN_4BIT: bool = True
"""Load base model in 4-bit NF4 quantisation (required for QLoRA)."""


# ---------------------------------------------------------------------------
# [eval] / [eval.thresholds] defaults — frame decision c36 is the single
# source of truth for the threshold baseline values; docs/benchmarks.md and
# docs/specs cite these constants rather than restating the numbers.
# ---------------------------------------------------------------------------

DEFAULT_EVAL_HOLDOUT_FRACTION: float = 0.0
"""Fraction of the training set carved into a held-out eval split. 0.0 = off."""

DEFAULT_EVAL_SEED: int = DEFAULT_SEED
"""Random seed for the eval holdout split. Shares the training default."""

DEFAULT_EVAL_STEPS: int = 0
"""Run an in-training eval every N steps. 0 = disabled."""

DEFAULT_EVAL_PERPLEXITY: bool = False
"""Whether to compute held-out perplexity/loss during eval."""

DEFAULT_EVAL_TOOL_CALL_FAMILY: str = ""
"""Tool-call parser family override (e.g. "qwen3", "lfm2"). "" = auto-detect."""

DEFAULT_REGRESSION_DROP_PP: float = 2.0
"""c36 baseline: max allowed regression-suite drop, in percentage points."""

DEFAULT_COMPLIANCE_MIN_PCT: float = 95.0
"""c36 baseline: minimum structured-output/tool-call compliance percentage."""

DEFAULT_LATENCY_MAX_RATIO: float = 1.10
"""c36 baseline: max per-item median latency, as a ratio of the base model's."""

DEFAULT_MIN_SUITE_ROWS: int = 100
"""c36 baseline: minimum suite size before a delta counts as pass/fail."""


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Typed representation of a ``run.toml`` fine-tune configuration.

    Required fields must be supplied in the ``[run]`` section of the TOML
    file. Optional hyperparameter fields default to the Spark-friendly values
    documented in the module-level constants above.
    """

    # Required — no defaults; load_config validates their presence.
    model: str
    dataset: str
    output: str

    # Optional with Spark-friendly defaults
    method: str = DEFAULT_METHOD

    # LoRA / QLoRA hyperparameters
    lora_r: int = DEFAULT_LORA_R
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    batch_size: int = DEFAULT_BATCH_SIZE
    grad_accum: int = DEFAULT_GRAD_ACCUM
    max_steps: int = DEFAULT_MAX_STEPS
    seed: int = DEFAULT_SEED
    load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT

    # PEFT target-module selection: a literal name list, a literal regex
    # string, or a "preset:<name>" reference. Stored as written in the TOML —
    # resolving "preset:<name>" to its regex happens at the call site via
    # sloth.tune.presets.resolve_target_modules(), not here.
    target_modules: list[str] | str | None = None

    # Column mapping for an external ``hf:<org>/<name>[:split]`` dataset — see
    # the "External datasets" section of this module's docstring. ``None`` when
    # the config has no ``[run.dataset_map]`` table (the common case for a
    # local JSONL ``dataset``).
    dataset_map: "dict[str, str] | None" = None

    # [eval] / [eval.thresholds] — both None when the config omits [eval]
    # entirely, so a pre-existing config's compute_config_hash() is untouched
    # (the hash formula drops None-valued top-level fields; see
    # sloth.tune.registry.compute_config_hash).
    eval: "EvalConfig | None" = None
    thresholds: "ThresholdsConfig | None" = None


@dataclass(frozen=True)
class EvalConfig:
    """Typed, frozen representation of a ``[eval]`` TOML section.

    Only constructed when the config file has an ``[eval]`` section (even an
    empty one) — its absence leaves :attr:`RunConfig.eval` as ``None``.
    """

    holdout_fraction: float = DEFAULT_EVAL_HOLDOUT_FRACTION
    seed: int = DEFAULT_EVAL_SEED
    eval_steps: int = DEFAULT_EVAL_STEPS
    perplexity: bool = DEFAULT_EVAL_PERPLEXITY
    tool_call_family: str = DEFAULT_EVAL_TOOL_CALL_FAMILY


@dataclass(frozen=True)
class ThresholdsConfig:
    """Typed, frozen representation of the ``[eval.thresholds]`` TOML section.

    Defaults are the c36 frame-decision baseline (regression drop 2 pp,
    compliance >= 95%, latency within 10% of base, minimum suite size 100
    rows) — the single source these numbers are cited from elsewhere.
    """

    regression_drop_pp: float = DEFAULT_REGRESSION_DROP_PP
    compliance_min_pct: float = DEFAULT_COMPLIANCE_MIN_PCT
    latency_max_ratio: float = DEFAULT_LATENCY_MAX_RATIO
    min_suite_rows: int = DEFAULT_MIN_SUITE_ROWS


# ---------------------------------------------------------------------------
# [eval] / [eval.thresholds] known keys — unknown keys are rejected the same
# way an unrecognised [hyperparameters] value would be (CliError + hint).
# ---------------------------------------------------------------------------

_EVAL_KNOWN_KEYS: frozenset[str] = frozenset(
    {"holdout_fraction", "seed", "eval_steps", "perplexity", "tool_call_family", "thresholds"}
)
_THRESHOLDS_KNOWN_KEYS: frozenset[str] = frozenset(
    {"regression_drop_pp", "compliance_min_pct", "latency_max_ratio", "min_suite_rows"}
)

#: Accepted keys in ``[run.dataset_map]`` — one chat-schema key, three task-schema keys.
_DATASET_MAP_KNOWN_KEYS: frozenset[str] = frozenset(
    {"messages", "task", "input", "expected_output"}
)


def _reject_unknown_keys(section: dict, known: frozenset[str], section_name: str) -> None:
    """Raise ``CliError(code=1)`` naming any key in *section* not in *known*."""
    unknown = sorted(set(section) - known)
    if unknown:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"unknown key(s) {unknown} in [{section_name}] section",
            remediation=(
                f"Remove the unrecognised key(s) from [{section_name}], or fix the typo. "
                f"Accepted keys: {sorted(known)}."
            ),
        )


# ---------------------------------------------------------------------------
# Hyperparameter validation
# ---------------------------------------------------------------------------


def _require_int(hp: dict, key: str, default: int, *, minimum: int) -> int:
    """Return ``hp[key]`` as an ``int >= minimum``, falling back to *default*.

    Rejects non-int values (and ``bool``, which is an ``int`` subclass) and
    out-of-range values with a ``CliError(code=1)`` that names the key — so a
    malformed ``run.toml`` fails here, not deep inside the ML stack.
    """
    value = hp.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"hyperparameter '{key}' must be an integer, got {type(value).__name__}",
            remediation=f"Set `{key} = <int>` (>= {minimum}) in the [hyperparameters] section.",
        )
    if value < minimum:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"hyperparameter '{key}' must be >= {minimum}, got {value}",
            remediation=f"Set `{key}` to an integer >= {minimum} in [hyperparameters].",
        )
    return value


def _require_float(
    hp: dict,
    key: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return ``hp[key]`` as a ``float`` in ``[minimum, maximum]``, else *default*.

    Accepts ``int`` or ``float`` (but not ``bool``) and validates the optional
    inclusive bounds, raising ``CliError(code=1)`` naming the key on failure.
    """
    value = hp.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"hyperparameter '{key}' must be a number, got {type(value).__name__}",
            remediation=f"Set `{key} = <number>` in the [hyperparameters] section.",
        )
    fvalue = float(value)
    if (minimum is not None and fvalue < minimum) or (maximum is not None and fvalue > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"hyperparameter '{key}' must be in {bound}, got {fvalue}",
            remediation=f"Set `{key}` to a number in {bound} in [hyperparameters].",
        )
    return fvalue


def _require_bool(hp: dict, key: str, default: bool) -> bool:
    """Return ``hp[key]`` as a ``bool``, else *default*; raise on any other type."""
    value = hp.get(key, default)
    if not isinstance(value, bool):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"hyperparameter '{key}' must be a boolean, got {type(value).__name__}",
            remediation=f"Set `{key} = true` or `{key} = false` in [hyperparameters].",
        )
    return value


def _require_str(hp: dict, key: str, default: str) -> str:
    """Return ``hp[key]`` as a ``str``, else *default*; raise on any other type."""
    value = hp.get(key, default)
    if not isinstance(value, str):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"'{key}' must be a string, got {type(value).__name__}",
            remediation=f'Set `{key} = "<value>"`.',
        )
    return value


_TARGET_MODULES_REMEDIATION = (
    "Set `target_modules` in [hyperparameters] to one of three forms: "
    "a non-empty list of non-empty strings (e.g. "
    '`target_modules = ["q_proj", "k_proj"]`), a single non-empty regex '
    'string (e.g. `target_modules = "model\\.layers\\.\\d+\\.self_attn\\..*"`), '
    'or a known preset name as `"preset:<name>"` (e.g. `target_modules = "preset:lfm2"`). '
    "Omit the key entirely to leave target_modules unset."
)


def _require_target_modules(hp: dict) -> list[str] | str | None:
    """Return ``hp["target_modules"]``, validated, or ``None`` if absent.

    Accepted forms:

    - absent -> ``None``
    - a non-empty ``list[str]`` of non-empty strings -> returned unchanged
    - a non-empty ``str`` (a regex, or ``"preset:<name>"``) -> returned
      unchanged (preset names are checked against
      :data:`sloth.tune.presets.PRESETS` here so an unknown preset fails at
      load time, before any GPU spend)

    Anything else (wrong type, empty list/string, unknown preset) raises
    ``CliError(code=1)`` whose remediation names all three accepted forms.
    """
    if "target_modules" not in hp:
        return None

    value = hp["target_modules"]

    if isinstance(value, list):
        if not value or not all(isinstance(item, str) and item for item in value):
            raise CliError(
                code=EXIT_USER_ERROR,
                message="hyperparameter 'target_modules' list must be non-empty "
                "and contain only non-empty strings",
                remediation=_TARGET_MODULES_REMEDIATION,
            )
        return value

    if isinstance(value, str):
        if not value:
            raise CliError(
                code=EXIT_USER_ERROR,
                message="hyperparameter 'target_modules' string must be non-empty",
                remediation=_TARGET_MODULES_REMEDIATION,
            )
        if value.startswith("preset:"):
            # Import locally to avoid any risk of a module-level import cycle
            # between config.py and presets.py; both are pure stdlib so the
            # cost is negligible.
            from sloth.tune.presets import PRESETS

            name = value[len("preset:") :]
            if name not in PRESETS:
                raise CliError(
                    code=EXIT_USER_ERROR,
                    message=f"hyperparameter 'target_modules' references unknown preset "
                    f"'{value}'. Known presets: {sorted(PRESETS)}.",
                    remediation=_TARGET_MODULES_REMEDIATION,
                )
        return value

    raise CliError(
        code=EXIT_USER_ERROR,
        message="hyperparameter 'target_modules' must be a list of strings, a regex "
        f"string, or a 'preset:<name>' string, got {type(value).__name__}",
        remediation=_TARGET_MODULES_REMEDIATION,
    )


# ---------------------------------------------------------------------------
# [run.dataset_map] parsing
# ---------------------------------------------------------------------------

_DATASET_MAP_REMEDIATION = (
    "Set [run.dataset_map] keys to column names in the hub dataset, e.g. "
    '`messages = "conversations"` for the chat schema, or '
    '`task = "instruction"` / `input = "context"` / '
    '`expected_output = "response"` for the task schema. '
    f"Accepted keys: {sorted(_DATASET_MAP_KNOWN_KEYS)}."
)


def _require_dataset_map(run_section: dict) -> "dict[str, str] | None":
    """Return ``run_section["dataset_map"]`` validated, or ``None`` if absent.

    ``[run.dataset_map]`` documents how an ``hf:<org>/<name>[:split]`` dataset's
    hub columns map onto the chat/task schema fields. Absent -> ``None`` (the
    common case for a local JSONL ``dataset``). Present but not a table, an
    unknown key, or a non-string value all raise ``CliError(code=1)``.
    """
    if "dataset_map" not in run_section:
        return None

    value = run_section["dataset_map"]
    if not isinstance(value, dict):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"'dataset_map' must be a table ([run.dataset_map]), got {type(value).__name__}"
            ),
            remediation=_DATASET_MAP_REMEDIATION,
        )
    _reject_unknown_keys(value, _DATASET_MAP_KNOWN_KEYS, "run.dataset_map")
    for key, column in value.items():
        if not isinstance(column, str) or not column:
            raise CliError(
                code=EXIT_USER_ERROR,
                message=(
                    f"[run.dataset_map] key '{key}' must map to a non-empty string "
                    f"column name, got {column!r}"
                ),
                remediation=_DATASET_MAP_REMEDIATION,
            )
    return dict(value)


# ---------------------------------------------------------------------------
# [eval] / [eval.thresholds] parsing
# ---------------------------------------------------------------------------


def _parse_thresholds(eval_section: dict) -> ThresholdsConfig:
    """Return a :class:`ThresholdsConfig` built from ``eval_section["thresholds"]``.

    Missing keys (or a wholly absent ``thresholds`` sub-table) fall back to
    the c36 baseline defaults. Raises ``CliError(code=1)`` if ``thresholds``
    is present but not a table, or on any unknown/malformed key.
    """
    thresholds_section = eval_section.get("thresholds", {})
    if not isinstance(thresholds_section, dict):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                "'thresholds' must be a table ([eval.thresholds]), got "
                f"{type(thresholds_section).__name__}"
            ),
            remediation="Define thresholds under an [eval.thresholds] section, not as a scalar.",
        )
    _reject_unknown_keys(thresholds_section, _THRESHOLDS_KNOWN_KEYS, "eval.thresholds")

    return ThresholdsConfig(
        regression_drop_pp=_require_float(
            thresholds_section, "regression_drop_pp", DEFAULT_REGRESSION_DROP_PP, minimum=0.0
        ),
        compliance_min_pct=_require_float(
            thresholds_section,
            "compliance_min_pct",
            DEFAULT_COMPLIANCE_MIN_PCT,
            minimum=0.0,
            maximum=100.0,
        ),
        latency_max_ratio=_require_float(
            thresholds_section, "latency_max_ratio", DEFAULT_LATENCY_MAX_RATIO, minimum=0.0
        ),
        min_suite_rows=_require_int(
            thresholds_section, "min_suite_rows", DEFAULT_MIN_SUITE_ROWS, minimum=0
        ),
    )


def _parse_eval(raw: dict) -> tuple["EvalConfig | None", "ThresholdsConfig | None"]:
    """Return ``(eval, thresholds)`` parsed from ``raw["eval"]``.

    Both are ``None`` when *raw* has no ``[eval]`` section at all — this
    keeps :func:`sloth.tune.registry.compute_config_hash` stable for every
    pre-existing config, since it drops ``None``-valued top-level fields.
    Raises ``CliError(code=1)`` on any unknown or malformed key in either
    ``[eval]`` or ``[eval.thresholds]``.
    """
    if "eval" not in raw:
        return None, None

    eval_section = raw["eval"] or {}
    _reject_unknown_keys(eval_section, _EVAL_KNOWN_KEYS, "eval")

    thresholds = _parse_thresholds(eval_section)
    eval_config = EvalConfig(
        holdout_fraction=_require_float(
            eval_section,
            "holdout_fraction",
            DEFAULT_EVAL_HOLDOUT_FRACTION,
            minimum=0.0,
            maximum=1.0,
        ),
        seed=_require_int(eval_section, "seed", DEFAULT_EVAL_SEED, minimum=0),
        eval_steps=_require_int(eval_section, "eval_steps", DEFAULT_EVAL_STEPS, minimum=0),
        perplexity=_require_bool(eval_section, "perplexity", DEFAULT_EVAL_PERPLEXITY),
        tool_call_family=_require_str(
            eval_section, "tool_call_family", DEFAULT_EVAL_TOOL_CALL_FAMILY
        ),
    )
    return eval_config, thresholds


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> RunConfig:
    """Parse *path* as TOML and return a validated :class:`RunConfig`.

    Raises:
        CliError(code=2): if the file cannot be read or is not valid TOML.
        CliError(code=1): if a required key is absent or ``method`` is invalid.
    """
    path = Path(path)

    # --- read & parse -------------------------------------------------------
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"Config file not found: {path}",
            remediation=(
                f"Create {path} with a [run] section containing "
                "model, dataset, and output keys. "
                "Run `sloth explain train` for an annotated template."
            ),
        )
    except tomllib.TOMLDecodeError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"Config file is not valid TOML: {path} — {exc}",
            remediation=(
                "Fix the TOML syntax error reported above. "
                "Run `python -c \"import tomllib; tomllib.load(open('run.toml','rb'))\"` "
                "to iterate quickly."
            ),
        )

    # --- required keys ------------------------------------------------------
    run_section: dict = raw.get("run", {})
    for key in ("model", "dataset", "output"):
        if key not in run_section:
            raise CliError(
                code=EXIT_USER_ERROR,
                message=f"Missing required key '{key}' in [run] section of {path}",
                remediation=(
                    f'Add `{key} = "<value>"` under the [run] header in {path}. '
                    "Run `sloth explain train` for an annotated template."
                ),
            )

    # --- method validation --------------------------------------------------
    method = run_section.get("method", DEFAULT_METHOD)
    if method not in VALID_METHODS:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"Invalid method '{method}' in {path}. "
                f"Accepted values: {sorted(VALID_METHODS)}."
            ),
            remediation=(
                'Set `method = "lora"` or `method = "qlora"` in the [run] section. '
                "Full fine-tuning of large dense models is out of scope for unsloth-cli."
            ),
        )

    # --- hyperparameters (all optional, but type/range-checked) -------------
    hp: dict = raw.get("hyperparameters", {})

    # --- eval / thresholds (both optional; absent [eval] -> both None) -----
    eval_config, thresholds_config = _parse_eval(raw)

    return RunConfig(
        model=run_section["model"],
        dataset=run_section["dataset"],
        output=run_section["output"],
        method=method,
        lora_r=_require_int(hp, "lora_r", DEFAULT_LORA_R, minimum=1),
        lora_alpha=_require_int(hp, "lora_alpha", DEFAULT_LORA_ALPHA, minimum=1),
        lora_dropout=_require_float(
            hp, "lora_dropout", DEFAULT_LORA_DROPOUT, minimum=0.0, maximum=1.0
        ),
        learning_rate=_require_float(hp, "learning_rate", DEFAULT_LEARNING_RATE, minimum=0.0),
        max_seq_len=_require_int(hp, "max_seq_len", DEFAULT_MAX_SEQ_LEN, minimum=1),
        batch_size=_require_int(hp, "batch_size", DEFAULT_BATCH_SIZE, minimum=1),
        grad_accum=_require_int(hp, "grad_accum", DEFAULT_GRAD_ACCUM, minimum=1),
        max_steps=_require_int(hp, "max_steps", DEFAULT_MAX_STEPS, minimum=1),
        seed=_require_int(hp, "seed", DEFAULT_SEED, minimum=0),
        load_in_4bit=_require_bool(hp, "load_in_4bit", DEFAULT_LOAD_IN_4BIT),
        target_modules=_require_target_modules(hp),
        dataset_map=_require_dataset_map(run_section),
        eval=eval_config,
        thresholds=thresholds_config,
    )
