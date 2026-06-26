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
    )
