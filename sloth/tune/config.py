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

    # --- hyperparameters (all optional) -------------------------------------
    hp: dict = raw.get("hyperparameters", {})

    return RunConfig(
        model=run_section["model"],
        dataset=run_section["dataset"],
        output=run_section["output"],
        method=method,
        lora_r=hp.get("lora_r", DEFAULT_LORA_R),
        lora_alpha=hp.get("lora_alpha", DEFAULT_LORA_ALPHA),
        lora_dropout=hp.get("lora_dropout", DEFAULT_LORA_DROPOUT),
        learning_rate=hp.get("learning_rate", DEFAULT_LEARNING_RATE),
        max_seq_len=hp.get("max_seq_len", DEFAULT_MAX_SEQ_LEN),
        batch_size=hp.get("batch_size", DEFAULT_BATCH_SIZE),
        grad_accum=hp.get("grad_accum", DEFAULT_GRAD_ACCUM),
        max_steps=hp.get("max_steps", DEFAULT_MAX_STEPS),
        seed=hp.get("seed", DEFAULT_SEED),
        load_in_4bit=hp.get("load_in_4bit", DEFAULT_LOAD_IN_4BIT),
    )
