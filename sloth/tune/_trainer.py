"""Lazy LoRA/QLoRA trainer adapter — the ONLY module that touches torch/unsloth.

The heavy ML stack (``unsloth``, ``torch``, ``trl``) is imported **only** inside
:func:`_load_backend`, never at module top level. Importing this module — or the
``sloth`` package — stays torch-free, so the introspection verbs keep working on
a machine with no GPU and no ML stack installed (the repo's load-bearing
"zero runtime dependencies" rule).

Public entry point
------------------
:func:`run_training` resolves a :class:`~sloth.tune.config.RunConfig` into a
training *plan* and, unless ``dry_run`` is set, runs the adapter job.

Out-of-scope policy (documented contract)
----------------------------------------
* **Dry-run never raises on scope.** It returns the resolved plan with the
  scope decision embedded under ``plan["scope"]`` so the calling verb can warn,
  downgrade, or decide. A dry-run never imports torch.
* **A real (non-dry-run) run hard-refuses an out-of-scope request** by raising
  ``CliError(code=1)`` *before* importing the heavy backend — so no GPU time is
  spent setting up a job that this tool will not run (full fine-tuning of large
  dense models is explicitly out of scope; use ``lora``/``qlora``).

Missing-backend policy
----------------------
When the ML stack is absent, :func:`_load_backend` raises ``ImportError`` and
:func:`run_training` converts it into ``CliError(code=2)`` carrying the
``uv tool install unsloth-cli`` install hint. Isolating the heavy import in a
tiny helper makes it monkeypatchable: tests inject an ``ImportError`` (to assert
the install-hint path) or a fake backend (to exercise the real flow) without a
GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sloth.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from sloth.tune.config import RunConfig
from sloth.tune.datasets import detect_schema, validate_dataset
from sloth.tune.metadata import write_metadata
from sloth.tune.scope import check_scope

_INSTALL_HINT = (
    "The fine-tuning backend (unsloth + torch + trl) ships with unsloth-cli. "
    "Reinstall it with `uv tool install unsloth-cli` (or `uv sync` in a checkout) "
    "on a CUDA-capable machine, then re-run."
)

_NGC_HINT = (
    "No GPU accelerator was found. Run the trainer inside the NVIDIA NGC container: "
    "nvcr.io/nvidia/pytorch:25.11-py3 "
    "(which includes the required CUDA drivers, unsloth, torch, and trl)."
)


# ---------------------------------------------------------------------------
# Plan construction (pure — no torch)
# ---------------------------------------------------------------------------


def _resolved_hyperparameters(config: RunConfig) -> dict[str, Any]:
    """Return the fully-resolved hyperparameter mapping for *config*."""
    return {
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "learning_rate": config.learning_rate,
        "max_seq_len": config.max_seq_len,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum,
        "max_steps": config.max_steps,
        "seed": config.seed,
        "load_in_4bit": config.load_in_4bit,
    }


def _build_plan(config: RunConfig, *, dry_run: bool, scope) -> dict[str, Any]:
    """Build the resolved training-plan dict (model, method, hparams, scope)."""
    return {
        "model": config.model,
        "method": config.method,
        "dataset": config.dataset,
        "output": config.output,
        "hyperparameters": _resolved_hyperparameters(config),
        "scope": {
            "ok": scope.ok,
            "out_of_scope": scope.out_of_scope,
            "warning": scope.warning,
            "downgrade_to": scope.downgrade_to,
            "message": scope.message,
        },
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# Heavy backend (the ONLY place torch/unsloth/trl are imported)
# ---------------------------------------------------------------------------


@dataclass
class _Backend:
    """Bundle of the lazily-imported ML callables used by the real training path.

    Field names are snake_case (not the PascalCase of the imported classes) to
    satisfy the field-naming convention; each holds the corresponding callable.
    """

    fast_language_model: Any  # unsloth.FastLanguageModel
    sft_trainer: Any  # trl.SFTTrainer
    sft_config: Any  # trl.SFTConfig
    torch: Any


def _load_backend() -> _Backend:
    """Import the heavy ML stack and return it as a :class:`_Backend`.

    This is the single seam where ``unsloth``/``torch``/``trl`` enter the
    process. Isolated so tests can monkeypatch it: raising ``ImportError`` here
    (or from a patched stand-in) is converted by :func:`run_training` into
    ``CliError(code=2)``; returning a fake exercises the real flow GPU-free.

    Raises:
        ImportError: if any component of the ML stack is unavailable.
    """
    import torch  # noqa: PLC0415 — intentional lazy import
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415
    from unsloth import FastLanguageModel  # noqa: PLC0415

    return _Backend(
        fast_language_model=FastLanguageModel,
        sft_trainer=SFTTrainer,
        sft_config=SFTConfig,
        torch=torch,
    )


# ---------------------------------------------------------------------------
# Dataset loading for the real path (pure — no torch)
# ---------------------------------------------------------------------------


def _detect_dataset_schema(path: Path) -> str:
    """Sniff the schema (``"chat"``/``"task"``) from the first record of *path*."""
    try:
        with path.open(encoding="utf-8") as fh:
            first_record = None
            for line in fh:
                stripped = line.strip()
                if stripped:
                    first_record = json.loads(stripped)
                    break
    except OSError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"cannot open dataset file {path}",
            remediation="Check that the file exists and is readable.",
        ) from exc
    except json.JSONDecodeError as exc:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"dataset {path}: first line is not valid JSON — {exc.msg}",
            remediation="Each line of the dataset must be a JSON object.",
        ) from exc

    schema = detect_schema(first_record) if first_record is not None else None
    if schema is None:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"cannot detect a known schema for dataset {path}",
            remediation=(
                'Use the chat schema ({"messages": [...]}) or the task schema '
                '({"task", "input", "expected_output"}).'
            ),
        )
    return schema


def _load_train_records(dataset: str) -> list[dict]:
    """Validate the dataset and return its parsed records (raises CliError on failure)."""
    path = Path(dataset)
    schema = _detect_dataset_schema(path)
    return validate_dataset(path, schema=schema)


# ---------------------------------------------------------------------------
# Real training path (uses the lazily-loaded backend; not GPU-tested in CI)
# ---------------------------------------------------------------------------


def _run_real(config: RunConfig, plan: dict[str, Any], backend: _Backend) -> dict[str, Any]:
    """Load the model, apply LoRA/QLoRA, train, save the adapter, write metadata."""
    load_in_4bit = bool(config.load_in_4bit) or config.method == "qlora"

    # Validate + load the dataset BEFORE the expensive model load, so a schema or
    # empty-dataset failure surfaces a CliError without spending any GPU/model-load
    # time ("validate before spending GPU").
    train_records = _load_train_records(config.dataset)

    # Wrap the raw list[dict] as a datasets.Dataset so SFTTrainer gets the typed
    # object it expects. Lazy-imported here (not at module top) so:
    #   (a) the module stays importable without ``datasets`` installed, and
    #   (b) tests can monkeypatch sys.modules["datasets"] to inject a fake.
    from datasets import Dataset  # noqa: PLC0415 — intentional lazy import

    train_dataset = Dataset.from_list(train_records)

    try:
        model, tokenizer = backend.fast_language_model.from_pretrained(
            model_name=config.model,
            max_seq_length=config.max_seq_len,
            load_in_4bit=load_in_4bit,
            dtype=None,
        )
        model = backend.fast_language_model.get_peft_model(
            model,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            random_state=config.seed,
        )

        sft_config = backend.sft_config(
            output_dir=config.output,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accum,
            learning_rate=config.learning_rate,
            max_steps=config.max_steps,
            seed=config.seed,
        )
        trainer = backend.sft_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            args=sft_config,
        )
        trainer.train()
    except NotImplementedError as exc:
        # Unsloth raises NotImplementedError (message: "cannot find any torch
        # accelerator") when no GPU is available. Map it to a user-actionable
        # CliError so the CLI can surface a clear remediation instead of a
        # "file a bug" generic error (code=1).
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"No GPU accelerator found — the ML backend raised: {exc}",
            remediation=_NGC_HINT,
        ) from exc

    output_dir = Path(config.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta_path = write_metadata(
        output_dir,
        model=config.model,
        method=config.method,
        dataset_path=Path(config.dataset),
        hyperparameters=plan["hyperparameters"],
    )

    result = dict(plan)
    result["status"] = "trained"
    result["adapter_dir"] = str(output_dir)
    result["metadata_path"] = str(meta_path)
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_training(config: RunConfig, *, dry_run: bool = False) -> dict[str, Any]:
    """Resolve *config* into a training plan and (unless ``dry_run``) run the job.

    Parameters
    ----------
    config:
        The validated run configuration (see :class:`sloth.tune.config.RunConfig`).
    dry_run:
        When ``True``, return the resolved plan immediately without importing the
        heavy backend. The plan carries the scope decision so the caller can warn
        or downgrade.

    Returns
    -------
    dict
        The resolved training plan. For a real run the dict additionally carries
        ``status``, ``adapter_dir``, and ``metadata_path``.

    Raises
    ------
    CliError(code=1)
        For a non-dry-run *out-of-scope* request (hard refusal, before any heavy
        import).
    CliError(code=2)
        When the ML backend is not installed (the ``uv tool install unsloth-cli``
        install hint is attached).
    """
    scope = check_scope(config.model, config.method)
    plan = _build_plan(config, dry_run=dry_run, scope=scope)

    if dry_run:
        return plan

    # Hard refusal for out-of-scope real runs — before importing torch.
    if scope.out_of_scope:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"Refusing to train: {scope.message}",
            remediation=scope.warning or "Switch to method='lora' or method='qlora'.",
        )

    try:
        backend = _load_backend()
    except ImportError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message="The fine-tuning backend (unsloth + torch + trl) is not installed.",
            remediation=_INSTALL_HINT,
        ) from exc

    return _run_real(config, plan, backend)
