"""``sloth train`` — the integrator verb that ties the ``sloth.tune`` core together.

Flow::

    load_config  ->  validate dataset  ->  scope-guard  ->  dry-run | train

1. **load_config** — parse the ``run.toml`` into a :class:`~sloth.tune.config.RunConfig`
   (``CliError`` propagates on a missing/invalid config).
2. **validate dataset** — sniff the schema and run
   :func:`~sloth.tune.datasets.validate_dataset` *before any GPU work* so a
   malformed dataset fails fast with ``CliError(code=1)`` ("validate before
   spending GPU").
3. **scope-guard** — :func:`~sloth.tune.scope.check_scope` classifies the
   (model, method) request. An out-of-scope request (e.g. full fine-tuning of a
   large dense model) emits its warning EXPLICITLY to stderr via
   :func:`emit_diagnostic` and is then hard-refused with ``CliError(code=1)``.
4. **dry-run | train** — :func:`~sloth.tune._trainer.run_training` resolves the
   plan. ``--dry-run`` returns the plan without importing torch; a real run
   delegates to the trainer, which loads the backend, trains the adapter, and
   writes ``training_metadata.json`` next to the adapter output.

This module imports no torch/unsloth — the heavy stack lives only inside the
trainer's ``_load_backend`` seam, lazily. Importing ``train`` stays torch-free.

Dataset-schema choice
---------------------
The schema is **inferred from the first non-blank record** via
:func:`~sloth.tune.datasets.detect_schema` (``"chat"`` when a ``messages`` key is
present, ``"task"`` for the ``task``/``input``/``expected_output`` shape).  When
the file is empty, unreadable, or the first record is inconclusive, validation
falls back to the **chat** schema (the documented default) and lets
``validate_dataset`` raise the authoritative ``CliError``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sloth.cli._errors import EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_diagnostic, emit_result
from sloth.tune._trainer import run_training
from sloth.tune.config import load_config
from sloth.tune.datasets import detect_schema, validate_dataset
from sloth.tune.scope import check_scope

#: Schema assumed when the dataset's first record cannot be classified.
DEFAULT_SCHEMA = "chat"


# ---------------------------------------------------------------------------
# Schema inference (pure — no torch)
# ---------------------------------------------------------------------------


def _resolve_schema(dataset: str) -> str:
    """Infer the dataset schema from its first non-blank record.

    Returns ``"chat"`` or ``"task"`` when the first record is classifiable,
    otherwise the :data:`DEFAULT_SCHEMA`. Never raises: any read/parse failure
    is deferred to :func:`validate_dataset`, which surfaces the authoritative
    ``CliError`` once it re-reads the file.
    """
    path = Path(dataset)
    try:
        with path.open(encoding="utf-8") as fh:
            record: Any = None
            for line in fh:
                stripped = line.strip()
                if stripped:
                    record = json.loads(stripped)
                    break
    except (OSError, json.JSONDecodeError):
        return DEFAULT_SCHEMA
    if record is None:
        return DEFAULT_SCHEMA
    return detect_schema(record) or DEFAULT_SCHEMA


# ---------------------------------------------------------------------------
# Plan rendering (text mode)
# ---------------------------------------------------------------------------


def _render_plan_text(plan: dict[str, Any]) -> str:
    """Render the resolved training plan as human-readable text for stdout."""
    mode = "dry-run" if plan.get("dry_run") else "train"
    lines = [
        f"plan: {mode}",
        f"model:   {plan.get('model')}",
        f"method:  {plan.get('method')}",
        f"dataset: {plan.get('dataset')}",
        f"output:  {plan.get('output')}",
        "hyperparameters:",
    ]
    for key, value in plan.get("hyperparameters", {}).items():
        lines.append(f"  {key}: {value}")

    # Real-run extras (present only after the trainer ran the job).
    if plan.get("status"):
        lines.append(f"status:   {plan['status']}")
    if plan.get("adapter_dir"):
        lines.append(f"adapter:  {plan['adapter_dir']}")
    if plan.get("metadata_path"):
        lines.append(f"metadata: {plan['metadata_path']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    """Handler for ``sloth train``.

    Returns ``0`` on success; every failure raises :class:`CliError`.
    """
    json_mode = bool(getattr(args, "json", False))
    dry_run = bool(getattr(args, "dry_run", False))

    # 1) Load + validate the run config (CliError propagates on bad/missing file).
    config = load_config(args.config)

    # 2) Validate the dataset BEFORE any GPU work ("validate before spending GPU").
    schema = _resolve_schema(config.dataset)
    validate_dataset(config.dataset, schema)

    # 3) Scope-guard: warn explicitly, then hard-refuse an out-of-scope request.
    scope = check_scope(config.model, config.method)
    if scope.warning:
        emit_diagnostic(scope.warning)
    if scope.out_of_scope:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"refusing to train: {scope.message}",
            remediation=(
                scope.warning or "Switch to an adapter method: set method='lora' or method='qlora'."
            ),
        )

    # 4) Resolve the plan (dry-run) or run the real job (delegates to the trainer,
    #    which loads the backend, trains, and writes metadata next to the adapter).
    plan = run_training(config, dry_run=dry_run)

    if json_mode:
        emit_result(plan, json_mode=True)
    else:
        emit_result(_render_plan_text(plan), json_mode=False)
    return 0


# ---------------------------------------------------------------------------
# Subparser registration
# ---------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``train`` subparser on *sub*."""
    p = sub.add_parser(
        "train",
        help="Validate a dataset and run (or plan) a LoRA/QLoRA adapter job.",
    )
    p.add_argument(
        "--config",
        required=True,
        metavar="TOML",
        help="Path to the run.toml describing the model, dataset, output, and method.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and resolve the plan without importing torch or training.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_train)
