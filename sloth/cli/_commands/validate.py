"""``sloth validate`` — standalone dataset validator.

Calls the SAME :func:`~sloth.tune.datasets.validate_dataset` function that
``sloth train`` uses, so validation rules are shared (no duplication).

Usage::

    sloth validate --dataset data/train.jsonl
    sloth validate --dataset data/train.jsonl --schema task
    sloth validate --dataset data/train.jsonl --json

Exit codes:
    0 — dataset is valid (report emitted to stdout)
    1 — invalid dataset, missing file, or bad schema (error to stderr)
    2 — environment error (dataset file exists but cannot be opened)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sloth.cli._errors import EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_diagnostic, emit_result
from sloth.tune.datasets import detect_schema, validate_dataset

#: Schema assumed when the dataset's first record cannot be classified.
DEFAULT_SCHEMA = "chat"


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int | None:
    """Handler for ``sloth validate``.

    Validates a JSONL dataset file against the requested schema (or auto-detected
    schema) and emits a structured report on success.  Uses the *same*
    :func:`~sloth.tune.datasets.validate_dataset` function that ``sloth train``
    calls, so rules are shared.

    Returns ``None`` (exit 0) on success; raises :class:`CliError` on failure.
    """
    json_mode = bool(getattr(args, "json", False))
    dataset_path = Path(args.dataset)
    schema = args.schema

    # --- check file exists --------------------------------------------------
    if not dataset_path.is_file():
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"dataset file not found: {dataset_path}",
            remediation="Pass an existing JSONL file with --dataset <path>.",
        )

    # --- resolve schema -----------------------------------------------------
    if schema is None:
        # Auto-detect from the first record (same logic as train).
        try:
            with dataset_path.open(encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if stripped:
                        record = json.loads(stripped)
                        break
                else:
                    record = None
        except (OSError, json.JSONDecodeError):
            record = None

        schema = detect_schema(record) if record else DEFAULT_SCHEMA
        emit_diagnostic(f"auto-detected schema: {schema!r}")
    # else: schema was passed explicitly and already validated by argparse choices.

    # --- validate (shared code path with train) -----------------------------
    records = validate_dataset(dataset_path, schema)

    # --- emit report --------------------------------------------------------
    report: dict[str, Any] = {
        "valid": True,
        "schema": schema,
        "line_count": len(records),
    }

    if json_mode:
        emit_result(report, json_mode=True)
    else:
        lines = [
            f"dataset: {dataset_path}",
            f"schema:  {schema}",
            f"records: {len(records)}",
            "status:  valid",
        ]
        emit_result("\n".join(lines), json_mode=False)

    return None


# ---------------------------------------------------------------------------
# Subparser registration
# ---------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``validate`` subparser on *sub*."""
    p = sub.add_parser(
        "validate",
        help="Validate a JSONL dataset file against the chat or task schema.",
        description=(
            "Validate a JSONL dataset file against the chat or task schema. "
            "Uses the same validation rules as ``sloth train``."
        ),
    )
    p.add_argument(
        "--dataset",
        required=True,
        metavar="PATH",
        help="Path to the JSONL dataset file.",
    )
    p.add_argument(
        "--schema",
        choices=["chat", "task"],
        default=None,
        help=("Schema to validate against (default: auto-detect from the first record)."),
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_validate)
