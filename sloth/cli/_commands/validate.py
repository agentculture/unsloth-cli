"""``sloth validate`` — standalone dataset / eval-suite validator.

Calls the SAME :func:`~sloth.tune.datasets.validate_dataset` function that
``sloth train`` uses (via ``--dataset``), and the SAME
:func:`~sloth.tune.datasets.validate_suite` function that ``sloth eval`` uses
for its pre-launch check (via ``--suite``) — so validation rules never drift
between the verbs that enforce them and this standalone checker.

Exactly one of ``--dataset`` / ``--suite`` is required (checked here rather than
via an argparse mutually-exclusive group, so calling :func:`cmd_validate`
directly gets the same contract):

* ``--dataset PATH`` — a single JSONL training dataset (chat or task schema,
  auto-detected unless ``--schema`` is passed).
* ``--suite PATH`` — a single JSONL eval suite, or a directory of them (every
  ``*.jsonl`` child is validated); defaults to the ``task`` schema and reports
  per-file record counts plus a total.

Usage::

    sloth validate --dataset data/train.jsonl
    sloth validate --dataset data/train.jsonl --schema task
    sloth validate --dataset data/train.jsonl --json
    sloth validate --suite data/eval.jsonl
    sloth validate --suite examples/eval/ --json

Exit codes:
    0 — dataset/suite is valid (report emitted to stdout)
    1 — invalid dataset/suite, missing file, bad schema, or both/neither of
        --dataset and --suite passed (error to stderr)
    2 — environment error (dataset file exists but cannot be opened)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sloth.cli._errors import EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_diagnostic, emit_result
from sloth.tune.datasets import detect_schema, validate_dataset, validate_suite

#: Schema assumed when the dataset's first record cannot be classified.
DEFAULT_SCHEMA = "chat"

#: Schema a --suite is validated against (eval suites are always task-schema).
SUITE_SCHEMA = "task"


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------


def _cmd_validate_suite(args: argparse.Namespace) -> int | None:
    """Handler for ``sloth validate --suite`` — delegates to :func:`validate_suite`.

    Uses the *same* :func:`~sloth.tune.datasets.validate_suite` function that
    ``sloth eval`` calls for its pre-launch check, so a suite that passes here
    is guaranteed to pass ``sloth eval``'s host-side validation too. ``--suite``
    is **always** validated against the task schema — ``sloth eval`` never
    accepts anything else — so an explicit ``--schema`` other than ``task``
    is a user error rather than being silently honoured (that would let a
    suite be reported "valid" here and still be rejected by ``sloth eval``).
    """
    json_mode = bool(getattr(args, "json", False))
    suite_arg = args.suite
    requested_schema = getattr(args, "schema", None)
    if requested_schema is not None and requested_schema != SUITE_SCHEMA:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"--schema {requested_schema} is not supported with --suite "
                f"(eval suites are always validated against the {SUITE_SCHEMA!r} schema)"
            ),
            remediation=("Drop --schema when using --suite, or pass --schema task explicitly."),
        )
    schema = SUITE_SCHEMA

    report = validate_suite(suite_arg, schema=schema)
    files = report["files"]
    total = report["total_records"]

    payload: dict[str, Any] = {
        "valid": True,
        "schema": schema,
        "files": files,
        "total_records": total,
    }

    if json_mode:
        emit_result(payload, json_mode=True)
    else:
        lines = [f"suite:  {suite_arg}", f"schema: {schema}"]
        for entry in files:
            lines.append(f"  {entry['path']}: {entry['line_count']} records")
        lines.append(f"total:  {total} records across {len(files)} file(s)")
        lines.append("status: valid")
        emit_result("\n".join(lines), json_mode=False)

    return None


def _require_exactly_one_target(dataset_arg: str | None, suite_arg: str | None) -> None:
    """Raise ``CliError(code=1)`` unless exactly one of the two args is set."""
    if dataset_arg and suite_arg:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="--dataset and --suite are mutually exclusive; pass exactly one",
            remediation=(
                "Use --dataset <path> to validate a training dataset, or "
                "--suite <path> to validate an eval suite (file or directory)."
            ),
        )
    if not dataset_arg and not suite_arg:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="one of --dataset or --suite is required",
            remediation=(
                "Pass --dataset <path> to validate a training dataset, or "
                "--suite <path> to validate an eval suite (file or directory)."
            ),
        )


def _detect_dataset_schema(dataset_path: Path) -> str:
    """Auto-detect the schema from the dataset's first non-blank record.

    Same logic ``sloth train`` uses; falls back to :data:`DEFAULT_SCHEMA` when
    the file is empty, unreadable, or its first record is not valid JSON.
    """
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
    return schema


def _cmd_validate_dataset(args: argparse.Namespace) -> None:
    """Handler for ``sloth validate --dataset`` — delegates to :func:`validate_dataset`.

    Uses the *same* :func:`~sloth.tune.datasets.validate_dataset` function that
    ``sloth train`` calls internally, so the accepted rules never drift.
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

    # --- resolve schema -------------------------------------------------
    # else: schema was passed explicitly and already validated by argparse choices.
    if schema is None:
        schema = _detect_dataset_schema(dataset_path)

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


def cmd_validate(args: argparse.Namespace) -> int | None:
    """Handler for ``sloth validate``.

    Dispatches to one of two shared validators depending on which flag was
    passed — exactly one of ``--dataset`` / ``--suite`` is required:

    * ``--dataset`` validates a JSONL training dataset against the requested
      schema (or auto-detected schema) — see :func:`_cmd_validate_dataset`.
    * ``--suite`` validates a JSONL eval suite (a file, or a directory of them),
      via the *same* :func:`~sloth.tune.datasets.validate_suite` function
      ``sloth eval`` calls before launching the container — see
      :func:`_cmd_validate_suite`.

    Returns ``None`` (exit 0) on success; raises :class:`CliError` on failure.
    """
    dataset_arg = getattr(args, "dataset", None)
    suite_arg = getattr(args, "suite", None)

    _require_exactly_one_target(dataset_arg, suite_arg)

    if suite_arg is not None:
        return _cmd_validate_suite(args)

    return _cmd_validate_dataset(args)


# ---------------------------------------------------------------------------
# Subparser registration
# ---------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``validate`` subparser on *sub*."""
    p = sub.add_parser(
        "validate",
        help="Validate a JSONL dataset (--dataset) or eval suite (--suite).",
        description=(
            "Validate a JSONL dataset file (--dataset) against the chat or task "
            "schema, or an eval suite (--suite; a file or a directory of them) "
            "against the task schema. Exactly one of --dataset / --suite is "
            "required. Uses the same validation rules as ``sloth train`` and "
            "``sloth eval``, respectively."
        ),
    )
    p.add_argument(
        "--dataset",
        default=None,
        metavar="PATH",
        help="Path to a JSONL training dataset file. Mutually exclusive with --suite.",
    )
    p.add_argument(
        "--suite",
        default=None,
        metavar="PATH",
        help=(
            "Path to a task-schema JSONL eval suite, or a directory of them "
            "(every *.jsonl child is validated). Mutually exclusive with --dataset."
        ),
    )
    p.add_argument(
        "--schema",
        choices=["chat", "task"],
        default=None,
        help=(
            "Schema to validate against. With --dataset: default auto-detect "
            "from the first record. With --suite: always 'task' (eval suites "
            "are task-schema only); passing anything else with --suite exits 1."
        ),
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_validate)
