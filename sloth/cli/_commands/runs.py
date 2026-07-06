"""``sloth runs`` — the run-registry noun: list/show past training runs.

Reads ``<runs-root>/runs.jsonl`` (see :mod:`sloth.tune.registry` for the file
format, the runs-root rule, and the run_id scheme) so an agent can enumerate
and inspect past ``sloth train`` runs from the CLI alone — no directory
walking.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sloth.cli._commands.overview import emit_overview
from sloth.cli._errors import EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_diagnostic, emit_result
from sloth.tune import registry as registry_mod

_VERBS = [
    "runs list — newest-first table of every registered run",
    "runs show <run_id> — the full registry record + whether its output dir still exists",
    "runs overview — this description",
]


def _resolve_runs_root(args: argparse.Namespace) -> Path:
    raw = getattr(args, "runs_root", None)
    return Path(raw) if raw else Path.cwd()


def _emit_registry_diagnostics(diagnostics: list[str]) -> None:
    for message in diagnostics:
        emit_diagnostic(message)


# ---------------------------------------------------------------------------
# runs list
# ---------------------------------------------------------------------------


def cmd_runs_list(args: argparse.Namespace) -> int:
    """Handler for ``sloth runs list``.

    Missing/empty registry -> an honest empty list (exit 0), never an error.
    A corrupt line is skipped with a diagnostic on stderr (never a crash).
    """
    runs_root = _resolve_runs_root(args)
    records, diagnostics = registry_mod.read_registry(runs_root)
    _emit_registry_diagnostics(diagnostics)

    records_sorted = sorted(records, key=lambda r: str(r.get("started", "")), reverse=True)
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result(records_sorted, json_mode=True)
        return 0

    if not records_sorted:
        emit_result(f"no runs registered under {runs_root}", json_mode=False)
        return 0

    header = f"{'run_id':<32} {'status':<8} {'model':<24} {'method':<6} started"
    lines = [header, "-" * len(header)]
    for rec in records_sorted:
        lines.append(
            f"{str(rec.get('run_id', '?')):<32} {str(rec.get('status', '?')):<8} "
            f"{str(rec.get('model', '?')):<24} {str(rec.get('method', '?')):<6} "
            f"{rec.get('started', '?')}"
        )
    emit_result("\n".join(lines), json_mode=False)
    return 0


# ---------------------------------------------------------------------------
# runs show
# ---------------------------------------------------------------------------


def cmd_runs_show(args: argparse.Namespace) -> int:
    """Handler for ``sloth runs show <run_id>``.

    Emits the full registry record plus ``output_dir_exists`` (whether the
    recorded output directory is still present on disk). Raises
    ``CliError(code=1)`` when *run_id* is not found in the registry.
    """
    runs_root = _resolve_runs_root(args)
    record = registry_mod.find_run(runs_root, args.run_id)
    if record is None:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"run_id '{args.run_id}' not found in " f"{registry_mod.registry_path(runs_root)}"
            ),
            remediation=f"List known run_ids with: sloth runs list --runs-root {runs_root}",
        )

    output_dir = record.get("output_dir", "")
    report: dict[str, Any] = dict(record)
    report["output_dir_exists"] = bool(output_dir) and Path(output_dir).is_dir()

    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result(report, json_mode=True)
        return 0

    lines = [f"{key}: {value}" for key, value in report.items()]
    emit_result("\n".join(lines), json_mode=False)
    return 0


# ---------------------------------------------------------------------------
# runs overview
# ---------------------------------------------------------------------------


def _runs_sections() -> list[dict[str, object]]:
    return [
        {"title": "Verbs", "items": list(_VERBS)},
        {
            "title": "Registry file",
            "items": [
                "location: <runs-root>/runs.jsonl "
                "(runs-root = the parent dir of a run's output dir)",
                "one JSON line per run: {run_id, config_hash, output_dir, model, "
                "method, dataset{sha256,line_count}, started, finished, status}",
                "status: running -> ok | failed (no pid tracking in v1 — a killed "
                "train leaves status 'running', reported honestly as-is)",
            ],
        },
    ]


def cmd_runs_overview(args: argparse.Namespace) -> int:
    emit_overview(
        "unsloth-cli runs",
        _runs_sections(),
        json_mode=bool(getattr(args, "json", False)),
    )
    return 0


# ---------------------------------------------------------------------------
# Subparser registration
# ---------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``runs`` subparser (noun with list/show/overview verbs) on *sub*."""
    p = sub.add_parser(
        "runs",
        help="Enumerate and inspect past training runs (see 'sloth runs overview').",
        description="Read the run registry (<runs-root>/runs.jsonl) to list or show past runs.",
    )

    def _no_verb(_args: argparse.Namespace) -> int:
        # `sloth runs` with no sub-verb prints usage (mirrors `config`'s
        # no-sub-verb behaviour) rather than crashing on a missing `args.func`.
        p.print_help()
        return 0

    p.set_defaults(func=_no_verb, json=False)
    sub_runs = p.add_subparsers(dest="runs_command")

    # --- runs list -----------------------------------------------------------
    p_list = sub_runs.add_parser("list", help="Newest-first table of every registered run.")
    p_list.add_argument(
        "--runs-root",
        dest="runs_root",
        default=None,
        metavar="DIR",
        help="Directory containing runs.jsonl (default: the current directory).",
    )
    p_list.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p_list.set_defaults(func=cmd_runs_list)

    # --- runs show -------------------------------------------------------------
    p_show = sub_runs.add_parser("show", help="Show the full registry record for a run_id.")
    p_show.add_argument("run_id", help="The run_id to show (see 'sloth runs list').")
    p_show.add_argument(
        "--runs-root",
        dest="runs_root",
        default=None,
        metavar="DIR",
        help="Directory containing runs.jsonl (default: the current directory).",
    )
    p_show.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p_show.set_defaults(func=cmd_runs_show)

    # --- runs overview -----------------------------------------------------
    p_overview = sub_runs.add_parser(
        "overview", help="Describe the 'runs' noun (registry file shape)."
    )
    p_overview.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p_overview.set_defaults(func=cmd_runs_overview)
