"""``sloth summarize <run_id|output_dir>`` — one JSON summary of a past run.

Joins ``training_metadata.json`` with the final/best loss + step count from
the run's newest ``checkpoint-N/trainer_state.json`` (see
:mod:`sloth.tune.summary`). *target* may be either a ``run_id`` from the
registry (see :mod:`sloth.tune.registry`) or a literal output directory path —
an existing directory takes precedence when a string happens to be both.

This is a **global** verb (a sibling of ``train``/``eval``/``export``), not
nested under a noun.
"""

from __future__ import annotations

import argparse

from sloth.cli._output import emit_result
from sloth.tune.registry import resolve_target
from sloth.tune.summary import build_summary


def cmd_summarize(args: argparse.Namespace) -> int:
    """Handler for ``sloth summarize``.

    Resolves *target* to an output directory (a run_id or a literal path),
    then builds and emits the joined summary. Raises ``CliError(code=1)`` when
    *target* resolves to neither.
    """
    output_dir = resolve_target(args.target, args.runs_root)
    summary = build_summary(output_dir)

    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result(summary, json_mode=True)
        return 0

    lines = [f"output_dir: {summary['output_dir']}"]
    metadata = summary.get("metadata")
    if metadata:
        lines.append(f"model:      {metadata.get('model')}")
        lines.append(f"method:     {metadata.get('method')}")
        lines.append(f"dataset:    {metadata.get('dataset')}")
        lines.append("hyperparameters:")
        for key, value in (metadata.get("hyperparameters") or {}).items():
            lines.append(f"  {key}: {value}")
    training = summary.get("training")
    if training:
        lines.append(f"checkpoint:  {training.get('checkpoint')}")
        lines.append(f"final_step:  {training.get('final_step')}")
        lines.append(f"final_loss:  {training.get('final_loss')}")
        if training.get("best_metric") is not None:
            lines.append(f"best_metric: {training.get('best_metric')}")
    for note in summary.get("notes") or []:
        lines.append(f"note: {note}")
    emit_result("\n".join(lines), json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``summarize`` subparser on *sub*."""
    p = sub.add_parser(
        "summarize",
        help="Summarize a past run: training_metadata.json + trainer_state.json.",
        description=(
            "Join training_metadata.json with the newest checkpoint's "
            "trainer_state.json into one JSON summary. Accepts a run_id "
            "(from 'sloth runs list') or a literal output directory."
        ),
    )
    p.add_argument("target", help="A run_id (see 'sloth runs list') or an output directory.")
    p.add_argument(
        "--runs-root",
        dest="runs_root",
        default=None,
        metavar="DIR",
        help="Directory containing runs.jsonl, used to resolve a run_id (default: cwd).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_summarize)
