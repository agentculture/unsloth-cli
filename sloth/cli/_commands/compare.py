"""``sloth compare <a> <b>`` — side-by-side comparison of two past runs.

Resolves both *a* and *b* the same way ``sloth summarize`` does (a run_id or a
literal output directory; see :func:`sloth.tune.registry.resolve_target`),
builds a summary for each (:func:`sloth.tune.summary.build_summary`), and
reports the config/hyperparameter keys that differ between the two
``training_metadata.json`` records alongside the two full summaries.

This is a **global** verb (a sibling of ``train``/``eval``/``export``), not
nested under a noun.
"""

from __future__ import annotations

import argparse
from typing import Any, Iterable

from sloth.cli._output import emit_result
from sloth.tune.registry import resolve_target
from sloth.tune.summary import build_summary

#: Top-level metadata keys (outside hyperparameters/dataset) compared directly.
_METADATA_TOP_KEYS = ("model", "method")


def _dict_key_deltas(a: dict[str, Any], b: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """Return ``{key: {"a": val, "b": val}}`` for every *key* whose value in *a*
    differs from its value in *b*. A key absent from either dict compares
    against ``None``.
    """
    deltas: dict[str, Any] = {}
    for key in keys:
        va, vb = a.get(key), b.get(key)
        if va != vb:
            deltas[key] = {"a": va, "b": vb}
    return deltas


def _config_deltas(meta_a: dict[str, Any] | None, meta_b: dict[str, Any] | None) -> dict[str, Any]:
    """Return ``{key: {"a": val, "b": val}}`` for every top-level, dataset, or
    hyperparameter key that differs between the two metadata dicts.

    A key absent on one side compares against ``None``. Missing metadata on
    either side (``None``) is treated as an empty record, so a delta report is
    still produced (naming what *is* known) rather than raised as an error.
    """
    a_top = meta_a or {}
    b_top = meta_b or {}
    deltas = _dict_key_deltas(a_top, b_top, _METADATA_TOP_KEYS)

    a_hp = (meta_a or {}).get("hyperparameters") or {}
    b_hp = (meta_b or {}).get("hyperparameters") or {}
    deltas.update(_dict_key_deltas(a_hp, b_hp, sorted(set(a_hp) | set(b_hp))))

    a_ds = (meta_a or {}).get("dataset") or {}
    b_ds = (meta_b or {}).get("dataset") or {}
    if a_ds.get("sha256") != b_ds.get("sha256"):
        deltas["dataset"] = {"a": a_ds, "b": b_ds}

    return deltas


def cmd_compare(args: argparse.Namespace) -> int:
    """Handler for ``sloth compare <a> <b>``.

    Resolves both targets (a run_id or a literal output directory each),
    builds a summary for each, and reports the config deltas plus both
    summaries. Raises ``CliError(code=1)`` when either target is unresolvable.
    """
    dir_a = resolve_target(args.a, args.runs_root)
    dir_b = resolve_target(args.b, args.runs_root)

    summary_a = build_summary(dir_a)
    summary_b = build_summary(dir_b)
    deltas = _config_deltas(summary_a.get("metadata"), summary_b.get("metadata"))

    report: dict[str, Any] = {"a": summary_a, "b": summary_b, "deltas": deltas}

    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result(report, json_mode=True)
    else:
        lines = [f"a: {summary_a['output_dir']}", f"b: {summary_b['output_dir']}", "", "deltas:"]
        if deltas:
            for key, vals in deltas.items():
                lines.append(f"  {key}: a={vals['a']!r} b={vals['b']!r}")
        else:
            lines.append("  (none — configs match)")
        emit_result("\n".join(lines), json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``compare`` subparser on *sub*."""
    p = sub.add_parser(
        "compare",
        help="Side-by-side comparison of two past runs (config deltas + summaries).",
        description=(
            "Compare two runs: config/hyperparameter deltas plus each run's "
            "'sloth summarize' summary. Each of <a>/<b> is a run_id or output directory."
        ),
    )
    p.add_argument("a", help="First run: a run_id or an output directory.")
    p.add_argument("b", help="Second run: a run_id or an output directory.")
    p.add_argument(
        "--runs-root",
        dest="runs_root",
        default=None,
        metavar="DIR",
        help="Directory containing runs.jsonl, used to resolve a run_id (default: cwd).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_compare)
