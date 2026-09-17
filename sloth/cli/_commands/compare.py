"""``sloth compare`` — side-by-side comparison of two runs, or of an adapter
against its base model.

Two modes:

``sloth compare <a> <b>``
    Resolves both *a* and *b* the same way ``sloth summarize`` does (a run_id
    or a literal output directory; see
    :func:`sloth.tune.registry.resolve_target`), builds a summary for each
    (:func:`sloth.tune.summary.build_summary`), and reports the
    config/hyperparameter keys that differ between the two
    ``training_metadata.json`` records alongside the two full summaries.

``sloth compare --base <hf-id-or-dir> <adapter-dir>``
    Scores the **base** model on every suite the adapter already has results
    for, then reports per-suite, per-metric deltas and checks them against the
    ``[eval.thresholds]`` gate. The two models are evaluated in **two separate,
    sequential container invocations** — the adapter first, the base second —
    so the two are never resident on the GPU at the same time. The base run's
    results land under ``<adapter-dir>/eval-base/<suite>.json`` (via ``sloth
    eval --results-dir``), leaving the adapter's own ``eval/`` untouched.

This is a **global** verb (a sibling of ``train``/``eval``/``export``), not
nested under a noun.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from sloth.cli._errors import EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_result
from sloth.tune import container
from sloth.tune.config import ThresholdsConfig, load_config
from sloth.tune.registry import resolve_target
from sloth.tune.summary import build_eval_summary, build_summary, read_eval

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


def _export_fingerprint(summary: dict[str, Any]) -> dict[str, Any]:
    """Reduce a run's ``exports`` list to a comparable fingerprint: how many
    exports it has, and which formats are present."""
    exports = summary.get("exports") or []
    formats = sorted({e.get("format") for e in exports if isinstance(e, dict) and e.get("format")})
    return {"count": len(exports), "formats": formats}


def _export_deltas(summary_a: dict[str, Any], summary_b: dict[str, Any]) -> dict[str, Any]:
    """Return ``{"exports": {"a": fingerprint, "b": fingerprint}}`` when the two
    runs' export presence/formats differ, else ``{}``."""
    fp_a = _export_fingerprint(summary_a)
    fp_b = _export_fingerprint(summary_b)
    same = fp_a["count"] == fp_b["count"] and fp_a["formats"] == fp_b["formats"]
    return {} if same else {"exports": {"a": fp_a, "b": fp_b}}


#: Aggregate eval fields compared between two runs' eval.json summaries.
_EVAL_KEYS = ("exact_match_pct", "f1")


def _is_number(value: Any) -> bool:
    """True for a real number — ``bool`` is an ``int`` in Python, and
    ``base_load_in_4bit`` is a flag, not a metric."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _suite_metric_deltas(
    suites_a: dict[str, Any], suites_b: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return ``{suite: {metric: {"a", "b", "delta"}}}`` over both sides' suites.

    Generic over metric names: every key either side records for a suite is
    reported, so a metric a scorer adds later (``compliance_pct``,
    ``median_latency_ms``, ``tokens_per_s``, ``perplexity``, …) shows up here
    without this function knowing it in advance. ``delta`` is ``b - a`` when
    both sides are numeric, else ``None`` (a flag such as
    ``base_load_in_4bit``, or a metric only one side recorded). Suites present
    on only one side are still reported, with ``None`` for the missing side.
    """
    deltas: dict[str, dict[str, dict[str, Any]]] = {}
    for suite in sorted(set(suites_a) | set(suites_b)):
        metrics_a = suites_a.get(suite) or {}
        metrics_b = suites_b.get(suite) or {}
        if not isinstance(metrics_a, dict) or not isinstance(metrics_b, dict):
            continue
        per_metric: dict[str, dict[str, Any]] = {}
        for key in sorted(set(metrics_a) | set(metrics_b)):
            va, vb = metrics_a.get(key), metrics_b.get(key)
            delta = vb - va if _is_number(va) and _is_number(vb) else None
            per_metric[key] = {"a": va, "b": vb, "delta": delta}
        deltas[suite] = per_metric
    return deltas


def _eval_deltas(summary_a: dict[str, Any], summary_b: dict[str, Any]) -> dict[str, Any]:
    """Return ``{"eval": {"a": {...}, "b": {...}, "suites": {...}}}`` when BOTH
    runs have eval results (i.e. ``summary["eval"]`` is not ``None`` on either
    side) AND anything about them differs.

    ``a``/``b`` keep the flat, backward-compatible ``_EVAL_KEYS`` projection;
    ``suites`` adds the per-suite, per-metric breakdown
    (:func:`_suite_metric_deltas`) so two runs that differ only inside one
    named suite, or only on a metric outside ``_EVAL_KEYS``, still report it.

    Read-only: never recomputes a metric, just re-shapes the numbers
    :func:`sloth.tune.summary.build_summary` already read. Returns ``{}`` when
    either side lacks an eval — nothing to compare, so nothing is shown,
    exactly like the rest of this module's delta helpers degrade silently — or
    when the two sides' metrics are identical throughout.
    """
    eval_a = summary_a.get("eval")
    eval_b = summary_b.get("eval")
    if eval_a is None or eval_b is None:
        return {}
    proj_a = {key: eval_a.get(key) for key in _EVAL_KEYS}
    proj_b = {key: eval_b.get(key) for key in _EVAL_KEYS}
    suite_deltas = _suite_metric_deltas(eval_a.get("suites") or {}, eval_b.get("suites") or {})
    differs = proj_a != proj_b or any(
        entry["a"] != entry["b"] for metrics in suite_deltas.values() for entry in metrics.values()
    )
    if not differs:
        return {}
    return {"eval": {"a": proj_a, "b": proj_b, "suites": suite_deltas}}


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


# ---------------------------------------------------------------------------
# --base mode: adapter vs. its base model
# ---------------------------------------------------------------------------

#: Directory (under the adapter) the base model's per-suite results are written
#: to, so they never collide with the adapter's own ``eval/``.
BASE_EVAL_DIR = "eval-base"

#: The accuracy metric a regression-tagged suite is gated on (percentage points).
_ACCURACY_KEY = "exact_match_pct"
#: Contract-compliance percentage, gated on the ADAPTER side only.
_COMPLIANCE_KEY = "compliance_pct"
#: Median per-item latency; gated as an adapter/base ratio.
_LATENCY_KEY = "median_latency_ms"
#: Row count of a suite.
_ROWS_KEY = "total"
#: The base model's load precision, which must match across both result sets.
_PRECISION_KEY = "base_load_in_4bit"

#: A suite named exactly this is regression-tagged without any flag.
_REGRESSION_SUITE = "regression"


def _repo_root() -> Path:
    """Return the unsloth-cli checkout root (bind-mounted into the container).

    ``sloth/cli/_commands/compare.py`` → ``parents[3]`` is the dir holding the
    ``sloth/`` package, exactly as :func:`sloth.cli._commands.eval._repo_root`
    resolves it for its own launches.
    """
    return Path(__file__).resolve().parents[3]


def _suite_files(payload: dict[str, Any]) -> list[str]:
    """Return the suite files a recorded eval payload was scored over.

    Tried in order: the per-file ``files[].path`` entries the seams record, the
    ``suite_paths`` list the legacy flat layout carries, then a bare ``path``
    (a single-file suite payload). Returns ``[]`` when the payload records no
    provenance at all — that suite cannot be re-run and is skipped.
    """
    files = payload.get("files")
    if isinstance(files, list):
        paths = [e["path"] for e in files if isinstance(e, dict) and isinstance(e.get("path"), str)]
        if paths:
            return paths
    suite_paths = payload.get("suite_paths")
    if isinstance(suite_paths, list):
        paths = [p for p in suite_paths if isinstance(p, str) and p]
        if paths:
            return paths
    path = payload.get("path")
    return [path] if isinstance(path, str) and path else []


def _adapter_suites(adapter_dir: Path) -> tuple[list[str], int | None]:
    """Return ``(suite files, batch size)`` describing what the adapter was scored on.

    The suites an adapter "has" are its recorded ``eval/*.json`` files; their
    provenance (:func:`_suite_files`) is what the base model is re-scored over,
    so both sides see exactly the same rows. The recorded ``batch_size`` is
    reused for both runs so the latency comparison is apples-to-apples.

    Raises ``CliError(code=1)`` when the adapter has no eval results, or when
    none of them record which files they were scored over.
    """
    suites = read_eval(adapter_dir)
    if not suites:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"no eval results found under {adapter_dir}",
            remediation=(
                f"Run `sloth eval --adapter {adapter_dir} --suite <suite.jsonl>` first: "
                "`compare --base` re-scores the base model on the suites the adapter "
                "already has results for."
            ),
        )

    files: list[str] = []
    batch_size: int | None = None
    for payload in suites.values():
        for path in _suite_files(payload):
            if path not in files:
                files.append(path)
        recorded = payload.get("batch_size")
        if batch_size is None and isinstance(recorded, int) and not isinstance(recorded, bool):
            batch_size = recorded
    if not files:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"the eval results under {adapter_dir} record no suite file paths",
            remediation=(
                f"Re-run `sloth eval --adapter {adapter_dir} --suite <suite.jsonl>` so the "
                "result files record which suites they were scored over, then compare again."
            ),
        )
    return files, batch_size


def _launch_eval(
    flag: str,
    reference: str,
    suite_files: list[str],
    *,
    results_dir: Path | None,
    batch_size: int | None,
    workdir: Path,
) -> dict[str, Any]:
    """Run one ``sloth eval`` inside the NGC container and return its result dict.

    One invocation, one model: the caller runs these strictly sequentially, so
    the base and the adapter are never resident on the GPU at the same time.
    ``--in-container`` is appended exactly as
    :func:`sloth.cli._commands.eval._launch_container` appends it, so the
    container never recurses into another docker launch. Identity mounts are
    added for every existing local path involved (a Hugging Face repo id has
    none). Raises :class:`CliError` on any container failure — including the
    ``code=2`` out-of-memory mapping, which propagates untouched.
    """
    sloth_args = ["eval", flag, reference]
    for path in suite_files:
        sloth_args += ["--suite", str(Path(path).resolve())]
    if batch_size is not None:
        sloth_args += ["--batch-size", str(batch_size)]
    if results_dir is not None:
        sloth_args += ["--results-dir", str(results_dir.resolve())]
    sloth_args += ["--json", "--in-container"]

    mount_dirs = {Path(p).resolve().parent for p in suite_files}
    reference_path = Path(reference)
    if reference_path.exists():
        mount_dirs.add(reference_path.resolve().parent)
    if results_dir is not None:
        mount_dirs.add(results_dir.resolve())
    return container.launch(
        sloth_args,
        workdir=str(workdir),
        checkout=str(_repo_root()),
        extra_mounts=[(str(p), str(p)) for p in sorted(mount_dirs)],
    )


def _resolve_thresholds(config_path: str | None) -> tuple[ThresholdsConfig, str]:
    """Return ``(thresholds, source)`` for the gate.

    With ``--config`` the run config's ``[eval.thresholds]`` section wins (a
    config without that section still yields the dataclass defaults). Without
    it, the c36 baseline defaults apply — ``ThresholdsConfig()`` is the single
    place those numbers live.
    """
    if not config_path:
        return ThresholdsConfig(), "default"
    return (load_config(config_path).thresholds or ThresholdsConfig()), str(config_path)


def _check_thresholds(
    suites_a: dict[str, Any],
    suites_b: dict[str, Any],
    thresholds: ThresholdsConfig,
    regression_suites: set[str],
) -> list[dict[str, Any]]:
    """Return one entry per failed threshold check (empty = the gate passes).

    Four gates plus the precision guard, each skipped for a suite that does not
    record the metric it needs (a missing number is never read as a failure):

    * ``regression`` — a regression-tagged suite whose accuracy drops by more
      than ``regression_drop_pp`` percentage points;
    * ``compliance`` — the ADAPTER's ``compliance_pct`` below
      ``compliance_min_pct``;
    * ``latency`` — adapter/base median latency ratio above
      ``latency_max_ratio``;
    * ``min_suite_rows`` — either side's suite holding fewer than
      ``min_suite_rows`` rows;
    * ``precision`` — ``base_load_in_4bit`` recorded differently by the two
      result sets, which would make every other number incomparable. Only
      flagged when BOTH sides state it: ``None`` means "not stated", not
      "differs".

    Only suites **both** sides recorded are gated: a suite one side does not
    have is not a comparison, so it is reported in the deltas (with ``None``
    for the missing side) but never turned into a threshold failure. That also
    keeps a pre-suite-keyed ``eval.json`` — which
    :func:`sloth.tune.summary.read_eval` surfaces as the pseudo-suite
    ``legacy`` — from failing a gate it has no base-side counterpart for.
    """
    failures: list[dict[str, Any]] = []
    for suite in sorted(set(suites_a) & set(suites_b)):
        a = suites_a.get(suite) or {}
        b = suites_b.get(suite) or {}

        acc_a, acc_b = a.get(_ACCURACY_KEY), b.get(_ACCURACY_KEY)
        if suite in regression_suites and _is_number(acc_a) and _is_number(acc_b):
            drop = acc_a - acc_b
            if drop > thresholds.regression_drop_pp:
                failures.append(
                    {
                        "check": "regression",
                        "suite": suite,
                        "message": (
                            f"{suite}: {_ACCURACY_KEY} dropped {drop:.2f} pp "
                            f"({acc_a} -> {acc_b}), over the "
                            f"{thresholds.regression_drop_pp} pp allowance"
                        ),
                    }
                )

        compliance = b.get(_COMPLIANCE_KEY)
        if _is_number(compliance) and compliance < thresholds.compliance_min_pct:
            failures.append(
                {
                    "check": "compliance",
                    "suite": suite,
                    "message": (
                        f"{suite}: adapter {_COMPLIANCE_KEY} {compliance} is below the "
                        f"{thresholds.compliance_min_pct} minimum"
                    ),
                }
            )

        lat_a, lat_b = a.get(_LATENCY_KEY), b.get(_LATENCY_KEY)
        if _is_number(lat_a) and _is_number(lat_b) and lat_a > 0:
            ratio = lat_b / lat_a
            if ratio > thresholds.latency_max_ratio:
                failures.append(
                    {
                        "check": "latency",
                        "suite": suite,
                        "message": (
                            f"{suite}: median latency ratio {ratio:.3f} "
                            f"({lat_b} ms vs {lat_a} ms) is over the "
                            f"{thresholds.latency_max_ratio} maximum"
                        ),
                    }
                )

        for side, metrics in (("base", a), ("adapter", b)):
            rows = metrics.get(_ROWS_KEY)
            if _is_number(rows) and rows < thresholds.min_suite_rows:
                failures.append(
                    {
                        "check": "min_suite_rows",
                        "suite": suite,
                        "message": (
                            f"{suite}: the {side} result holds {rows} row(s), under the "
                            f"{thresholds.min_suite_rows}-row minimum"
                        ),
                    }
                )

        prec_a, prec_b = a.get(_PRECISION_KEY), b.get(_PRECISION_KEY)
        if prec_a is not None and prec_b is not None and prec_a != prec_b:
            failures.append(
                {
                    "check": "precision",
                    "suite": suite,
                    "message": (
                        f"{suite}: {_PRECISION_KEY} differs between the two result sets "
                        f"(base={prec_a}, adapter={prec_b})"
                    ),
                }
            )
    return failures


def _render_base_text(report: dict[str, Any]) -> str:
    """Render the ``--base`` report for stdout in text mode."""
    lines = [
        f"base:    {report['a'].get('model')}",
        f"adapter: {report['b'].get('output_dir')}",
        "",
    ]
    for suite, per_metric in report["deltas"]["suites"].items():
        lines.append(f"suite {suite}:")
        for metric, values in per_metric.items():
            delta = values["delta"]
            shown = "" if delta is None else f" delta={delta:+g}"
            lines.append(f"  {metric}: base={values['a']!r} adapter={values['b']!r}{shown}")
    applied = report["thresholds"]
    lines.append("")
    lines.append(
        "thresholds ({source}): regression_drop_pp={regression_drop_pp} "
        "compliance_min_pct={compliance_min_pct} latency_max_ratio={latency_max_ratio} "
        "min_suite_rows={min_suite_rows}".format(**applied)
    )
    verdict = report["verdict"]
    lines.append(f"verdict: {'passed' if verdict['passed'] else 'FAILED'}")
    for failure in verdict["failures"]:
        lines.append(f"  - {failure['check']}: {failure['message']}")
    return "\n".join(lines)


def _cmd_compare_base(args: argparse.Namespace) -> int:
    """Handler for ``sloth compare --base <hf-id-or-dir> <adapter-dir>``.

    Two sequential container invocations — the adapter, then the base — then a
    per-suite, per-metric delta report gated on ``[eval.thresholds]``. The
    report is emitted on stdout **before** any threshold failure is raised, so
    ``--json`` consumers get the numbers and the applied thresholds even on the
    exit-1 path.
    """
    if getattr(args, "b", None):
        raise CliError(
            code=EXIT_USER_ERROR,
            message="--base takes a single positional adapter directory, not two targets",
            remediation=(
                "Use `sloth compare --base <hf-id-or-dir> <adapter-dir>` to compare an "
                "adapter against its base model, or `sloth compare <a> <b>` (no --base) "
                "to compare two runs."
            ),
        )

    adapter_dir = resolve_target(args.a, args.runs_root)
    thresholds, source = _resolve_thresholds(getattr(args, "config", None))
    suite_files, batch_size = _adapter_suites(adapter_dir)
    base_dir = adapter_dir / BASE_EVAL_DIR

    # Sequential, one model at a time: the adapter is re-scored first (its
    # results stay under <adapter>/eval/), then the base (under eval-base/).
    # A failure on the second run leaves the first run's results on disk.
    _launch_eval(
        "--adapter",
        str(adapter_dir.resolve()),
        suite_files,
        results_dir=None,
        batch_size=batch_size,
        workdir=adapter_dir.resolve(),
    )
    _launch_eval(
        "--model",
        args.base,
        suite_files,
        results_dir=base_dir,
        batch_size=batch_size,
        workdir=adapter_dir.resolve(),
    )

    eval_b = build_eval_summary(adapter_dir) or {}
    eval_a = build_eval_summary(adapter_dir, subdir=BASE_EVAL_DIR) or {}
    summary_a = {"model": args.base, "output_dir": str(base_dir), "eval": eval_a or None}
    summary_b = build_summary(adapter_dir)

    suites_a = eval_a.get("suites") or {}
    suites_b = eval_b.get("suites") or {}
    regression_suites = {_REGRESSION_SUITE, *(getattr(args, "regression_suite", None) or [])}
    failures = _check_thresholds(suites_a, suites_b, thresholds, regression_suites)

    applied = dict(asdict(thresholds))
    applied["source"] = source
    applied["regression_suites"] = sorted(regression_suites & (set(suites_a) | set(suites_b)))
    report: dict[str, Any] = {
        "a": summary_a,
        "b": summary_b,
        "deltas": {"suites": _suite_metric_deltas(suites_a, suites_b)},
        "thresholds": applied,
        "verdict": {"passed": not failures, "failures": failures},
    }

    json_mode = bool(getattr(args, "json", False))
    emit_result(report if json_mode else _render_base_text(report), json_mode=json_mode)

    if failures:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"{len(failures)} threshold check(s) failed: "
                + ", ".join(sorted({f["check"] for f in failures}))
            ),
            remediation=(
                "Failed checks: "
                + "; ".join(f["message"] for f in failures)
                + ". Retrain, widen the gate in [eval.thresholds] of your run config, "
                "or pass a --config whose thresholds match this comparison."
            ),
        )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Handler for ``sloth compare``.

    With ``--base``, delegates to :func:`_cmd_compare_base` (adapter vs. its
    base model). Otherwise resolves both targets (a run_id or a literal output
    directory each), builds a summary for each, and reports the config deltas
    plus both summaries. Raises ``CliError(code=1)`` when a target is
    unresolvable, and when the second positional is missing without ``--base``.
    """
    if getattr(args, "base", None):
        return _cmd_compare_base(args)
    if not getattr(args, "b", None):
        raise CliError(
            code=EXIT_USER_ERROR,
            message="compare needs two targets: <a> <b>",
            remediation=(
                "Pass a second run_id or output directory: `sloth compare <a> <b>`. "
                "To compare an adapter against its base model instead, use "
                "`sloth compare --base <hf-id-or-dir> <adapter-dir>`."
            ),
        )

    dir_a = resolve_target(args.a, args.runs_root)
    dir_b = resolve_target(args.b, args.runs_root)

    summary_a = build_summary(dir_a)
    summary_b = build_summary(dir_b)
    deltas = _config_deltas(summary_a.get("metadata"), summary_b.get("metadata"))
    deltas.update(_export_deltas(summary_a, summary_b))
    deltas.update(_eval_deltas(summary_a, summary_b))

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
        help="Compare two past runs, or an adapter against its base model (--base).",
        description=(
            "Compare two runs: config/hyperparameter deltas plus each run's "
            "'sloth summarize' summary. Each of <a>/<b> is a run_id or output directory. "
            "With --base <hf-id-or-dir>, compare a single adapter against its base model "
            "instead: per-suite, per-metric deltas gated on [eval.thresholds]."
        ),
    )
    p.add_argument("a", help="First run: a run_id or an output directory.")
    p.add_argument(
        "b",
        nargs="?",
        default=None,
        help=(
            "Second run: a run_id or an output directory. Required without --base; "
            "omitted with it (the single positional is then the adapter)."
        ),
    )
    p.add_argument(
        "--base",
        default=None,
        metavar="REF",
        help=(
            "Compare <a> (an adapter directory) against this base model — a Hugging "
            "Face repo id or a local model directory — on every suite the adapter "
            "already has results for. Two sequential container runs; the base "
            "model's results land under <adapter>/eval-base/."
        ),
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "Run config (TOML) whose [eval.thresholds] section gates the --base "
            "comparison (default: the built-in baseline thresholds)."
        ),
    )
    p.add_argument(
        "--regression-suite",
        dest="regression_suite",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Treat this suite as regression-tagged for the --base gate (repeatable). "
            "A suite named 'regression' is tagged without any flag."
        ),
    )
    p.add_argument(
        "--runs-root",
        dest="runs_root",
        default=None,
        metavar="DIR",
        help="Directory containing runs.jsonl, used to resolve a run_id (default: cwd).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_compare)
