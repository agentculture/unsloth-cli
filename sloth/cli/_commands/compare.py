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
    so the two are never resident on the GPU at the same time. Neither run
    touches the adapter's own ``eval/``: it is the record the suite list comes
    from. Both write through ``sloth eval --results-dir`` into a sibling
    directory of their own — the adapter's re-eval to
    ``<adapter-dir>/eval-compare/<suite>.json``, the base's to
    ``<adapter-dir>/eval-base/<suite>.json`` — each emptied of stale ``*.json``
    immediately before its run.

This is a **global** verb (a sibling of ``train``/``eval``/``export``), not
nested under a noun.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from sloth.cli._errors import EXIT_USER_ERROR, CliError
from sloth.cli._output import emit_result
from sloth.tune import container
from sloth.tune.config import ThresholdsConfig, load_config
from sloth.tune.metrics import EVAL_JSON_DIR
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


def _dataset_identity(dataset: dict[str, Any] | None) -> Any:
    """Return a normalized identity for a metadata ``dataset`` record.

    Two runs' datasets are "the same" when their identities compare equal:

    * a **local** file is identified by its ``sha256`` — the digest
      :func:`sloth.tune.metadata.write_metadata` embedded;
    * a **hub** dataset (``dataset.hf_id`` present, or ``dataset.source ==
      "hf"``) carries no digest at all, so it is identified by the
      ``(hf_id, split, revision)`` tuple that determines which rows
      ``load_dataset`` returns. Comparing ``sha256`` alone read every pair of
      hub datasets as identical (``None == None``), so a run that switched
      split or revision reported no dataset delta.

    Anything else (a record with neither) falls back to the record itself, so
    two differing shapes still compare unequal rather than silently matching.
    """
    ds = dataset or {}
    # Always a (kind, value) pair so every branch returns the same shape.
    if ds.get("hf_id") or ds.get("source") == "hf":
        return ("hf", (ds.get("hf_id"), ds.get("split"), ds.get("revision")))
    if ds.get("sha256") is not None:
        return ("sha256", ds.get("sha256"))
    return ("raw", tuple(sorted((str(k), str(v)) for k, v in ds.items())))


def _config_deltas(meta_a: dict[str, Any] | None, meta_b: dict[str, Any] | None) -> dict[str, Any]:
    """Return ``{key: {"a": val, "b": val}}`` for every top-level, dataset, or
    hyperparameter key that differs between the two metadata dicts.

    A key absent on one side compares against ``None``. Missing metadata on
    either side (``None``) is treated as an empty record, so a delta report is
    still produced (naming what *is* known) rather than raised as an error.
    The ``dataset`` key is compared on a normalized identity
    (:func:`_dataset_identity`) so a changed hub split/revision is reported
    just like a changed local-file digest.
    """
    a_top = meta_a or {}
    b_top = meta_b or {}
    deltas = _dict_key_deltas(a_top, b_top, _METADATA_TOP_KEYS)

    a_hp = (meta_a or {}).get("hyperparameters") or {}
    b_hp = (meta_b or {}).get("hyperparameters") or {}
    deltas.update(_dict_key_deltas(a_hp, b_hp, sorted(set(a_hp) | set(b_hp))))

    a_ds = (meta_a or {}).get("dataset") or {}
    b_ds = (meta_b or {}).get("dataset") or {}
    if _dataset_identity(a_ds) != _dataset_identity(b_ds):
        deltas["dataset"] = {"a": a_ds, "b": b_ds}

    return deltas


# ---------------------------------------------------------------------------
# --base mode: adapter vs. its base model
# ---------------------------------------------------------------------------

#: Directory (under the adapter) the base model's per-suite results are written
#: to, so they never collide with the adapter's own ``eval/``.
BASE_EVAL_DIR = "eval-base"

#: Directory (under the adapter) this comparison's ADAPTER re-eval is written
#: to. The adapter's own ``eval/`` is the suite list the comparison is built
#: from — overwriting it with the re-run would destroy the record the next
#: comparison reads (and lose results scored with other settings).
ADAPTER_EVAL_DIR = "eval-compare"

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


def _resolve_recorded_suite(raw: str, adapter_dir: Path) -> str:
    """Return the absolute path of a suite path recorded in an eval result file.

    New result files record canonical absolute paths (see
    :func:`sloth.tune.metrics.canonicalize_result_paths`), which resolve to
    themselves. A **legacy** file may hold a path relative to whatever directory
    the eval was run from; it is looked for next to the result file
    (``<adapter>/eval/``), then under the adapter directory, then against the
    current working directory.

    Raises ``CliError(code=1)`` when nothing exists at any candidate: mounting a
    nonexistent path into the container would fail obscurely an hour later,
    whereas re-running ``sloth eval`` re-records the path canonically.
    """
    path = Path(raw)
    candidates = (
        [path]
        if path.is_absolute()
        else [adapter_dir / EVAL_JSON_DIR / raw, adapter_dir / raw, Path.cwd() / raw]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    raise CliError(
        code=EXIT_USER_ERROR,
        message=f"a recorded eval suite file no longer exists: {raw}",
        remediation=(
            f"The eval results under {adapter_dir} name a suite file that cannot be found "
            f"(looked in {', '.join(str(c) for c in candidates)}). Re-run "
            f"`sloth eval --adapter {adapter_dir} --suite <suite.jsonl>` so the result "
            "files record the suite paths as they exist now, then compare again."
        ),
    )


def _adapter_suites(adapter_dir: Path) -> tuple[list[str], int | None]:
    """Return ``(suite files, batch size)`` describing what the adapter was scored on.

    The suites an adapter "has" are its recorded ``eval/*.json`` files; their
    provenance (:func:`_suite_files`) is what the base model is re-scored over,
    so both sides see exactly the same rows. The recorded ``batch_size`` is
    reused for both runs so the latency comparison is apples-to-apples.

    Every recorded path is resolved to an absolute, existing file
    (:func:`_resolve_recorded_suite`) before it is handed to a container mount.

    Raises ``CliError(code=1)`` when the adapter has no eval results, when none
    of them record which files they were scored over, or when a recorded file
    no longer exists.
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
        for raw in _suite_files(payload):
            path = _resolve_recorded_suite(raw, adapter_dir)
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


def _clear_results_dir(results_dir: Path) -> None:
    """Remove every ``*.json`` under *results_dir* before a run writes into it.

    The report must describe **this** invocation only. A suite that was scored
    by an earlier comparison (different suites, different base model, different
    batch size) and is not re-scored now would otherwise still be read back and
    silently gated against the fresh other side. Unremovable files are left
    alone — the run that follows overwrites what it rewrites anyway.
    """
    for stale in results_dir.glob("*.json"):
        try:
            stale.unlink()
        except OSError:  # pragma: no cover - defensive; the writability check precedes this
            pass


def _ensure_writable_results_dir(base_dir: Path) -> None:
    """Create a results directory AS THE HOST USER and require it writable.

    Docker creates a bind-mounted host path that does not exist yet as *root*,
    and the container runs as the host uid, so its writer would get EACCES on
    every suite and the base side would come back empty (live-measured
    2026-09-17, plan risk r15). A leftover root-owned directory from such a run
    is the same trap, so this runs *before* the adapter re-eval: failing here
    costs seconds, failing after the base run costs the whole hour.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(base_dir, os.W_OK):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"base results directory is not writable: {base_dir}",
            remediation=(
                "It was probably created as root by an earlier container run. Remove it "
                f"(`rmdir {base_dir}` works when it is empty) or chown it to your user, "
                "then re-run."
            ),
        )


def _resolve_base_reference(base: str) -> str:
    """Return what ``--base`` should be forwarded to the container as.

    A **local** path is resolved to an absolute one: the container runs with a
    different working directory than the host, so a relative ``--base ./base``
    would resolve inside the container to something else (or to nothing), and
    the identity mount is computed from the same path. A **Hugging Face repo
    id** (nothing local answers to that name) is passed through verbatim —
    resolving it would turn ``org/name`` into a nonexistent local path.
    """
    path = Path(base)
    return str(path.resolve()) if path.exists() else base


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
        candidates: list[dict[str, Any] | None] = [
            _regression_failure(suite, a, b, thresholds, regression_suites),
            _compliance_failure(suite, b, thresholds),
            _latency_failure(suite, a, b, thresholds),
            *_min_rows_failures(suite, a, b, thresholds),
            _precision_failure(suite, a, b, thresholds),
        ]
        failures.extend(entry for entry in candidates if entry is not None)
    return failures


def _regression_failure(
    suite: str,
    a: dict[str, Any],
    b: dict[str, Any],
    thresholds: ThresholdsConfig,
    regression_suites: set[str],
) -> dict[str, Any] | None:
    """The ``regression`` gate: a tagged suite's accuracy dropping too far."""
    acc_a, acc_b = a.get(_ACCURACY_KEY), b.get(_ACCURACY_KEY)
    if not (suite in regression_suites and _is_number(acc_a) and _is_number(acc_b)):
        return None
    drop = acc_a - acc_b
    if drop <= thresholds.regression_drop_pp:
        return None
    return {
        "check": "regression",
        "suite": suite,
        "message": (
            f"{suite}: {_ACCURACY_KEY} dropped {drop:.2f} pp "
            f"({acc_a} -> {acc_b}), over the "
            f"{thresholds.regression_drop_pp} pp allowance"
        ),
    }


def _compliance_failure(
    suite: str, b: dict[str, Any], thresholds: ThresholdsConfig
) -> dict[str, Any] | None:
    """The ``compliance`` gate: the adapter's ``compliance_pct`` under the floor."""
    compliance = b.get(_COMPLIANCE_KEY)
    if not (_is_number(compliance) and compliance < thresholds.compliance_min_pct):
        return None
    return {
        "check": "compliance",
        "suite": suite,
        "message": (
            f"{suite}: adapter {_COMPLIANCE_KEY} {compliance} is below the "
            f"{thresholds.compliance_min_pct} minimum"
        ),
    }


def _latency_failure(
    suite: str, a: dict[str, Any], b: dict[str, Any], thresholds: ThresholdsConfig
) -> dict[str, Any] | None:
    """The ``latency`` gate: the adapter/base median-latency ratio over the cap."""
    lat_a, lat_b = a.get(_LATENCY_KEY), b.get(_LATENCY_KEY)
    if not (_is_number(lat_a) and _is_number(lat_b) and lat_a > 0):
        return None
    ratio = lat_b / lat_a
    if ratio <= thresholds.latency_max_ratio:
        return None
    return {
        "check": "latency",
        "suite": suite,
        "message": (
            f"{suite}: median latency ratio {ratio:.3f} "
            f"({lat_b} ms vs {lat_a} ms) is over the "
            f"{thresholds.latency_max_ratio} maximum"
        ),
    }


def _min_rows_failures(
    suite: str, a: dict[str, Any], b: dict[str, Any], thresholds: ThresholdsConfig
) -> list[dict[str, Any]]:
    """The ``min_suite_rows`` gate, checked on both sides (base first)."""
    entries: list[dict[str, Any]] = []
    for side, side_metrics in (("base", a), ("adapter", b)):
        rows = side_metrics.get(_ROWS_KEY)
        if _is_number(rows) and rows < thresholds.min_suite_rows:
            entries.append(
                {
                    "check": "min_suite_rows",
                    "suite": suite,
                    "message": (
                        f"{suite}: the {side} result holds {rows} row(s), under the "
                        f"{thresholds.min_suite_rows}-row minimum"
                    ),
                }
            )
    return entries


def _precision_failure(
    suite: str, a: dict[str, Any], b: dict[str, Any], thresholds: ThresholdsConfig
) -> dict[str, Any] | None:
    """The ``precision`` guard: both sides stating a different ``base_load_in_4bit``."""
    del thresholds  # uniform gate signature; this guard reads no threshold number
    prec_a, prec_b = a.get(_PRECISION_KEY), b.get(_PRECISION_KEY)
    if prec_a is None or prec_b is None or prec_a == prec_b:
        return None
    return {
        "check": "precision",
        "suite": suite,
        "message": (
            f"{suite}: {_PRECISION_KEY} differs between the two result sets "
            f"(base={prec_a}, adapter={prec_b})"
        ),
    }


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

    Two sequential container invocations — the adapter (into ``eval-compare/``),
    then the base (into ``eval-base/``), leaving ``eval/`` untouched — then a
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
    base_ref = _resolve_base_reference(args.base)
    suite_files, batch_size = _adapter_suites(adapter_dir)
    base_dir = adapter_dir / BASE_EVAL_DIR
    compare_dir = adapter_dir / ADAPTER_EVAL_DIR
    # base_dir first: a leftover root-owned eval-base/ is the r15 trap, and its
    # message is the one that names the directory to remove.
    _ensure_writable_results_dir(base_dir)
    _ensure_writable_results_dir(compare_dir)

    # Sequential, one model at a time: the adapter is re-scored first (into
    # eval-compare/, leaving the eval/ record this comparison's suite list came
    # from untouched), then the base (into eval-base/). Each directory is
    # emptied of *.json immediately before its run, so only this invocation's
    # artifacts are read back. A failure on the second run leaves the first
    # run's results on disk.
    _clear_results_dir(compare_dir)
    _launch_eval(
        "--adapter",
        str(adapter_dir.resolve()),
        suite_files,
        results_dir=compare_dir,
        batch_size=batch_size,
        workdir=adapter_dir.resolve(),
    )
    # Created as the host user (and checked writable) before the adapter run —
    # see _ensure_writable_results_dir.
    _clear_results_dir(base_dir)
    _launch_eval(
        "--model",
        base_ref,
        suite_files,
        results_dir=base_dir,
        batch_size=batch_size,
        workdir=adapter_dir.resolve(),
    )

    eval_b = build_eval_summary(adapter_dir, subdir=ADAPTER_EVAL_DIR) or {}
    eval_a = build_eval_summary(adapter_dir, subdir=BASE_EVAL_DIR) or {}
    if not (eval_a.get("suites") or {}):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"the base evaluation of {base_ref} produced no results under {base_dir}",
            remediation=(
                "The second container run wrote nothing readable (check its stderr above "
                "for 'could not write' notes or a load error); a compare with no base side "
                "cannot pass. Re-run, or evaluate the base with "
                f"`sloth eval --model {base_ref} --results-dir {base_dir} ...` first."
            ),
        )
    summary_a = {"model": base_ref, "output_dir": str(base_dir), "eval": eval_a or None}
    summary_b = build_summary(adapter_dir)
    # The adapter side of the report is THIS comparison's re-eval, not whatever
    # <adapter>/eval/ happens to hold from an older run.
    summary_b["eval"] = eval_b or None

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
            "already has results for. Two sequential container runs; the adapter's "
            "re-eval lands under <adapter>/eval-compare/ and the base model's under "
            "<adapter>/eval-base/, leaving <adapter>/eval/ untouched."
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
