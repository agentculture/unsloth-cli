"""Pure-stdlib scoring for ``sloth eval`` — exact match, token-F1, and ``eval.json``.

This module is **pure stdlib**: no torch, no transformers, no third-party import
of any kind (``tests/test_lazy_import.py`` and
``tests/test_packaging_import_light.py`` both assert it). Scoring is therefore
importable and testable on a machine with no ML stack at all, exactly like
:mod:`sloth.tune.datasets` / :mod:`sloth.tune.config` / :mod:`sloth.tune.scope`.

Two consumers share it — :func:`sloth.tune._trainer.run_eval` (``--adapter``) and
:func:`sloth.tune._exporter.run_eval_model` (``--model``) — so both report the
*same* shape:

``{total, exact_match, exact_match_pct, f1, results, files}``

where ``files`` carries one entry per scored suite file
(``{path, total, exact_match, exact_match_pct, f1, results}``) and the top-level
fields are the **aggregate** across every file.

Metrics
-------
* **exact match** — ``prediction.strip() == expected_output.strip()`` (unchanged
  from the original inline comparison, so existing scores stay comparable).
* **token F1** — SQuAD-style bag-of-tokens F1 over lowercased ``\\w+`` tokens,
  counted as a *multiset* intersection so repeated tokens are not over-credited.
  ``token_f1("a b c", "a b d") == 2/3`` (``0.667`` to three decimals).

The per-row dict returned by :func:`score_records` is deliberately **open**:
callers may pass ``extra_metrics`` (a callable or a ``{name: callable}`` map) to
add more numeric metric keys per row without this module knowing their names in
advance. :func:`summarize` / :func:`aggregate` fold *any* numeric metric key
found on the rows into the summary as its mean — ``exact_match_pct`` and ``f1``
are kept as named fields for backward compatibility, but a brand-new metric
name registered only in a test or a future scorer shows up in the aggregate
automatically.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Union

#: Filename written into the evaluated directory by the *legacy* call shape of
#: :func:`write_eval_json`. Kept exported verbatim for readers of the old
#: single-file layout (see ``t5`` for the legacy-read side of this contract).
EVAL_JSON_NAME = "eval.json"

#: Directory (relative to the evaluated target) holding the newer, suite-keyed
#: result files written by the new call shape of :func:`write_eval_json`.
EVAL_JSON_DIR = "eval"

#: Schema version stamped onto every suite-keyed result file.
SCHEMA_VERSION = 2

#: Decimal places for the reported F1 figures (percentages keep the legacy 2).
_F1_PLACES = 4

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

#: Row keys that are never treated as foldable numeric metrics.
_NON_METRIC_KEYS = frozenset(
    {"index", "task", "input", "expected_output", "prediction", "exact_match", "file"}
)

#: Either a single ``(record, prediction) -> {name: value}`` callable, or a
#: ``{name: (record, prediction) -> value}`` mapping of per-metric callables.
ExtraMetrics = Union[
    Callable[[dict[str, Any], str], Mapping[str, Any]],
    Mapping[str, Callable[[dict[str, Any], str], Any]],
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Split *text* into lowercased word tokens (``\\w+``), ignoring punctuation."""
    return _TOKEN_RE.findall((text or "").lower())


def token_f1(prediction: str, reference: str) -> float:
    """Return the SQuAD-style token F1 of *prediction* against *reference*.

    Both sides are tokenized with :func:`tokenize` and compared as **multisets**
    (``Counter`` intersection), so a prediction that repeats a token five times
    gets credit only for the number of times the reference contains it.

    Conventions at the edges: two empty token lists score ``1.0`` (a blank
    expectation matched by a blank prediction is a success), and exactly one
    empty side scores ``0.0``. The value is **unrounded** — callers round for
    display (see :func:`summarize`).
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Per-record scoring + aggregation
# ---------------------------------------------------------------------------


def _resolve_extra_metrics(
    extra_metrics: ExtraMetrics, record: dict[str, Any], prediction: str
) -> dict[str, Any]:
    """Normalize *extra_metrics* (a callable or a ``{name: callable}`` map) to a dict."""
    if callable(extra_metrics):
        computed = extra_metrics(record, prediction)
        return dict(computed) if computed else {}
    return {name: fn(record, prediction) for name, fn in extra_metrics.items()}


def score_records(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[str],
    *,
    start_index: int = 0,
    source: str | None = None,
    extra_metrics: ExtraMetrics | None = None,
) -> list[dict[str, Any]]:
    """Score *predictions* against *records* and return one **open** result dict per item.

    Each entry keeps the historical keys (``index``, ``task``, ``input``,
    ``expected_output``, ``prediction``, ``exact_match``) and adds ``f1`` plus,
    when *source* is given, the ``file`` the record came from. ``start_index``
    makes indices unique across a multi-file suite: the aggregate ``results``
    list is the concatenation of the per-file lists, so indices run ``0..n-1``
    over the *whole* suite while each per-file slice keeps its own entries.

    *extra_metrics*, when given, is either a single
    ``(record, prediction) -> {name: value}`` callable or a
    ``{name: (record, prediction) -> value}`` mapping of per-metric callables.
    Whatever keys it returns are merged onto the row dict, so a scorer can add
    a brand-new numeric metric without this module ever naming it — see
    :func:`summarize` for how those extra keys are folded into the aggregate.
    """
    results: list[dict[str, Any]] = []
    for offset, (record, prediction) in enumerate(zip(records, predictions)):
        expected = record["expected_output"]
        entry: dict[str, Any] = {
            "index": start_index + offset,
            "task": record["task"],
            "input": record["input"],
            "expected_output": expected,
            "prediction": prediction,
            "exact_match": prediction.strip() == expected.strip(),
            "f1": round(token_f1(prediction, expected), _F1_PLACES),
        }
        if source is not None:
            entry["file"] = source
        if extra_metrics is not None:
            entry.update(_resolve_extra_metrics(extra_metrics, record, prediction))
        results.append(entry)
    return results


def _is_metric_number(value: Any) -> bool:
    """Return whether *value* counts as a foldable numeric metric.

    ``bool`` is an ``int`` subclass in Python, but a boolean row field (e.g.
    ``exact_match``) is a flag, not a metric — so it is excluded.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _generic_metric_keys(results: Sequence[dict[str, Any]]) -> list[str]:
    """Return the sorted row keys that :func:`summarize` folds in generically."""
    keys: set[str] = set()
    for row in results:
        for key, value in row.items():
            if key not in _NON_METRIC_KEYS and key != "f1" and _is_metric_number(value):
                keys.add(key)
    return sorted(keys)


def _fold_generic_metrics(results: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Mean every generic numeric metric across the rows that reported it.

    Keys are visited in sorted order and each mean is rounded to
    :data:`_F1_PLACES`; a key no row reported a number for scores ``0.0``.
    """
    folded: dict[str, float] = {}
    for key in _generic_metric_keys(results):
        values = [float(row[key]) for row in results if _is_metric_number(row.get(key))]
        folded[key] = round(sum(values) / len(values), _F1_PLACES) if values else 0.0
    return folded


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return ``{total, exact_match, exact_match_pct, f1, ...}`` over scored *results*.

    ``exact_match_pct`` keeps the legacy 2-decimal rounding; ``f1`` is the mean
    per-record F1 rounded to four places. An empty list scores ``0`` / ``0.0``
    rather than raising, so an empty suite file still reports a well-formed
    entry.

    Any **other** numeric key present on the row dicts (added by a scorer's
    ``extra_metrics``, see :func:`score_records`) is folded in generically as
    its mean across rows that reported it, rounded to the same four places —
    without this function needing to know the metric's name in advance.
    """
    total = len(results)
    exact = sum(1 for r in results if r["exact_match"])
    mean_f1 = sum(float(r.get("f1", 0.0)) for r in results) / total if total else 0.0
    payload: dict[str, Any] = {
        "total": total,
        "exact_match": exact,
        "exact_match_pct": round(exact / total * 100, 2) if total else 0.0,
        "f1": round(mean_f1, _F1_PLACES),
    }

    payload.update(_fold_generic_metrics(results))
    return payload


def aggregate(files: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build the full eval payload from per-file entries produced by :func:`file_entry`.

    The returned dict is the aggregate (``total``, ``exact_match``,
    ``exact_match_pct``, ``f1``, plus any other numeric metric folded in by
    :func:`summarize`) over every record of every file, plus the concatenated
    ``results`` and the ``files`` list itself. Callers add their own
    target-specific fields (``model_dir``/``quant_method``/``quant_format``).
    """
    all_results: list[dict[str, Any]] = []
    for entry in files:
        all_results.extend(entry["results"])
    payload = summarize(all_results)
    payload["results"] = all_results
    payload["files"] = list(files)
    return payload


def file_entry(path: Any, results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return one ``files`` entry: the file's path plus its own scores and results."""
    entry = summarize(results)
    entry["path"] = str(path)
    entry["results"] = list(results)
    return entry


# ---------------------------------------------------------------------------
# eval.json
# ---------------------------------------------------------------------------

_SUITE_NAME_RE = re.compile(r"[^a-z0-9-]+")


def sanitize_suite_name(suite: str | Path) -> str:
    """Return the suite file/dir stem, sanitised to ``[a-z0-9-]``.

    The suite's basename (its final path component, extension stripped) is
    lowercased and every run of characters outside ``[a-z0-9-]`` is collapsed
    to a single ``-``; leading/trailing ``-`` are stripped. An empty result
    (e.g. a suite name made entirely of punctuation) falls back to ``"suite"``
    so callers always get a non-empty, filesystem-safe name.
    """
    stem = Path(suite).stem or Path(suite).name
    lowered = stem.lower()
    sanitized = _SUITE_NAME_RE.sub("-", lowered).strip("-")
    return sanitized or "suite"


def write_eval_json(
    directory: Path,
    suite_or_payload: str | Path | dict[str, Any],
    payload: dict[str, Any] | None = None,
    *,
    suite_paths: Sequence[Any] = (),
    target: str | None = None,
    batch_size: int | None = None,
    base_load_in_4bit: bool | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Write an eval result file and return its path.

    This function supports **two call shapes**:

    Legacy (unchanged behaviour, kept so ``sloth.tune._trainer``/``_exporter``
    never had to change)::

        write_eval_json(directory, payload, *, suite_paths=[...], target="adapter")

    writes ``<directory>/eval.json`` (see :data:`EVAL_JSON_NAME`), overwriting
    any prior run — the historical single-file layout.

    New, suite-keyed shape::

        write_eval_json(directory, suite, payload, batch_size=8)

    writes ``<directory>/eval/<sanitised-suite-name>.json`` (see
    :func:`sanitize_suite_name`), carrying ``schema_version`` (:data:`SCHEMA_VERSION`),
    ``suite``, ``batch_size``, ``target``, ``written_at`` and
    ``base_load_in_4bit`` alongside *payload*'s own keys. A second call with a
    *different* suite name writes a sibling file and leaves the first one
    untouched — each suite owns its own result file.

    The two shapes are told apart by whether *payload* was supplied: passing
    only two positional arguments is the legacy shape (the second argument is
    the result payload itself); passing three is the new shape (the second
    argument is the suite name/path, the third is the payload).
    """
    when = timestamp or datetime.now(timezone.utc)
    directory = Path(directory)

    if payload is None:
        legacy_payload = suite_or_payload
        if not isinstance(legacy_payload, dict):
            raise TypeError(
                "write_eval_json(directory, payload, *, suite_paths=..., target=...) "
                "requires payload to be a dict when called with two positional arguments"
            )
        record = dict(legacy_payload)
        record["suite_paths"] = [str(p) for p in suite_paths]
        record["target"] = target
        record["written_at"] = when.isoformat()
        destination = directory / EVAL_JSON_NAME
        destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return destination

    suite_name = sanitize_suite_name(suite_or_payload)
    record = dict(payload)
    record["schema_version"] = SCHEMA_VERSION
    record["suite"] = suite_name
    record["batch_size"] = batch_size
    record["target"] = target
    record["written_at"] = when.isoformat()
    record["base_load_in_4bit"] = base_load_in_4bit
    eval_dir = directory / EVAL_JSON_DIR
    eval_dir.mkdir(parents=True, exist_ok=True)
    destination = eval_dir / f"{suite_name}.json"
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return destination
