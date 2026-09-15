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
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

#: Filename written into the evaluated directory by :func:`write_eval_json`.
EVAL_JSON_NAME = "eval.json"

#: Decimal places for the reported F1 figures (percentages keep the legacy 2).
_F1_PLACES = 4

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


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


def score_records(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[str],
    *,
    start_index: int = 0,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Score *predictions* against *records* and return one result dict per item.

    Each entry keeps the historical keys (``index``, ``task``, ``input``,
    ``expected_output``, ``prediction``, ``exact_match``) and adds ``f1`` plus,
    when *source* is given, the ``file`` the record came from. ``start_index``
    makes indices unique across a multi-file suite: the aggregate ``results``
    list is the concatenation of the per-file lists, so indices run ``0..n-1``
    over the *whole* suite while each per-file slice keeps its own entries.
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
        results.append(entry)
    return results


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return ``{total, exact_match, exact_match_pct, f1}`` over scored *results*.

    ``exact_match_pct`` keeps the legacy 2-decimal rounding; ``f1`` is the mean
    per-record F1 rounded to four places. An empty list scores ``0`` / ``0.0``
    rather than raising, so an empty suite file still reports a well-formed
    entry.
    """
    total = len(results)
    exact = sum(1 for r in results if r["exact_match"])
    mean_f1 = sum(float(r.get("f1", 0.0)) for r in results) / total if total else 0.0
    return {
        "total": total,
        "exact_match": exact,
        "exact_match_pct": round(exact / total * 100, 2) if total else 0.0,
        "f1": round(mean_f1, _F1_PLACES),
    }


def aggregate(files: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build the full eval payload from per-file entries produced by :func:`file_entry`.

    The returned dict is the aggregate (``total``, ``exact_match``,
    ``exact_match_pct``, ``f1``) over every record of every file, plus the
    concatenated ``results`` and the ``files`` list itself. Callers add their own
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


def write_eval_json(
    directory: Path,
    payload: dict[str, Any],
    *,
    suite_paths: Sequence[Any],
    target: str,
    timestamp: datetime | None = None,
) -> Path:
    """Write ``<directory>/eval.json`` and return its path (overwriting any prior run).

    The file's schema is the stdout result dict plus three provenance fields:
    ``suite_paths`` (every scored file, in order), ``target`` (``"adapter"`` or
    ``"model"``) and ``written_at`` (ISO-8601 UTC). Re-running an eval overwrites
    it, so the file always describes the most recent run of that directory.
    """
    when = timestamp or datetime.now(timezone.utc)
    record = dict(payload)
    record["suite_paths"] = [str(p) for p in suite_paths]
    record["target"] = target
    record["written_at"] = when.isoformat()
    destination = directory / EVAL_JSON_NAME
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return destination
