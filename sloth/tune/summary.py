"""Per-run training summaries (pure stdlib, no torch).

Joins ``training_metadata.json`` (:func:`sloth.tune.metadata.read_metadata`)
with the loss/step history from the run's newest ``checkpoint-N/``
``trainer_state.json`` (written by the HF ``Trainer``/``SFTTrainer`` used in
:mod:`sloth.tune._trainer`), any discovered exports, and the aggregate
``eval.json`` scores (written by ``sloth eval`` via
:func:`sloth.tune.metrics.write_eval_json`) into one JSON-able summary dict.

All of these are OPTIONAL — a run that never checkpointed, was never
evaluated, or whose metadata file is missing/unreadable, still gets a
best-effort summary rather than an error; :func:`build_summary` records what
was skipped (and why) in its ``notes`` list (a missing ``eval.json`` is the
one exception — it degrades silently, with no note, exactly like a
still-empty ``exports`` list).

This module is read-only: it never computes or recomputes a metric, it only
reads the numbers ``sloth eval`` already wrote to disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sloth.cli._errors import CliError
from sloth.tune.metadata import read_metadata
from sloth.tune.metrics import EVAL_JSON_DIR, EVAL_JSON_NAME

_CHECKPOINT_PREFIX = "checkpoint-"


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def _checkpoint_step(dirname: str) -> int | None:
    """Parse the step number out of a ``checkpoint-N`` directory name."""
    if not dirname.startswith(_CHECKPOINT_PREFIX):
        return None
    suffix = dirname[len(_CHECKPOINT_PREFIX) :]
    return int(suffix) if suffix.isdigit() else None


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the ``checkpoint-N`` dir with the HIGHEST N under *output_dir*.

    Returns ``None`` when *output_dir* does not exist or holds no
    ``checkpoint-N`` directory (e.g. a run that never saved one).
    """
    root = Path(output_dir)
    if not root.is_dir():
        return None
    best_step = -1
    best_dir: Path | None = None
    for child in root.iterdir():
        if not child.is_dir():
            continue
        step = _checkpoint_step(child.name)
        if step is not None and step > best_step:
            best_step = step
            best_dir = child
    return best_dir


def read_trainer_state(checkpoint_dir: Path) -> dict[str, Any] | None:
    """Read+parse ``trainer_state.json`` inside *checkpoint_dir*.

    Returns ``None`` on absence or a parse failure — tolerated, never raised
    (a summary degrades gracefully rather than crashing on a partial
    checkpoint).
    """
    state_path = checkpoint_dir / "trainer_state.json"
    try:
        raw = state_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _training_progress(state: dict[str, Any]) -> dict[str, Any]:
    """Extract the summary-relevant fields from a parsed ``trainer_state.json``."""
    log_history = state.get("log_history") or []
    final_loss = None
    for entry in reversed(log_history):
        if isinstance(entry, dict) and "loss" in entry:
            final_loss = entry["loss"]
            break
    return {
        "final_step": state.get("global_step"),
        "final_loss": final_loss,
        "best_metric": state.get("best_metric"),
        "best_model_checkpoint": state.get("best_model_checkpoint"),
    }


# ---------------------------------------------------------------------------
# Eval discovery
# ---------------------------------------------------------------------------

#: The suite key used for a flat, legacy ``eval.json`` (predates suite-keyed
#: ``eval/<suite>.json`` files — see :func:`sloth.tune.metrics.write_eval_json`).
_LEGACY_SUITE_NAME = "legacy"

#: Bookkeeping keys on an eval payload that are never folded in as a numeric
#: metric, even when their value happens to be a number (``schema_version``).
#: ``batch_size`` and ``base_load_in_4bit`` are also bookkeeping, but they are
#: handled explicitly by :func:`_eval_summary` (rendered even when ``None``),
#: not silently dropped, so they are not listed here.
_EVAL_METRIC_EXCLUDE_KEYS = frozenset({"schema_version"})


def _read_json_file(path: Path) -> dict[str, Any] | None:
    """Read+parse one JSON object file, tolerating absence/corruption.

    Returns ``None`` on a missing file, an OS error, invalid UTF-8, a JSON
    decode failure, or a payload that isn't a JSON object — never raises.
    Shared by every eval-result reader in this module.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def read_eval(output_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Read every eval result recorded for *output_dir* as ``{suite: payload}``.

    Two sources are merged:

    * ``<output_dir>/eval/<suite>.json`` — the newer, suite-keyed files
      written by :func:`sloth.tune.metrics.write_eval_json`'s three-argument
      call shape. Each file's own ``suite`` field names its key (falling back
      to the file's stem when that field is missing/not a string).
    * ``<output_dir>/eval.json`` — the flat, legacy single-file layout (the
      two-argument call shape). When present it is added under the suite key
      ``"legacy"`` (see :data:`_LEGACY_SUITE_NAME`), so a pre-existing run dir
      that predates suite-keying still summarizes exactly as before (h19).

    Absence of either source is not an error, and a corrupt individual file
    is skipped rather than failing the whole read (tolerated, never raised,
    exactly like :func:`read_trainer_state`). A run with no eval results at
    all returns ``{}``. Read-only: this module never computes or recomputes a
    metric, it only reads the numbers ``sloth eval`` already wrote.
    """
    output_path = Path(output_dir)
    suites: dict[str, dict[str, Any]] = {}

    eval_dir = output_path / EVAL_JSON_DIR
    if eval_dir.is_dir():
        for path in sorted(eval_dir.glob("*.json")):
            payload = _read_json_file(path)
            if payload is None:
                continue
            suite = payload.get("suite")
            suite_name = suite if isinstance(suite, str) and suite else path.stem
            suites[suite_name] = payload

    legacy_payload = _read_json_file(output_path / EVAL_JSON_NAME)
    if legacy_payload is not None:
        suites[_LEGACY_SUITE_NAME] = legacy_payload

    return suites


def _eval_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Reduce one suite's eval payload to its summary block.

    Every top-level numeric field (``exact_match_pct``, ``f1``, ``total``,
    ``exact_match``, and any custom metric a scorer added — see
    :func:`sloth.tune.metrics.summarize`) is folded in generically: any key
    whose value is a ``bool``, ``list``, or ``dict`` is skipped, and
    :data:`_EVAL_METRIC_EXCLUDE_KEYS` drops pure bookkeeping fields, so a
    brand-new metric name shows up here without this function ever knowing it
    in advance. On top of that generic fold, this always adds the suite's
    file count (``len(payload["files"])``) and, when the key is present on
    *payload* (even if its value is ``None``), ``batch_size`` and
    ``base_load_in_4bit`` ("base precision") verbatim.
    """
    summary: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _EVAL_METRIC_EXCLUDE_KEYS:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (list, dict)):
            continue
        if isinstance(value, (int, float)):
            summary[key] = value

    files = payload.get("files")
    summary["file_count"] = len(files) if isinstance(files, list) else 0
    if "batch_size" in payload:
        summary["batch_size"] = payload.get("batch_size")
    if "base_load_in_4bit" in payload:
        summary["base_load_in_4bit"] = payload.get("base_load_in_4bit")
    return summary


def _select_target_suite(suites: dict[str, dict[str, Any]]) -> str:
    """Pick the suite whose ``exact_match_pct``/``f1`` back the top-level,
    backward-compatible fields on ``build_summary()["eval"]``.

    Prefers a suite literally named ``"target"`` (the target-task suite, once
    ``sloth eval`` wires suite names through); otherwise falls back to the
    first suite in *suites* (insertion order — the ``eval/*.json`` files in
    glob-sorted order, then ``"legacy"`` last). For a run with only a legacy
    ``eval.json`` (no ``eval/`` dir at all) that first/only suite IS
    ``"legacy"``, so existing readers of ``summary["eval"]["f1"]`` keep
    working unchanged (h19).
    """
    if "target" in suites:
        return "target"
    return next(iter(suites))


def _build_eval_block(suites: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Build the ``eval`` block of :func:`build_summary` from *suites*.

    Returns ``None`` when *suites* is empty (no eval results at all — no
    note, degrades silently, exactly like the rest of this module). Otherwise
    returns ``{"suites": {name: summary, ...}, "exact_match_pct": ..., "f1":
    ...}`` — the per-suite summaries keyed by suite name, plus the legacy
    flat ``exact_match_pct``/``f1`` fields taken from the selected suite (see
    :func:`_select_target_suite`) so old readers of ``summary["eval"]["f1"]``
    keep working without change.
    """
    if not suites:
        return None
    suite_summaries = {name: _eval_summary(payload) for name, payload in suites.items()}
    chosen = suite_summaries[_select_target_suite(suites)]
    return {
        "suites": suite_summaries,
        "exact_match_pct": chosen.get("exact_match_pct"),
        "f1": chosen.get("f1"),
    }


# ---------------------------------------------------------------------------
# Export discovery
# ---------------------------------------------------------------------------

_EXPORTS_INDEX_NAME = "exports.json"
_EXPORT_RECORD_NAME = "export.json"


def _load_exports_index(output_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read ``<output_dir>/exports.json`` (the index ``_exporter._append_index``
    writes), tolerating absence/corruption exactly like the rest of this module.

    Returns ``([], [])`` when the file is absent (not an error — a run with no
    exports yet). A present-but-unparseable file, or one that does not decode
    to a JSON list, degrades to ``([], [note])``.
    """
    index_path = output_dir / _EXPORTS_INDEX_NAME
    if not index_path.is_file():
        return [], []
    try:
        raw = index_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return [], [f"{_EXPORTS_INDEX_NAME} is missing or unreadable — export index omitted"]
    if not isinstance(parsed, list):
        return [], [f"{_EXPORTS_INDEX_NAME} does not contain a JSON list — export index omitted"]
    return [entry for entry in parsed if isinstance(entry, dict)], []


def _find_export_json_files(output_dir: Path) -> list[Path]:
    """Every ``export.json`` nested anywhere under *output_dir*, sorted for
    determinism. Returns ``[]`` when *output_dir* does not exist."""
    if not output_dir.is_dir():
        return []
    return sorted(output_dir.rglob(_EXPORT_RECORD_NAME))


def _read_export_record(path: Path) -> dict[str, Any] | None:
    """Read+parse one ``export.json``. Returns ``None`` on absence/corruption
    (tolerated, never raised)."""
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _export_identity(record: dict[str, Any]) -> tuple[Any, Any]:
    """A record's dedup key: (format, timestamp) — the pair a single
    ``run_export`` call stamps identically onto both the ``exports.json``
    entry and the record's own ``export.json``."""
    return record.get("format"), record.get("timestamp")


def discover_exports(output_dir: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Discover every export recorded for a run.

    Joins two sources:

    * ``<output_dir>/exports.json`` — the append-only index
      :func:`sloth.tune._exporter._append_index` maintains.
    * any ``export.json`` nested under *output_dir* — the per-export record
      :func:`sloth.tune._exporter._write_export_json` writes into each
      export's own output directory (present even when that directory was
      never, or not yet, mirrored into the index).

    A record discovered via a nested ``export.json`` is merged with its
    matching index entry (filling in ``export_json``/``output`` paths) rather
    than duplicated, keyed on ``(format, timestamp)``. A genuinely new record
    (no matching index entry — e.g. a corrupt/missing index) is appended.

    Every returned entry carries ``export_json`` and ``output`` keys — ``None``
    when the record came only from the index and no matching file was found
    on disk.

    Never raises: a missing/corrupt index or a corrupt individual
    ``export.json`` degrades gracefully, with a note explaining what was
    skipped, exactly like the rest of this module.
    """
    output_path = Path(output_dir)
    notes: list[str] = []

    index_entries, index_notes = _load_exports_index(output_path)
    notes.extend(index_notes)

    exports: list[dict[str, Any]] = []
    for entry in index_entries:
        merged = dict(entry)
        merged.setdefault("export_json", None)
        merged.setdefault("output", None)
        exports.append(merged)

    by_identity: dict[tuple[Any, Any], dict[str, Any]] = {
        _export_identity(entry): entry for entry in exports
    }

    for path in _find_export_json_files(output_path):
        record = _read_export_record(path)
        if record is None:
            notes.append(f"{_EXPORT_RECORD_NAME} at {path} is missing or unreadable — skipped")
            continue
        identity = _export_identity(record)
        existing = by_identity.get(identity)
        if existing is not None and existing.get("export_json") is None:
            existing["export_json"] = str(path)
            existing["output"] = str(path.parent)
            continue
        if existing is not None:
            continue
        new_entry = dict(record)
        new_entry["export_json"] = str(path)
        new_entry["output"] = str(path.parent)
        exports.append(new_entry)
        by_identity[identity] = new_entry

    return exports, notes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_summary(output_dir: str | Path) -> dict[str, Any]:
    """Build the one-JSON-object summary for *output_dir*.

    Shape::

        {
            "output_dir": str,
            "metadata": dict | None,   # sloth.tune.metadata.read_metadata() shape
            "training": {
                "checkpoint": str,       # e.g. "checkpoint-60"
                "final_step": int | None,
                "final_loss": float | None,
                "best_metric": float | None,
                "best_model_checkpoint": str | None,
            } | None,
            "exports": list[dict],    # discover_exports() — [] when none found;
                                       # each entry additionally carries an
                                       # "eval" key (see below) when that
                                       # export's own output dir has an
                                       # eval.json
            "eval": {
                "suites": {
                    "<suite name>": {          # e.g. "target", "holdout", "legacy"
                        "exact_match_pct": float,
                        "f1": float,
                        "total": int,          # plus any other numeric metric key
                        "file_count": int,
                        "batch_size": int | None,          # when present on the suite's file
                        "base_load_in_4bit": bool | None,  # when present on the suite's file
                    },
                    ...
                },
                "exact_match_pct": float | None,  # from the "target" suite when present,
                "f1": float | None,               # else the first suite (see
                                                   # _select_target_suite) — kept so
                                                   # existing readers of
                                                   # summary["eval"]["f1"] keep working
            } | None,                 # read_eval(output_dir) — None, silently,
                                       # when no eval results (neither eval/*.json nor a
                                       # legacy eval.json) are present
            "notes": list[str],       # what was skipped/degraded, and why
        }

    Never raises: a missing/unreadable ``training_metadata.json`` or
    ``trainer_state.json`` degrades that half to ``None`` plus a note in
    ``notes`` — the other half (and the overall summary) is still returned.
    Likewise a missing/corrupt export index or record degrades ``exports`` to
    ``[]`` (or a partial list) plus a note, never an exception. Missing eval
    results (at *output_dir* or at any export's own output dir) are not an
    error and add no note — they degrade to ``None`` (or an absent ``"eval"``
    key on an export entry) silently, exactly like an export-free run's empty
    ``exports`` list. A run dir that predates suite-keyed eval results (only a
    flat ``eval.json``, no ``eval/`` dir) still summarizes with unchanged
    ``exact_match_pct``/``f1`` (h19) — it becomes the single ``"legacy"`` suite.
    """
    output_path = Path(output_dir)
    notes: list[str] = []

    metadata: dict[str, Any] | None
    try:
        metadata = read_metadata(output_path)
    except CliError:
        metadata = None
        notes.append("no training_metadata.json found — metadata omitted")

    training: dict[str, Any] | None = None
    checkpoint_dir = find_latest_checkpoint(output_path)
    if checkpoint_dir is None:
        notes.append("no checkpoint-N directory found — no trainer_state.json to read")
    else:
        state = read_trainer_state(checkpoint_dir)
        if state is None:
            notes.append(f"trainer_state.json in {checkpoint_dir.name} is missing or unreadable")
        else:
            training = _training_progress(state)
            training["checkpoint"] = checkpoint_dir.name

    exports, export_notes = discover_exports(output_path)
    notes.extend(export_notes)

    eval_summary = _build_eval_block(read_eval(output_path))

    for export in exports:
        export_output = export.get("output")
        if not export_output or not isinstance(export_output, (str, os.PathLike)):
            continue
        export_eval_block = _build_eval_block(read_eval(export_output))
        if export_eval_block is not None:
            export["eval"] = export_eval_block

    return {
        "output_dir": str(output_path),
        "metadata": metadata,
        "training": training,
        "exports": exports,
        "eval": eval_summary,
        "notes": notes,
    }
