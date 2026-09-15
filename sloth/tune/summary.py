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

_EVAL_JSON_NAME = "eval.json"


def read_eval(output_dir: str | Path) -> dict[str, Any] | None:
    """Read+parse ``eval.json`` directly inside *output_dir* (the file
    :func:`sloth.tune.metrics.write_eval_json` writes for ``sloth eval``).

    Returns ``None`` on absence, a decode failure, or a parse failure —
    tolerated, never raised, exactly like :func:`read_trainer_state`.
    Read-only: this module never computes or recomputes a metric, it only
    reads the numbers ``sloth eval`` already wrote.
    """
    eval_path = Path(output_dir) / _EVAL_JSON_NAME
    try:
        raw = eval_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _eval_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Reduce a parsed ``eval.json`` payload to the aggregate fields a
    summary cares about: ``exact_match_pct``, ``f1``, and the suite file
    count (``len(payload["files"])``)."""
    files = payload.get("files")
    file_count = len(files) if isinstance(files, list) else 0
    return {
        "exact_match_pct": payload.get("exact_match_pct"),
        "f1": payload.get("f1"),
        "file_count": file_count,
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
                "exact_match_pct": float | None,
                "f1": float | None,
                "file_count": int,
            } | None,                 # read_eval(output_dir) — None, silently,
                                       # when no eval.json is present
            "notes": list[str],       # what was skipped/degraded, and why
        }

    Never raises: a missing/unreadable ``training_metadata.json`` or
    ``trainer_state.json`` degrades that half to ``None`` plus a note in
    ``notes`` — the other half (and the overall summary) is still returned.
    Likewise a missing/corrupt export index or record degrades ``exports`` to
    ``[]`` (or a partial list) plus a note, never an exception. A missing
    ``eval.json`` (at *output_dir* or at any export's own output dir) is not
    an error and adds no note — it degrades to ``None`` (or an absent
    ``"eval"`` key on an export entry) silently, exactly like an export-free
    run's empty ``exports`` list.
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

    eval_payload = read_eval(output_path)
    eval_summary = _eval_summary(eval_payload) if eval_payload is not None else None

    for export in exports:
        export_output = export.get("output")
        if not export_output or not isinstance(export_output, (str, os.PathLike)):
            continue
        export_eval_payload = read_eval(export_output)
        if export_eval_payload is not None:
            export["eval"] = _eval_summary(export_eval_payload)

    return {
        "output_dir": str(output_path),
        "metadata": metadata,
        "training": training,
        "exports": exports,
        "eval": eval_summary,
        "notes": notes,
    }
