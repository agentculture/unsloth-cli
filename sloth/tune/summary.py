"""Per-run training summaries (pure stdlib, no torch).

Joins ``training_metadata.json`` (:func:`sloth.tune.metadata.read_metadata`)
with the loss/step history from the run's newest ``checkpoint-N/``
``trainer_state.json`` (written by the HF ``Trainer``/``SFTTrainer`` used in
:mod:`sloth.tune._trainer`) into one JSON-able summary dict.

Both halves are OPTIONAL — a run that never checkpointed, or whose metadata
file is missing/unreadable, still gets a best-effort summary rather than an
error; :func:`build_summary` records what was skipped (and why) in its
``notes`` list.
"""

from __future__ import annotations

import json
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
            "notes": list[str],       # what was skipped/degraded, and why
        }

    Never raises: a missing/unreadable ``training_metadata.json`` or
    ``trainer_state.json`` degrades that half to ``None`` plus a note in
    ``notes`` — the other half (and the overall summary) is still returned.
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

    return {
        "output_dir": str(output_path),
        "metadata": metadata,
        "training": training,
        "notes": notes,
    }
