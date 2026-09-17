"""Training-metadata writer for adapter fine-tuning runs (pure stdlib, no torch).

Public API
----------
dataset_digest(path) -> tuple[str, int]
    Stream *path* and return ``(sha256_hexdigest, line_count)``.
    Raises :class:`sloth.cli._errors.CliError` (code 2) if the file is unreadable.

write_metadata(adapter_dir, *, model, method, dataset_path, hyperparameters, timestamp=None) -> Path
    Write ``adapter_dir/training_metadata.json`` containing every field required
    by the training-metadata contract and return the path.  When *timestamp* is
    ``None`` the current UTC time is used (injectable for deterministic tests).

read_metadata(adapter_dir) -> dict
    Read and return the metadata dict previously written by :func:`write_metadata`.
    Raises :class:`sloth.cli._errors.CliError` (code 2) if the file is missing or unreadable.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sloth.cli._errors import EXIT_ENV_ERROR, CliError

_METADATA_FILENAME = "training_metadata.json"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def dataset_digest(path: Path) -> tuple[str, int]:
    """Return ``(sha256_hexdigest, line_count)`` for the dataset at *path*.

    Streams the file one line at a time so large datasets do not require loading
    the entire file into memory.  ``line_count`` is the number of **non-blank**
    lines — i.e. the number of JSONL records — matching
    :func:`sloth.tune.datasets.validate_dataset`, which skips blank lines.
    Counting records (rather than ``b"\\n"`` bytes) keeps the count correct when
    the final record lacks a trailing newline and ignores blank separator lines.

    Raises
    ------
    CliError
        code=2 when *path* does not exist or cannot be read.
    """
    h = hashlib.sha256()
    line_count = 0
    try:
        with path.open("rb") as fh:
            for raw_line in fh:
                h.update(raw_line)
                if raw_line.strip():
                    line_count += 1
    except OSError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"Cannot read dataset file: {path}",
            remediation="Check that the path exists and is readable.",
        ) from exc
    return h.hexdigest(), line_count


#: Prefix marking a dataset value as a Hugging Face Hub dataset id rather than a
#: local JSONL path — mirrors ``sloth.tune._trainer.HF_DATASET_PREFIX`` (kept as
#: a separate literal so this module stays free of any import from ``_trainer``).
_HF_DATASET_PREFIX = "hf:"

#: Split recorded when a ``hf:<org>/<name>`` spec omits an explicit ``:<split>``.
_DEFAULT_HF_SPLIT = "train"

#: Revision recorded for a hub dataset when the caller does not pin one — the
#: implicit ref ``datasets.load_dataset`` resolves to when no revision is given.
_DEFAULT_HF_REVISION = "main"


def _hf_dataset_record(dataset_spec: str, *, hf_revision: str | None) -> dict[str, Any]:
    """Return the ``{"hf_id", "split", "revision"}`` record for a ``hf:`` spec."""
    body = dataset_spec[len(_HF_DATASET_PREFIX) :]
    hf_id, _, split = body.partition(":")
    return {
        "hf_id": hf_id,
        "split": split or _DEFAULT_HF_SPLIT,
        "revision": hf_revision or _DEFAULT_HF_REVISION,
    }


#: Keys a ``log_history`` entry may use for the *training* loss. ``"loss"`` is
#: what transformers logs per ``logging_steps``; ``"train_loss"`` is the single
#: summary entry appended when training finishes.
_TRAIN_LOSS_KEYS: tuple[str, ...] = ("loss", "train_loss")


def fold_log_history(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold a ``trainer.state.log_history`` list into the metadata loss record.

    transformers logs training and evaluation separately — ``{"loss", "step"}``
    entries every ``logging_steps`` and ``{"eval_loss", "step"}`` entries every
    ``eval_steps`` — so the same step can appear twice. They are merged here by
    ``step`` (first-seen order preserved) into
    ``[{"step", "train_loss", "eval_loss"}]``, with a missing side recorded as
    ``None``. Entries carrying no ``step`` or neither loss (e.g. the
    ``train_runtime`` summary) are ignored.

    Returns
    -------
    dict
        ``{"loss_history": [...], "final_train_loss": float|None,
        "final_eval_loss": float|None}`` — the finals being the last non-``None``
        value of each series.
    """
    merged: dict[Any, dict[str, Any]] = {}
    for entry in log_history:
        if not isinstance(entry, dict) or "step" not in entry:
            continue
        train_loss = next((entry[k] for k in _TRAIN_LOSS_KEYS if k in entry), None)
        eval_loss = entry.get("eval_loss")
        if train_loss is None and eval_loss is None:
            continue
        step = entry["step"]
        row = merged.setdefault(step, {"step": step, "train_loss": None, "eval_loss": None})
        if train_loss is not None:
            row["train_loss"] = train_loss
        if eval_loss is not None:
            row["eval_loss"] = eval_loss

    loss_history = list(merged.values())
    final_train_loss = next(
        (row["train_loss"] for row in reversed(loss_history) if row["train_loss"] is not None),
        None,
    )
    final_eval_loss = next(
        (row["eval_loss"] for row in reversed(loss_history) if row["eval_loss"] is not None),
        None,
    )
    return {
        "loss_history": loss_history,
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
    }


def write_metadata(
    adapter_dir: Path,
    *,
    model: str,
    method: str,
    dataset_path: "Path | str",
    hyperparameters: dict[str, Any],
    timestamp: str | None = None,
    hf_revision: str | None = None,
    log_history: list[dict[str, Any]] | None = None,
    holdout: dict[str, Any] | None = None,
) -> Path:
    """Write ``adapter_dir/training_metadata.json`` and return the path.

    Parameters
    ----------
    adapter_dir:
        Directory that contains (or will contain) the saved adapter weights.
        The metadata file is written beside the adapter output.
    model:
        Base model identifier, e.g. ``"unsloth/Qwen3-4B"``.
    method:
        Adapter method: ``"lora"`` or ``"qlora"``.
    dataset_path:
        Either a path to the local JSONL training dataset (its sha256 and line
        count are computed and embedded in the metadata), or a
        ``"hf:<org>/<name>[:split]"`` string naming a Hugging Face Hub dataset
        — recorded instead as ``{"hf_id", "split", "revision"}`` (no sha256:
        the dataset is not a fixed local file).
    hyperparameters:
        Mapping of training hyperparameters (rank, lora_alpha, epochs, …).
    timestamp:
        ISO-8601 string to stamp the record.  When ``None`` (the default) the
        current UTC time is used.  Pass an explicit value in tests for a
        deterministic round-trip.
    hf_revision:
        Revision/ref to record for a ``hf:`` *dataset_path*. Ignored for a
        local dataset path. Defaults to ``"main"`` when not given.
    log_history:
        ``trainer.state.log_history`` from the finished run. When given it is
        folded by :func:`fold_log_history` into the ``loss_history``,
        ``final_train_loss`` and ``final_eval_loss`` keys. ``None`` (the
        default) omits all three.
    holdout:
        Provenance of the training-time eval split — ``{"fraction", "seed",
        "train_path", "holdout_path", "train_count", "holdout_count",
        "eval_steps"}`` — recorded verbatim so the split is reproducible.
        ``None`` (the default) omits the key entirely.

    Returns
    -------
    Path
        The path of the written ``training_metadata.json`` file.

    Raises
    ------
    CliError
        code=2 when a local *dataset_path* cannot be read.
    """
    dataset_str = str(dataset_path)
    if dataset_str.startswith(_HF_DATASET_PREFIX):
        dataset_record: dict[str, Any] = _hf_dataset_record(dataset_str, hf_revision=hf_revision)
    else:
        sha256, line_count = dataset_digest(Path(dataset_path))
        dataset_record = {
            "path": dataset_str,
            "sha256": sha256,
            "line_count": line_count,
        }

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    record: dict[str, Any] = {
        "model": model,
        "method": method,
        "dataset": dataset_record,
        "hyperparameters": hyperparameters,
        "timestamp": timestamp,
    }
    if log_history is not None:
        record.update(fold_log_history(log_history))
    if holdout is not None:
        record["holdout"] = holdout

    out_path = adapter_dir / _METADATA_FILENAME
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out_path


def read_metadata(adapter_dir: Path) -> dict[str, Any]:
    """Read and return the metadata dict from *adapter_dir/training_metadata.json*.

    Raises
    ------
    CliError
        code=2 when the metadata file is missing or cannot be parsed.
    """
    meta_path = adapter_dir / _METADATA_FILENAME
    try:
        raw = meta_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"Cannot read metadata file: {meta_path}",
            remediation=(
                "Ensure the adapter directory exists and contains a "
                f"'{_METADATA_FILENAME}' file produced by write_metadata()."
            ),
        ) from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"Metadata file is not valid JSON: {meta_path}",
            remediation="The file may be corrupted; re-run the training job to regenerate it.",
        ) from exc
