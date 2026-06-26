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


def write_metadata(
    adapter_dir: Path,
    *,
    model: str,
    method: str,
    dataset_path: Path,
    hyperparameters: dict[str, Any],
    timestamp: str | None = None,
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
        Path to the JSONL training dataset.  Its sha256 and line count are
        computed and embedded in the metadata.
    hyperparameters:
        Mapping of training hyperparameters (rank, lora_alpha, epochs, …).
    timestamp:
        ISO-8601 string to stamp the record.  When ``None`` (the default) the
        current UTC time is used.  Pass an explicit value in tests for a
        deterministic round-trip.

    Returns
    -------
    Path
        The path of the written ``training_metadata.json`` file.

    Raises
    ------
    CliError
        code=2 when *dataset_path* cannot be read.
    """
    sha256, line_count = dataset_digest(dataset_path)

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    record: dict[str, Any] = {
        "model": model,
        "method": method,
        "dataset": {
            "sha256": sha256,
            "line_count": line_count,
        },
        "hyperparameters": hyperparameters,
        "timestamp": timestamp,
    }

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
