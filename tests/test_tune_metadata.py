"""Tests for sloth.tune.metadata — training-metadata writer (pure stdlib, no torch).

Criteria verified:
  1. After a simulated run, the written metadata file records model, method,
     dataset sha256 + line count, hyperparameters, and an ISO-8601 timestamp.
  2. write → read round-trips every field exactly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sloth.tune.metadata import dataset_digest, read_metadata, write_metadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "dataset.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _expected_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# dataset_digest
# ---------------------------------------------------------------------------


class TestDatasetDigest:
    def test_returns_sha256_and_line_count(self, tmp_path: Path) -> None:
        dataset = _make_dataset(tmp_path, ['{"a": 1}', '{"b": 2}', '{"c": 3}'])
        digest, count = dataset_digest(dataset)
        assert digest == _expected_sha256(dataset)
        assert count == 3

    def test_single_line(self, tmp_path: Path) -> None:
        dataset = _make_dataset(tmp_path, ['{"x": 0}'])
        _, count = dataset_digest(dataset)
        assert count == 1

    def test_sha256_changes_with_content(self, tmp_path: Path) -> None:
        d1 = tmp_path / "a.jsonl"
        d1.write_text('{"row": 1}\n', encoding="utf-8")
        d2 = tmp_path / "b.jsonl"
        d2.write_text('{"row": 2}\n', encoding="utf-8")
        sha1, _ = dataset_digest(d1)
        sha2, _ = dataset_digest(d2)
        assert sha1 != sha2

    def test_missing_file_raises_cli_error(self, tmp_path: Path) -> None:
        from sloth.cli._errors import CliError

        with pytest.raises(CliError) as exc_info:
            dataset_digest(tmp_path / "does_not_exist.jsonl")
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# write_metadata / read_metadata
# ---------------------------------------------------------------------------


HYPERPARAMS = {"rank": 16, "lora_alpha": 32, "epochs": 3, "learning_rate": 2e-4}
FIXED_TS = "2026-06-26T10:00:00+00:00"


class TestWriteMetadata:
    def test_creates_file_in_adapter_dir(self, tmp_path: Path) -> None:
        dataset = _make_dataset(tmp_path, ['{"msg": "hi"}'])
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        result = write_metadata(
            adapter_dir,
            model="unsloth/Qwen3-4B",
            method="lora",
            dataset_path=dataset,
            hyperparameters=HYPERPARAMS,
            timestamp=FIXED_TS,
        )
        assert result == adapter_dir / "training_metadata.json"
        assert result.exists()

    def test_metadata_contains_required_fields(self, tmp_path: Path) -> None:
        dataset = _make_dataset(tmp_path, ['{"a": 1}', '{"b": 2}'])
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        write_metadata(
            adapter_dir,
            model="unsloth/Qwen3-4B",
            method="qlora",
            dataset_path=dataset,
            hyperparameters=HYPERPARAMS,
            timestamp=FIXED_TS,
        )
        meta_path = adapter_dir / "training_metadata.json"
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["model"] == "unsloth/Qwen3-4B"
        assert data["method"] == "qlora"
        assert "dataset" in data
        assert data["dataset"]["sha256"] == _expected_sha256(dataset)
        assert data["dataset"]["line_count"] == 2
        assert data["hyperparameters"] == HYPERPARAMS
        assert data["timestamp"] == FIXED_TS

    def test_timestamp_defaults_to_utc_iso(self, tmp_path: Path) -> None:
        dataset = _make_dataset(tmp_path, ['{"x": 1}'])
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        write_metadata(
            adapter_dir,
            model="m",
            method="lora",
            dataset_path=dataset,
            hyperparameters={},
        )
        data = json.loads((adapter_dir / "training_metadata.json").read_text(encoding="utf-8"))
        ts = data["timestamp"]
        # Must look like an ISO-8601 datetime with timezone offset.
        assert "T" in ts
        assert ts.endswith("+00:00") or ts.endswith("Z")

    def test_output_is_pretty_json(self, tmp_path: Path) -> None:
        dataset = _make_dataset(tmp_path, ['{"y": 9}'])
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        write_metadata(
            adapter_dir,
            model="m",
            method="lora",
            dataset_path=dataset,
            hyperparameters={},
            timestamp=FIXED_TS,
        )
        raw = (adapter_dir / "training_metadata.json").read_text(encoding="utf-8")
        # Pretty JSON has newlines.
        assert "\n" in raw


class TestRoundTrip:
    def test_write_then_read_equals(self, tmp_path: Path) -> None:
        """Criterion 2: write → read round-trips every field exactly."""
        dataset = _make_dataset(
            tmp_path,
            ['{"role": "user", "content": "hi"}', '{"role": "assistant", "content": "hello"}'],
        )
        adapter_dir = tmp_path / "my_adapter"
        adapter_dir.mkdir()
        write_metadata(
            adapter_dir,
            model="unsloth/Qwen3-9B",
            method="lora",
            dataset_path=dataset,
            hyperparameters=HYPERPARAMS,
            timestamp=FIXED_TS,
        )
        recovered = read_metadata(adapter_dir)

        assert recovered["model"] == "unsloth/Qwen3-9B"
        assert recovered["method"] == "lora"
        assert recovered["dataset"]["sha256"] == _expected_sha256(dataset)
        assert recovered["dataset"]["line_count"] == 2
        assert recovered["hyperparameters"] == HYPERPARAMS
        assert recovered["timestamp"] == FIXED_TS

    def test_read_missing_raises_cli_error(self, tmp_path: Path) -> None:
        from sloth.cli._errors import CliError

        adapter_dir = tmp_path / "no_adapter"
        adapter_dir.mkdir()
        with pytest.raises(CliError) as exc_info:
            read_metadata(adapter_dir)
        assert exc_info.value.code == 2


class TestDatasetDigestLineCount:
    """line_count must equal the number of JSONL records, matching validate_dataset."""

    def test_no_trailing_newline_is_not_off_by_one(self, tmp_path: Path) -> None:
        path = tmp_path / "no_trailing.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}', encoding="utf-8")  # no final \n
        _, count = dataset_digest(path)
        assert count == 3

    def test_blank_lines_are_not_counted(self, tmp_path: Path) -> None:
        path = tmp_path / "with_blanks.jsonl"
        path.write_text('{"a": 1}\n\n{"b": 2}\n   \n', encoding="utf-8")
        _, count = dataset_digest(path)
        assert count == 2
