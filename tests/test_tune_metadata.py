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

    def test_metadata_records_dataset_path(self, tmp_path: Path) -> None:
        """Deviation d1: dataset.path is recorded exactly as given, so quantized
        exports can calibrate on the training dataset by default."""
        dataset = _make_dataset(tmp_path, ['{"a": 1}'])
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        write_metadata(
            adapter_dir,
            model="unsloth/Qwen3-4B",
            method="lora",
            dataset_path=dataset,
            hyperparameters=HYPERPARAMS,
            timestamp=FIXED_TS,
        )
        data = json.loads((adapter_dir / "training_metadata.json").read_text(encoding="utf-8"))
        assert data["dataset"]["path"] == str(dataset)
        # existing keys unchanged, plus the dataset-kind marker
        assert set(data["dataset"]) == {"path", "sha256", "line_count", "source"}

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
        assert recovered["dataset"]["path"] == str(dataset)
        assert recovered["hyperparameters"] == HYPERPARAMS
        assert recovered["timestamp"] == FIXED_TS

    def test_read_missing_raises_cli_error(self, tmp_path: Path) -> None:
        from sloth.cli._errors import CliError

        adapter_dir = tmp_path / "no_adapter"
        adapter_dir.mkdir()
        with pytest.raises(CliError) as exc_info:
            read_metadata(adapter_dir)
        assert exc_info.value.code == 2

    def test_legacy_metadata_without_path_key_still_reads(self, tmp_path: Path) -> None:
        """Older metadata files written before deviation d1 lack dataset.path;
        read_metadata must not raise on the missing key."""
        adapter_dir = tmp_path / "legacy_adapter"
        adapter_dir.mkdir()
        legacy_record = {
            "model": "unsloth/Qwen3-4B",
            "method": "lora",
            "dataset": {"sha256": "deadbeef", "line_count": 5},
            "hyperparameters": HYPERPARAMS,
            "timestamp": FIXED_TS,
        }
        (adapter_dir / "training_metadata.json").write_text(
            json.dumps(legacy_record, indent=2), encoding="utf-8"
        )
        recovered = read_metadata(adapter_dir)
        assert recovered["dataset"]["sha256"] == "deadbeef"
        assert "path" not in recovered["dataset"]


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


# ---------------------------------------------------------------------------
# write_metadata for an external (hf:) dataset — t11 / c34
# ---------------------------------------------------------------------------


class TestWriteMetadataHfDataset:
    def test_records_hf_id_split_and_revision_instead_of_sha256(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        write_metadata(
            adapter_dir,
            model="unsloth/Qwen3-4B",
            method="qlora",
            dataset_path="hf:my-org/my-dataset:train",
            hyperparameters=HYPERPARAMS,
            timestamp=FIXED_TS,
            hf_revision="abc123",
        )
        data = json.loads((adapter_dir / "training_metadata.json").read_text(encoding="utf-8"))
        assert data["dataset"] == {
            "hf_id": "my-org/my-dataset",
            "split": "train",
            "revision": "abc123",
            "source": "hf",
        }
        assert "sha256" not in data["dataset"]
        assert "line_count" not in data["dataset"]
        assert "path" not in data["dataset"]

    def test_split_defaults_to_train_when_omitted(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        write_metadata(
            adapter_dir,
            model="m",
            method="lora",
            dataset_path="hf:my-org/my-dataset",
            hyperparameters={},
            timestamp=FIXED_TS,
        )
        data = json.loads((adapter_dir / "training_metadata.json").read_text(encoding="utf-8"))
        assert data["dataset"]["hf_id"] == "my-org/my-dataset"
        assert data["dataset"]["split"] == "train"

    def test_revision_defaults_to_main_when_not_given(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        write_metadata(
            adapter_dir,
            model="m",
            method="lora",
            dataset_path="hf:my-org/my-dataset:validation",
            hyperparameters={},
            timestamp=FIXED_TS,
        )
        data = json.loads((adapter_dir / "training_metadata.json").read_text(encoding="utf-8"))
        assert data["dataset"]["split"] == "validation"
        assert data["dataset"]["revision"] == "main"

    def test_round_trips_through_read_metadata(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        write_metadata(
            adapter_dir,
            model="m",
            method="qlora",
            dataset_path="hf:my-org/my-dataset:train",
            hyperparameters={},
            timestamp=FIXED_TS,
            hf_revision="v2",
        )
        recovered = read_metadata(adapter_dir)
        assert recovered["dataset"] == {
            "hf_id": "my-org/my-dataset",
            "split": "train",
            "revision": "v2",
            "source": "hf",
        }


# ---------------------------------------------------------------------------
# Training-time eval: loss_history + holdout provenance (t7)
# ---------------------------------------------------------------------------


class TestWriteMetadataLossHistory:
    """``log_history`` from ``trainer.state`` is folded into the metadata record."""

    def _write(self, tmp_path: Path, **extra) -> dict:
        dataset = _make_dataset(tmp_path, ['{"task": "t", "input": "i", "expected_output": "o"}'])
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        write_metadata(
            adapter_dir,
            model="unsloth/Qwen3-4B",
            method="lora",
            dataset_path=dataset,
            hyperparameters={"lora_r": 8},
            timestamp="2026-01-01T00:00:00+00:00",
            **extra,
        )
        return read_metadata(adapter_dir)

    def test_absent_when_no_log_history_given(self, tmp_path: Path) -> None:
        data = self._write(tmp_path)
        assert "loss_history" not in data
        assert "holdout" not in data

    def test_merges_train_and_eval_entries_by_step(self, tmp_path: Path) -> None:
        log_history = [
            {"loss": 2.0, "step": 1},
            {"loss": 1.5, "step": 2},
            {"eval_loss": 1.9, "step": 2},
            {"loss": 1.0, "step": 3},
            {"eval_loss": 1.2, "step": 4},
            {"train_runtime": 0.4, "step": 4},
        ]
        data = self._write(tmp_path, log_history=log_history)

        assert data["loss_history"] == [
            {"step": 1, "train_loss": 2.0, "eval_loss": None},
            {"step": 2, "train_loss": 1.5, "eval_loss": 1.9},
            {"step": 3, "train_loss": 1.0, "eval_loss": None},
            {"step": 4, "train_loss": None, "eval_loss": 1.2},
        ]
        assert data["final_train_loss"] == 1.0
        assert data["final_eval_loss"] == 1.2

    def test_final_train_loss_key_is_honoured(self, tmp_path: Path) -> None:
        """HF appends a summary entry keyed ``train_loss`` — it counts as a train loss."""
        data = self._write(
            tmp_path,
            log_history=[{"loss": 3.0, "step": 1}, {"train_loss": 2.5, "step": 2}],
        )
        assert data["final_train_loss"] == 2.5
        assert data["final_eval_loss"] is None

    def test_empty_log_history_records_empty_list_and_null_finals(self, tmp_path: Path) -> None:
        data = self._write(tmp_path, log_history=[])
        assert data["loss_history"] == []
        assert data["final_train_loss"] is None
        assert data["final_eval_loss"] is None

    def test_holdout_block_is_recorded_verbatim(self, tmp_path: Path) -> None:
        holdout = {
            "fraction": 0.2,
            "seed": 7,
            "train_path": str(tmp_path / "d.train.jsonl"),
            "holdout_path": str(tmp_path / "d.holdout.jsonl"),
            "train_count": 8,
            "holdout_count": 2,
            "eval_steps": 15,
        }
        data = self._write(tmp_path, holdout=holdout)
        assert data["holdout"] == holdout


# ---------------------------------------------------------------------------
# dataset.source + resolved.load_in_4bit (qodo PR #29 findings 2 and 4)
# ---------------------------------------------------------------------------


class TestDatasetSourceField:
    """Every dataset record says *what kind* of dataset it was.

    A hub dataset carries no ``dataset.path``, so a consumer that keys off the
    path alone (the eval-side train/eval overlap check) silently skipped hub
    runs. ``dataset.source`` states it outright.
    """

    def test_local_dataset_records_source_file(self, tmp_path: Path) -> None:
        dataset = _make_dataset(tmp_path, ['{"a": 1}'])
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
        data = json.loads((adapter_dir / "training_metadata.json").read_text(encoding="utf-8"))
        assert data["dataset"]["source"] == "file"
        assert data["dataset"]["path"] == str(dataset)

    def test_hf_dataset_records_source_hf(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        write_metadata(
            adapter_dir,
            model="m",
            method="qlora",
            dataset_path="hf:my-org/my-dataset:train",
            hyperparameters={},
            timestamp=FIXED_TS,
        )
        data = json.loads((adapter_dir / "training_metadata.json").read_text(encoding="utf-8"))
        assert data["dataset"]["source"] == "hf"
        assert "path" not in data["dataset"]


class TestResolvedLoadIn4bit:
    """``resolved.load_in_4bit`` is the *effective* precision the run trained at.

    ``hyperparameters.load_in_4bit`` is the raw config value; training forces
    4-bit for a ``qlora`` method regardless, so a downstream bench reading the
    raw value alone would benchmark a QLoRA adapter at the wrong precision.
    """

    def test_qlora_run_resolves_to_true_even_when_the_flag_is_off(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        write_metadata(
            adapter_dir,
            model="m",
            method="qlora",
            dataset_path="hf:my-org/my-dataset",
            hyperparameters={"load_in_4bit": False},
            timestamp=FIXED_TS,
        )
        data = json.loads((adapter_dir / "training_metadata.json").read_text(encoding="utf-8"))
        assert data["resolved"]["load_in_4bit"] is True

    def test_lora_run_resolves_to_the_configured_flag(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        write_metadata(
            adapter_dir,
            model="m",
            method="lora",
            dataset_path="hf:my-org/my-dataset",
            hyperparameters={"load_in_4bit": False},
            timestamp=FIXED_TS,
        )
        data = json.loads((adapter_dir / "training_metadata.json").read_text(encoding="utf-8"))
        assert data["resolved"]["load_in_4bit"] is False

    def test_lora_run_with_the_flag_on_resolves_to_true(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        write_metadata(
            adapter_dir,
            model="m",
            method="lora",
            dataset_path="hf:my-org/my-dataset",
            hyperparameters={"load_in_4bit": True},
            timestamp=FIXED_TS,
        )
        data = json.loads((adapter_dir / "training_metadata.json").read_text(encoding="utf-8"))
        assert data["resolved"]["load_in_4bit"] is True
