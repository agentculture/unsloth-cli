"""Tests for :mod:`sloth.tune.summary` export discovery (t10).

Covers:
* ``build_summary`` discovers ``<output_dir>/exports.json`` (the index
  ``sloth.tune._exporter._append_index`` writes) and returns its entries
  under an ``"exports"`` key.
* ``build_summary`` also discovers any ``export.json`` file nested under the
  run dir (the per-export record ``_write_export_json`` writes into each
  export's own output directory), even when it is not (yet, or ever)
  mirrored into ``exports.json``.
* Entries already present in ``exports.json`` are not duplicated when the
  matching ``export.json`` is also found on disk.
* A missing/corrupt ``exports.json`` degrades to an empty list plus a note,
  never an exception.
* A run with no exports at all returns an empty list (no note necessary).
"""

from __future__ import annotations

import json
from pathlib import Path

from sloth.tune.summary import build_summary, discover_exports

_RECORD_A = {
    "format": "gguf",
    "quant": ["q4_k_m"],
    "base": "unsloth/Qwen3-4B",
    "adapter": "ADAPTER_DIR",
    "files": {"model.Q4_K_M.gguf": 123456},
    "calibration": None,
    "versions": {"unsloth": "2026.7.1"},
    "timestamp": "2026-07-06T00:00:00+00:00",
}

_RECORD_B = {
    "format": "awq",
    "quant": ["int4"],
    "base": "unsloth/Qwen3-4B",
    "adapter": "ADAPTER_DIR",
    "files": {"model.safetensors": 999, "config.json": 42},
    "calibration": {"source": "train.jsonl", "count": 128},
    "versions": {"llmcompressor": "0.11.0"},
    "timestamp": "2026-07-06T01:00:00+00:00",
}


def _record(base: dict, adapter: Path) -> dict:
    rec = dict(base)
    rec["adapter"] = str(adapter)
    return rec


def _write_exports_index(adapter: Path, records: list[dict]) -> None:
    (adapter / "exports.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def _write_export_json(export_dir: Path, record: dict) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / "export.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# exports.json discovery
# ---------------------------------------------------------------------------


def test_discover_exports_reads_exports_json_index(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    record_a = _record(_RECORD_A, adapter)
    record_b = _record(_RECORD_B, adapter)
    _write_exports_index(adapter, [record_a, record_b])

    exports, notes = discover_exports(adapter)

    assert notes == []
    assert len(exports) == 2
    formats = {e["format"] for e in exports}
    assert formats == {"gguf", "awq"}


def test_discover_exports_no_index_no_files_is_empty(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    exports, notes = discover_exports(adapter)

    assert exports == []
    assert notes == []


def test_discover_exports_corrupt_index_degrades_with_note(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "exports.json").write_text("not json", encoding="utf-8")

    exports, notes = discover_exports(adapter)

    assert exports == []
    assert any("exports.json" in note for note in notes)


def test_discover_exports_index_not_a_list_degrades_with_note(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "exports.json").write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    exports, notes = discover_exports(adapter)

    assert exports == []
    assert notes


# ---------------------------------------------------------------------------
# export.json discovery under the run dir
# ---------------------------------------------------------------------------


def test_discover_exports_finds_export_json_under_run_dir(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    record_a = _record(_RECORD_A, adapter)
    export_dir = adapter / "gguf-export" / ".partial"
    export_json = _write_export_json(export_dir, record_a)

    exports, notes = discover_exports(adapter)

    assert len(exports) == 1
    assert exports[0]["format"] == "gguf"
    assert exports[0]["export_json"] == str(export_json)
    assert exports[0]["output"] == str(export_dir)
    assert notes == []


def test_discover_exports_does_not_duplicate_indexed_export(tmp_path: Path) -> None:
    """When a record already appears in exports.json AND its export.json is
    also found nested under the run dir, it must appear exactly once."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    record_a = _record(_RECORD_A, adapter)
    export_dir = adapter / "gguf-export"
    export_json = _write_export_json(export_dir, record_a)
    _write_exports_index(adapter, [record_a])

    exports, _notes = discover_exports(adapter)

    assert len(exports) == 1
    assert exports[0]["export_json"] == str(export_json)
    assert exports[0]["output"] == str(export_dir)


def test_discover_exports_corrupt_export_json_file_is_skipped_with_note(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    export_dir = adapter / "bad-export"
    export_dir.mkdir(parents=True)
    (export_dir / "export.json").write_text("not json", encoding="utf-8")

    exports, notes = discover_exports(adapter)

    assert exports == []
    assert any("export.json" in note for note in notes)


# ---------------------------------------------------------------------------
# build_summary integration
# ---------------------------------------------------------------------------


def test_build_summary_includes_exports_key(tmp_path: Path) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    record_a = _record(_RECORD_A, output_dir)
    _write_exports_index(output_dir, [record_a])

    summary = build_summary(output_dir)

    assert "exports" in summary
    assert len(summary["exports"]) == 1
    assert summary["exports"][0]["format"] == "gguf"


def test_build_summary_exports_empty_when_none_present(tmp_path: Path) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()

    summary = build_summary(output_dir)

    assert summary["exports"] == []
