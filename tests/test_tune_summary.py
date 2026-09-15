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

Also covers (t6): ``build_summary`` reading ``eval.json`` (written by
``sloth eval``, see :func:`sloth.tune.metrics.write_eval_json`) — both at the
run's own output dir and at each discovered export's own output dir — into
an ``"eval"`` block ``{exact_match_pct, f1, file_count}``, omitted silently
(``None``, no note, no key on an export entry) when absent.
"""

from __future__ import annotations

import json
from pathlib import Path

from sloth.tune.summary import build_summary, discover_exports, read_eval

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


# ---------------------------------------------------------------------------
# eval.json discovery (t6)
# ---------------------------------------------------------------------------

_EVAL_PAYLOAD = {
    "total": 4,
    "exact_match": 3,
    "exact_match_pct": 75.0,
    "f1": 0.82,
    "results": [{"input": "x", "expected": "y", "actual": "y", "exact_match": True, "f1": 1.0}],
    "files": [
        {
            "path": "suite_a.jsonl",
            "total": 2,
            "exact_match": 2,
            "exact_match_pct": 100.0,
            "f1": 1.0,
            "results": [],
        },
        {
            "path": "suite_b.jsonl",
            "total": 2,
            "exact_match": 1,
            "exact_match_pct": 50.0,
            "f1": 0.64,
            "results": [],
        },
    ],
    "suite_paths": ["suite_a.jsonl", "suite_b.jsonl"],
    "target": "adapter",
    "written_at": "2026-07-06T02:00:00+00:00",
}


def _write_eval_json(directory: Path, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "eval.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_read_eval_reads_parsed_payload(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    _write_eval_json(adapter, _EVAL_PAYLOAD)

    payload = read_eval(adapter)

    assert payload is not None
    assert payload["exact_match_pct"] == 75.0
    assert len(payload["files"]) == 2


def test_read_eval_missing_file_returns_none(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    assert read_eval(adapter) is None


def test_read_eval_corrupt_file_returns_none(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "eval.json").write_text("not json", encoding="utf-8")

    assert read_eval(adapter) is None


def test_build_summary_includes_eval_block(tmp_path: Path) -> None:
    output_dir = tmp_path / "adapter"
    _write_eval_json(output_dir, _EVAL_PAYLOAD)

    summary = build_summary(output_dir)

    assert summary["eval"] == {"exact_match_pct": 75.0, "f1": 0.82, "file_count": 2}
    assert summary["notes"] == [
        "no training_metadata.json found — metadata omitted",
        "no checkpoint-N directory found — no trainer_state.json to read",
    ]


def test_build_summary_eval_none_when_absent_no_note(tmp_path: Path) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()

    summary = build_summary(output_dir)

    assert summary["eval"] is None
    assert not any("eval" in note for note in summary["notes"])


def test_build_summary_attaches_eval_to_export_entry(tmp_path: Path) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    record_a = _record(_RECORD_A, output_dir)
    export_dir = output_dir / "gguf-export"
    _write_export_json(export_dir, record_a)
    _write_eval_json(export_dir, _EVAL_PAYLOAD)

    summary = build_summary(output_dir)

    assert len(summary["exports"]) == 1
    export_eval = summary["exports"][0]["eval"]
    assert export_eval == {"exact_match_pct": 75.0, "f1": 0.82, "file_count": 2}


def test_build_summary_export_entry_omits_eval_key_when_absent(tmp_path: Path) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    record_a = _record(_RECORD_A, output_dir)
    export_dir = output_dir / "gguf-export"
    _write_export_json(export_dir, record_a)

    summary = build_summary(output_dir)

    assert len(summary["exports"]) == 1
    assert "eval" not in summary["exports"][0]


def test_build_summary_export_index_with_int_output_is_skipped(tmp_path: Path) -> None:
    """qodo finding: a malformed export index entry whose 'output' is an int
    (not a str/PathLike) must be skipped rather than reaching Path() and
    raising TypeError."""
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    entry = dict(_record(_RECORD_A, output_dir))
    entry["output"] = 12345
    _write_exports_index(output_dir, [entry])

    summary = build_summary(output_dir)

    assert len(summary["exports"]) == 1
    assert "eval" not in summary["exports"][0]


def test_build_summary_export_index_with_list_output_is_skipped(tmp_path: Path) -> None:
    """Same as above but with a list 'output' value."""
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    entry = dict(_record(_RECORD_B, output_dir))
    entry["output"] = ["not", "a", "path"]
    _write_exports_index(output_dir, [entry])

    summary = build_summary(output_dir)

    assert len(summary["exports"]) == 1
    assert "eval" not in summary["exports"][0]


def test_read_eval_invalid_utf8_returns_none(tmp_path: Path) -> None:
    """qodo finding: read_eval must tolerate invalid UTF-8 (UnicodeDecodeError)
    exactly like it tolerates OSError/JSON errors — never raise."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "eval.json").write_bytes(b"\xff\xfe\x00invalid-utf8")

    assert read_eval(adapter) is None


def test_build_summary_invalid_utf8_eval_at_run_level_degrades(tmp_path: Path) -> None:
    """Invalid UTF-8 in the run's own eval.json must not crash build_summary;
    the eval block just degrades to None."""
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    (output_dir / "eval.json").write_bytes(b"\xff\xfe\x00invalid-utf8")

    summary = build_summary(output_dir)

    assert summary["eval"] is None


def test_build_summary_invalid_utf8_eval_at_export_level_degrades(tmp_path: Path) -> None:
    """Invalid UTF-8 in an export's own eval.json must not crash build_summary;
    that export entry just has no 'eval' key."""
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    record_a = _record(_RECORD_A, output_dir)
    export_dir = output_dir / "gguf-export"
    _write_export_json(export_dir, record_a)
    (export_dir / "eval.json").write_bytes(b"\xff\xfe\x00invalid-utf8")

    summary = build_summary(output_dir)

    assert len(summary["exports"]) == 1
    assert "eval" not in summary["exports"][0]
