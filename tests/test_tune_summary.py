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

from sloth.tune.metrics import EVAL_JSON_DIR
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
    """A flat legacy eval.json (no eval/ dir) reads as the single 'legacy' suite."""
    adapter = tmp_path / "adapter"
    _write_eval_json(adapter, _EVAL_PAYLOAD)

    suites = read_eval(adapter)

    assert set(suites) == {"legacy"}
    assert suites["legacy"]["exact_match_pct"] == 75.0
    assert len(suites["legacy"]["files"]) == 2


def test_read_eval_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    assert read_eval(adapter) == {}


def test_read_eval_corrupt_legacy_file_is_skipped(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "eval.json").write_text("not json", encoding="utf-8")

    assert read_eval(adapter) == {}


def _write_suite_eval_json(
    adapter: Path, suite: str, payload: dict, *, batch_size: int | None = 8
) -> Path:
    eval_dir = adapter / EVAL_JSON_DIR
    eval_dir.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record["schema_version"] = 2
    record["suite"] = suite
    record["batch_size"] = batch_size
    record["target"] = "adapter"
    record["written_at"] = "2026-07-06T02:00:00+00:00"
    record["base_load_in_4bit"] = True
    path = eval_dir / f"{suite}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def test_read_eval_reads_suite_keyed_files(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    _write_suite_eval_json(adapter, "holdout", _EVAL_PAYLOAD)

    suites = read_eval(adapter)

    assert set(suites) == {"holdout"}
    assert suites["holdout"]["batch_size"] == 8
    assert suites["holdout"]["base_load_in_4bit"] is True


def test_read_eval_mixed_dir_reads_both_suite_and_legacy(tmp_path: Path) -> None:
    """h19/c33: a run dir with BOTH the new eval/<suite>.json layout and an
    old flat eval.json (predating suite-keying) reports both."""
    adapter = tmp_path / "adapter"
    _write_suite_eval_json(adapter, "holdout", _EVAL_PAYLOAD)
    _write_eval_json(adapter, _EVAL_PAYLOAD)

    suites = read_eval(adapter)

    assert set(suites) == {"holdout", "legacy"}


def test_read_eval_corrupt_suite_file_is_skipped(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    eval_dir = adapter / EVAL_JSON_DIR
    eval_dir.mkdir(parents=True)
    (eval_dir / "holdout.json").write_text("not json", encoding="utf-8")

    assert read_eval(adapter) == {}


def test_build_summary_includes_eval_block(tmp_path: Path) -> None:
    """Legacy-only run dir (predates suite-keying, e.g. a072b88): summarizes
    with unchanged top-level exact_match_pct/f1 (h19), as the sole 'legacy'
    suite."""
    output_dir = tmp_path / "adapter"
    _write_eval_json(output_dir, _EVAL_PAYLOAD)

    summary = build_summary(output_dir)

    assert summary["eval"]["exact_match_pct"] == 75.0
    assert summary["eval"]["f1"] == 0.82
    assert set(summary["eval"]["suites"]) == {"legacy"}
    assert summary["eval"]["suites"]["legacy"]["exact_match_pct"] == 75.0
    assert summary["eval"]["suites"]["legacy"]["f1"] == 0.82
    assert summary["eval"]["suites"]["legacy"]["file_count"] == 2
    assert summary["notes"] == [
        "no training_metadata.json found — metadata omitted",
        "no checkpoint-N directory found — no trainer_state.json to read",
    ]


def test_build_summary_mixed_eval_reports_both_suites(tmp_path: Path) -> None:
    output_dir = tmp_path / "adapter"
    _write_suite_eval_json(output_dir, "holdout", _EVAL_PAYLOAD)
    _write_eval_json(output_dir, _EVAL_PAYLOAD)

    summary = build_summary(output_dir)

    assert set(summary["eval"]["suites"]) == {"holdout", "legacy"}


def test_build_summary_prefers_target_suite_for_top_level_fields(tmp_path: Path) -> None:
    output_dir = tmp_path / "adapter"
    other_payload = dict(_EVAL_PAYLOAD)
    other_payload["exact_match_pct"] = 10.0
    other_payload["f1"] = 0.1
    _write_suite_eval_json(output_dir, "holdout", other_payload)
    target_payload = dict(_EVAL_PAYLOAD)
    target_payload["exact_match_pct"] = 99.0
    target_payload["f1"] = 0.99
    _write_suite_eval_json(output_dir, "target", target_payload)

    summary = build_summary(output_dir)

    assert summary["eval"]["exact_match_pct"] == 99.0
    assert summary["eval"]["f1"] == 0.99


def test_build_summary_falls_back_to_first_suite_when_no_target(tmp_path: Path) -> None:
    output_dir = tmp_path / "adapter"
    alpha_payload = dict(_EVAL_PAYLOAD)
    alpha_payload["exact_match_pct"] = 11.0
    alpha_payload["f1"] = 0.11
    _write_suite_eval_json(output_dir, "alpha", alpha_payload)
    beta_payload = dict(_EVAL_PAYLOAD)
    beta_payload["exact_match_pct"] = 22.0
    beta_payload["f1"] = 0.22
    _write_suite_eval_json(output_dir, "beta", beta_payload)

    summary = build_summary(output_dir)

    # glob-sorted: "alpha" < "beta" -> alpha is the first/only-fallback suite.
    assert summary["eval"]["exact_match_pct"] == 11.0
    assert summary["eval"]["f1"] == 0.11


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
    assert export_eval["exact_match_pct"] == 75.0
    assert export_eval["f1"] == 0.82
    assert export_eval["suites"]["legacy"]["file_count"] == 2


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


def test_read_eval_invalid_utf8_returns_empty_dict(tmp_path: Path) -> None:
    """qodo finding: read_eval must tolerate invalid UTF-8 (UnicodeDecodeError)
    exactly like it tolerates OSError/JSON errors — never raise."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "eval.json").write_bytes(b"\xff\xfe\x00invalid-utf8")

    assert read_eval(adapter) == {}


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
