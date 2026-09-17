"""Tests for :mod:`sloth.tune.metrics` — the open metrics dict + suite-keyed eval.json.

Covers the t1 acceptance criteria from ``docs/plans/`` (full-benchmark-suite,
task t1):

* :func:`score_records` returns an **open** per-row dict — an ``extra_metrics``
  callable/mapping can add a brand-new numeric metric key, and
  :func:`summarize` / :func:`aggregate` fold it into the summary generically,
  without this module ever naming it.
* :func:`write_eval_json` supports the new suite-keyed call shape
  (``write_eval_json(directory, suite, payload, batch_size=...)``), writing
  ``eval/<suite>.json`` with ``schema_version=2``, ``suite``, ``batch_size``,
  ``target``, ``written_at`` and ``base_load_in_4bit`` — and a second call
  with a different suite leaves the first file intact.
* The legacy two-positional-argument call shape
  (``write_eval_json(directory, payload, *, suite_paths=..., target=...)``)
  keeps writing ``eval.json`` unchanged, so ``_trainer.py``/``_exporter.py``
  never had to change.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from sloth.tune import metrics

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _records() -> list[dict[str, str]]:
    return [
        {"task": "t", "input": "i1", "expected_output": "a b c"},
        {"task": "t", "input": "i2", "expected_output": "x y z"},
    ]


# ---------------------------------------------------------------------------
# score_records is an open dict — extra metrics fold into the aggregate
# ---------------------------------------------------------------------------


class TestOpenMetricsDict:
    def test_score_records_with_no_extra_metrics_keeps_legacy_shape(self) -> None:
        scored = metrics.score_records(_records(), ["a b c", "x y q"])
        assert set(scored[0]) == {
            "index",
            "task",
            "input",
            "expected_output",
            "prediction",
            "exact_match",
            "f1",
        }

    def test_extra_metrics_callable_adds_a_new_key_per_row(self) -> None:
        def fake_metric(record: dict, prediction: str) -> dict:
            return {"length_ratio": len(prediction) / max(len(record["expected_output"]), 1)}

        scored = metrics.score_records(_records(), ["a b c", "x y"], extra_metrics=fake_metric)
        assert "length_ratio" in scored[0]
        assert "length_ratio" in scored[1]

    def test_extra_metrics_mapping_of_callables_adds_a_new_key_per_row(self) -> None:
        extra = {"always_one": lambda record, prediction: 1.0}
        scored = metrics.score_records(_records(), ["a b c", "x y z"], extra_metrics=extra)
        assert scored[0]["always_one"] == 1.0
        assert scored[1]["always_one"] == 1.0

    def test_fake_metric_registered_in_a_test_appears_in_summarize(self) -> None:
        """A brand-new numeric metric name, never known to metrics.py, is folded in."""
        extra = {"banana_score": lambda record, prediction: 3.0}
        scored = metrics.score_records(_records(), ["a b c", "x y q"], extra_metrics=extra)
        summary = metrics.summarize(scored)
        assert summary["banana_score"] == 3.0
        # The named legacy fields are untouched.
        assert summary["exact_match_pct"] == 50.0
        assert "f1" in summary

    def test_fake_metric_registered_in_a_test_appears_in_aggregate(self) -> None:
        """The acceptance criterion, verbatim: a fake metric shows up in aggregate()."""
        extra = {"banana_score": lambda record, prediction: 5.0}
        scored_a = metrics.score_records(
            [_records()[0]], ["a b c"], extra_metrics=extra, source="a.jsonl"
        )
        scored_b = metrics.score_records(
            [_records()[1]], ["x y z"], extra_metrics=extra, source="b.jsonl", start_index=1
        )
        files = [metrics.file_entry("a.jsonl", scored_a), metrics.file_entry("b.jsonl", scored_b)]
        result = metrics.aggregate(files)
        assert result["banana_score"] == 5.0
        # Per-file entries fold the metric too, since file_entry -> summarize.
        assert files[0]["banana_score"] == 5.0

    def test_extra_metric_mean_is_computed_across_varying_values(self) -> None:
        extra = {"score": lambda record, prediction: 1.0 if "a" in prediction else 0.0}
        scored = metrics.score_records(_records(), ["a b c", "x y z"], extra_metrics=extra)
        summary = metrics.summarize(scored)
        assert summary["score"] == 0.5

    def test_summarize_ignores_non_numeric_extra_values(self) -> None:
        """A non-numeric extra field (e.g. a debug string) is not folded as a mean."""
        extra = {"note": lambda record, prediction: "some debug string"}
        scored = metrics.score_records(_records(), ["a b c", "x y z"], extra_metrics=extra)
        summary = metrics.summarize(scored)
        assert "note" not in summary

    def test_summarize_on_empty_results_has_no_extra_keys(self) -> None:
        summary = metrics.summarize([])
        assert summary["total"] == 0
        assert summary["exact_match_pct"] == 0.0
        assert summary["f1"] == 0.0


# ---------------------------------------------------------------------------
# write_eval_json — new suite-keyed shape
# ---------------------------------------------------------------------------


class TestWriteEvalJsonSuiteKeyed:
    def _payload(self) -> dict:
        return {"total": 2, "exact_match": 1, "exact_match_pct": 50.0, "f1": 0.833}

    def test_writes_eval_dir_suite_json(self, tmp_path: Path) -> None:
        destination = metrics.write_eval_json(
            tmp_path, "my-suite", self._payload(), batch_size=4, target="adapter"
        )
        assert destination == tmp_path / "eval" / "my-suite.json"
        assert destination.exists()

    def test_written_file_carries_the_required_schema_fields(self, tmp_path: Path) -> None:
        destination = metrics.write_eval_json(
            tmp_path,
            "suite-a",
            self._payload(),
            batch_size=8,
            target="adapter",
            base_load_in_4bit=True,
        )
        written = json.loads(destination.read_text(encoding="utf-8"))
        assert written["schema_version"] == 2
        assert written["suite"] == "suite-a"
        assert written["batch_size"] == 8
        assert written["target"] == "adapter"
        assert written["base_load_in_4bit"] is True
        assert written["written_at"].startswith("20")
        # Original payload keys survive.
        assert written["total"] == 2
        assert written["f1"] == 0.833

    def test_suite_name_is_sanitised_to_lowercase_alnum_dash(self, tmp_path: Path) -> None:
        destination = metrics.write_eval_json(
            tmp_path, "My Suite_v2!!.jsonl", self._payload(), batch_size=1
        )
        assert destination.name == "my-suite-v2.json"

    def test_suite_name_from_a_full_path_uses_the_stem(self, tmp_path: Path) -> None:
        suite_path = tmp_path / "suites" / "Reverse_Task.jsonl"
        destination = metrics.write_eval_json(tmp_path, suite_path, self._payload(), batch_size=1)
        assert destination.name == "reverse-task.json"

    def test_second_suite_leaves_the_first_file_intact(self, tmp_path: Path) -> None:
        first = metrics.write_eval_json(tmp_path, "suite-one", self._payload(), batch_size=2)
        first_contents = first.read_text(encoding="utf-8")

        second_payload = {"total": 5, "exact_match": 5, "exact_match_pct": 100.0, "f1": 1.0}
        second = metrics.write_eval_json(tmp_path, "suite-two", second_payload, batch_size=2)

        assert first.exists()
        assert second.exists()
        assert first != second
        assert first.read_text(encoding="utf-8") == first_contents
        written_first = json.loads(first.read_text(encoding="utf-8"))
        written_second = json.loads(second.read_text(encoding="utf-8"))
        assert written_first["suite"] == "suite-one"
        assert written_second["suite"] == "suite-two"
        assert written_first["total"] == 2
        assert written_second["total"] == 5

    def test_rewriting_the_same_suite_overwrites_only_that_file(self, tmp_path: Path) -> None:
        metrics.write_eval_json(tmp_path, "suite-one", self._payload(), batch_size=2)
        metrics.write_eval_json(tmp_path, "suite-two", self._payload(), batch_size=2)
        updated_payload = {"total": 9, "exact_match": 9, "exact_match_pct": 100.0, "f1": 1.0}
        metrics.write_eval_json(tmp_path, "suite-one", updated_payload, batch_size=2)

        written_one = json.loads((tmp_path / "eval" / "suite-one.json").read_text())
        written_two = json.loads((tmp_path / "eval" / "suite-two.json").read_text())
        assert written_one["total"] == 9
        assert written_two["total"] == 2


# ---------------------------------------------------------------------------
# write_eval_json — legacy shape stays byte-for-byte compatible
# ---------------------------------------------------------------------------


class TestWriteEvalJsonLegacyShape:
    def test_legacy_two_positional_call_writes_eval_json(self, tmp_path: Path) -> None:
        payload = {"total": 1, "exact_match": 1, "exact_match_pct": 100.0, "f1": 1.0}
        destination = metrics.write_eval_json(
            tmp_path, payload, suite_paths=["a.jsonl"], target="adapter"
        )
        assert destination == tmp_path / metrics.EVAL_JSON_NAME
        written = json.loads(destination.read_text(encoding="utf-8"))
        # Recorded canonically (Qodo r5) so a later reader in another cwd finds it.
        assert written["suite_paths"] == [str(Path("a.jsonl").resolve())]
        assert written["target"] == "adapter"
        assert written["written_at"].startswith("20")
        # No new-schema fields leak into the legacy shape.
        assert "schema_version" not in written
        assert "suite" not in written
        assert "batch_size" not in written

    def test_legacy_call_overwrites_in_place_as_before(self, tmp_path: Path) -> None:
        first_payload = {"total": 1, "exact_match": 1, "exact_match_pct": 100.0, "f1": 1.0}
        metrics.write_eval_json(tmp_path, first_payload, suite_paths=["a.jsonl"], target="adapter")
        second_payload = {"total": 2, "exact_match": 1, "exact_match_pct": 50.0, "f1": 0.5}
        metrics.write_eval_json(tmp_path, second_payload, suite_paths=["b.jsonl"], target="model")

        written = json.loads((tmp_path / metrics.EVAL_JSON_NAME).read_text(encoding="utf-8"))
        assert written["total"] == 2
        assert written["suite_paths"] == [str(Path("b.jsonl").resolve())]
        assert written["target"] == "model"

    def test_eval_json_name_constant_is_still_exported_and_unchanged(self) -> None:
        assert metrics.EVAL_JSON_NAME == "eval.json"


# ---------------------------------------------------------------------------
# sanitize_suite_name
# ---------------------------------------------------------------------------


class TestSanitizeSuiteName:
    def test_lowercases_and_replaces_invalid_chars(self) -> None:
        assert metrics.sanitize_suite_name("My Suite!!") == "my-suite"

    def test_collapses_runs_of_invalid_chars(self) -> None:
        assert metrics.sanitize_suite_name("a___b") == "a-b"

    def test_strips_leading_and_trailing_dashes(self) -> None:
        assert metrics.sanitize_suite_name("--weird--") == "weird"

    def test_uses_path_stem_not_full_path(self) -> None:
        assert metrics.sanitize_suite_name(Path("/a/b/c/Suite.jsonl")) == "suite"

    def test_falls_back_to_suite_for_all_punctuation_input(self) -> None:
        assert metrics.sanitize_suite_name("!!!") == "suite"

    def test_already_clean_name_is_unchanged(self) -> None:
        assert metrics.sanitize_suite_name("reverse-task-v2") == "reverse-task-v2"


# ---------------------------------------------------------------------------
# Existing legacy behaviour untouched (score_records / summarize / file_entry)
# ---------------------------------------------------------------------------


class TestLegacyBehaviourUnchanged:
    def test_summarize_reports_totals_and_mean_f1(self) -> None:
        scored = metrics.score_records(_records(), ["a b c", "x y q"])
        summary = metrics.summarize(scored)
        assert summary["total"] == 2
        assert summary["exact_match"] == 1
        assert summary["exact_match_pct"] == 50.0
        assert round(summary["f1"], 3) == 0.833

    def test_score_records_indices_continue_across_files(self) -> None:
        scored = metrics.score_records(
            [{"task": "t", "input": "i", "expected_output": "a"}],
            ["a"],
            start_index=7,
            source="s.jsonl",
        )
        assert scored[0]["index"] == 7
        assert scored[0]["file"] == "s.jsonl"

    def test_file_entry_carries_path_and_scores(self) -> None:
        scored = metrics.score_records(_records(), ["a b c", "x y z"])
        entry = metrics.file_entry("s.jsonl", scored)
        assert entry["path"] == "s.jsonl"
        assert entry["total"] == 2
        assert entry["results"] == scored

    def test_aggregate_concatenates_results_and_files(self) -> None:
        scored_a = metrics.score_records([_records()[0]], ["a b c"])
        scored_b = metrics.score_records([_records()[1]], ["x y z"])
        files = [metrics.file_entry("a.jsonl", scored_a), metrics.file_entry("b.jsonl", scored_b)]
        result = metrics.aggregate(files)
        assert result["total"] == 2
        assert len(result["results"]) == 2
        assert [f["path"] for f in result["files"]] == ["a.jsonl", "b.jsonl"]


# ---------------------------------------------------------------------------
# Module stays pure stdlib (acceptance criterion #3)
# ---------------------------------------------------------------------------


def test_metrics_module_imports_only_stdlib() -> None:
    """metrics.py must not gain a single non-stdlib top-level import."""
    source = (_REPO_ROOT / "sloth" / "tune" / "metrics.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    non_stdlib = {r for r in roots if r not in sys.stdlib_module_names}
    assert not non_stdlib, f"metrics.py imports non-stdlib modules: {sorted(non_stdlib)}"


class TestResultPathsAreCanonicalAtWriteTime:
    """Qodo r5: a result file must record suite paths that survive a later
    ``sloth compare`` run from a different working directory."""

    def test_suite_keyed_shape_resolves_file_paths(self, tmp_path: Path, monkeypatch) -> None:
        suite = tmp_path / "regression.jsonl"
        suite.write_text('{"task": "t", "input": "i", "expected_output": "o"}\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        destination = metrics.write_eval_json(
            tmp_path,
            "regression",
            {"total": 1, "path": "regression.jsonl", "files": [{"path": "regression.jsonl"}]},
            batch_size=1,
        )
        record = json.loads(destination.read_text(encoding="utf-8"))
        assert record["files"][0]["path"] == str(suite.resolve())
        assert record["path"] == str(suite.resolve())

    def test_legacy_shape_resolves_suite_paths(self, tmp_path: Path, monkeypatch) -> None:
        suite = tmp_path / "regression.jsonl"
        suite.write_text('{"task": "t", "input": "i", "expected_output": "o"}\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        destination = metrics.write_eval_json(
            tmp_path, {"total": 1}, suite_paths=["regression.jsonl"], target="adapter"
        )
        record = json.loads(destination.read_text(encoding="utf-8"))
        assert record["suite_paths"] == [str(suite.resolve())]
