"""Tests for the shipped example suites and the demo corpus.

Covers the three things that make ``examples/`` trustworthy:

1. every generated suite file validates under its declared schema and meets its
   minimum row count;
2. the demo corpus shares no scorable ``(prompt, expected_output)`` pair with
   any eval suite (``overlap_check`` returns an empty list);
3. ``examples/generate_suites.py`` is deterministic — regenerating into a temp
   directory reproduces the committed bytes exactly.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pytest

from sloth.tune.datasets import overlap_check, validate_suite

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"
EVAL_DIR = EXAMPLES / "eval"

sys.path.insert(0, str(EXAMPLES))

import generate_mmlu_subset  # noqa: E402  (path-dependent import of the committed generator)
import generate_suites  # noqa: E402  (path-dependent import of the committed generator)

#: (path, schema, minimum row count) for every generated example file.
GENERATED = [
    (EVAL_DIR / "regression.jsonl", "task", 100),
    (EVAL_DIR / "instruction-following.jsonl", "instruction", 50),
    (EVAL_DIR / "structured-output.jsonl", "structured", 50),
    (EVAL_DIR / "tool-call.jsonl", "toolcall", 30),
    (EXAMPLES / "demo-corpus.jsonl", "chat", 500),
]

#: The three hand-authored suites that predate this task.
HAND_AUTHORED = [
    EVAL_DIR / "agentculture-terms.jsonl",
    EVAL_DIR / "cli-contract.jsonl",
    EVAL_DIR / "task-format.jsonl",
]


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.mark.parametrize("path, schema, minimum", GENERATED, ids=lambda v: getattr(v, "name", v))
def test_generated_file_validates_and_meets_minimum(path: Path, schema: str, minimum: int) -> None:
    """Each generated file parses under its schema and carries enough rows."""
    report = validate_suite(path, schema=schema)
    assert report["total_records"] >= minimum, f"{path.name} has too few rows"


def test_hand_authored_suites_still_validate_as_task_schema() -> None:
    """The pre-existing suites are untouched and still task-schema."""
    for path in HAND_AUTHORED:
        assert validate_suite(path, schema="task")["total_records"] > 0


def test_instruction_suite_has_refusal_rows() -> None:
    """At least ten rows carry a ``must_refuse`` constraint."""
    rows = _read_rows(EVAL_DIR / "instruction-following.jsonl")
    refusals = [
        row
        for row in rows
        if any("must_refuse" in constraint for constraint in row.get("constraints", []))
    ]
    assert len(refusals) >= 10


def test_instruction_suite_uses_every_constraint_kind() -> None:
    """All six constraint keys appear somewhere in the suite."""
    rows = _read_rows(EVAL_DIR / "instruction-following.jsonl")
    seen = {key for row in rows for constraint in row["constraints"] for key in constraint}
    assert seen == {
        "must_refuse",
        "max_words",
        "min_words",
        "must_contain",
        "must_not_contain",
        "json_only",
    }


def test_tool_call_inputs_name_their_tool() -> None:
    """Each tool-call row states the tool it expects, so a base model can attempt it."""
    rows = _read_rows(EVAL_DIR / "tool-call.jsonl")
    for row in rows:
        assert row["expected_tool_call"]["name"] in row["input"]
        assert isinstance(row["expected_tool_call"]["arguments"], dict)


def test_demo_corpus_does_not_overlap_any_suite() -> None:
    """The corpus teaches the suites without reproducing a single scored row."""
    suites = HAND_AUTHORED + [path for path, _, _ in GENERATED[:4]]
    assert overlap_check(EXAMPLES / "demo-corpus.jsonl", suites) == []


def test_generator_is_deterministic(tmp_path: Path) -> None:
    """Two regenerations agree with each other and with the committed files."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    counts_a = generate_suites.generate(first)
    counts_b = generate_suites.generate(second)
    assert counts_a == counts_b

    for relative in generate_suites.BUILDERS:
        generated = (first / relative).read_bytes()
        assert generated == (second / relative).read_bytes()
        assert (
            generated == (EXAMPLES / relative).read_bytes()
        ), f"{relative} is stale — rerun `uv run python examples/generate_suites.py`"


# ---------------------------------------------------------------------------
# mmlu-subset.jsonl — the committed, offline MMLU-*style* letter-choice suite
# ---------------------------------------------------------------------------

MMLU_SUBSET = EVAL_DIR / "mmlu-subset.jsonl"


def test_mmlu_subset_validates_and_is_large_enough() -> None:
    """The subset is a task-schema suite of at least 100 rows."""
    report = validate_suite(MMLU_SUBSET, schema="task")
    assert report["total_records"] >= 100


def test_mmlu_subset_spans_at_least_ten_subjects() -> None:
    """A single-subject file would not exercise MMLU's breadth."""
    rows = _read_rows(MMLU_SUBSET)
    assert len({row["task"] for row in rows}) >= 10


def test_mmlu_subset_rows_are_lettered_choices() -> None:
    """Every row offers A-D options, asks for a letter, and expects one."""
    for row in _read_rows(MMLU_SUBSET):
        assert row["expected_output"] in {"A", "B", "C", "D"}
        assert row["input"].rstrip().endswith("Answer with the letter.")
        for letter in ("A", "B", "C", "D"):
            assert f"\n{letter}. " in row["input"]


def test_mmlu_subset_answer_key_is_balanced() -> None:
    """No letter is over-represented, so guessing a fixed letter scores chance."""
    rows = _read_rows(MMLU_SUBSET)
    counts = collections.Counter(row["expected_output"] for row in rows)
    assert set(counts) == {"A", "B", "C", "D"}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_mmlu_subset_scores_with_letter_choice_extraction() -> None:
    """The suite scores through the ordinary eval path on the *letter* picked.

    ``sloth eval --suite examples/eval/mmlu-subset.jsonl`` detects an all-letter
    suite and adds ``choice_match`` / ``choice_acc_pct``: ``"Answer: B"`` counts
    as a hit even though it is not an exact-match string.
    """
    from sloth.tune import metrics
    from sloth.tune._trainer import _extra_metrics_for, build_file_entry, is_choice_suite

    rows = _read_rows(MMLU_SUBSET)[:4]
    assert is_choice_suite(rows)
    extra = _extra_metrics_for("task", records=rows)
    predictions = [f"Answer: {row['expected_output']}" for row in rows[:3]] + ["Z"]
    scored = metrics.score_records(rows, predictions, extra_metrics=extra)

    assert [row["choice_match"] for row in scored] == [True, True, True, False]
    assert build_file_entry(MMLU_SUBSET, scored)["choice_acc_pct"] == 75.0


def test_mmlu_subset_does_not_overlap_the_demo_corpus() -> None:
    """Nothing the demo corpus trains on is scored by the subset."""
    assert overlap_check(EXAMPLES / "demo-corpus.jsonl", [MMLU_SUBSET]) == []


def test_mmlu_subset_generator_is_deterministic(tmp_path: Path) -> None:
    """Two regenerations agree with each other and with the committed file."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert generate_mmlu_subset.generate(first) == generate_mmlu_subset.generate(second)

    relative = generate_mmlu_subset.RELATIVE_PATH
    generated = (first / relative).read_bytes()
    assert generated == (second / relative).read_bytes()
    assert (
        generated == (EXAMPLES / relative).read_bytes()
    ), f"{relative} is stale \u2014 rerun `uv run python examples/generate_mmlu_subset.py`"
