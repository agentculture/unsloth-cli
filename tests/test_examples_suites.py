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

import json
import sys
from pathlib import Path

import pytest

from sloth.tune.datasets import overlap_check, validate_suite

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"
EVAL_DIR = EXAMPLES / "eval"

sys.path.insert(0, str(EXAMPLES))

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
