"""Tests for sloth.tune.datasets — dataset validation (chat + task JSONL schemas).

Acceptance criteria:
  1. Valid chat and task JSONL pass validation and return parsed records.
  2. Malformed lines raise CliError with the 1-based line number and a remediation hint.
  3. torch is NOT imported as a side-effect of running the validator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sloth.cli._errors import CliError
from sloth.tune.datasets import (
    detect_schema,
    overlap_check,
    split_holdout,
    validate_dataset,
    validate_suite,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_jsonl(tmp_path: Path, records: list[dict], filename: str = "data.jsonl") -> Path:
    p = tmp_path / filename
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Criterion 1 — valid data passes
# ---------------------------------------------------------------------------


class TestValidChatJSONL:
    def test_single_line(self, tmp_path: Path) -> None:
        record = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
        path = write_jsonl(tmp_path, [record])
        result = validate_dataset(path, "chat")
        assert result == [record]

    def test_multiple_lines(self, tmp_path: Path) -> None:
        records = [
            {
                "messages": [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                ]
            },
            {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "q2"}]},
        ]
        path = write_jsonl(tmp_path, records)
        result = validate_dataset(path, "chat")
        assert result == records

    def test_all_valid_roles(self, tmp_path: Path) -> None:
        record = {
            "messages": [
                {"role": "system", "content": "You are a helper."},
                {"role": "user", "content": "Explain X."},
                {"role": "assistant", "content": "Sure."},
            ]
        }
        path = write_jsonl(tmp_path, [record])
        result = validate_dataset(path, "chat")
        assert len(result) == 1

    def test_path_as_string(self, tmp_path: Path) -> None:
        record = {"messages": [{"role": "user", "content": "hi"}]}
        path = write_jsonl(tmp_path, [record])
        result = validate_dataset(str(path), "chat")
        assert result == [record]

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        record = {"messages": [{"role": "user", "content": "hi"}]}
        p = tmp_path / "data.jsonl"
        p.write_text(json.dumps(record) + "\n\n", encoding="utf-8")
        result = validate_dataset(p, "chat")
        assert result == [record]


class TestValidTaskJSONL:
    def test_single_line(self, tmp_path: Path) -> None:
        record = {"task": "summarise", "input": "long text", "expected_output": "short"}
        path = write_jsonl(tmp_path, [record])
        result = validate_dataset(path, "task")
        assert result == [record]

    def test_multiple_lines(self, tmp_path: Path) -> None:
        records = [
            {"task": "t1", "input": "i1", "expected_output": "o1"},
            {"task": "t2", "input": "i2", "expected_output": "o2"},
        ]
        path = write_jsonl(tmp_path, records)
        result = validate_dataset(path, "task")
        assert result == records


# ---------------------------------------------------------------------------
# Criterion 2 — invalid data raises CliError with line number + remediation
# ---------------------------------------------------------------------------


class TestInvalidJSONLine:
    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        p.write_text("not json\n", encoding="utf-8")
        with pytest.raises(CliError) as exc_info:
            validate_dataset(p, "chat")
        err = exc_info.value
        assert err.code == 1
        assert "line 1" in err.message
        assert err.remediation

    def test_invalid_json_on_second_line(self, tmp_path: Path) -> None:
        good = {"messages": [{"role": "user", "content": "hi"}]}
        p = tmp_path / "bad.jsonl"
        p.write_text(json.dumps(good) + "\nnot json\n", encoding="utf-8")
        with pytest.raises(CliError) as exc_info:
            validate_dataset(p, "chat")
        assert "line 2" in exc_info.value.message


class TestChatSchemaViolations:
    def test_missing_messages_key(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, [{"role": "user", "content": "hi"}])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "chat")
        err = exc_info.value
        assert err.code == 1
        assert "line 1" in err.message
        assert err.remediation

    def test_messages_not_a_list(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, [{"messages": "not a list"}])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "chat")
        assert "line 1" in exc_info.value.message

    def test_empty_messages_list(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, [{"messages": []}])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "chat")
        assert "line 1" in exc_info.value.message

    def test_message_missing_role(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, [{"messages": [{"content": "hi"}]}])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "chat")
        assert "line 1" in exc_info.value.message

    def test_message_missing_content(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, [{"messages": [{"role": "user"}]}])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "chat")
        assert "line 1" in exc_info.value.message

    def test_bad_role_value(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, [{"messages": [{"role": "alien", "content": "hi"}]}])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "chat")
        assert "line 1" in exc_info.value.message

    def test_role_not_a_string(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, [{"messages": [{"role": 42, "content": "hi"}]}])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "chat")
        assert "line 1" in exc_info.value.message

    def test_content_not_a_string(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, [{"messages": [{"role": "user", "content": 123}]}])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "chat")
        assert "line 1" in exc_info.value.message

    def test_extra_top_level_keys_are_rejected(self, tmp_path: Path) -> None:
        record = {
            "messages": [{"role": "user", "content": "hi"}],
            "unexpected_key": "value",
        }
        path = write_jsonl(tmp_path, [record])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "chat")
        assert "line 1" in exc_info.value.message

    def test_error_on_third_line(self, tmp_path: Path) -> None:
        good = {"messages": [{"role": "user", "content": "hi"}]}
        bad = {"messages": []}
        records = [good, good, bad]
        path = write_jsonl(tmp_path, records)
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "chat")
        assert "line 3" in exc_info.value.message


class TestTaskSchemaViolations:
    def test_missing_task_key(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, [{"input": "i", "expected_output": "o"}])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "task")
        err = exc_info.value
        assert err.code == 1
        assert "line 1" in err.message
        assert err.remediation

    def test_missing_input_key(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, [{"task": "t", "expected_output": "o"}])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "task")
        assert "line 1" in exc_info.value.message

    def test_missing_expected_output_key(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, [{"task": "t", "input": "i"}])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "task")
        assert "line 1" in exc_info.value.message

    def test_task_not_a_string(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, [{"task": 1, "input": "i", "expected_output": "o"}])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "task")
        assert "line 1" in exc_info.value.message

    def test_extra_top_level_keys_are_rejected(self, tmp_path: Path) -> None:
        record = {"task": "t", "input": "i", "expected_output": "o", "bonus": "x"}
        path = write_jsonl(tmp_path, [record])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "task")
        assert "line 1" in exc_info.value.message


class TestUnknownSchema:
    def test_unknown_schema_raises(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, [{"messages": [{"role": "user", "content": "hi"}]}])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "unknown_schema")
        err = exc_info.value
        assert err.code == 1
        assert err.remediation


class TestFileErrors:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CliError) as exc_info:
            validate_dataset(tmp_path / "nonexistent.jsonl", "chat")
        err = exc_info.value
        assert err.code == 2
        assert err.remediation


# ---------------------------------------------------------------------------
# Criterion 3 — torch is NOT imported as a side-effect
# ---------------------------------------------------------------------------


class TestNoTorchImport:
    def test_torch_absent_from_sys_modules(self, tmp_path: Path) -> None:
        """Importing and running the validator must not pull torch into sys.modules."""
        record = {"messages": [{"role": "user", "content": "hi"}]}
        path = write_jsonl(tmp_path, [record])
        validate_dataset(path, "chat")
        assert "torch" not in sys.modules, "torch must not be imported by the dataset validator"


# ---------------------------------------------------------------------------
# detect_schema helper
# ---------------------------------------------------------------------------


class TestDetectSchema:
    def test_detects_chat(self) -> None:
        record = {"messages": [{"role": "user", "content": "hi"}]}
        assert detect_schema(record) == "chat"

    def test_detects_task(self) -> None:
        record = {"task": "t", "input": "i", "expected_output": "o"}
        assert detect_schema(record) == "task"

    def test_unknown_returns_none(self) -> None:
        assert detect_schema({"foo": "bar"}) is None


# ---------------------------------------------------------------------------
# Empty / blank-only datasets fail fast (validate before spending GPU)
# ---------------------------------------------------------------------------


class TestEmptyDataset:
    def test_empty_file_raises_user_error(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "chat")
        assert exc_info.value.code == 1
        assert "no records" in str(exc_info.value.message)

    def test_blank_only_file_raises_user_error(self, tmp_path: Path) -> None:
        path = tmp_path / "blank.jsonl"
        path.write_text("\n   \n\n", encoding="utf-8")
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "task")
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# split_holdout — seeded, deterministic, scorable holdout
# ---------------------------------------------------------------------------


class TestSplitHoldout:
    def _task_records(self, n: int) -> list[dict]:
        return [
            {"task": f"t{i}", "input": f"in{i}", "expected_output": f"out{i}"} for i in range(n)
        ]

    def test_deterministic_with_same_seed(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, self._task_records(20))
        result1 = split_holdout(path, fraction=0.25, seed=42)
        train1 = Path(result1["train_path"]).read_text(encoding="utf-8")
        holdout1 = Path(result1["holdout_path"]).read_text(encoding="utf-8")

        result2 = split_holdout(path, fraction=0.25, seed=42)
        train2 = Path(result2["train_path"]).read_text(encoding="utf-8")
        holdout2 = Path(result2["holdout_path"]).read_text(encoding="utf-8")

        assert train1 == train2
        assert holdout1 == holdout2
        assert result1["train_count"] == result2["train_count"]
        assert result1["holdout_count"] == result2["holdout_count"]

    def test_different_seed_can_differ(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, self._task_records(20))
        result1 = split_holdout(path, fraction=0.5, seed=1)
        # split_holdout writes to a fixed <stem>.holdout.jsonl path, so read
        # this call's output before the next call overwrites it.
        holdout1 = Path(result1["holdout_path"]).read_text(encoding="utf-8")
        result2 = split_holdout(path, fraction=0.5, seed=3)
        holdout2 = Path(result2["holdout_path"]).read_text(encoding="utf-8")
        # Not a hard guarantee for all seeds, but true for this fixture/seed pair.
        assert holdout1 != holdout2

    def test_writes_stem_train_and_holdout_files(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, self._task_records(10), filename="mydata.jsonl")
        result = split_holdout(path, fraction=0.3, seed=7)
        assert Path(result["train_path"]) == tmp_path / "mydata.train.jsonl"
        assert Path(result["holdout_path"]) == tmp_path / "mydata.holdout.jsonl"
        assert Path(result["train_path"]).exists()
        assert Path(result["holdout_path"]).exists()

    def test_counts_sum_to_total(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, self._task_records(13))
        result = split_holdout(path, fraction=0.2, seed=3)
        assert result["train_count"] + result["holdout_count"] == 13

    def test_chat_rows_split_into_scorable_task_rows(self, tmp_path: Path) -> None:
        records = [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello there"},
                ]
            }
            for _ in range(6)
        ]
        path = write_jsonl(tmp_path, records)
        result = split_holdout(path, fraction=0.5, seed=5)
        holdout_lines = (
            Path(result["holdout_path"]).read_text(encoding="utf-8").strip().splitlines()
        )
        assert holdout_lines
        for line in holdout_lines:
            row = json.loads(line)
            assert set(row.keys()) == {"task", "input", "expected_output"}
            assert row["expected_output"] == "hello there"
            assert "system: You are helpful." in row["input"]
            assert "user: hi" in row["input"]
            assert "assistant" not in row["input"]

        train_lines = Path(result["train_path"]).read_text(encoding="utf-8").strip().splitlines()
        for line in train_lines:
            row = json.loads(line)
            assert set(row.keys()) == {"task", "input", "expected_output"}

    def test_invalid_fraction_raises(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, self._task_records(5))
        with pytest.raises(CliError) as exc_info:
            split_holdout(path, fraction=1.5, seed=1)
        assert exc_info.value.code == 1
        assert exc_info.value.remediation

    def test_invalid_fraction_zero_raises(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, self._task_records(5))
        with pytest.raises(CliError):
            split_holdout(path, fraction=0.0, seed=1)


# ---------------------------------------------------------------------------
# overlap_check — cross-schema duplicate detection between train and suites
# ---------------------------------------------------------------------------


class TestOverlapCheck:
    def test_no_overlap_returns_empty(self, tmp_path: Path) -> None:
        train = write_jsonl(
            tmp_path,
            [{"task": "t", "input": "a", "expected_output": "b"}],
            filename="train.jsonl",
        )
        suite = write_jsonl(
            tmp_path,
            [{"task": "t", "input": "c", "expected_output": "d"}],
            filename="suite.jsonl",
        )
        assert overlap_check(train, [suite]) == []

    def test_task_duplicate_detected(self, tmp_path: Path) -> None:
        train = write_jsonl(
            tmp_path,
            [{"task": "t", "input": "same-in", "expected_output": "same-out"}],
            filename="train.jsonl",
        )
        suite = write_jsonl(
            tmp_path,
            [{"task": "other", "input": "same-in", "expected_output": "same-out"}],
            filename="suite.jsonl",
        )
        findings = overlap_check(train, [suite])
        assert len(findings) == 2
        assert any(str(train) in f and f.endswith(":1") for f in findings)
        assert any(str(suite) in f and f.endswith(":1") for f in findings)

    def test_cross_schema_duplicate_detected(self, tmp_path: Path) -> None:
        train = write_jsonl(
            tmp_path,
            [
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "same-out"},
                    ]
                }
            ],
            filename="train.jsonl",
        )
        suite = write_jsonl(
            tmp_path,
            [{"task": "t", "input": "user: q", "expected_output": "same-out"}],
            filename="suite.jsonl",
        )
        findings = overlap_check(train, [suite])
        assert len(findings) == 2

    def test_multiple_suites_scanned(self, tmp_path: Path) -> None:
        train = write_jsonl(
            tmp_path,
            [{"task": "t", "input": "x", "expected_output": "y"}],
            filename="train.jsonl",
        )
        suite_a = write_jsonl(
            tmp_path,
            [{"task": "t", "input": "no-match", "expected_output": "z"}],
            filename="suite_a.jsonl",
        )
        suite_b = write_jsonl(
            tmp_path,
            [{"task": "t", "input": "x", "expected_output": "y"}],
            filename="suite_b.jsonl",
        )
        findings = overlap_check(train, [suite_a, suite_b])
        assert len(findings) == 2
        assert any(str(suite_b) in f for f in findings)
        assert not any(str(suite_a) in f for f in findings)

    def test_does_not_raise_cli_error(self, tmp_path: Path) -> None:
        """overlap_check only returns findings; raising is a CLI-caller responsibility."""
        train = write_jsonl(
            tmp_path,
            [{"task": "t", "input": "same", "expected_output": "same"}],
            filename="train.jsonl",
        )
        suite = write_jsonl(
            tmp_path,
            [{"task": "t", "input": "same", "expected_output": "same"}],
            filename="suite.jsonl",
        )
        findings = overlap_check(train, [suite])
        assert isinstance(findings, list)
        assert all(isinstance(f, str) for f in findings)


# ---------------------------------------------------------------------------
# instruction schema — task keys + constraints[]
# ---------------------------------------------------------------------------


class TestInstructionSchema:
    def test_valid_record_no_constraints(self, tmp_path: Path) -> None:
        record = {"task": "t", "input": "i", "expected_output": "o"}
        path = write_jsonl(tmp_path, [record])
        result = validate_dataset(path, "instruction")
        assert result == [record]

    def test_valid_record_with_constraints(self, tmp_path: Path) -> None:
        record = {
            "task": "t",
            "input": "i",
            "expected_output": "o",
            "constraints": [
                {"max_words": 50},
                {"min_words": 5},
                {"must_contain": "hello"},
                {"must_not_contain": "goodbye"},
                {"must_refuse": False},
                {"json_only": True},
            ],
        }
        path = write_jsonl(tmp_path, [record])
        result = validate_dataset(path, "instruction")
        assert result == [record]

    def test_unknown_constraint_key_rejected(self, tmp_path: Path) -> None:
        record = {
            "task": "t",
            "input": "i",
            "expected_output": "o",
            "constraints": [{"bogus_constraint": 1}],
        }
        path = write_jsonl(tmp_path, [record])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "instruction")
        assert "bogus_constraint" in exc_info.value.message

    def test_wrong_constraint_value_type_rejected(self, tmp_path: Path) -> None:
        record = {
            "task": "t",
            "input": "i",
            "expected_output": "o",
            "constraints": [{"max_words": "fifty"}],
        }
        path = write_jsonl(tmp_path, [record])
        with pytest.raises(CliError):
            validate_dataset(path, "instruction")

    def test_constraints_must_be_a_list(self, tmp_path: Path) -> None:
        record = {"task": "t", "input": "i", "expected_output": "o", "constraints": "nope"}
        path = write_jsonl(tmp_path, [record])
        with pytest.raises(CliError):
            validate_dataset(path, "instruction")

    def test_extra_top_level_key_rejected(self, tmp_path: Path) -> None:
        record = {"task": "t", "input": "i", "expected_output": "o", "bogus": 1}
        path = write_jsonl(tmp_path, [record])
        with pytest.raises(CliError):
            validate_dataset(path, "instruction")


# ---------------------------------------------------------------------------
# structured schema — {task, input, json_schema} with a JSON-Schema subset
# ---------------------------------------------------------------------------


class TestStructuredSchema:
    def test_valid_minimal_record(self, tmp_path: Path) -> None:
        record = {"task": "t", "input": "i", "json_schema": {"type": "object"}}
        path = write_jsonl(tmp_path, [record])
        result = validate_dataset(path, "structured")
        assert result == [record]

    def test_valid_full_subset(self, tmp_path: Path) -> None:
        record = {
            "task": "t",
            "input": "i",
            "json_schema": {
                "type": "object",
                "required": ["name", "tags"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "status": {"type": "string", "enum": ["ok", "fail"]},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
        path = write_jsonl(tmp_path, [record])
        result = validate_dataset(path, "structured")
        assert result == [record]

    def test_unknown_top_level_keyword_rejected(self, tmp_path: Path) -> None:
        record = {"task": "t", "input": "i", "json_schema": {"type": "object", "minimum": 1}}
        path = write_jsonl(tmp_path, [record])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "structured")
        assert "minimum" in exc_info.value.message

    def test_unknown_nested_keyword_rejected(self, tmp_path: Path) -> None:
        record = {
            "task": "t",
            "input": "i",
            "json_schema": {
                "type": "object",
                "properties": {"name": {"type": "string", "pattern": "^a"}},
            },
        }
        path = write_jsonl(tmp_path, [record])
        with pytest.raises(CliError) as exc_info:
            validate_dataset(path, "structured")
        assert "pattern" in exc_info.value.message

    def test_json_schema_must_be_object(self, tmp_path: Path) -> None:
        record = {"task": "t", "input": "i", "json_schema": "nope"}
        path = write_jsonl(tmp_path, [record])
        with pytest.raises(CliError):
            validate_dataset(path, "structured")

    def test_missing_json_schema_key(self, tmp_path: Path) -> None:
        record = {"task": "t", "input": "i"}
        path = write_jsonl(tmp_path, [record])
        with pytest.raises(CliError):
            validate_dataset(path, "structured")


# ---------------------------------------------------------------------------
# toolcall schema — {task, input, expected_tool_call: {name, arguments}}
# ---------------------------------------------------------------------------


class TestToolcallSchema:
    def test_valid_record(self, tmp_path: Path) -> None:
        record = {
            "task": "t",
            "input": "i",
            "expected_tool_call": {"name": "search", "arguments": {"query": "cats"}},
        }
        path = write_jsonl(tmp_path, [record])
        result = validate_dataset(path, "toolcall")
        assert result == [record]

    def test_missing_expected_tool_call(self, tmp_path: Path) -> None:
        record = {"task": "t", "input": "i"}
        path = write_jsonl(tmp_path, [record])
        with pytest.raises(CliError):
            validate_dataset(path, "toolcall")

    def test_expected_tool_call_missing_name(self, tmp_path: Path) -> None:
        record = {"task": "t", "input": "i", "expected_tool_call": {"arguments": {}}}
        path = write_jsonl(tmp_path, [record])
        with pytest.raises(CliError):
            validate_dataset(path, "toolcall")

    def test_expected_tool_call_missing_arguments(self, tmp_path: Path) -> None:
        record = {"task": "t", "input": "i", "expected_tool_call": {"name": "search"}}
        path = write_jsonl(tmp_path, [record])
        with pytest.raises(CliError):
            validate_dataset(path, "toolcall")

    def test_expected_tool_call_name_not_a_string(self, tmp_path: Path) -> None:
        record = {"task": "t", "input": "i", "expected_tool_call": {"name": 1, "arguments": {}}}
        path = write_jsonl(tmp_path, [record])
        with pytest.raises(CliError):
            validate_dataset(path, "toolcall")

    def test_expected_tool_call_arguments_not_a_dict(self, tmp_path: Path) -> None:
        record = {
            "task": "t",
            "input": "i",
            "expected_tool_call": {"name": "search", "arguments": "nope"},
        }
        path = write_jsonl(tmp_path, [record])
        with pytest.raises(CliError):
            validate_dataset(path, "toolcall")

    def test_extra_top_level_key_rejected(self, tmp_path: Path) -> None:
        record = {
            "task": "t",
            "input": "i",
            "expected_tool_call": {"name": "search", "arguments": {}},
            "bogus": 1,
        }
        path = write_jsonl(tmp_path, [record])
        with pytest.raises(CliError):
            validate_dataset(path, "toolcall")


# ---------------------------------------------------------------------------
# validate_suite works with the new schemas too
# ---------------------------------------------------------------------------


class TestValidateSuiteNewSchemas:
    def test_instruction_suite(self, tmp_path: Path) -> None:
        record = {"task": "t", "input": "i", "expected_output": "o", "constraints": []}
        path = write_jsonl(tmp_path, [record])
        report = validate_suite(path, schema="instruction")
        assert report["total_records"] == 1

    def test_structured_suite(self, tmp_path: Path) -> None:
        record = {"task": "t", "input": "i", "json_schema": {"type": "string"}}
        path = write_jsonl(tmp_path, [record])
        report = validate_suite(path, schema="structured")
        assert report["total_records"] == 1

    def test_toolcall_suite(self, tmp_path: Path) -> None:
        record = {
            "task": "t",
            "input": "i",
            "expected_tool_call": {"name": "n", "arguments": {}},
        }
        path = write_jsonl(tmp_path, [record])
        report = validate_suite(path, schema="toolcall")
        assert report["total_records"] == 1


# ---------------------------------------------------------------------------
# detect_schema stays backwards compatible with existing chat/task detection
# ---------------------------------------------------------------------------


class TestDetectSchemaBackwardsCompatible:
    def test_chat_still_detects(self) -> None:
        assert detect_schema({"messages": [{"role": "user", "content": "hi"}]}) == "chat"

    def test_task_still_detects(self) -> None:
        assert detect_schema({"task": "t", "input": "i", "expected_output": "o"}) == "task"


class TestDetectSchemaFiveWay:
    """Deviation d1: detect_schema is the single five-way detector."""

    def test_each_schema_is_detected_by_its_discriminating_key(self) -> None:
        from sloth.tune.datasets import detect_schema

        assert detect_schema({"messages": []}) == "chat"
        assert detect_schema({"task": "t", "input": "i", "expected_output": "o"}) == "task"
        assert (
            detect_schema({"task": "t", "input": "i", "expected_output": "o", "constraints": []})
            == "instruction"
        )
        assert detect_schema({"task": "t", "input": "i", "json_schema": {}}) == "structured"
        assert detect_schema({"task": "t", "input": "i", "expected_tool_call": {}}) == "toolcall"
        assert detect_schema({"foo": 1}) is None
        assert detect_schema("not a dict") is None  # type: ignore[arg-type]

    def test_detect_file_schema_reads_first_record_and_never_raises(self, tmp_path) -> None:
        from sloth.tune.datasets import detect_file_schema

        f = tmp_path / "s.jsonl"
        f.write_text('\n{"task": "t", "input": "i", "json_schema": {}}\n', encoding="utf-8")
        assert detect_file_schema(f) == "structured"
        (tmp_path / "bad.jsonl").write_text("not json\n", encoding="utf-8")
        assert detect_file_schema(tmp_path / "bad.jsonl") is None
        assert detect_file_schema(tmp_path / "missing.jsonl") is None

    def test_validate_suite_auto_mixes_schemas_and_falls_back_to_task(self, tmp_path) -> None:
        from sloth.cli._errors import CliError
        from sloth.tune.datasets import validate_suite

        d = tmp_path / "suite"
        d.mkdir()
        (d / "a.jsonl").write_text(
            '{"task": "t", "input": "i", "expected_output": "o"}\n', encoding="utf-8"
        )
        (d / "b.jsonl").write_text('{"messages": [{"role": "user", "content": "x"}]}\n')
        report = validate_suite(d, schema="auto")
        assert [f["schema"] for f in report["files"]] == ["task", "chat"]
        (d / "c.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(CliError) as exc_info:
            validate_suite(d, schema="auto")
        assert "c.jsonl" in exc_info.value.message
        assert "line 1" in exc_info.value.message
