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
from sloth.tune.datasets import detect_schema, validate_dataset

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
