"""Tests for sloth.tune.scorers — constraints, JSON-schema subset, tool-call
parsers (qwen3, lfm2), and lazy GLEU/BLEU."""

from __future__ import annotations

import sys

import pytest

from sloth.cli._errors import CliError
from sloth.tune.scorers import (
    TOOL_CALL_PARSERS,
    bleu,
    check_json_subset,
    family_for_model,
    gleu,
    parse_tool_call,
    score_constraints,
    score_tool_call,
)

# ---------------------------------------------------------------------------
# score_constraints
# ---------------------------------------------------------------------------


class TestScoreConstraints:
    def test_all_pass(self) -> None:
        result = score_constraints(
            "the quick brown fox",
            [
                {"kind": "max_words", "value": 10},
                {"kind": "min_words", "value": 2},
                {"kind": "must_contain", "value": "fox"},
                {"kind": "must_not_contain", "value": "dog"},
            ],
        )
        assert result == {"passed": True, "failed": []}

    def test_max_words_fails(self) -> None:
        result = score_constraints("one two three four", [{"kind": "max_words", "value": 2}])
        assert result == {"passed": False, "failed": ["max_words"]}

    def test_min_words_fails(self) -> None:
        result = score_constraints("one", [{"kind": "min_words", "value": 3}])
        assert result == {"passed": False, "failed": ["min_words"]}

    def test_must_contain_fails(self) -> None:
        result = score_constraints("hello world", [{"kind": "must_contain", "value": "goodbye"}])
        assert result == {"passed": False, "failed": ["must_contain"]}

    def test_must_not_contain_fails(self) -> None:
        result = score_constraints("hello world", [{"kind": "must_not_contain", "value": "world"}])
        assert result == {"passed": False, "failed": ["must_not_contain"]}

    def test_must_refuse_passes_on_refusal(self) -> None:
        result = score_constraints(
            "I'm sorry, but I can't help with that request.",
            [{"kind": "must_refuse"}],
        )
        assert result == {"passed": True, "failed": []}

    def test_must_refuse_fails_on_compliance(self) -> None:
        result = score_constraints("Sure, here you go: 42.", [{"kind": "must_refuse"}])
        assert result == {"passed": False, "failed": ["must_refuse"]}

    def test_json_only_passes(self) -> None:
        result = score_constraints('{"a": 1}', [{"kind": "json_only"}])
        assert result == {"passed": True, "failed": []}

    def test_json_only_fails_on_prose(self) -> None:
        result = score_constraints("not json at all", [{"kind": "json_only"}])
        assert result == {"passed": False, "failed": ["json_only"]}

    def test_multiple_failures_listed_in_order(self) -> None:
        result = score_constraints(
            "hello world",
            [
                {"kind": "must_contain", "value": "goodbye"},
                {"kind": "max_words", "value": 1},
            ],
        )
        assert result == {"passed": False, "failed": ["must_contain", "max_words"]}

    def test_unknown_kind_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown constraint kind"):
            score_constraints("x", [{"kind": "bogus"}])


# ---------------------------------------------------------------------------
# check_json_subset
# ---------------------------------------------------------------------------


class TestCheckJsonSubset:
    def test_valid_object(self) -> None:
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        result = check_json_subset('{"name": "Ada", "age": 30}', schema)
        assert result == {"valid": True, "errors": []}

    def test_invalid_json_string(self) -> None:
        result = check_json_subset("not json", {"type": "object"})
        assert result["valid"] is False
        assert len(result["errors"]) == 1
        assert "invalid JSON" in result["errors"][0]

    def test_type_mismatch(self) -> None:
        result = check_json_subset("42", {"type": "string"})
        assert result["valid"] is False
        assert "expected string" in result["errors"][0]

    def test_bool_rejected_for_integer_type(self) -> None:
        result = check_json_subset("true", {"type": "integer"})
        assert result["valid"] is False
        assert "bool" in result["errors"][0]

    def test_boolean_type_accepts_bool(self) -> None:
        result = check_json_subset("true", {"type": "boolean"})
        assert result == {"valid": True, "errors": []}

    def test_missing_required_property(self) -> None:
        schema = {"type": "object", "required": ["name"]}
        result = check_json_subset("{}", schema)
        assert result["valid"] is False
        assert "missing required property 'name'" in result["errors"][0]

    def test_enum_violation(self) -> None:
        result = check_json_subset('"red"', {"enum": ["a", "b"]})
        assert result["valid"] is False
        assert "not in enum" in result["errors"][0]

    def test_enum_ok(self) -> None:
        result = check_json_subset('"a"', {"enum": ["a", "b"]})
        assert result == {"valid": True, "errors": []}

    def test_items_recursion(self) -> None:
        schema = {"type": "array", "items": {"type": "integer"}}
        result = check_json_subset("[1, 2, 3]", schema)
        assert result == {"valid": True, "errors": []}

    def test_items_recursion_reports_index(self) -> None:
        schema = {"type": "array", "items": {"type": "integer"}}
        result = check_json_subset("[1, 2, true]", schema)
        assert result["valid"] is False
        assert "$[2]" in result["errors"][0]

    def test_additional_properties_false_rejects_extra(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": False,
        }
        result = check_json_subset('{"a": 1, "b": 2}', schema)
        assert result["valid"] is False
        assert "additional property 'b' not allowed" in result["errors"][0]

    def test_additional_properties_schema_recurses(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": {"type": "string"},
        }
        result = check_json_subset('{"a": 1, "b": 2}', schema)
        assert result["valid"] is False
        assert "$.b" in result["errors"][0]

    def test_nested_properties(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {"id": {"type": "integer"}},
                }
            },
        }
        result = check_json_subset('{"user": {}}', schema)
        assert result["valid"] is False
        assert "$.user: missing required property 'id'" in result["errors"][0]

    def test_unknown_keyword_named_in_error(self) -> None:
        result = check_json_subset("{}", {"type": "object", "minLength": 3})
        assert result["valid"] is False
        assert any("minLength" in e for e in result["errors"])

    def test_accepts_pre_parsed_value(self) -> None:
        result = check_json_subset({"a": 1}, {"type": "object"})
        assert result == {"valid": True, "errors": []}


# ---------------------------------------------------------------------------
# parse_tool_call — qwen3
# ---------------------------------------------------------------------------


class TestParseToolCallQwen3:
    def test_parses_object_arguments(self) -> None:
        prediction = (
            '<tool_call>\n{"name": "get_weather", '
            '"arguments": {"city": "Paris", "unit": "celsius"}}\n</tool_call>'
        )
        result = parse_tool_call(prediction, "qwen3")
        assert result == {
            "name": "get_weather",
            "arguments": {"city": "Paris", "unit": "celsius"},
        }

    def test_parses_string_encoded_arguments(self) -> None:
        prediction = (
            '<tool_call>\n{"name": "search", "arguments": "{\\"q\\": \\"cats\\"}"}\n</tool_call>'
        )
        result = parse_tool_call(prediction, "qwen3")
        assert result == {"name": "search", "arguments": {"q": "cats"}}

    def test_missing_block_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_call"):
            parse_tool_call("no tool call here", "qwen3")

    def test_surrounding_prose_ignored(self) -> None:
        prediction = (
            "Sure, let me check.\n"
            '<tool_call>\n{"name": "ping", "arguments": {}}\n</tool_call>\nDone.'
        )
        result = parse_tool_call(prediction, "qwen3")
        assert result == {"name": "ping", "arguments": {}}


# ---------------------------------------------------------------------------
# parse_tool_call — lfm2
# ---------------------------------------------------------------------------


class TestParseToolCallLfm2:
    def test_parses_string_and_number_args(self) -> None:
        prediction = "<|tool_call_start|>[get_weather(city='Paris', days=3)]<|tool_call_end|>"
        result = parse_tool_call(prediction, "lfm2")
        assert result == {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}

    def test_parses_mapping_arg_rendered_as_json(self) -> None:
        prediction = (
            '<|tool_call_start|>[configure(options={"retries": 2, "verbose": true})]'
            "<|tool_call_end|>"
        )
        result = parse_tool_call(prediction, "lfm2")
        assert result == {
            "name": "configure",
            "arguments": {"options": {"retries": 2, "verbose": True}},
        }

    def test_parses_no_args_call(self) -> None:
        prediction = "<|tool_call_start|>[ping()]<|tool_call_end|>"
        result = parse_tool_call(prediction, "lfm2")
        assert result == {"name": "ping", "arguments": {}}

    def test_parses_escaped_quote_in_string_arg(self) -> None:
        prediction = "<|tool_call_start|>[say(text='it\\'s me')]<|tool_call_end|>"
        result = parse_tool_call(prediction, "lfm2")
        assert result == {"name": "say", "arguments": {"text": "it's me"}}

    def test_missing_block_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_call_start"):
            parse_tool_call("no tool call here", "lfm2")


# ---------------------------------------------------------------------------
# parse_tool_call — registry
# ---------------------------------------------------------------------------


class TestParseToolCallRegistry:
    def test_unknown_family_lists_supported(self) -> None:
        with pytest.raises(ValueError, match="qwen3.*lfm2|lfm2.*qwen3"):
            parse_tool_call("anything", "bogus-family")

    def test_registry_has_both_families(self) -> None:
        assert set(TOOL_CALL_PARSERS) >= {"qwen3", "lfm2"}


# ---------------------------------------------------------------------------
# family_for_model
# ---------------------------------------------------------------------------


class TestFamilyForModel:
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("unsloth/Qwen3-1.7B", "qwen3"),
            ("Qwen/Qwen3.5-4B", "qwen3"),
            ("LiquidAI/LFM2.5-1.2B-Instruct", "lfm2"),
            ("unsloth/LFM2.5-1.2B-Instruct", "lfm2"),
        ],
    )
    def test_detects_by_substring(self, model_id: str, expected: str) -> None:
        assert family_for_model(model_id) == expected

    def test_override_takes_precedence(self) -> None:
        assert family_for_model("unsloth/Qwen3-1.7B", override="lfm2") == "lfm2"

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_call_family"):
            family_for_model("some/unknown-model")


# ---------------------------------------------------------------------------
# score_tool_call
# ---------------------------------------------------------------------------


class TestScoreToolCall:
    def test_full_match(self) -> None:
        prediction = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        )
        result = score_tool_call(
            prediction, {"name": "get_weather", "arguments": {"city": "Paris"}}, "qwen3"
        )
        assert result == {"matched": True, "name_ok": True, "arguments_ok": True}

    def test_name_mismatch(self) -> None:
        prediction = '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'
        result = score_tool_call(prediction, {"name": "get_weather", "arguments": {}}, "qwen3")
        assert result == {"matched": False, "name_ok": False, "arguments_ok": True}

    def test_arguments_mismatch(self) -> None:
        prediction = "<|tool_call_start|>[get_weather(city='London')]<|tool_call_end|>"
        result = score_tool_call(
            prediction, {"name": "get_weather", "arguments": {"city": "Paris"}}, "lfm2"
        )
        assert result == {"matched": False, "name_ok": True, "arguments_ok": False}

    def test_unparseable_prediction_scores_false(self) -> None:
        result = score_tool_call("garbage output", {"name": "x", "arguments": {}}, "qwen3")
        assert result == {"matched": False, "name_ok": False, "arguments_ok": False}


# ---------------------------------------------------------------------------
# gleu / bleu — lazy sacrebleu import
# ---------------------------------------------------------------------------


class TestLazySacrebleu:
    def test_gleu_raises_cli_error_when_sacrebleu_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "sacrebleu", None)
        with pytest.raises(CliError) as excinfo:
            gleu("the cat sat", "the cat sat")
        assert excinfo.value.code == 2
        assert "sacrebleu" in excinfo.value.remediation

    def test_bleu_raises_cli_error_when_sacrebleu_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "sacrebleu", None)
        with pytest.raises(CliError) as excinfo:
            bleu("the cat sat", "the cat sat")
        assert excinfo.value.code == 2
        assert "sacrebleu" in excinfo.value.remediation

    def test_gleu_uses_sacrebleu_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeSacrebleu()
        monkeypatch.setitem(sys.modules, "sacrebleu", fake)
        score = gleu("the cat sat on the mat", "the cat sat on the mat")
        assert score == pytest.approx(100.0)

    def test_gleu_partial_overlap_scores_between_0_and_100(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeSacrebleu()
        monkeypatch.setitem(sys.modules, "sacrebleu", fake)
        score = gleu("completely different words here", "the cat sat on the mat")
        assert 0.0 <= score < 100.0

    def test_bleu_uses_sacrebleu_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeSacrebleu()
        monkeypatch.setitem(sys.modules, "sacrebleu", fake)
        score = bleu("the cat sat on the mat", "the cat sat on the mat")
        assert score == pytest.approx(100.0)


class _FakeScoreResult:
    def __init__(self, score: float) -> None:
        self.score = score


class _FakeBLEU:
    """Minimal stand-in for ``sacrebleu.BLEU`` — just a whitespace tokenizer."""

    def tokenizer(self, text: str) -> str:
        return text


class _FakeSacrebleu:
    """Minimal stand-in for the ``sacrebleu`` module used by gleu()/bleu()."""

    def BLEU(self) -> _FakeBLEU:  # noqa: N802 - mirrors sacrebleu's class name
        return _FakeBLEU()

    def sentence_bleu(self, prediction: str, references: list[str]) -> _FakeScoreResult:
        ref = references[0]
        score = 100.0 if prediction.strip() == ref.strip() else 0.0
        return _FakeScoreResult(score)
