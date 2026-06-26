"""Tests for sloth.tune.scope — model-scope guard (adapter vs. out-of-scope full FT)."""

from __future__ import annotations

import pytest

from sloth.tune.scope import (
    LARGE_DENSE_THRESHOLD_B,
    SUPPORTED_ADAPTER_METHODS,
    ScopeResult,
    check_scope,
)

# ---------------------------------------------------------------------------
# Acceptance criterion 1 — LoRA/QLoRA on small/medium Qwen → OK
# ---------------------------------------------------------------------------


class TestAdapterMethodsAreInScope:
    """LoRA/QLoRA targets on supported small/medium Qwen models must return ok=True."""

    @pytest.mark.parametrize(
        "model,method",
        [
            ("unsloth/Qwen3-4B", "lora"),
            ("unsloth/Qwen3-4B", "qlora"),
            ("unsloth/Qwen3-9B", "lora"),
            ("unsloth/Qwen3-9B", "qlora"),
            ("Qwen/Qwen3-4B-Instruct", "lora"),
            ("Qwen/Qwen3.6-9B", "qlora"),
            ("qwen3-4b", "lora"),
            ("some/path/qwen2.5-7b-instruct", "lora"),
        ],
    )
    def test_ok_result(self, model: str, method: str) -> None:
        result = check_scope(model, method)
        assert isinstance(result, ScopeResult)
        assert result.ok is True
        assert result.out_of_scope is False
        assert result.warning is None

    def test_lora_is_supported_method(self) -> None:
        assert "lora" in SUPPORTED_ADAPTER_METHODS

    def test_qlora_is_supported_method(self) -> None:
        assert "qlora" in SUPPORTED_ADAPTER_METHODS

    def test_result_has_message(self) -> None:
        result = check_scope("unsloth/Qwen3-4B", "lora")
        assert isinstance(result.message, str)
        assert result.message  # non-empty

    def test_downgrade_to_is_none_for_ok(self) -> None:
        result = check_scope("unsloth/Qwen3-4B", "qlora")
        assert result.downgrade_to is None


# ---------------------------------------------------------------------------
# Acceptance criterion 2 — Full fine-tuning → out-of-scope warning
# ---------------------------------------------------------------------------


class TestFullFineTuningIsOutOfScope:
    """Full fine-tuning requests must return out_of_scope=True with a warning."""

    @pytest.mark.parametrize(
        "model,method",
        [
            ("unsloth/Qwen3-4B", "full"),
            ("unsloth/Qwen3-4B", "full_ft"),
            ("Qwen/Qwen3-9B", "full"),
        ],
    )
    def test_full_ft_returns_out_of_scope(self, model: str, method: str) -> None:
        result = check_scope(model, method)
        assert isinstance(result, ScopeResult)
        assert result.out_of_scope is True
        assert result.ok is False

    @pytest.mark.parametrize("method", ["full", "full_ft"])
    def test_full_ft_warning_is_string(self, method: str) -> None:
        result = check_scope("unsloth/Qwen3-4B", method)
        assert isinstance(result.warning, str)
        assert result.warning  # non-empty

    @pytest.mark.parametrize("method", ["full", "full_ft"])
    def test_full_ft_warning_mentions_lora(self, method: str) -> None:
        """Warning must guide the user toward LoRA/QLoRA."""
        result = check_scope("unsloth/Qwen3-4B", method)
        lower = result.warning.lower()
        assert "lora" in lower

    @pytest.mark.parametrize("method", ["full", "full_ft"])
    def test_full_ft_downgrade_suggested(self, method: str) -> None:
        """Scope result should recommend a downgrade method."""
        result = check_scope("unsloth/Qwen3-4B", method)
        assert result.downgrade_to in SUPPORTED_ADAPTER_METHODS


# ---------------------------------------------------------------------------
# Large dense model + full FT → out-of-scope
# ---------------------------------------------------------------------------


class TestLargeDenseFullFTIsOutOfScope:
    """Large dense models with full FT must always be out of scope."""

    @pytest.mark.parametrize(
        "model,method",
        [
            ("Qwen/Qwen3-72B", "full"),
            ("unsloth/Qwen3.6-27B-dense", "full"),
            ("some-org/qwen2-57b-a14b", "full"),
            ("Qwen/Qwen3-32B", "full"),
        ],
    )
    def test_large_dense_full_ft_out_of_scope(self, model: str, method: str) -> None:
        result = check_scope(model, method)
        assert result.out_of_scope is True
        assert result.ok is False
        assert result.warning is not None


# ---------------------------------------------------------------------------
# Large dense model + adapter method → OK (adapters are fine on large models)
# ---------------------------------------------------------------------------


class TestLargeDenseAdapterIsOK:
    """Adapter methods on large models are allowed — adapters are always OK."""

    @pytest.mark.parametrize(
        "model,method",
        [
            ("Qwen/Qwen3-72B", "lora"),
            ("unsloth/Qwen3.6-27B-dense", "qlora"),
        ],
    )
    def test_large_model_with_adapter_is_ok(self, model: str, method: str) -> None:
        result = check_scope(model, method)
        assert result.ok is True
        assert result.out_of_scope is False


# ---------------------------------------------------------------------------
# Threshold constant is exported and sane
# ---------------------------------------------------------------------------


class TestThresholdConstant:
    def test_large_dense_threshold_is_positive_int(self) -> None:
        assert isinstance(LARGE_DENSE_THRESHOLD_B, (int, float))
        assert LARGE_DENSE_THRESHOLD_B > 0

    def test_threshold_is_at_least_10b(self) -> None:
        """Threshold should exclude 9B (still small/medium) but catch 27B+."""
        assert LARGE_DENSE_THRESHOLD_B > 9


# ---------------------------------------------------------------------------
# Unknown / unrecognised method
# ---------------------------------------------------------------------------


class TestUnknownMethod:
    """Unrecognised method strings should be treated as out-of-scope."""

    @pytest.mark.parametrize("method", ["sft", "dpo", "ppo", "reward_modeling"])
    def test_unsupported_method_is_out_of_scope(self, method: str) -> None:
        result = check_scope("unsloth/Qwen3-4B", method)
        assert result.out_of_scope is True
        assert result.ok is False
        assert result.warning is not None
