"""Unit tests for core research functions.

Tests thinking budget parsing and system prompt generation.
Run with: uv run pytest tests/ -v
"""

import pytest
from datetime import date

from gemini_research_mcp.config import (
    DEFAULT_MODEL,
    DEFAULT_THINKING_BUDGET,
    THINKING_BUDGETS,
    get_thinking_budget,
    default_system_prompt,
)

# Aliases matching old test expectations
_get_thinking_budget = get_thinking_budget
_default_system_prompt = default_system_prompt


class TestThinkingBudget:
    """Test thinking budget parsing."""

    @pytest.mark.parametrize("level,expected", [
        ("minimal", 0),
        ("low", 2048),
        ("medium", 8192),
        ("high", 16384),
        ("max", 24576),
        ("dynamic", -1),
    ])
    def test_named_levels(self, level: str, expected: int):
        """Named levels should map to correct token counts."""
        assert _get_thinking_budget(level) == expected

    def test_integer_passthrough(self):
        """Integer values should pass through unchanged."""
        assert _get_thinking_budget(4096) == 4096
        assert _get_thinking_budget(0) == 0
        assert _get_thinking_budget(24576) == 24576

    def test_unknown_level_uses_default(self):
        """Unknown level names should use default budget."""
        assert _get_thinking_budget("unknown") == DEFAULT_THINKING_BUDGET
        assert _get_thinking_budget("super_high") == DEFAULT_THINKING_BUDGET

    def test_thinking_budgets_dict_complete(self):
        """THINKING_BUDGETS should have all expected levels."""
        expected = {"minimal", "low", "medium", "high", "max", "dynamic"}
        assert set(THINKING_BUDGETS.keys()) == expected


class TestSystemPrompt:
    """Test default system prompt generation."""

    def test_includes_current_date(self):
        """System prompt should include current date."""
        prompt = _default_system_prompt()
        today = date.today().strftime("%B %d, %Y")
        assert today in prompt

    def test_mentions_research_role(self):
        """System prompt should establish research analyst role."""
        prompt = _default_system_prompt()
        assert "research" in prompt.lower()
        assert "analyst" in prompt.lower() or "expert" in prompt.lower()

    def test_mentions_citations(self):
        """System prompt should mention citing sources."""
        prompt = _default_system_prompt()
        assert "cite" in prompt.lower() or "source" in prompt.lower()

    def test_reasonable_length(self):
        """System prompt should be reasonably sized."""
        prompt = _default_system_prompt()
        assert 200 < len(prompt) < 2000, f"Prompt length {len(prompt)} seems unusual"


class TestConstants:
    """Test module constants."""

    def test_default_model_is_gemini(self):
        """Default model should be a Gemini model."""
        assert "gemini" in DEFAULT_MODEL.lower()

    def test_default_thinking_budget_is_medium(self):
        """Default thinking budget should be medium (8192)."""
        assert DEFAULT_THINKING_BUDGET == 8192
        assert DEFAULT_THINKING_BUDGET == THINKING_BUDGETS["medium"]
