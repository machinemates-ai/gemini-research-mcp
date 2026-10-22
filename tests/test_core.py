"""Unit tests for core research functions.

Tests thinking level parsing and system prompt generation.
Run with: uv run pytest tests/ -v
"""

import types
from datetime import date
from unittest.mock import AsyncMock

import pytest
from google.genai.types import ThinkingLevel

from gemini_research_mcp.config import (
    DEFAULT_MODEL,
    DEFAULT_THINKING_LEVEL,
    default_system_prompt,
)
from gemini_research_mcp.quick import (
    THINKING_LEVEL_MAP,
    _extract_sources_from_text,
    _get_thinking_level,
    quick_research,
)


class TestThinkingLevel:
    """Test thinking level parsing for Gemini 3 models."""

    @pytest.mark.parametrize("level,expected", [
        ("minimal", ThinkingLevel.MINIMAL),
        ("low", ThinkingLevel.LOW),
        ("medium", ThinkingLevel.MEDIUM),
        ("high", ThinkingLevel.HIGH),
    ])
    def test_named_levels(self, level: str, expected: ThinkingLevel):
        """Named levels should map to correct ThinkingLevel enum."""
        assert _get_thinking_level(level) == expected

    def test_case_insensitive(self):
        """Level names should be case-insensitive."""
        assert _get_thinking_level("HIGH") == ThinkingLevel.HIGH
        assert _get_thinking_level("High") == ThinkingLevel.HIGH

    def test_unknown_level_defaults_to_high(self):
        """Unknown level names should default to HIGH."""
        assert _get_thinking_level("unknown") == ThinkingLevel.HIGH
        assert _get_thinking_level("max") == ThinkingLevel.HIGH  # 'max' not valid for Gemini 3

    def test_level_map_complete(self):
        """THINKING_LEVEL_MAP should have all expected levels."""
        expected = {"minimal", "low", "medium", "high"}
        assert set(THINKING_LEVEL_MAP.keys()) == expected

    def test_source_fallback_extracts_markdown_links(self):
        """Link extraction should populate sources when grounding metadata is unavailable."""
        sources = _extract_sources_from_text(
            "See the [Python docs](https://docs.python.org/3/library/dataclasses.html)."
        )

        assert len(sources) == 1
        assert sources[0].title == "Python docs"
        assert sources[0].uri == "https://docs.python.org/3/library/dataclasses.html"

    def test_default_is_high(self):
        """Default thinking level should be 'high'."""
        assert DEFAULT_THINKING_LEVEL == "high"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured,explicit,expected",
    [
        (None, None, "gemini-4-argon"),
        ("gemini-3.8-flash", None, "gemini-3.8-flash"),
        ("gemini-3.8-flash", "custom-model", "custom-model"),
    ],
)
async def test_shared_model_routing(
    monkeypatch: pytest.MonkeyPatch,
    configured: str | None,
    explicit: str | None,
    expected: str,
) -> None:
    """Both provider paths honor model precedence without reducing web thinking."""
    from gemini_research_mcp import deep, quick

    if configured is None:
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
    else:
        monkeypatch.setenv("GEMINI_MODEL", configured)

    generate = AsyncMock(return_value=types.SimpleNamespace(text="answer", candidates=[]))
    create = AsyncMock(return_value=types.SimpleNamespace(output_text="answer"))
    client = types.SimpleNamespace(
        aio=types.SimpleNamespace(
            models=types.SimpleNamespace(generate_content=generate),
            interactions=types.SimpleNamespace(create=create),
        )
    )
    monkeypatch.setattr(quick, "get_api_key", lambda: "synthetic-key")
    monkeypatch.setattr(quick.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(deep, "_get_healthy_client", lambda: client)
    monkeypatch.setattr(deep, "_client_health", None)

    result = await quick_research("synthetic query", model=explicit, thinking_level="minimal")
    followup = await deep.research_followup("synthetic-id", "synthetic query", model=explicit)

    assert result.text == followup == "answer"
    generate.assert_awaited_once()
    create.assert_awaited_once_with(
        input="synthetic query", model=expected, previous_interaction_id="synthetic-id"
    )
    assert generate.await_args.kwargs["model"] == expected
    config = generate.await_args.kwargs["config"]
    assert config.thinking_config.thinking_level == ThinkingLevel.HIGH
    assert config.tools[0].google_search is not None


class TestSystemPrompt:
    """Test default system prompt generation."""

    def test_includes_current_date(self):
        """System prompt should include current date."""
        prompt = default_system_prompt()
        today = date.today().strftime("%B %d, %Y")
        assert today in prompt

    def test_mentions_research_role(self):
        """System prompt should establish research analyst role."""
        prompt = default_system_prompt()
        assert "research" in prompt.lower()
        assert "analyst" in prompt.lower() or "expert" in prompt.lower()

    def test_mentions_citations(self):
        """System prompt should mention citing sources."""
        prompt = default_system_prompt()
        assert "cite" in prompt.lower() or "source" in prompt.lower()

    def test_reasonable_length(self):
        """System prompt should be reasonably sized."""
        prompt = default_system_prompt()
        assert 200 < len(prompt) < 2000, f"Prompt length {len(prompt)} seems unusual"


class TestConstants:
    """Test module constants."""

    def test_default_model_is_gemini(self):
        """Default model should be a Gemini model."""
        assert "gemini" in DEFAULT_MODEL.lower()

    def test_default_thinking_level_is_high(self):
        """Default thinking level should be 'high' for quality results."""
        assert DEFAULT_THINKING_LEVEL == "high"
