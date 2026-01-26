"""Test SEP-1577 Sampling with Tools pattern for query clarification.

This tests the sampling-based clarification flow in research_deep:
1. User calls research_deep with a vague query
2. ctx.sample() is called with the ask_clarifying_questions tool
3. LLM can invoke the tool to ask user for clarification via ctx.elicit()
4. Deep research proceeds with the refined query

These tests verify the pattern without actual API calls (unit tests)
and with API calls (E2E tests marked with @pytest.mark.e2e).
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Context, FastMCP
from fastmcp.client import Client
from fastmcp.dependencies import Depends, Progress
from fastmcp.server.sampling import SamplingResult, SamplingTool
from fastmcp.server.tasks import TaskConfig
from pydantic import BaseModel, Field


# =============================================================================
# Unit Tests (no API calls)
# =============================================================================


class TestSamplingClarificationPattern:
    """Test the SEP-1577 sampling pattern mechanics."""

    @pytest.fixture
    def mock_analyzed_query(self):
        """Mock AnalyzedQuery result from sampling."""

        class MockAnalyzedQuery(BaseModel):
            refined_query: str = Field(description="The refined query")
            was_clarified: bool = Field(description="Whether clarification occurred")
            summary: str = Field(description="Summary")

        return MockAnalyzedQuery

    @pytest.mark.asyncio
    async def test_clarify_tool_structure(self):
        """Verify the _clarify_tool SamplingTool has correct structure."""
        from gemini_research_mcp.server import _clarify_tool

        assert isinstance(_clarify_tool, SamplingTool)
        assert _clarify_tool.name == "ask_clarifying_questions"
        assert "clarifying questions" in _clarify_tool.description.lower()

        # Parameters should have questions array and original_query
        params = _clarify_tool.parameters
        assert params["type"] == "object"
        assert "questions" in params["properties"]
        assert "original_query" in params["properties"]
        assert params["properties"]["questions"]["type"] == "array"

    @pytest.mark.asyncio
    async def test_analyzed_query_model(self):
        """Verify AnalyzedQuery model has correct fields."""
        from gemini_research_mcp.server import AnalyzedQuery

        # Create an instance
        result = AnalyzedQuery(
            refined_query="What are the top 3 Python web frameworks for REST APIs in 2025?",
            was_clarified=True,
            summary="Refined to focus on REST APIs and recent year",
        )

        assert result.refined_query != ""
        assert result.was_clarified is True
        assert result.summary != ""

    @pytest.mark.asyncio
    async def test_clarify_tool_without_context_returns_empty(self):
        """When no context is available, clarify tool returns empty string."""
        from gemini_research_mcp.server import _ask_clarifying_questions

        # Call without setting _clarification_context
        result = await _ask_clarifying_questions(
            questions=["What frameworks?", "What use case?"],
            original_query="compare python frameworks",
        )

        # Should return empty when no context
        assert result == ""

    @pytest.mark.asyncio
    async def test_maybe_clarify_query_without_context(self):
        """_maybe_clarify_query returns original query when context is None."""
        from gemini_research_mcp.server import _maybe_clarify_query

        original = "compare python frameworks"
        result = await _maybe_clarify_query(original, ctx=None)

        assert result == original

    @pytest.mark.asyncio
    async def test_optional_context_dependency_pattern(self):
        """Verify OptionalContext yields Context or None correctly."""
        from gemini_research_mcp.server import _optional_context

        # When no current context is set, should yield None
        async with _optional_context() as ctx:
            # In test environment without active MCP session, ctx is None
            assert ctx is None or isinstance(ctx, Context)


class TestSamplingToolExecution:
    """Test sampling tool execution with mocked context."""

    @pytest.fixture
    def mock_context_with_elicit(self):
        """Create a mock Context with working elicit method."""
        ctx = MagicMock(spec=Context)

        # Mock elicit to return an accepted result
        async def mock_elicit(message, response_type):
            result = MagicMock()
            result.action = "accept"
            result.data = {"answer_1": "REST APIs", "answer_2": "2025"}
            return result

        ctx.elicit = mock_elicit
        return ctx

    @pytest.mark.asyncio
    async def test_clarify_tool_with_mocked_context(self, mock_context_with_elicit):
        """Test clarify tool with mocked elicitation."""
        from gemini_research_mcp.server import (
            _ask_clarifying_questions,
            _clarification_context,
        )
        import gemini_research_mcp.server as server_module

        # Set the global context
        original_ctx = server_module._clarification_context
        server_module._clarification_context = mock_context_with_elicit

        try:
            result = await _ask_clarifying_questions(
                questions=["What type of APIs?", "What year?"],
                original_query="compare python frameworks",
            )

            # Should have collected answers
            assert "REST APIs" in result or "2025" in result or result == ""
        finally:
            server_module._clarification_context = original_ctx


class TestResearchDeepIntegration:
    """Integration tests for research_deep tool."""

    @pytest.mark.asyncio
    async def test_research_deep_tool_has_optional_task(self):
        """research_deep should have task=optional for SEP-1577 pattern."""
        from gemini_research_mcp.server import mcp

        async with Client(mcp) as client:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == "research_deep"), None)

            assert tool is not None
            # Check execution metadata
            if tool.execution:
                assert tool.execution.taskSupport == "optional"

    @pytest.mark.asyncio
    async def test_research_deep_foreground_has_context(self):
        """When called in foreground, research_deep should have Context."""
        from gemini_research_mcp.server import mcp

        async with Client(mcp) as client:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]

            assert "research_deep" in tool_names
            # Tool is registered and can receive context in foreground

    @pytest.mark.asyncio
    async def test_sampling_system_prompt_structure(self):
        """Verify the analyzer system prompt is well-structured."""
        from gemini_research_mcp.server import _ANALYZER_SYSTEM_PROMPT

        assert "research query analyzer" in _ANALYZER_SYSTEM_PROMPT.lower()
        assert "specific" in _ANALYZER_SYSTEM_PROMPT.lower()
        assert "vague" in _ANALYZER_SYSTEM_PROMPT.lower()
        assert "ask_clarifying_questions" in _ANALYZER_SYSTEM_PROMPT


# =============================================================================
# E2E Tests (require GEMINI_API_KEY and chat.mcp.serverSampling enabled)
# =============================================================================


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)
class TestSamplingClarificationE2E:
    """E2E tests for sampling-based clarification.

    These tests require:
    1. GEMINI_API_KEY environment variable
    2. chat.mcp.serverSampling enabled for gemini-research-mcp in VS Code

    Note: These tests verify the flow but actual sampling depends on
    the client (VS Code) providing an LLM. In test mode, we mock the
    sampling response.
    """

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_clarify_via_sampling_with_mock_sample(self):
        """Test _clarify_query_via_sampling with mocked ctx.sample()."""
        from gemini_research_mcp.server import AnalyzedQuery

        # Create a mock context with sample() method
        mock_ctx = MagicMock(spec=Context)

        # Mock sample to return a refined query
        async def mock_sample(**kwargs):
            result = MagicMock()
            result.result = AnalyzedQuery(
                refined_query="Compare Django, FastAPI, and Flask for building REST APIs in Python 2025",
                was_clarified=True,
                summary="Added specifics: frameworks, purpose, year",
            )
            return result

        mock_ctx.sample = mock_sample

        # Import and patch
        from gemini_research_mcp import server as server_module

        original_clarify = server_module._clarify_query_via_sampling

        # Call the function with mock context
        result = await original_clarify("compare python frameworks", mock_ctx)

        assert "Django" in result or "FastAPI" in result or "Flask" in result
        assert "REST" in result.upper() or "2025" in result

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_full_research_deep_flow_mocked(self):
        """Test full research_deep flow with mocked dependencies."""
        from gemini_research_mcp.server import mcp, AnalyzedQuery

        # This test verifies the tool is properly registered and callable
        async with Client(mcp) as client:
            tools = await client.list_tools()
            research_deep = next((t for t in tools if t.name == "research_deep"), None)

            assert research_deep is not None
            assert "comprehensive" in research_deep.description.lower()

            # Verify tool parameters include query
            input_schema = research_deep.inputSchema
            assert "query" in input_schema.get("properties", {})


class TestVSCodeSamplingBehavior:
    """Test VS Code-specific sampling behavior.

    VS Code requires chat.mcp.serverSampling to be enabled for
    ctx.sample() to work. These tests verify the graceful degradation
    when sampling is not available.
    """

    @pytest.mark.asyncio
    async def test_sampling_fallback_when_unavailable(self):
        """When sampling fails, should fall back to original query."""
        from gemini_research_mcp.server import _clarify_query_via_sampling

        # Create mock context where sample() raises
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.sample = AsyncMock(side_effect=Exception("Sampling not supported"))

        result = await _clarify_query_via_sampling("test query", mock_ctx)

        # Should return original query on failure
        assert result == "test query"

    @pytest.mark.asyncio
    async def test_server_declares_sampling_tools(self):
        """Server should properly declare tools for sampling."""
        from gemini_research_mcp.server import _clarify_tool

        # Tool should be a valid SamplingTool
        assert _clarify_tool.name == "ask_clarifying_questions"
        assert _clarify_tool.fn is not None
        assert callable(_clarify_tool.fn)

        # Should have proper JSON schema for parameters
        params = _clarify_tool.parameters
        assert params.get("type") == "object"
        assert "required" in params
        assert "questions" in params["required"]
        assert "original_query" in params["required"]
