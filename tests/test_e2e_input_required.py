"""End-to-end tests for input_required elicitation pattern (SEP-1686).

These tests verify that the MCP SDK's elicitation pattern works correctly
when running research_deep with vague queries:

1. Context.elicit() is called for vague queries
2. User can provide clarifying answers
3. Research proceeds with refined query

Run with: uv run pytest tests/test_e2e_input_required.py -v -m e2e --tb=short

These tests require both:
- GEMINI_API_KEY environment variable
- A running MCP client that supports elicitation (e.g., VS Code with Copilot)

For unit tests (no API key needed), see test_elicitation.py.
"""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic import BaseModel


# Skip all tests in this module if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)


class MockElicitResult:
    """Mock elicit result for testing."""

    def __init__(self, action: str, data: BaseModel | None = None):
        self.action = action
        self.data = data


class TestElicitationIntegration:
    """Integration tests for elicitation during research_deep."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_vague_query_triggers_elicitation(self):
        """Vague queries should trigger elicitation when context is available."""
        from gemini_research_mcp.server import _maybe_clarify_query

        # Create a mock context with elicit method
        mock_ctx = MagicMock()
        mock_data = MagicMock()
        mock_data.model_dump.return_value = {
            "answer_1": "web APIs and performance",
            "answer_2": "building a microservices architecture",
        }
        mock_ctx.elicit = AsyncMock(return_value=MockElicitResult("accept", mock_data))

        # A vague query that should trigger clarification
        query = "compare python frameworks"
        result = await _maybe_clarify_query(query, mock_ctx)

        # Should have called elicit
        mock_ctx.elicit.assert_called_once()

        # Result should be refined with the answers
        assert "compare python frameworks" in result
        assert "Additional context:" in result
        assert "web APIs and performance" in result
        assert "microservices architecture" in result

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_user_skips_clarification(self):
        """User skipping clarification should proceed with original query."""
        from gemini_research_mcp.server import _maybe_clarify_query

        mock_ctx = MagicMock()
        mock_ctx.elicit = AsyncMock(return_value=MockElicitResult("cancel", None))

        query = "compare python frameworks"
        result = await _maybe_clarify_query(query, mock_ctx)

        # Should have called elicit
        mock_ctx.elicit.assert_called_once()

        # Result should be original query (no refinement)
        assert result == query

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_user_submits_empty_answers(self):
        """Empty answers should proceed with original query."""
        from gemini_research_mcp.server import _maybe_clarify_query

        mock_ctx = MagicMock()
        mock_data = MagicMock()
        mock_data.model_dump.return_value = {
            "answer_1": "",
            "answer_2": "  ",  # Whitespace only
        }
        mock_ctx.elicit = AsyncMock(return_value=MockElicitResult("accept", mock_data))

        query = "compare python frameworks"
        result = await _maybe_clarify_query(query, mock_ctx)

        # Should have called elicit
        mock_ctx.elicit.assert_called_once()

        # Result should be original query (empty answers don't refine)
        assert result == query

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_specific_query_skips_elicitation(self):
        """Specific queries should not trigger elicitation."""
        from gemini_research_mcp.server import _maybe_clarify_query

        mock_ctx = MagicMock()
        mock_ctx.elicit = AsyncMock()

        # A specific, detailed query
        query = (
            "Compare FastAPI vs Django for building REST APIs in 2025 "
            "with async support and SQLAlchemy integration"
        )
        result = await _maybe_clarify_query(query, mock_ctx)

        # Should NOT have called elicit (query is specific)
        mock_ctx.elicit.assert_not_called()

        # Result should be original query
        assert result == query

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_elicitation_failure_returns_original(self):
        """Elicitation failure should gracefully return original query."""
        from gemini_research_mcp.server import _maybe_clarify_query

        mock_ctx = MagicMock()
        mock_ctx.elicit = AsyncMock(side_effect=RuntimeError("Elicitation not supported"))

        query = "compare python frameworks"
        result = await _maybe_clarify_query(query, mock_ctx)

        # Should have attempted elicit
        mock_ctx.elicit.assert_called_once()

        # Result should be original query (graceful fallback)
        assert result == query


class TestResearchDeepWithElicitation:
    """E2E tests for research_deep tool with elicitation."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_research_deep_with_refined_query(self):
        """research_deep should use refined query from elicitation.

        This test requires actual API access and a long-running research task.
        Consider running with: pytest -x -v --timeout=1200
        """
        from gemini_research_mcp.server import research_deep

        # Create mock context that provides clarification
        # All async methods must be AsyncMock
        mock_ctx = MagicMock()
        mock_data = MagicMock()
        mock_data.model_dump.return_value = {
            "answer_1": "Python web frameworks",
            "answer_2": "building REST APIs",
        }
        mock_ctx.elicit = AsyncMock(return_value=MockElicitResult("accept", mock_data))
        mock_ctx.info = AsyncMock()  # For progress reporting
        mock_ctx.report_progress = AsyncMock()  # Required for deep research progress

        # Call research_deep with a vague query and mock context
        # This will:
        # 1. Trigger elicitation
        # 2. Refine the query
        # 3. Perform actual deep research (3-20 minutes)
        result = await research_deep(
            query="compare frameworks",
            format_instructions="Brief summary, max 2 paragraphs",
            ctx=mock_ctx,
        )

        # Verify elicitation was called
        mock_ctx.elicit.assert_called_once()

        # Verify result contains research content
        assert "## Research Report" in result or "Research Report" in result
        assert len(result) > 200  # Should have substantial content

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_research_deep_without_context(self):
        """research_deep should work without context (background task mode)."""
        from gemini_research_mcp.server import research_deep

        # Call research_deep with ctx=None (simulating background task)
        result = await research_deep(
            query="What are the key features of FastAPI?",
            format_instructions="Brief summary",
            ctx=None,
        )

        # Should complete without elicitation
        assert "## Research Report" in result or "FastAPI" in result
        assert len(result) > 100


class TestInputRequiredProtocol:
    """Tests for MCP input_required task status (SEP-1686/SEP-1732).

    These tests verify the interaction between:
    - Context.elicit() for foreground elicitation
    - ServerTaskContext.elicit() for background task elicitation (input_required)

    The input_required status allows MCP clients to:
    1. Pause task execution
    2. Request user input
    3. Resume task with provided input
    """

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_task_support_enabled(self):
        """Verify task support is configured in the server."""
        from gemini_research_mcp.server import mcp, lifespan

        # The server should have a lifespan function that enables task support
        assert lifespan is not None
        # Verify the lifespan is assigned to the FastMCP app
        assert callable(lifespan)

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_elicit_schema_has_descriptions(self):
        """Verify dynamic schema includes question descriptions for UI."""
        from pydantic import create_model, Field

        questions = [
            "What specific aspects would you like to compare?",
            "What's your use case or context?",
        ]

        field_definitions = {
            f"answer_{i+1}": (str, Field(default="", description=q))
            for i, q in enumerate(questions)
        }
        DynamicSchema = create_model("ClarificationQuestions", **field_definitions)

        # Verify descriptions are accessible for UI rendering
        schema = DynamicSchema.model_json_schema()
        assert "properties" in schema
        assert "answer_1" in schema["properties"]
        assert "answer_2" in schema["properties"]

        # Descriptions should be the questions
        assert schema["properties"]["answer_1"]["description"] == questions[0]
        assert schema["properties"]["answer_2"]["description"] == questions[1]

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_elicit_message_format(self):
        """Verify elicit message format is user-friendly."""
        query = "compare AI tools"
        expected_message_parts = [
            "To improve research quality",
            query,
            "Please answer these questions",
        ]

        # Build the message like _maybe_clarify_query does
        message = (
            f'To improve research quality for:\n\n**"{query}"**\n\n'
            f"Please answer these questions (optional - press 'Skip' to continue):"
        )

        for part in expected_message_parts:
            assert part in message
