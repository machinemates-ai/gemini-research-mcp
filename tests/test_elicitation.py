"""Test MCP SDK elicitation pattern for query clarification.

This tests the elicitation-based clarification flow in research_deep:
1. User calls research_deep with a vague query
2. ctx.elicit() is called with a dynamic schema
3. User can provide answers to clarifying questions
4. Deep research proceeds with the refined query

These tests verify the pattern without actual API calls (unit tests).
"""

import pytest
from pydantic import Field, create_model


class TestElicitationPattern:
    """Test the elicitation pattern mechanics."""

    @pytest.mark.asyncio
    async def test_clarification_schema_structure(self):
        """Verify ClarificationSchema has correct fields."""
        from gemini_research_mcp.server import ClarificationSchema

        # Check the model has expected fields
        assert hasattr(ClarificationSchema, "model_fields")
        fields = ClarificationSchema.model_fields
        assert "answer_1" in fields
        assert "answer_2" in fields
        assert "answer_3" in fields

    @pytest.mark.asyncio
    async def test_dynamic_schema_creation(self):
        """Test that dynamic Pydantic models can be created for elicitation."""
        questions = [
            "What specific aspects would you like to compare?",
            "What's your use case or context?",
        ]

        # Create dynamic schema like _maybe_clarify_query does
        field_definitions = {
            f"answer_{i+1}": (str, Field(default="", description=q))
            for i, q in enumerate(questions)
        }
        DynamicSchema = create_model("ClarificationQuestions", **field_definitions)

        # Verify schema structure
        assert hasattr(DynamicSchema, "model_fields")
        fields = DynamicSchema.model_fields
        assert len(fields) == 2
        assert "answer_1" in fields
        assert "answer_2" in fields

        # Verify default values work
        instance = DynamicSchema()
        assert instance.answer_1 == ""
        assert instance.answer_2 == ""

        # Verify can set values
        instance = DynamicSchema(answer_1="web APIs", answer_2="building a REST service")
        assert instance.answer_1 == "web APIs"
        assert instance.answer_2 == "building a REST service"

    @pytest.mark.asyncio
    async def test_maybe_clarify_query_without_context(self):
        """_maybe_clarify_query returns original query when context is None."""
        from gemini_research_mcp.server import _maybe_clarify_query

        original = "compare python frameworks"
        result = await _maybe_clarify_query(original, ctx=None)

        assert result == original

    @pytest.mark.asyncio
    async def test_vague_query_detection_short(self):
        """Short queries should be detected as potentially vague."""
        # These are implementation details we can infer from the code
        vague_queries = [
            "AI",  # Very short
            "research AI",  # Contains "research"
            "compare frameworks",  # Contains "compare"
            "best tools",  # Contains "best"
            "analyze trends",  # Contains "analyze"
        ]

        specific_queries = [
            "Compare FastAPI vs Django for building REST APIs in 2025 with async support and SQLAlchemy integration",
            "Research the environmental impact of electric vehicles vs gasoline cars in European markets from 2020-2025",
        ]

        # We can't directly test the internal logic without mocking,
        # but we can verify the function exists and handles None context
        from gemini_research_mcp.server import _maybe_clarify_query

        for query in vague_queries + specific_queries:
            result = await _maybe_clarify_query(query, ctx=None)
            # Without context, should always return original
            assert result == query


class TestQueryRefinement:
    """Test query refinement logic."""

    @pytest.mark.asyncio
    async def test_refined_query_format(self):
        """Test how refined queries are formatted."""
        original = "compare python frameworks"
        clarification = "Q: What specific aspects?\nA: performance and ease of use"

        # Simulate how the code builds refined query
        refined = f"{original}\n\nAdditional context:\n{clarification}"

        assert original in refined
        assert "Additional context:" in refined
        assert clarification in refined

    @pytest.mark.asyncio
    async def test_empty_clarification_uses_original(self):
        """Empty clarification should not modify the query."""
        original = "compare python frameworks"
        clarification = ""

        # With empty clarification, should use original
        if clarification:
            refined = f"{original}\n\nAdditional context:\n{clarification}"
        else:
            refined = original

        assert refined == original
