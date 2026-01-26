"""
Tests for the clarifier module (LLM-driven query clarification).

These are unit tests that don't require API keys - they test the
data structures and helper functions.
"""

import pytest

from gemini_research_mcp.clarifier import (
    CONFIDENCE_THRESHOLD,
    ClarifyingQuestion,
    QueryAnalysis,
    RefinedQuery,
    should_clarify,
)


class TestClarifyingQuestion:
    """Tests for ClarifyingQuestion dataclass."""

    def test_construction_minimal(self) -> None:
        """Test minimal construction."""
        q = ClarifyingQuestion(
            question="What time period?",
            purpose="Helps focus the research",
        )
        assert q.question == "What time period?"
        assert q.purpose == "Helps focus the research"
        assert q.priority == 1  # default
        assert q.default_answer is None

    def test_construction_full(self) -> None:
        """Test full construction with all fields."""
        q = ClarifyingQuestion(
            question="What industry?",
            purpose="Narrows scope",
            priority=2,
            default_answer="Technology",
        )
        assert q.question == "What industry?"
        assert q.purpose == "Narrows scope"
        assert q.priority == 2
        assert q.default_answer == "Technology"


class TestQueryAnalysis:
    """Tests for QueryAnalysis dataclass."""

    def test_construction_minimal(self) -> None:
        """Test minimal construction."""
        analysis = QueryAnalysis(
            needs_clarification=True,
            confidence=0.5,
        )
        assert analysis.needs_clarification is True
        assert analysis.confidence == 0.5
        assert analysis.questions == []
        assert analysis.detected_intent is None
        assert analysis.ambiguities == []

    def test_construction_full(self) -> None:
        """Test full construction with all fields."""
        questions = [
            ClarifyingQuestion(question="Q1", purpose="P1"),
            ClarifyingQuestion(question="Q2", purpose="P2"),
        ]
        analysis = QueryAnalysis(
            needs_clarification=True,
            confidence=0.3,
            questions=questions,
            detected_intent="Research AI trends",
            ambiguities=["Time period unclear", "Industry not specified"],
        )
        assert analysis.needs_clarification is True
        assert analysis.confidence == 0.3
        assert len(analysis.questions) == 2
        assert analysis.detected_intent == "Research AI trends"
        assert len(analysis.ambiguities) == 2


class TestRefinedQuery:
    """Tests for RefinedQuery dataclass."""

    def test_construction_minimal(self) -> None:
        """Test minimal construction."""
        refined = RefinedQuery(
            original_query="AI trends",
            refined_query="AI trends in healthcare for 2024",
            context_summary="Focused on healthcare, 2024",
        )
        assert refined.original_query == "AI trends"
        assert refined.refined_query == "AI trends in healthcare for 2024"
        assert refined.context_summary == "Focused on healthcare, 2024"
        assert refined.answers == {}

    def test_construction_full(self) -> None:
        """Test full construction with answers."""
        refined = RefinedQuery(
            original_query="AI trends",
            refined_query="AI trends in healthcare for 2024",
            context_summary="Focused on healthcare, 2024",
            answers={
                "What industry?": "Healthcare",
                "What time period?": "2024",
            },
        )
        assert len(refined.answers) == 2
        assert refined.answers["What industry?"] == "Healthcare"


class TestShouldClarify:
    """Tests for should_clarify() function."""

    def test_high_confidence_no_clarify(self) -> None:
        """High confidence should not trigger clarification."""
        analysis = QueryAnalysis(
            needs_clarification=True,
            confidence=0.8,  # Above threshold
            questions=[ClarifyingQuestion(question="Q", purpose="P")],
        )
        assert should_clarify(analysis) is False

    def test_threshold_boundary(self) -> None:
        """Test boundary at exactly the threshold."""
        # At threshold - should not clarify
        analysis = QueryAnalysis(
            needs_clarification=True,
            confidence=CONFIDENCE_THRESHOLD,
            questions=[ClarifyingQuestion(question="Q", purpose="P")],
        )
        assert should_clarify(analysis) is False

        # Just below threshold - should clarify
        analysis_below = QueryAnalysis(
            needs_clarification=True,
            confidence=CONFIDENCE_THRESHOLD - 0.01,
            questions=[ClarifyingQuestion(question="Q", purpose="P")],
        )
        assert should_clarify(analysis_below) is True

    def test_no_questions_no_clarify(self) -> None:
        """No questions means no clarification needed."""
        analysis = QueryAnalysis(
            needs_clarification=True,
            confidence=0.3,  # Low confidence
            questions=[],  # But no questions
        )
        assert should_clarify(analysis) is False

    def test_needs_clarification_false(self) -> None:
        """If needs_clarification is False, don't clarify."""
        analysis = QueryAnalysis(
            needs_clarification=False,  # Explicitly False
            confidence=0.3,
            questions=[ClarifyingQuestion(question="Q", purpose="P")],
        )
        assert should_clarify(analysis) is False

    def test_typical_clarification_case(self) -> None:
        """Test typical case where clarification is needed."""
        analysis = QueryAnalysis(
            needs_clarification=True,
            confidence=0.4,
            questions=[
                ClarifyingQuestion(question="What time period?", purpose="Focus scope"),
                ClarifyingQuestion(question="What industry?", purpose="Narrow domain"),
            ],
        )
        assert should_clarify(analysis) is True


class TestConfidenceThreshold:
    """Tests for the confidence threshold constant."""

    def test_threshold_value(self) -> None:
        """Threshold should be a reasonable value."""
        assert 0.5 <= CONFIDENCE_THRESHOLD <= 0.9
        # Default is 0.7 based on research (3-5 questions is optimal)
        assert CONFIDENCE_THRESHOLD == 0.7
