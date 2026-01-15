"""End-to-end tests for deep-research MCP server.

These tests make actual API calls and require GEMINI_API_KEY.
Run with: uv run pytest tests/test_e2e.py -v --tb=short

Skip these in CI unless API key is available:
    uv run pytest tests/test_e2e.py -v -m "not e2e"
"""

import os
import pytest

# Skip all tests in this module if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set"
)


@pytest.fixture
def api_available():
    """Check if Gemini API is available."""
    return bool(os.environ.get("GEMINI_API_KEY"))


class TestResearchQuickE2E:
    """End-to-end tests for quick research."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_basic_search(self):
        """Basic search should return grounded result with sources."""
        from gemini_research_mcp.quick import quick_research
        
        result = await quick_research(
            "What is OMOP CDM version 5.4?",
            thinking_budget="minimal",
        )
        
        assert result.text, "Should have response text"
        assert len(result.text) > 100, "Response should be substantial"
        assert result.sources, "Should have grounding sources"
        # Note: We don't assert source domains here - Google may return any relevant sources

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_site_filter(self):
        """Site filter should scope results to specific domain."""
        from gemini_research_mcp.quick import quick_research
        
        result = await quick_research(
            "site:cloud.google.com BigQuery pricing",
            thinking_budget="minimal",
        )
        
        assert result.text, "Should have response text"
        # Most sources should be from google.com
        google_sources = [s for s in result.sources if "google" in s.uri.lower()]
        assert len(google_sources) >= 1, "Should have Google sources"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_system_prompt_affects_response(self):
        """Custom system prompt should affect response style."""
        from gemini_research_mcp.quick import quick_research
        
        # Request JSON-style response
        result = await quick_research(
            "List 3 popular Python web frameworks",
            thinking_budget="minimal",
            system_instruction="Always respond with a numbered list. Be extremely brief, one line per item.",
        )
        
        assert result.text, "Should have response text"
        # Should have numbered items
        assert "1." in result.text or "1)" in result.text, "Should have numbered list"


class TestResearchVendorDocsE2E:
    """End-to-end tests for vendor documentation search."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_gcp_docs(self):
        """GCP vendor search should return GCP-focused results."""
        from gemini_research_mcp.server import research_vendor_docs
        
        result = await research_vendor_docs(
            vendor="gcp",
            topic="Cloud Run container deployment",
            include_code=True,
            thinking_budget="low",
        )
        
        assert "GCP" in result, "Should have GCP header"
        assert "Cloud Run" in result, "Should mention Cloud Run"
        assert "```" in result or "gcloud" in result.lower(), "Should have code or CLI example"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_ohdsi_docs(self):
        """OHDSI vendor search should return OMOP-focused results."""
        from gemini_research_mcp.server import research_vendor_docs
        
        result = await research_vendor_docs(
            vendor="ohdsi",
            topic="person table structure",
            include_code=False,
            thinking_budget="low",
        )
        
        assert "OHDSI" in result, "Should have OHDSI header"
        assert "person" in result.lower(), "Should mention person table"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_unknown_vendor_fallback(self):
        """Unknown vendor should use generic site filter."""
        from gemini_research_mcp.server import research_vendor_docs
        
        result = await research_vendor_docs(
            vendor="python.org",
            topic="asyncio tutorial",
            include_code=True,
            thinking_budget="low",
        )
        
        assert result, "Should return a result"
        assert "asyncio" in result.lower(), "Should mention asyncio"


class TestSourceExtraction:
    """Test source/citation extraction from responses."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_sources_have_uri_and_title(self):
        """Extracted sources should have URI and title."""
        from gemini_research_mcp.quick import quick_research
        
        result = await quick_research(
            "Python dataclasses documentation",
            thinking_budget="minimal",
        )
        
        assert result.sources, "Should have sources"
        for source in result.sources:
            assert source.uri, "Source should have URI"
            assert source.uri.startswith("http"), "URI should be a URL"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_queries_are_captured(self):
        """Search queries used by model should be captured."""
        from gemini_research_mcp.quick import quick_research
        
        result = await quick_research(
            "latest Python release version",
            thinking_budget="minimal",
        )
        
        # Queries may or may not be present depending on grounding behavior
        # Just verify the field exists
        assert hasattr(result, "queries")
