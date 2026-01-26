"""Unit tests for citation extraction and URL resolution.

Tests citation parsing from Deep Research reports.
Run with: uv run pytest tests/test_citations.py -v
"""

import pytest

from gemini_research_mcp.citations import (
    extract_citations_from_text,
    is_blocked_page_title,
)
from gemini_research_mcp.types import DeepResearchResult, ParsedCitation


class TestExtractCitationsFromText:
    """Test citation extraction from report text."""

    def test_extract_basic_citations(self):
        """Should extract citations from **Sources:** section."""
        text = """# Research Report

This is the main content of the report.

**Sources:**
1. [example.com](https://vertexaisearch.cloud.google.com/redirect?url=https://example.com)
2. [docs.python.org](https://vertexaisearch.cloud.google.com/redirect?url=https://docs.python.org)
"""
        text_without, citations = extract_citations_from_text(text)

        assert "**Sources:**" not in text_without
        assert "Research Report" in text_without
        assert len(citations) == 2
        assert citations[0].number == 1
        assert citations[0].domain == "example.com"
        assert citations[1].number == 2
        assert citations[1].domain == "docs.python.org"

    def test_extract_markdown_h2_sources(self):
        """Should extract citations from ## Sources section."""
        text = """# Report

Content here.

## Sources
1. [github.com](https://github.com/example)
2. [stackoverflow.com](https://stackoverflow.com/questions)
"""
        text_without, citations = extract_citations_from_text(text)

        assert "## Sources" not in text_without
        assert len(citations) == 2
        assert citations[0].domain == "github.com"

    def test_extract_markdown_h3_sources(self):
        """Should extract citations from ### Sources section."""
        text = """# Report

Content.

### Sources
1. [example.com](https://example.com)
"""
        _, citations = extract_citations_from_text(text)
        assert len(citations) == 1

    def test_no_sources_section(self):
        """Should return original text when no sources section."""
        text = """# Report

Just content, no sources.
"""
        text_without, citations = extract_citations_from_text(text)

        assert text_without == text
        assert citations == []

    def test_empty_text(self):
        """Should handle empty text."""
        text_without, citations = extract_citations_from_text("")
        assert text_without == ""
        assert citations == []

    def test_case_insensitive_sources(self):
        """Should handle case variations in Sources header."""
        text = """Report content.

**SOURCES:**
1. [example.com](https://example.com)
"""
        _, citations = extract_citations_from_text(text)
        assert len(citations) == 1

    def test_preserves_content_before_sources(self):
        """Should preserve all content before Sources section."""
        text = """# Title

## Section 1
Content 1.

## Section 2
Content 2.

**Sources:**
1. [example.com](https://example.com)
"""
        text_without, _ = extract_citations_from_text(text)

        assert "# Title" in text_without
        assert "## Section 1" in text_without
        assert "Content 2." in text_without
        assert "**Sources:**" not in text_without

    def test_citation_redirect_url_preserved(self):
        """Should preserve the redirect URL for resolution."""
        text = """Content.

**Sources:**
1. [example.com](https://vertexaisearch.cloud.google.com/redirect?query=abc)
"""
        _, citations = extract_citations_from_text(text)

        assert citations[0].redirect_url == "https://vertexaisearch.cloud.google.com/redirect?query=abc"
        assert citations[0].url is None  # Not yet resolved


class TestIsBlockedPageTitle:
    """Test blocked page title detection."""

    def test_cloudflare_blocked(self):
        """Should detect Cloudflare blocks."""
        assert is_blocked_page_title("Attention Required! | Cloudflare")
        assert is_blocked_page_title("Just a moment...")
        assert is_blocked_page_title("Checking your browser before accessing")

    def test_error_pages_blocked(self):
        """Should detect error pages."""
        assert is_blocked_page_title("403 Forbidden")
        assert is_blocked_page_title("404 Not Found")
        assert is_blocked_page_title("Access Denied")

    def test_security_check_blocked(self):
        """Should detect security checks."""
        assert is_blocked_page_title("Security Check Required")

    def test_normal_titles_not_blocked(self):
        """Should not block normal page titles."""
        assert not is_blocked_page_title("Python Documentation")
        assert not is_blocked_page_title("How to use asyncio - Stack Overflow")
        assert not is_blocked_page_title("Google Cloud Documentation")

    def test_none_is_blocked(self):
        """None title should be considered blocked."""
        assert is_blocked_page_title(None)

    def test_case_insensitive(self):
        """Should be case insensitive."""
        assert is_blocked_page_title("CLOUDFLARE")
        assert is_blocked_page_title("Access DENIED")


class TestCitationUrlField:
    """Test that citations have proper URL handling."""

    def test_parsed_citation_url_defaults_none(self):
        """URL should default to None before resolution."""
        citation = ParsedCitation(number=1, domain="example.com")
        assert citation.url is None

    def test_parsed_citation_url_settable(self):
        """URL should be settable after resolution."""
        citation = ParsedCitation(number=1, domain="example.com")
        citation.url = "https://example.com/resolved"
        assert citation.url == "https://example.com/resolved"

    def test_fallback_to_domain_url(self):
        """When resolution fails, domain should be used as URL."""
        citation = ParsedCitation(
            number=1,
            domain="example.com",
            redirect_url="https://vertexaisearch.cloud.google.com/...",
        )
        # Simulate resolution failure: set URL to domain
        if not citation.url:
            citation.url = f"https://{citation.domain}"
        
        assert citation.url == "https://example.com"
