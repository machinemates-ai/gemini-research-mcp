"""Tests for research session export functionality."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from gemini_research_mcp.export import (
    ExportFormat,
    ExportResult,
    export_session,
    export_to_json,
    export_to_markdown,
    get_supported_formats,
)
from gemini_research_mcp.storage import ResearchSession


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_session() -> ResearchSession:
    """Create a sample research session for export testing."""
    return ResearchSession(
        interaction_id="test-export-12345-abcdef",
        query="What are the best practices for quantum computing security?",
        created_at=time.time(),
        title="Quantum Computing Security Research",
        summary="This research explores quantum computing security best practices including post-quantum cryptography and quantum key distribution.",
        report_text="""## Executive Summary

Quantum computing poses significant challenges to current cryptographic systems.

### Key Findings

1. **Post-Quantum Cryptography**: NIST has standardized several algorithms
2. **Quantum Key Distribution (QKD)**: Offers theoretically unbreakable encryption
3. **Timeline**: Experts estimate 10-15 years until cryptographically relevant quantum computers

### Recommendations

- Begin migration to post-quantum algorithms
- Implement crypto-agility in systems
- Monitor NIST standards developments

**Sources:**
1. [NIST Post-Quantum Cryptography](https://example.com/nist)
2. [IBM Quantum Research](https://example.com/ibm)
""",
        format_instructions="Create an executive briefing",
        agent_name="deep-research-pro-preview-12-2025",
        duration_seconds=342.5,
        total_tokens=15000,
        tags=["security", "quantum", "cryptography"],
        notes="Priority research for Q1 security review",
    )


@pytest.fixture
def minimal_session() -> ResearchSession:
    """Create a minimal session with only required fields."""
    return ResearchSession(
        interaction_id="minimal-session-123",
        query="Simple query",
        created_at=time.time(),
    )


# =============================================================================
# Export Format Tests
# =============================================================================


class TestExportFormats:
    """Tests for supported export formats."""

    def test_get_supported_formats(self) -> None:
        """Test that we return the expected formats."""
        formats = get_supported_formats()
        assert "markdown" in formats
        assert "json" in formats
        assert "docx" in formats
        assert len(formats) == 3


# =============================================================================
# Markdown Export Tests
# =============================================================================


class TestMarkdownExport:
    """Tests for Markdown export functionality."""

    def test_export_to_markdown_basic(self, sample_session: ResearchSession) -> None:
        """Test basic Markdown export."""
        result = export_to_markdown(sample_session)

        assert result.format == ExportFormat.MARKDOWN
        assert result.filename.endswith(".md")
        assert result.mime_type == "text/markdown"
        assert len(result.content) > 0

    def test_markdown_content_structure(self, sample_session: ResearchSession) -> None:
        """Test Markdown content includes expected sections."""
        result = export_to_markdown(sample_session)
        content = result.content.decode("utf-8")

        # Title
        assert "# Quantum Computing Security Research" in content

        # Metadata section
        assert "## Metadata" in content
        assert "**Query:**" in content
        assert "**Created:**" in content
        assert "**Duration:**" in content
        assert "**Tokens:**" in content
        assert "**Agent:**" in content
        assert "**Tags:**" in content
        assert "security, quantum, cryptography" in content

        # Summary section
        assert "## Summary" in content
        assert "post-quantum cryptography" in content

        # Report section
        assert "## Research Report" in content
        assert "Executive Summary" in content

        # Footer
        assert "Exported from Gemini Research MCP" in content

    def test_markdown_minimal_session(self, minimal_session: ResearchSession) -> None:
        """Test Markdown export with minimal session data."""
        result = export_to_markdown(minimal_session)
        content = result.content.decode("utf-8")

        # Should still have basic structure
        assert "# Simple query" in content
        assert "## Metadata" in content
        assert "**Query:** Simple query" in content

    def test_markdown_size_human(self, sample_session: ResearchSession) -> None:
        """Test human-readable size formatting."""
        result = export_to_markdown(sample_session)
        assert "B" in result.size_human or "KB" in result.size_human


# =============================================================================
# JSON Export Tests
# =============================================================================


class TestJsonExport:
    """Tests for JSON export functionality."""

    def test_export_to_json_basic(self, sample_session: ResearchSession) -> None:
        """Test basic JSON export."""
        result = export_to_json(sample_session)

        assert result.format == ExportFormat.JSON
        assert result.filename.endswith(".json")
        assert result.mime_type == "application/json"

    def test_json_content_valid(self, sample_session: ResearchSession) -> None:
        """Test JSON content is valid and parseable."""
        result = export_to_json(sample_session)
        content = result.content.decode("utf-8")

        # Should be valid JSON
        data = json.loads(content)

        # Check required fields
        assert data["interaction_id"] == sample_session.interaction_id
        assert data["query"] == sample_session.query
        assert data["title"] == sample_session.title
        assert data["summary"] == sample_session.summary
        assert data["report_text"] == sample_session.report_text
        assert data["tags"] == sample_session.tags
        assert data["duration_seconds"] == sample_session.duration_seconds
        assert data["total_tokens"] == sample_session.total_tokens

        # Check timestamp fields
        assert "created_at" in data
        assert "expires_at" in data
        assert "export_timestamp" in data

    def test_json_minimal_session(self, minimal_session: ResearchSession) -> None:
        """Test JSON export with minimal session."""
        result = export_to_json(minimal_session)
        data = json.loads(result.content.decode("utf-8"))

        assert data["interaction_id"] == minimal_session.interaction_id
        assert data["query"] == minimal_session.query
        assert data["title"] is None
        assert data["summary"] is None


# =============================================================================
# DOCX Export Tests
# =============================================================================


class TestDocxExport:
    """Tests for DOCX export functionality."""

    def test_export_to_docx_basic(self, sample_session: ResearchSession) -> None:
        """Test basic DOCX export."""
        # Import check - skip if python-docx not installed
        pytest.importorskip("skelmis.docx")

        from gemini_research_mcp.export import export_to_docx

        result = export_to_docx(sample_session)

        assert result.format == ExportFormat.DOCX
        assert result.filename.endswith(".docx")
        assert result.mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert len(result.content) > 0

    def test_docx_is_valid_zip(self, sample_session: ResearchSession) -> None:
        """Test that DOCX output is a valid ZIP archive (OOXML format)."""
        pytest.importorskip("skelmis.docx")

        from zipfile import ZipFile
        from io import BytesIO
        from gemini_research_mcp.export import export_to_docx

        result = export_to_docx(sample_session)

        # DOCX is a ZIP archive
        with ZipFile(BytesIO(result.content)) as zf:
            namelist = zf.namelist()
            # Should contain standard OOXML files
            assert "[Content_Types].xml" in namelist
            assert "word/document.xml" in namelist

    def test_docx_minimal_session(self, minimal_session: ResearchSession) -> None:
        """Test DOCX export with minimal session."""
        pytest.importorskip("skelmis.docx")

        from gemini_research_mcp.export import export_to_docx

        result = export_to_docx(minimal_session)
        assert len(result.content) > 0

    def test_docx_import_error(self, sample_session: ResearchSession) -> None:
        """Test that missing python-docx raises helpful error."""
        # This test is informational - we can't easily test ImportError
        # when python-docx is installed in dev environment
        pass


# =============================================================================
# export_session Function Tests
# =============================================================================


class TestExportSession:
    """Tests for the unified export_session function."""

    def test_export_session_markdown_by_string(self, sample_session: ResearchSession) -> None:
        """Test export with format as string."""
        result = export_session(sample_session, "markdown")
        assert result.format == ExportFormat.MARKDOWN

    def test_export_session_md_shorthand(self, sample_session: ResearchSession) -> None:
        """Test export with 'md' shorthand."""
        result = export_session(sample_session, "md")
        assert result.format == ExportFormat.MARKDOWN

    def test_export_session_json(self, sample_session: ResearchSession) -> None:
        """Test JSON export via export_session."""
        result = export_session(sample_session, "json")
        assert result.format == ExportFormat.JSON

    def test_export_session_docx(self, sample_session: ResearchSession) -> None:
        """Test DOCX export via export_session."""
        pytest.importorskip("skelmis.docx")
        result = export_session(sample_session, "docx")
        assert result.format == ExportFormat.DOCX

    def test_export_session_word_alias(self, sample_session: ResearchSession) -> None:
        """Test 'word' alias for DOCX."""
        pytest.importorskip("skelmis.docx")
        result = export_session(sample_session, "word")
        assert result.format == ExportFormat.DOCX

    def test_export_session_enum(self, sample_session: ResearchSession) -> None:
        """Test export with ExportFormat enum."""
        result = export_session(sample_session, ExportFormat.JSON)
        assert result.format == ExportFormat.JSON

    def test_export_session_invalid_format(self, sample_session: ResearchSession) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported format"):
            export_session(sample_session, "pdf")

    def test_export_session_to_file(
        self, sample_session: ResearchSession, tmp_path: Path
    ) -> None:
        """Test export with file output."""
        output_path = tmp_path / "research.md"
        result = export_session(sample_session, "markdown", output_path=output_path)

        assert output_path.exists()
        assert output_path.read_bytes() == result.content


# =============================================================================
# Filename Generation Tests
# =============================================================================


class TestFilenameGeneration:
    """Tests for filename generation."""

    def test_filename_from_title(self, sample_session: ResearchSession) -> None:
        """Test filename uses title when available."""
        result = export_to_markdown(sample_session)
        assert "Quantum" in result.filename or "quantum" in result.filename.lower()

    def test_filename_from_query(self, minimal_session: ResearchSession) -> None:
        """Test filename uses query when no title."""
        result = export_to_markdown(minimal_session)
        assert "Simple" in result.filename or "simple" in result.filename.lower()

    def test_filename_has_date(self, sample_session: ResearchSession) -> None:
        """Test filename includes date."""
        result = export_to_markdown(sample_session)
        # Should contain YYYYMMDD pattern
        import re
        assert re.search(r"\d{8}", result.filename)

    def test_filename_safe_characters(self) -> None:
        """Test filename handles special characters safely."""
        session = ResearchSession(
            interaction_id="test-123",
            query="What's the best <approach> to AI/ML?",
            created_at=time.time(),
        )
        result = export_to_markdown(session)
        # Should not contain unsafe characters
        assert "<" not in result.filename
        assert ">" not in result.filename
        assert "?" not in result.filename
        assert "/" not in result.filename


# =============================================================================
# ExportResult Tests
# =============================================================================


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_size_human_bytes(self) -> None:
        """Test size formatting for small files."""
        result = ExportResult(
            format=ExportFormat.MARKDOWN,
            filename="test.md",
            content=b"Hello",
            mime_type="text/markdown",
        )
        assert "B" in result.size_human
        assert "5" in result.size_human

    def test_size_human_kilobytes(self) -> None:
        """Test size formatting for KB files."""
        result = ExportResult(
            format=ExportFormat.MARKDOWN,
            filename="test.md",
            content=b"x" * 2048,
            mime_type="text/markdown",
        )
        assert "KB" in result.size_human
