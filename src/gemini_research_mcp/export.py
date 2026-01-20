"""
Export research sessions to various formats.

Supports:
- Markdown (.md): Clean, readable format with full citations
- JSON (.json): Machine-readable with all metadata
- DOCX (.docx): Professional Word document via Pandoc

DOCX generation uses pypandoc (Pandoc wrapper) for high-quality conversion:
- Automatic Table of Contents generation
- Proper heading hierarchy and styling
- Native list handling (bullet/numbered)
- Full Markdown feature support
- Professional typography
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gemini_research_mcp.config import LOGGER_NAME

if TYPE_CHECKING:
    from gemini_research_mcp.storage import ResearchSession

logger = logging.getLogger(LOGGER_NAME)


class ExportFormat(str, Enum):
    """Supported export formats."""

    MARKDOWN = "markdown"
    JSON = "json"
    DOCX = "docx"


@dataclass
class ExportResult:
    """Result of an export operation."""

    format: ExportFormat
    filename: str
    content: bytes
    mime_type: str

    @property
    def size_human(self) -> str:
        """Human-readable file size."""
        size: float = len(self.content)
        for unit in ["B", "KB", "MB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB"


# =============================================================================
# Markdown Export
# =============================================================================


def _format_markdown_export(session: ResearchSession) -> str:
    """Format a research session as a Markdown document."""
    lines: list[str] = []

    # Title
    title = session.title or session.query[:60]
    lines.append(f"# {title}")
    lines.append("")

    # Metadata block
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Query:** {session.query}")
    lines.append(f"- **Created:** {session.created_at_iso}")
    if session.duration_seconds:
        mins = int(session.duration_seconds // 60)
        secs = int(session.duration_seconds % 60)
        lines.append(f"- **Duration:** {mins}m {secs}s")
    if session.total_tokens:
        lines.append(f"- **Tokens:** {session.total_tokens:,}")
    if session.agent_name:
        lines.append(f"- **Agent:** {session.agent_name}")
    if session.tags:
        lines.append(f"- **Tags:** {', '.join(session.tags)}")
    if session.notes:
        lines.append(f"- **Notes:** {session.notes}")
    lines.append(f"- **Interaction ID:** `{session.interaction_id}`")
    if session.expires_at_iso:
        lines.append(f"- **Expires:** {session.expires_at_iso}")
    lines.append("")

    # Summary
    if session.summary:
        lines.append("## Summary")
        lines.append("")
        lines.append(session.summary)
        lines.append("")

    # Full report
    if session.report_text:
        lines.append("## Research Report")
        lines.append("")
        lines.append(session.report_text)
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(
        f"*Exported from Gemini Research MCP on "
        f"{datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}*"
    )

    return "\n".join(lines)


def export_to_markdown(session: ResearchSession) -> ExportResult:
    """Export a research session to Markdown format."""
    content = _format_markdown_export(session)
    filename = _generate_filename(session, "md")

    return ExportResult(
        format=ExportFormat.MARKDOWN,
        filename=filename,
        content=content.encode("utf-8"),
        mime_type="text/markdown",
    )


# =============================================================================
# JSON Export
# =============================================================================


def _session_to_export_dict(session: ResearchSession) -> dict[str, Any]:
    """Convert session to export-friendly dictionary."""
    return {
        "interaction_id": session.interaction_id,
        "query": session.query,
        "title": session.title,
        "summary": session.summary,
        "report_text": session.report_text,
        "format_instructions": session.format_instructions,
        "agent_name": session.agent_name,
        "duration_seconds": session.duration_seconds,
        "total_tokens": session.total_tokens,
        "tags": session.tags,
        "notes": session.notes,
        "created_at": session.created_at_iso,
        "expires_at": session.expires_at_iso,
        "export_timestamp": datetime.now(tz=UTC).isoformat(),
    }


def export_to_json(session: ResearchSession) -> ExportResult:
    """Export a research session to JSON format."""
    data = _session_to_export_dict(session)
    content = json.dumps(data, indent=2, ensure_ascii=False)
    filename = _generate_filename(session, "json")

    return ExportResult(
        format=ExportFormat.JSON,
        filename=filename,
        content=content.encode("utf-8"),
        mime_type="application/json",
    )


# =============================================================================
# DOCX Export (using pypandoc)
# =============================================================================


def _build_docx_markdown(
    session: ResearchSession,
    *,
    include_toc: bool = True,
    include_cover_page: bool = True,
) -> str:
    """
    Build a complete Markdown document for Pandoc conversion.

    Includes YAML metadata block for title/author/date, optional TOC directive,
    cover page simulation, metadata section, and the research report.
    """
    lines: list[str] = []

    # YAML metadata block (Pandoc uses this for document properties)
    title = session.title or session.query[:60]
    lines.append("---")
    lines.append(f'title: "{title}"')
    lines.append('author: "Gemini Research MCP"')
    lines.append(
        f'date: "{datetime.fromtimestamp(session.created_at, tz=UTC).strftime("%B %d, %Y")}"'
    )
    if include_toc:
        lines.append("toc: true")
        lines.append("toc-depth: 3")
    lines.append("---")
    lines.append("")

    # Cover page content (if enabled)
    if include_cover_page:
        lines.append(f"# {title}")
        lines.append("")

        # Subtitle with query if different from title
        if session.title and session.query != session.title:
            lines.append(f'*Research Query: "{session.query}"*')
            lines.append("")

        # Key metrics
        date_str = datetime.fromtimestamp(session.created_at, tz=UTC).strftime(
            "%B %d, %Y"
        )
        lines.append(f"**Date:** {date_str}")
        lines.append("")

        if session.duration_seconds:
            mins = int(session.duration_seconds // 60)
            secs = int(session.duration_seconds % 60)
            lines.append(f"**Research Duration:** {mins}m {secs}s")
            lines.append("")

        if session.total_tokens:
            lines.append(f"**Tokens Used:** {session.total_tokens:,}")
            lines.append("")

        if session.agent_name:
            lines.append(f"**Agent:** {session.agent_name}")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("\\newpage")
        lines.append("")

    # Document Information section
    lines.append("# Document Information")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| **Research Query** | {session.query} |")
    lines.append(f"| **Created** | {session.created_at_iso} |")

    if session.duration_seconds:
        mins = int(session.duration_seconds // 60)
        secs = int(session.duration_seconds % 60)
        lines.append(f"| **Duration** | {mins}m {secs}s |")

    if session.total_tokens:
        lines.append(f"| **Tokens Used** | {session.total_tokens:,} |")

    if session.agent_name:
        lines.append(f"| **AI Agent** | {session.agent_name} |")

    if session.tags:
        lines.append(f"| **Tags** | {', '.join(session.tags)} |")

    if session.notes:
        lines.append(f"| **Notes** | {session.notes} |")

    lines.append(f"| **Session ID** | `{session.interaction_id}` |")

    if session.expires_at_iso:
        lines.append(f"| **Expires** | {session.expires_at_iso} |")

    lines.append("")

    # Executive Summary (if available)
    if session.summary:
        lines.append("# Executive Summary")
        lines.append("")
        lines.append("> " + session.summary.replace("\n", "\n> "))
        lines.append("")

    # Main Research Report
    if session.report_text:
        lines.append("\\newpage")
        lines.append("")
        lines.append("# Research Report")
        lines.append("")
        lines.append(session.report_text)
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(
        f"*This document was generated by Gemini Research MCP on "
        f"{datetime.now(tz=UTC).strftime('%Y-%m-%d at %H:%M:%S UTC')}*"
    )
    lines.append("")
    lines.append(f"*Session ID: {session.interaction_id}*")

    return "\n".join(lines)


def export_to_docx(
    session: ResearchSession,
    *,
    include_toc: bool = True,
    include_cover_page: bool = True,
    toc_levels: int = 3,
) -> ExportResult:
    """
    Export a research session to DOCX format using Pandoc.

    Creates a professionally formatted Word document with:
    - Cover page with title and metadata (optional)
    - Automatic Table of Contents (optional)
    - Executive summary as blockquote
    - Metadata table
    - Full research report with proper formatting
    - Support for all Markdown features

    Args:
        session: The research session to export
        include_toc: Whether to include a Table of Contents (default: True)
        include_cover_page: Whether to include a cover page (default: True)
        toc_levels: Number of heading levels to include in TOC (default: 3)

    Requires pypandoc package with Pandoc binary.
    Install with: pip install 'gemini-research-mcp[docx]'
    """
    try:
        import pypandoc  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "pypandoc is required for DOCX export. "
            "Install with: pip install 'gemini-research-mcp[docx]'"
        ) from e

    # Build the Markdown source
    markdown_source = _build_docx_markdown(
        session,
        include_toc=include_toc,
        include_cover_page=include_cover_page,
    )

    # Pandoc extra arguments
    extra_args = [
        "--standalone",
        f"--toc-depth={toc_levels}",
    ]

    if include_toc:
        extra_args.append("--toc")

    # Convert using pypandoc
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pypandoc.convert_text(
            markdown_source,
            "docx",
            format="markdown",
            outputfile=tmp_path,
            extra_args=extra_args,
        )

        # Read the generated DOCX
        content = Path(tmp_path).read_bytes()
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

    filename = _generate_filename(session, "docx")

    return ExportResult(
        format=ExportFormat.DOCX,
        filename=filename,
        content=content,
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _generate_filename(session: ResearchSession, extension: str) -> str:
    """Generate a safe filename from session metadata."""
    # Use title or first 40 chars of query
    base = session.title or session.query[:40]

    # Clean up for filename
    safe = re.sub(r"[^\w\s-]", "", base)  # Remove special chars
    safe = re.sub(r"\s+", "_", safe)  # Replace spaces with underscores
    safe = safe[:50]  # Limit length

    # Add timestamp
    timestamp = datetime.fromtimestamp(session.created_at, tz=UTC).strftime("%Y%m%d")

    return f"{safe}_{timestamp}.{extension}"


def export_session(
    session: ResearchSession,
    format: ExportFormat | str,
    output_path: Path | str | None = None,
) -> ExportResult:
    """
    Export a research session to the specified format.

    Args:
        session: The research session to export
        format: Export format (markdown, json, docx)
        output_path: Optional path to save the file (if None, returns bytes only)

    Returns:
        ExportResult with format, filename, content bytes, and mime_type
    """
    # Normalize format
    if isinstance(format, str):
        format_str = format.lower()
        if format_str in ("md", "markdown"):
            export_format = ExportFormat.MARKDOWN
        elif format_str == "json":
            export_format = ExportFormat.JSON
        elif format_str in ("docx", "word"):
            export_format = ExportFormat.DOCX
        else:
            raise ValueError(f"Unsupported format: {format}. Use: markdown, json, docx")
    else:
        export_format = format

    # Export
    if export_format == ExportFormat.MARKDOWN:
        result = export_to_markdown(session)
    elif export_format == ExportFormat.JSON:
        result = export_to_json(session)
    elif export_format == ExportFormat.DOCX:
        result = export_to_docx(session)
    else:
        raise ValueError(f"Unsupported format: {export_format}")

    # Save to file if path provided
    if output_path is not None:
        path = Path(output_path)
        path.write_bytes(result.content)
        logger.info("📄 Exported to %s (%s)", path, result.size_human)

    return result


# =============================================================================
# Convenience Functions
# =============================================================================


def get_supported_formats() -> list[str]:
    """Return list of supported export formats."""
    return [f.value for f in ExportFormat]
