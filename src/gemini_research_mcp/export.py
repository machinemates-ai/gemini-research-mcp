"""
Export research sessions to various formats.

Supports:
- Markdown (.md): Clean, readable format with full citations
- JSON (.json): Machine-readable with all metadata
- DOCX (.docx): Professional Word document with proper formatting

DOCX generation uses skelmis-docx (enhanced python-docx fork) to create
properly formatted Word documents with:
- Table of Contents with hyperlinks
- Headings, paragraphs, bullet/numbered lists
- Bold, italic, and code inline formatting
- External hyperlinks in citations
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from io import BytesIO
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
# DOCX Export (using python-docx)
# =============================================================================


def _parse_markdown_to_docx_elements(text: str) -> list[tuple[str, Any]]:
    """
    Parse markdown text into a list of (element_type, content) tuples.

    Supports:
    - Headings (## Heading)
    - Paragraphs
    - Bullet lists (- item)
    - Numbered lists (1. item)
    - Bold (**text**) and italic (*text*) inline
    """
    elements: list[tuple[str, Any]] = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Skip empty lines
        if not line.strip():
            i += 1
            continue

        # Headings
        if line.startswith("###"):
            elements.append(("heading3", line[3:].strip()))
        elif line.startswith("##"):
            elements.append(("heading2", line[2:].strip()))
        elif line.startswith("#"):
            elements.append(("heading1", line[1:].strip()))

        # Bullet lists
        elif line.strip().startswith("- "):
            elements.append(("bullet", line.strip()[2:]))

        # Numbered lists
        elif re.match(r"^\d+\.\s", line.strip()):
            content = re.sub(r"^\d+\.\s*", "", line.strip())
            elements.append(("number", content))

        # Regular paragraphs (collect consecutive non-special lines)
        else:
            para_lines = [line]
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                # Stop if we hit a special line
                if (
                    not next_line.strip()
                    or next_line.startswith("#")
                    or next_line.strip().startswith("- ")
                    or re.match(r"^\d+\.\s", next_line.strip())
                ):
                    break
                para_lines.append(next_line)
                j += 1
            elements.append(("paragraph", " ".join(para_lines)))
            i = j - 1  # Adjust for loop increment

        i += 1

    return elements


def _add_formatted_text(paragraph: Any, text: str) -> None:
    """
    Add text with inline formatting (bold/italic) to a python-docx paragraph.

    Handles:
    - **bold** or __bold__
    - *italic* or _italic_
    - `code` (monospace)
    """
    # Pattern to match bold, italic, and code
    pattern = r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)"
    parts = re.split(pattern, text)

    for part in parts:
        if not part:
            continue

        # Bold: **text**
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True

        # Italic: *text*
        elif part.startswith("*") and part.endswith("*"):
            run = paragraph.add_run(part[1:-1])
            run.italic = True

        # Code: `text`
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Courier New"

        # Plain text
        else:
            paragraph.add_run(part)


def export_to_docx(
    session: ResearchSession,
    *,
    include_toc: bool = True,
    toc_levels: int = 3,
) -> ExportResult:
    """
    Export a research session to DOCX format.

    Creates a professionally formatted Word document with:
    - Table of Contents with hyperlinks (optional)
    - Title and metadata section
    - Summary (if available)
    - Full research report with proper formatting

    Args:
        session: The research session to export
        include_toc: Whether to include a Table of Contents (default: True)
        toc_levels: Number of heading levels to include in TOC (default: 3)

    Requires skelmis-docx package (enhanced python-docx fork).
    """
    try:
        from skelmis.docx import Document
        from skelmis.docx.shared import Pt
    except ImportError as e:
        raise ImportError(
            "skelmis-docx is required for DOCX export. "
            "Install with: pip install skelmis-docx"
        ) from e

    doc = Document()

    # Title
    title = session.title or session.query[:60]
    doc.add_heading(title, 0)

    # Table of Contents (inserted after title)
    if include_toc:
        toc_para = doc.add_paragraph()
        toc_para.insert_table_of_contents(
            levels=toc_levels,
            format_table_as_links=True,
            show_page_numbers=True,
        )
        doc.add_paragraph()  # Space after TOC

    # Metadata section
    doc.add_heading("Metadata", level=1)

    meta_items = [
        ("Query", session.query),
        ("Created", session.created_at_iso),
    ]

    if session.duration_seconds:
        mins = int(session.duration_seconds // 60)
        secs = int(session.duration_seconds % 60)
        meta_items.append(("Duration", f"{mins}m {secs}s"))

    if session.total_tokens:
        meta_items.append(("Tokens", f"{session.total_tokens:,}"))

    if session.agent_name:
        meta_items.append(("Agent", session.agent_name))

    if session.tags:
        meta_items.append(("Tags", ", ".join(session.tags)))

    if session.notes:
        meta_items.append(("Notes", session.notes))

    meta_items.append(("Interaction ID", session.interaction_id))

    if session.expires_at_iso:
        meta_items.append(("Expires", session.expires_at_iso))

    for label, value in meta_items:
        p = doc.add_paragraph()
        run = p.add_run(f"{label}: ")
        run.bold = True
        p.add_run(str(value))

    # Summary section
    if session.summary:
        doc.add_heading("Summary", level=1)
        doc.add_paragraph(session.summary)

    # Research report section
    if session.report_text:
        doc.add_heading("Research Report", level=1)

        elements = _parse_markdown_to_docx_elements(session.report_text)

        for elem_type, content in elements:
            if elem_type == "heading1":
                doc.add_heading(content, level=1)
            elif elem_type == "heading2":
                doc.add_heading(content, level=2)
            elif elem_type == "heading3":
                doc.add_heading(content, level=3)
            elif elem_type == "bullet":
                p = doc.add_paragraph(style="List Bullet")
                _add_formatted_text(p, content)
            elif elem_type == "number":
                p = doc.add_paragraph(style="List Number")
                _add_formatted_text(p, content)
            elif elem_type == "paragraph":
                p = doc.add_paragraph()
                _add_formatted_text(p, content)

    # Footer
    doc.add_paragraph()
    footer = doc.add_paragraph()
    footer_text = (
        f"Exported from Gemini Research MCP on "
        f"{datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    run = footer.add_run(footer_text)
    run.italic = True
    run.font.size = Pt(9)

    # Save to bytes
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    filename = _generate_filename(session, "docx")

    return ExportResult(
        format=ExportFormat.DOCX,
        filename=filename,
        content=buffer.getvalue(),
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
