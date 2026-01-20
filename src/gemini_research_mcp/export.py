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
    - **bold** (asterisk style)
    - *italic* (asterisk style)
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


def _setup_docx_styles(doc: Any) -> None:
    """Configure document styles for professional appearance."""
    try:
        from skelmis.docx.shared import Pt, RGBColor
        from skelmis.docx.enum.style import WD_STYLE_TYPE
    except ImportError:
        return

    styles = doc.styles

    # Customize Heading 1 - Dark blue, larger
    try:
        h1 = styles["Heading 1"]
        h1.font.color.rgb = RGBColor(0x1A, 0x47, 0x80)  # Dark blue
        h1.font.size = Pt(18)
        h1.font.bold = True
    except KeyError:
        pass

    # Customize Heading 2 - Medium blue
    try:
        h2 = styles["Heading 2"]
        h2.font.color.rgb = RGBColor(0x2E, 0x5C, 0x9E)  # Medium blue
        h2.font.size = Pt(14)
        h2.font.bold = True
    except KeyError:
        pass

    # Customize Heading 3 - Light blue
    try:
        h3 = styles["Heading 3"]
        h3.font.color.rgb = RGBColor(0x3D, 0x7E, 0xBF)  # Light blue
        h3.font.size = Pt(12)
        h3.font.bold = True
    except KeyError:
        pass


def _add_cover_page(doc: Any, session: Any) -> None:
    """Add a professional cover page."""
    try:
        from skelmis.docx.shared import Pt
        from skelmis.docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        return

    # Add some vertical space at the top
    for _ in range(6):
        doc.add_paragraph()

    # Main title - centered, large
    title = session.title or session.query[:60]
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run(title)
    title_run.bold = True
    title_run.font.size = Pt(28)

    # Subtitle - the query if different from title
    if session.title and session.query != session.title:
        doc.add_paragraph()
        subtitle = doc.add_paragraph()
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sub_run = subtitle.add_run(f'Research Query: "{session.query}"')
        sub_run.italic = True
        sub_run.font.size = Pt(12)

    # Spacer
    for _ in range(4):
        doc.add_paragraph()

    # Research metadata summary - centered
    info_para = doc.add_paragraph()
    info_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Date
    date_run = info_para.add_run(
        datetime.fromtimestamp(session.created_at, tz=UTC).strftime("%B %d, %Y")
    )
    date_run.font.size = Pt(14)

    # Duration and tokens
    if session.duration_seconds:
        mins = int(session.duration_seconds // 60)
        secs = int(session.duration_seconds % 60)
        info_para.add_run("\n")
        dur_run = info_para.add_run(f"Research Duration: {mins}m {secs}s")
        dur_run.font.size = Pt(11)

    if session.total_tokens:
        info_para.add_run("\n")
        tok_run = info_para.add_run(f"Tokens Used: {session.total_tokens:,}")
        tok_run.font.size = Pt(11)

    # Agent
    if session.agent_name:
        info_para.add_run("\n")
        agent_run = info_para.add_run(f"Agent: {session.agent_name}")
        agent_run.font.size = Pt(11)

    # Add more space before footer
    for _ in range(8):
        doc.add_paragraph()

    # Generated by footer - centered
    footer_para = doc.add_paragraph()
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_run = footer_para.add_run("Generated by Gemini Research MCP")
    footer_run.italic = True
    footer_run.font.size = Pt(10)

    # Page break after cover
    doc.add_page_break()


def _add_static_toc(doc: Any, session: Any, toc_levels: int = 3) -> None:
    """
    Add a static Table of Contents by scanning the report headings.

    Unlike Word's dynamic TOC field, this creates actual text entries
    that don't require manual updating.
    """
    try:
        from skelmis.docx.shared import Pt, Inches
        from skelmis.docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        return

    doc.add_heading("Table of Contents", level=1)
    doc.add_paragraph()  # Space after heading

    # Build TOC entries from document structure
    toc_entries: list[tuple[int, str]] = []

    # Fixed sections that will appear
    toc_entries.append((1, "Document Information"))

    if session.summary:
        toc_entries.append((1, "Executive Summary"))

    if session.report_text:
        toc_entries.append((1, "Research Report"))

        # Parse headings from the report text
        elements = _parse_markdown_to_docx_elements(session.report_text)
        for elem_type, content in elements:
            if elem_type == "heading1" and toc_levels >= 1:
                toc_entries.append((1, content))
            elif elem_type == "heading2" and toc_levels >= 2:
                toc_entries.append((2, content))
            elif elem_type == "heading3" and toc_levels >= 3:
                toc_entries.append((3, content))

    toc_entries.append((1, "Citations"))

    # Render TOC entries with indentation
    for level, title in toc_entries:
        para = doc.add_paragraph()

        # Indentation based on level
        if level == 1:
            indent = 0
            font_size = 12
            bold = True
        elif level == 2:
            indent = 0.25
            font_size = 11
            bold = False
        else:  # level 3
            indent = 0.5
            font_size = 10
            bold = False

        para.paragraph_format.left_indent = Inches(indent)
        para.paragraph_format.space_after = Pt(4)

        run = para.add_run(title)
        run.font.size = Pt(font_size)
        run.bold = bold

    doc.add_paragraph()  # Space after TOC
    doc.add_page_break()


def _add_metadata_table(doc: Any, session: Any) -> None:
    """Add metadata as a formatted table."""
    try:
        from skelmis.docx.shared import Pt, Inches, RGBColor
        from skelmis.docx.oxml.ns import qn
        from skelmis.docx.oxml import OxmlElement
    except ImportError:
        return

    doc.add_heading("Document Information", level=1)

    meta_items = [
        ("Research Query", session.query),
        ("Created", session.created_at_iso),
    ]

    if session.duration_seconds:
        mins = int(session.duration_seconds // 60)
        secs = int(session.duration_seconds % 60)
        meta_items.append(("Duration", f"{mins}m {secs}s"))

    if session.total_tokens:
        meta_items.append(("Tokens Used", f"{session.total_tokens:,}"))

    if session.agent_name:
        meta_items.append(("AI Agent", session.agent_name))

    if session.tags:
        meta_items.append(("Tags", ", ".join(session.tags)))

    if session.notes:
        meta_items.append(("Notes", session.notes))

    meta_items.append(("Session ID", session.interaction_id))

    if session.expires_at_iso:
        meta_items.append(("Expires", session.expires_at_iso))

    # Create table
    table = doc.add_table(rows=len(meta_items), cols=2)
    table.style = "Table Grid"

    # Set column widths
    for row_idx, (label, value) in enumerate(meta_items):
        row = table.rows[row_idx]
        label_cell = row.cells[0]
        value_cell = row.cells[1]

        # Label cell - bold
        label_para = label_cell.paragraphs[0]
        label_run = label_para.add_run(label)
        label_run.bold = True
        label_run.font.size = Pt(10)

        # Value cell
        value_para = value_cell.paragraphs[0]
        value_run = value_para.add_run(str(value))
        value_run.font.size = Pt(10)

        # Shade label cell with light gray
        try:
            shading = OxmlElement("w:shd")
            shading.set(qn("w:fill"), "F0F0F0")
            label_cell._tc.get_or_add_tcPr().append(shading)
        except Exception:
            pass  # Skip shading if not supported

    doc.add_paragraph()  # Space after table


def _add_executive_summary(doc: Any, summary: str) -> None:
    """Add an executive summary with special formatting."""
    try:
        from skelmis.docx.shared import Pt, Inches
        from skelmis.docx.oxml.ns import qn
        from skelmis.docx.oxml import OxmlElement
    except ImportError:
        doc.add_heading("Executive Summary", level=1)
        doc.add_paragraph(summary)
        return

    doc.add_heading("Executive Summary", level=1)

    # Create a single-cell table for the summary box
    table = doc.add_table(rows=1, cols=1)

    cell = table.rows[0].cells[0]

    # Add summary text to the cell
    para = cell.paragraphs[0]
    run = para.add_run(summary)
    run.font.size = Pt(11)

    # Try to add light blue shading
    try:
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "E8F4FD")  # Light blue
        cell._tc.get_or_add_tcPr().append(shading)
    except Exception:
        pass

    doc.add_paragraph()  # Space after summary


def export_to_docx(
    session: ResearchSession,
    *,
    include_toc: bool = True,
    include_cover_page: bool = True,
    toc_levels: int = 3,
) -> ExportResult:
    """
    Export a research session to DOCX format.

    Creates a professionally formatted Word document with:
    - Cover page with title and metadata (optional)
    - Table of Contents with hyperlinks (optional)
    - Executive summary with highlighted box
    - Metadata table with styled cells
    - Full research report with proper formatting
    - Styled headings with color scheme
    - Page numbers in footer

    Args:
        session: The research session to export
        include_toc: Whether to include a Table of Contents (default: True)
        include_cover_page: Whether to include a cover page (default: True)
        toc_levels: Number of heading levels to include in TOC (default: 3)

    Requires skelmis-docx package (enhanced python-docx fork).
    """
    try:
        from skelmis.docx import Document
        from skelmis.docx.shared import Pt
    except ImportError as e:
        raise ImportError(
            "skelmis-docx is required for DOCX export. "
            "Install with: pip install 'gemini-research-mcp[docx]'"
        ) from e

    doc = Document()

    # Setup custom styles
    _setup_docx_styles(doc)

    # Cover page
    if include_cover_page:
        _add_cover_page(doc, session)

    # Table of Contents (static version - no manual update required)
    if include_toc:
        _add_static_toc(doc, session, toc_levels)

    # Metadata as table
    _add_metadata_table(doc, session)

    # Executive Summary
    if session.summary:
        _add_executive_summary(doc, session.summary)

    # Page break before main report
    if session.report_text:
        doc.add_page_break()

    # Research report section
    if session.report_text:
        doc.add_heading("Research Report", level=1)
        doc.add_paragraph()  # Space after heading

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

    # Final page break and footer
    doc.add_paragraph()
    doc.add_paragraph()

    # Horizontal rule simulation
    hr_para = doc.add_paragraph()
    hr_para.add_run("─" * 60)

    # Footer with export info
    footer = doc.add_paragraph()
    footer_text = (
        f"This document was generated by Gemini Research MCP\n"
        f"Exported on {datetime.now(tz=UTC).strftime('%Y-%m-%d at %H:%M:%S UTC')}\n"
        f"Session ID: {session.interaction_id}"
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
