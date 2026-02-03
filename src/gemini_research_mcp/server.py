"""
Gemini Research MCP Server

Provides AI-powered research tools via Gemini:
- research_web: Fast grounded web search (5-30 seconds) - Gemini + Google Search
- research_deep: Comprehensive multi-step research (3-20 minutes) - Deep Research Agent
- research_followup: Ask follow-up questions about completed research

Architecture:
- MCP SDK with experimental task support for background tasks (MCP Tasks / SEP-1732)
- ServerTaskContext for elicitation during background tasks (input_required pattern)
- Progress reporting via task status updates
"""

# NOTE: Do NOT use `from __future__ import annotations` with FastMCP/Pydantic
# as it breaks type resolution for Annotated parameters in tool functions

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from mcp.server.experimental.task_support import TaskSupport
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import (
    BlobResourceContents,
    EmbeddedResource,
    Icon,
    TextContent,
    TextResourceContents,
    ToolAnnotations,
)
from pydantic import AnyUrl, BaseModel, Field

from gemini_research_mcp import __version__
from gemini_research_mcp.citations import process_citations
from gemini_research_mcp.config import LOGGER_NAME, get_deep_research_agent, get_model
from gemini_research_mcp.content import fetch_webpage as _fetch_webpage
from gemini_research_mcp.deep import (
    deep_research_stream,
    get_research_status,
)
from gemini_research_mcp.deep import research_followup as _research_followup
from gemini_research_mcp.export import (
    ExportFormat,
    ExportResult,
    export_session,
)
from gemini_research_mcp.quick import (
    generate_session_metadata,
    generate_title_from_query,
    quick_research,
    semantic_match_session,
)
from gemini_research_mcp.storage import (
    ResearchStatus,
    get_research_session,
    list_research_sessions,
    list_resumable_sessions,
    save_research_session,
    update_research_session,
)
from gemini_research_mcp.types import DeepResearchError, DeepResearchResult

# Configure logging
logger = logging.getLogger(LOGGER_NAME)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Gemini sparkle icon as SVG data URI (official star/sparkle design with gradient)
GEMINI_ICON_DATA_URI = (
    "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZp"
    "ZXdCb3g9IjAgMCAyOCAyOCI+PGRlZnM+PGxpbmVhckdyYWRpZW50IGlkPSJnIiB4MT0iMCUiIHkxPSIw"
    "JSIgeDI9IjEwMCUiIHkyPSIxMDAlIj48c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjNDE4N0Y0"
    "Ii8+PHN0b3Agb2Zmc2V0PSI1MCUiIHN0b3AtY29sb3I9IiM5QjcyRkYiLz48c3RvcCBvZmZzZXQ9IjEw"
    "MCUiIHN0b3AtY29sb3I9IiNENDRGQjciLz48L2xpbmVhckdyYWRpZW50PjwvZGVmcz48cGF0aCBmaWxs"
    "PSJ1cmwoI2cpIiBkPSJNMTQgMEMxNCA5LjM3MyA0LjM3MyAxNCAwIDE0YzQuMzczIDAgMTQgNC42Mjcg"
    "MTQgMTQgMC05LjM3MyA5LjYyNy0xNCAxNC0xNC00LjM3MyAwLTE0LTQuNjI3LTE0LTE0eiIvPjwvc3Zn"
    "Pg=="
)


# =============================================================================
# Ephemeral Export Cache
# =============================================================================

# TTL for exported files (1 hour)
EXPORT_TTL_SECONDS = 3600


@dataclass
class ExportCacheEntry:
    """Cached export result with TTL."""

    result: ExportResult
    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_expired(self) -> bool:
        """Check if the export has expired."""
        return datetime.now(UTC) > self.created_at + timedelta(seconds=EXPORT_TTL_SECONDS)


# In-memory cache for exports (keyed by export_id)
_export_cache: dict[str, ExportCacheEntry] = {}


def _cache_export(result: ExportResult, session_id: str) -> str:
    """Cache an export and return its unique ID."""
    # Clean up expired entries
    expired_keys = [k for k, v in _export_cache.items() if v.is_expired]
    for key in expired_keys:
        del _export_cache[key]

    export_id = str(uuid.uuid4())[:12]
    _export_cache[export_id] = ExportCacheEntry(result=result, session_id=session_id)
    logger.info("   💾 Cached export %s (%s)", export_id, result.size_human)
    return export_id


def _get_cached_export(export_id: str) -> ExportCacheEntry | None:
    """Retrieve a cached export, or None if expired/missing."""
    entry = _export_cache.get(export_id)
    if entry and entry.is_expired:
        del _export_cache[export_id]
        return None
    return entry


# =============================================================================
# Task Support Configuration
# =============================================================================

# Global TaskSupport instance for the server
_task_support: TaskSupport | None = None


def get_task_support() -> TaskSupport:
    """Get the task support instance."""
    if _task_support is None:
        raise RuntimeError("TaskSupport not initialized. Server must be started with lifespan.")
    return _task_support


@asynccontextmanager
async def lifespan(app: FastMCP) -> AsyncIterator[None]:
    """Initialize task support and check for resumable sessions on startup."""
    global _task_support

    # Enable experimental task support on the low-level server
    _task_support = app._mcp_server.experimental.enable_tasks()

    logger.info("✅ Experimental task support enabled")

    # Check for resumable sessions from previous runs
    try:
        resumable = list_resumable_sessions(limit=10)
        if resumable:
            logger.info("=" * 60)
            logger.info("🔄 RESUMABLE SESSIONS FOUND (%d)", len(resumable))
            for session in resumable:
                status_emoji = "⏳" if session.status == ResearchStatus.IN_PROGRESS else "⚠️"
                logger.info(
                    "   %s [%s] %s",
                    status_emoji,
                    session.interaction_id[:12],
                    session.query[:60],
                )
            logger.info("   💡 Use resume_research tool to recover these sessions")
            logger.info("=" * 60)
    except Exception as e:
        logger.warning("Failed to check for resumable sessions: %s", e)

    async with _task_support.run():
        yield


# =============================================================================
# Server Instance
# =============================================================================

mcp = FastMCP(
    name="Gemini Research",
    icons=[Icon(src=GEMINI_ICON_DATA_URI, mimeType="image/svg+xml", sizes=["any"])],
    instructions="""
Gemini Research MCP Server - AI-powered research toolkit

## Quick Lookup (research_web)
Fast web research with Gemini grounding (5-30 seconds).
Use for: fact-checking, current events, documentation, "what is", "how to".

## Deep Research (research_deep)
Comprehensive autonomous research agent (3-20 minutes).
Use for: research reports, competitive analysis, "compare", "analyze", "investigate".
- Automatically asks clarifying questions for vague queries
- Runs as background task with progress updates
- Returns comprehensive report with citations
- Sessions are saved at START for resume support if interrupted

## Follow-up (research_followup)
Continue conversation with any previous deep research session.
Use for: "elaborate", "clarify", "summarize", follow-up questions.
- Automatically finds the matching session based on your question
- No need to track interaction_ids manually
- Sessions last 55 days (paid tier)

## Fetch Webpage (fetch_webpage)
Extract content from any URL as clean Markdown (sub-second).
Use for: reading articles, documentation, blog posts, extracting linked content.
- High-quality extraction via trafilatura (F1: 0.937)
- SSRF protection (blocks private IPs, cloud metadata)
- Falls back to basic HTML parsing if trafilatura unavailable

## Resume Research (resume_research)
Recover interrupted or in-progress research sessions.
Use for: VS Code disconnections, network issues, checking ongoing research.
- Lists sessions that can be resumed (in_progress/interrupted)
- Checks Gemini API for completion status
- Recovers completed reports that were interrupted during delivery

## List Sessions (list_research_sessions_tool)
Returns JSON list of previous research sessions with queries and summaries.
Use to answer "what research did I do about X?" questions.
Shows session status (completed, in_progress, failed, interrupted).

## Export (export_research_session)
Export completed research to Markdown, JSON, or Word (DOCX) format.
Use for: sharing reports, archiving research, creating deliverables.

**Workflow:**
- Simple questions → research_web
- Complex questions → research_deep
- Read a URL → fetch_webpage
- VS Code disconnected during research? → resume_research
- "What did I research about X?" → list_research_sessions_tool
- Continue old research → research_followup (auto-matches session)
- Export for sharing → export_research_session
""",
    lifespan=lifespan,
)


# =============================================================================
# Helper Functions
# =============================================================================


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


# =============================================================================
# Helper Functions - Report Formatting
# =============================================================================


def _format_deep_research_report(
    result: DeepResearchResult, interaction_id: str, elapsed: float
) -> str:
    """Format a deep research result into a markdown report."""
    lines = ["## Research Report"]

    if result.text:
        lines.append(result.text)
    else:
        lines.append("*No report available.*")

    # Usage stats
    if result.usage:
        lines.extend(["", "## Usage"])
        if result.usage.total_tokens:
            lines.append(f"- Total tokens: {result.usage.total_tokens}")
        if result.usage.total_cost:
            lines.append(f"- Estimated cost: ${result.usage.total_cost:.4f}")

    # Duration
    lines.extend(
        [
            "",
            "---",
            f"- Duration: {_format_duration(elapsed)}",
            f"- Interaction ID: `{interaction_id}`",
        ]
    )

    return "\n".join(lines)


# =============================================================================
# Tools
# =============================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def research_web(
    query: Annotated[str, "Search query or question to research on the web"],
    include_thoughts: Annotated[bool, "Include thinking summary in response"] = False,
) -> str:
    """
    Fast web research with Gemini grounding. Returns answer with citations in seconds.

    Always uses thorough reasoning (thinking_level=high) for quality results.

    Use for: quick lookups, fact-checking, current events, documentation, "what is",
    "how to", real-time information, news, API references, error messages.

    Args:
        query: Search query or question to research
        include_thoughts: Include thinking summary in response

    Returns:
        Research results with sources as markdown text
    """
    logger.info("🔎 research_web: %s", query[:100])
    start = time.time()

    try:
        result = await quick_research(
            query=query,
            include_thoughts=include_thoughts,
        )
        elapsed = time.time() - start
        logger.info("   ✅ Completed in %.1fs", elapsed)

        # Format response
        lines = []

        # Main response
        if result.text:
            lines.append(result.text)

        # Sources section
        if result.sources:
            lines.extend(["", "---", "### Sources"])
            for i, source in enumerate(result.sources, 1):
                title = source.title or source.uri
                lines.append(f"{i}. [{title}]({source.uri})")

        # Search queries used
        if result.queries:
            lines.extend(["", "### Search Queries"])
            for q in result.queries:
                lines.append(f"- {q}")

        # Thinking summary (if requested)
        if result.thinking_summary:
            lines.extend(["", "### Thinking Summary", result.thinking_summary])

        # Metadata
        lines.extend(
            [
                "",
                "---",
                f"*Completed in {_format_duration(elapsed)}*",
            ]
        )

        return "\n".join(lines)

    except Exception as e:
        logger.exception("research_web failed: %s", e)
        return f"❌ Research failed: {e}"


# =============================================================================
# Fetch Webpage Tool (FastMCP 3.0 pattern)
# =============================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def fetch_webpage(
    url: Annotated[str, "URL of the webpage to fetch and extract content from"],
) -> str:
    """
    Fetch and extract content from a webpage as Markdown.

    Uses trafilatura for high-quality content extraction (F1: 0.937).
    Falls back to basic HTML parsing if trafilatura is unavailable.

    Security: Blocks requests to private IPs, localhost, and cloud metadata endpoints
    (SSRF protection).

    Use for: Reading articles, documentation, blog posts, extracting content from
    URLs found in research results.

    Args:
        url: The URL to fetch (must be http/https, no private IPs)

    Returns:
        Extracted content as Markdown, or error message if fetch failed
    """
    logger.info("🌐 fetch_webpage: %s", url[:100])

    result = await _fetch_webpage(url)

    if result.error:
        return f"❌ Failed to fetch: {result.error}"

    # Format the response
    lines = []

    if result.title:
        lines.append(f"# {result.title}")
        lines.append("")

    lines.append(result.content)

    if result.word_count:
        lines.extend([
            "",
            "---",
            f"*Extracted {result.word_count:,} words from {url}*",
        ])

    return "\n".join(lines)


# =============================================================================
# SEP-1686: Elicitation via Context.elicit()
# =============================================================================
#
# The MCP SDK's FastMCP Context supports elicitation directly via ctx.elicit().
# This works in foreground (non-task) mode. For background tasks (SEP-1732),
# the ServerTaskContext provides elicit() with input_required status management.
#
# Current implementation: Foreground elicitation via ctx.elicit()
# TODO: Background task elicitation via ServerTaskContext when client supports it


class ClarificationSchema(BaseModel):
    """Schema for clarification question answers."""

    answer_1: str = Field(default="", description="Answer to first clarifying question")
    answer_2: str = Field(default="", description="Answer to second clarifying question")
    answer_3: str = Field(default="", description="Answer to third clarifying question")


async def _maybe_clarify_query(
    query: str,
    ctx: Context[Any, Any, Any] | None,
) -> str:
    """
    Analyze query and optionally ask clarifying questions via ctx.elicit().

    Uses heuristics to detect vague queries and prompts for clarification.

    Args:
        query: The research query
        ctx: MCP Context (None when running in background task)

    Returns the refined query, or original if clarification was skipped/unavailable.
    """
    if ctx is None:
        logger.info("🔍 Skipping clarification (no context)")
        return query

    # Simple heuristics for detecting vague queries
    query_lower = query.lower()
    query_len = len(query)

    is_vague = False
    questions: list[str] = []

    # Comprehensive queries (200+ chars with multiple sentences) skip clarification
    # This catches detailed requests with format_instructions, multiple criteria, etc.
    has_multiple_points = query.count("(") >= 2 or query.count(",") >= 3
    is_comprehensive = query_len >= 200 and has_multiple_points

    if is_comprehensive:
        logger.info("   ✅ Query is comprehensive (%d chars), skipping clarification", query_len)
        return query

    # Very short queries are often vague
    if query_len < 30:
        is_vague = True
        questions.append("Can you provide more context about what you're looking for?")

    # Generic comparative terms (only for short queries)
    comparative_terms = ["compare", "vs", "versus", "best", "top"]
    has_comparative = any(term in query_lower for term in comparative_terms)
    if has_comparative and query_len < 100 and not any(c.isdigit() for c in query):
        is_vague = True
        questions.append("What specific aspects would you like to compare?")
        questions.append("What's your use case or context?")

    # Generic topic terms (only for short queries)
    has_topic_term = any(term in query_lower for term in ["research", "analyze", "investigate"])
    if has_topic_term and query_len < 100:
        is_vague = True
        questions.append("What specific angle or focus area interests you?")
        questions.append("What's the timeframe or scope you're interested in?")

    # "Best practices" without context (only for short queries)
    if "best practice" in query_lower and query_len < 100:
        is_vague = True
        questions.append("What industry or domain are you in?")
        questions.append("What's the scale or context (startup, enterprise, etc.)?")

    if not is_vague or not questions:
        logger.info("   ✅ Query is specific enough, no clarification needed")
        return query

    # Trim to 3 questions max
    questions = questions[:3]
    logger.info("   🎯 Query may need clarification: %d questions", len(questions))

    try:
        # Build dynamic schema with actual questions as descriptions
        from pydantic import create_model

        field_definitions = {
            f"answer_{i+1}": (str, Field(default="", description=q))
            for i, q in enumerate(questions)
        }
        DynamicSchema = create_model("ClarificationQuestions", **field_definitions)  # type: ignore

        message = (
            f"To improve research quality for:\n\n**\"{query}\"**\n\n"
            f"Please answer these questions (optional - press 'Skip' to continue):"
        )

        result = await ctx.elicit(
            message=message,
            schema=DynamicSchema,
        )

        if result.action == "accept" and result.data:
            data = result.data.model_dump() if hasattr(result.data, "model_dump") else {}
            answers = [data.get(f"answer_{i + 1}", "") for i in range(len(questions))]
            non_empty = [a for a in answers if a.strip()]

            if non_empty:
                logger.info("   ✨ User provided %d/%d answers", len(non_empty), len(questions))
                clarification = "\n".join(
                    f"Q: {q}\nA: {a}"
                    for q, a in zip(questions, answers, strict=False)
                    if a.strip()
                )
                refined = f"{query}\n\nAdditional context:\n{clarification}"
                logger.info("   📝 Refined query: %s", refined[:100])
                return refined
            else:
                logger.info("   ⏭️ User submitted but answers empty")
        else:
            logger.info("   ⏭️ User skipped/cancelled clarification")

    except Exception as e:
        logger.warning("   ⚠️ Elicitation failed: %s", e)

    return query


# =============================================================================
# Deep Research Tool
# =============================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def research_deep(
    query: Annotated[str, "Research question or topic to investigate thoroughly"],
    format_instructions: Annotated[
        str | None,
        "Optional report format (e.g., 'executive briefing', 'comparison table')",
    ] = None,
    file_search_store_names: Annotated[
        list[str] | None,
        "Optional: Gemini File Search store names to search your own data alongside web",
    ] = None,
    auto_refine: Annotated[
        bool,
        "If True, run automatic critique cycle and append refinements to fill gaps",
    ] = False,
    ctx: Context[Any, Any, Any] | None = None,
) -> str:
    """
    Comprehensive autonomous research agent. Takes 3-20 minutes.

    Use for: research reports, competitive analysis, "compare X vs Y", "analyze",
    "investigate", literature review, multi-source synthesis.

    For vague queries, the tool automatically asks clarifying questions
    to refine the research scope before starting (when elicitation is available).

    Args:
        query: Research question or topic (can be vague - clarification is automatic)
        format_instructions: Optional report structure/tone guidance
        file_search_store_names: Optional file stores for RAG over your own data
        auto_refine: Run automatic quality critique and fill gaps (adds ~30s-2min)

    Returns:
        Comprehensive research report with citations
    """
    logger.info("🔬 research_deep: %s", query[:100])
    if format_instructions:
        logger.info("   📝 Format: %s", format_instructions[:80])
    if file_search_store_names:
        logger.info("   📁 File search stores: %s", file_search_store_names)
    if auto_refine:
        logger.info("   🔄 Auto-refine: enabled")

    # Resolve template key to full template instructions
    effective_format = format_instructions
    if format_instructions:
        from gemini_research_mcp.templates import get_template

        template = get_template(format_instructions)
        if template:
            logger.info("   📋 Using template: %s", template.name)
            effective_format = str(template)

    start = time.time()

    # ==========================================================================
    # Phase 1: Query Clarification (if ctx available)
    # ==========================================================================
    effective_query = await _maybe_clarify_query(query, ctx)

    if effective_query != query:
        logger.info("   ✨ Using refined query")
        logger.info("=" * 60)
        logger.info("📋 FINAL CONSOLIDATED QUERY:")
        for line in effective_query.split("\n"):
            logger.info("   %s", line)
        logger.info("=" * 60)

    # ==========================================================================
    # Phase 2: Deep Research Execution
    # ==========================================================================
    if ctx is not None:
        await ctx.info("Starting deep research...")

    try:
        thought_count = 0
        action_count = 0
        interaction_id: str | None = None
        session_saved = False  # Track if we saved session at start
        initial_title: str | None = None  # Generated title for the session

        # Consume the stream to get interaction_id and track progress
        async for event in deep_research_stream(
            query=effective_query,
            format_instructions=effective_format,
            file_search_store_names=file_search_store_names,
        ):
            if event.interaction_id:
                interaction_id = event.interaction_id
                logger.info("   📋 interaction_id: %s", interaction_id)

                # === RESUME SUPPORT: Save session at START with in_progress status ===
                if not session_saved:
                    try:
                        # Generate a proper title from the query (fast, ~$0.0001)
                        initial_title = await generate_title_from_query(effective_query)
                        if not initial_title:
                            initial_title = effective_query[:60]  # Fallback
                        logger.info("   📝 Generated title: %s", initial_title)

                        save_research_session(
                            interaction_id=interaction_id,
                            query=effective_query,
                            title=initial_title,
                            format_instructions=format_instructions,
                            agent_name=get_deep_research_agent(),
                            status=ResearchStatus.IN_PROGRESS,
                        )
                        session_saved = True
                        logger.info("   💾 Session saved (in_progress) for resume support")
                    except Exception as save_error:
                        logger.warning("⚠️ Failed to save session at start: %s", save_error)

            # Track events for progress
            if event.event_type == "thought":
                thought_count += 1
                content = event.content or ""
                short = content[:55] + "..." if len(content) > 55 else content
                if ctx:
                    await ctx.report_progress(
                        progress=min(50, thought_count * 5),
                        total=100,
                        message=f"[{thought_count}] 🧠 {short}",
                    )
            elif event.event_type == "action":
                action_count += 1
                content = event.content or ""
                short = content[:55] + "..." if len(content) > 55 else content
                if ctx:
                    await ctx.info(f"[{action_count}] 🔍 {short}")
            elif event.event_type == "start":
                if ctx:
                    await ctx.info("🚀 Research agent autonomous investigation started")
            elif event.event_type == "error":
                logger.error("   Stream error: %s", event.content)
                # Mark session as failed if we have interaction_id
                if interaction_id and session_saved:
                    with contextlib.suppress(Exception):
                        update_research_session(
                            interaction_id,
                            status=ResearchStatus.FAILED,
                        )

        if not interaction_id:
            raise ValueError("No interaction_id received from stream")

        logger.info("   📊 Stream consumed: %d thoughts, %d actions", thought_count, action_count)

        if ctx:
            await ctx.info("Waiting for research completion...")

        # Poll for completion
        max_wait = 1200  # 20 minutes max
        poll_interval = 10  # 10 seconds between polls
        poll_start = time.time()

        while time.time() - poll_start < max_wait:
            result = await get_research_status(interaction_id)

            raw_status = "unknown"
            if result.raw_interaction:
                raw_status = getattr(result.raw_interaction, "status", "unknown")

            elapsed = time.time() - start

            if raw_status == "completed":
                logger.info("   ✅ Research completed in %s", _format_duration(elapsed))

                result = await process_citations(result, resolve_urls=True)

                # ============================================================
                # Auto-refine: Run critique cycle and append findings
                # ============================================================
                if auto_refine and result.text and interaction_id:
                    logger.info("   🔄 AUTO_REFINE: Running critique cycle...")
                    if ctx:
                        await ctx.info("Running quality critique...")

                    from gemini_research_mcp.deep import critique_research

                    critique = await critique_research(effective_query, result.text)

                    if critique.needs_refinement and critique.follow_up_questions:
                        logger.info(
                            "   🔄 Found %d gaps, running %d follow-ups",
                            len(critique.gaps),
                            len(critique.follow_up_questions),
                        )
                        if ctx:
                            await ctx.info(
                                f"Refining: {len(critique.follow_up_questions)} follow-up queries"
                            )

                        refinements: list[str] = []
                        for i, question in enumerate(critique.follow_up_questions[:3], 1):
                            try:
                                followup_response = await _research_followup(
                                    interaction_id, question
                                )
                                if followup_response:
                                    refinements.append(f"### {question}\n\n{followup_response}")
                            except Exception as e:
                                logger.warning("   ⚠️ Follow-up %d failed: %s", i, e)

                        if refinements:
                            appendix = "\n\n---\n\n## Refinements\n\n" + "\n\n".join(refinements)
                            result.text = (result.text or "") + appendix
                            logger.info("   ✅ Appended %d refinements", len(refinements))
                    else:
                        logger.info("   ✅ Report passed quality check")

                # Auto-save session for later follow-up
                total_tokens = None
                if result.usage and result.usage.total_tokens:
                    total_tokens = result.usage.total_tokens

                # Generate title and summary in one call (~$0.0003/call)
                metadata = await generate_session_metadata(
                    text=result.text or "",
                    query=effective_query,
                )

                # Update session with completion data (session saved at start)
                try:
                    update_research_session(
                        interaction_id,
                        title=metadata.title or None,
                        summary=metadata.summary or None,
                        report_text=result.text,
                        duration_seconds=elapsed,
                        total_tokens=total_tokens,
                        status=ResearchStatus.COMPLETED,
                    )
                    logger.info("   💾 Session updated (completed)")
                except Exception as save_error:
                    logger.warning(
                        "⚠️ Failed to update session (research succeeded): %s",
                        save_error,
                    )

                return _format_deep_research_report(result, interaction_id, elapsed)

            elif raw_status in ("failed", "cancelled"):
                logger.error("   ❌ Research %s after %s", raw_status, _format_duration(elapsed))
                # Mark session as failed
                if session_saved:
                    with contextlib.suppress(Exception):
                        update_research_session(
                            interaction_id,
                            status=ResearchStatus.FAILED,
                            duration_seconds=elapsed,
                        )
                raise DeepResearchError(
                    code=f"RESEARCH_{raw_status.upper()}",
                    message=f"Research {raw_status} after {_format_duration(elapsed)}",
                )
            else:
                # Still working - report progress
                if ctx:
                    progress_pct = min(90, int(50 + (elapsed / max_wait) * 40))
                    await ctx.report_progress(
                        progress=progress_pct,
                        total=100,
                        message=f"⏳ Researching... ({_format_duration(elapsed)})",
                    )

            await asyncio.sleep(poll_interval)

        # Timeout
        elapsed = time.time() - start
        raise DeepResearchError(
            code="TIMEOUT",
            message=(
                f"Research timed out after {_format_duration(elapsed)}. "
                f"Interaction ID: {interaction_id}"
            ),
            details={"interaction_id": interaction_id},
        )

    except DeepResearchError:
        raise
    except Exception as e:
        logger.exception("research_deep failed: %s", e)
        raise DeepResearchError(
            code="INTERNAL_ERROR",
            message=str(e),
        ) from e


# =============================================================================
# research_deep_planned: Plan → Approve → Execute workflow
# Inspired by ADK Deep Search's interactive_planner_agent
# =============================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def research_deep_planned(
    query: Annotated[str, "Research question or topic to investigate thoroughly"],
    format_instructions: Annotated[
        str | None,
        "Optional report format (e.g., 'executive briefing', 'comparison table')",
    ] = None,
    ctx: Context[Any, Any, Any] | None = None,
) -> str:
    """
    Deep research with explicit plan approval step before execution.

    Generates a research plan first, presents it for approval via MCP elicitation,
    then executes the approved plan. This gives you control over the research direction.

    Workflow:
    1. Generate research plan (5-15 seconds)
    2. Present plan for approval (user interaction)
    3. Execute approved plan (3-20 minutes)

    Use for: When you want to review/modify the research approach before investing time.

    Args:
        query: Research question or topic
        format_instructions: Optional report structure/tone guidance

    Returns:
        Comprehensive research report with citations
    """
    from gemini_research_mcp.templates import RESEARCH_PLAN_PROMPT

    logger.info("📋 research_deep_planned: %s", query[:100])
    start = time.time()

    # Phase 1: Generate research plan using quick_research
    if ctx:
        await ctx.info("Generating research plan...")

    try:
        plan_prompt = RESEARCH_PLAN_PROMPT.format(query=query)
        plan_result = await quick_research(
            query=plan_prompt,
            include_thoughts=False,
        )
        plan_text = plan_result.text
        logger.info("   📝 Generated plan in %.1fs", time.time() - start)

    except Exception as e:
        logger.exception("Plan generation failed: %s", e)
        return f"❌ Failed to generate research plan: {e}"

    # Phase 2: Present plan for approval via MCP elicitation
    if ctx is None:
        # No context available - proceed without approval
        logger.warning("   ⚠️ No context for elicitation, proceeding with plan")
        approved_plan = plan_text
    else:
        try:
            from pydantic import create_model

            PlanApproval = create_model(
                "PlanApproval",
                approved_plan=(
                    str,
                    Field(
                        default=plan_text,
                        description="Edit the plan or leave as-is to approve",
                    ),
                ),
            )

            message = (
                f"## Research Plan for: \"{query}\"\n\n"
                f"{plan_text}\n\n"
                f"---\n\n"
                f"Edit the plan below or press **Continue** to approve and start research.\n"
                f"Press **Cancel** to abort."
            )

            result = await ctx.elicit(
                message=message,
                schema=PlanApproval,
            )

            if result.action == "cancel":
                return "❌ Research cancelled by user."

            if result.action == "accept" and result.data:
                data = result.data.model_dump() if hasattr(result.data, "model_dump") else {}
                approved_plan = data.get("approved_plan", plan_text)
                if approved_plan != plan_text:
                    logger.info("   ✏️ User modified the plan")
            else:
                approved_plan = plan_text

        except Exception as e:
            logger.warning("   ⚠️ Elicitation failed: %s, proceeding with plan", e)
            approved_plan = plan_text

    # Phase 3: Execute approved plan with research_deep
    logger.info("   🚀 Executing approved plan...")
    if ctx:
        await ctx.info("Starting research with approved plan...")

    # Combine query with approved plan as format instructions
    combined_instructions = (
        f"Follow this research plan:\n\n{approved_plan}"
        + (f"\n\nAdditional formatting:\n{format_instructions}" if format_instructions else "")
    )

    # Use the research_deep function directly
    return await research_deep(  # type: ignore[no-any-return]
        query=query,
        format_instructions=combined_instructions,
        ctx=ctx,
    )


# =============================================================================
# list_format_templates: Show available format instruction templates
# =============================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_format_templates(
    category: Annotated[
        str | None,
        "Filter by category: 'business', 'analysis', 'technical', or 'academic'",
    ] = None,
) -> str:
    """
    List available format instruction templates for research_deep.

    Pre-built templates help generate consistently structured reports:
    - executive_briefing: C-suite summary with key findings and recommendations
    - competitive_analysis: Deep dive comparison of competitors
    - comparison_table: Side-by-side feature comparison with verdict
    - technical_overview: Technical explanation for engineers
    - literature_review: Academic-style synthesis of research

    Use the template name with research_deep's format_instructions parameter:
    Example: research_deep(query="...", format_instructions="executive_briefing")

    Returns:
        JSON list of available templates with descriptions
    """
    from gemini_research_mcp.templates import ALL_TEMPLATES, TEMPLATES_BY_CATEGORY, TemplateCategory

    logger.info("📋 list_format_templates: category=%s", category)

    if category:
        try:
            cat = TemplateCategory(category.lower())
            templates = TEMPLATES_BY_CATEGORY.get(cat, [])
            template_list = [
                {
                    "key": key,
                    "name": t.name,
                    "description": t.description,
                    "category": t.category.value,
                }
                for key, t in ALL_TEMPLATES.items()
                if t in templates
            ]
        except ValueError:
            return json.dumps({
                "error": f"Invalid category: {category}",
                "valid_categories": [c.value for c in TemplateCategory],
            })
    else:
        template_list = [
            {
                "key": key,
                "name": t.name,
                "description": t.description,
                "category": t.category.value,
            }
            for key, t in ALL_TEMPLATES.items()
        ]

    return json.dumps({
        "templates": template_list,
        "count": len(template_list),
        "usage": "Pass template key to research_deep's format_instructions parameter",
        "example": 'research_deep(query="...", format_instructions="executive_briefing")',
    }, indent=2)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def research_followup(
    query: Annotated[
        str, "Follow-up question about previous research (e.g., 'elaborate on surface codes')"
    ],
    interaction_id: Annotated[
        str | None,
        "Optional: specific interaction_id. If not provided, auto-matches from sessions.",
    ] = None,
    model: Annotated[
        str, "Model to use for follow-up. Default: gemini-3-pro-preview"
    ] = "gemini-3-pro-preview",
) -> str:
    """
    Continue conversation after deep research. Ask follow-up questions without restarting.

    The tool automatically finds the relevant research session based on your question.
    You can optionally provide an interaction_id for direct reference.

    Use for: "clarify", "elaborate", "summarize", "explain more", "what about",
    continue discussion, ask more questions about completed research results.

    Args:
        query: Your follow-up question
        interaction_id: Optional specific session ID (from list_research_sessions)
        model: Model to use (default: gemini-3-pro-preview)

    Returns:
        Response to the follow-up question
    """
    logger.info("💬 research_followup: query=%s, id=%s", query[:100], interaction_id)

    try:
        # If no interaction_id provided, find the best matching session
        previous_interaction_id = interaction_id
        if not previous_interaction_id:
            sessions = list_research_sessions(limit=20, include_expired=False)
            if not sessions:
                return "❌ No research sessions found. Complete a deep research first."

            # Build session list for semantic matching
            session_dicts = [
                {
                    "id": s.interaction_id,
                    "query": s.query,
                    "summary": s.summary or s.query[:100],
                }
                for s in sessions
            ]

            matched_id = await semantic_match_session(query, session_dicts)
            if matched_id:
                previous_interaction_id = matched_id
                # Find the matched session for logging
                matched_session = next(
                    (s for s in sessions if s.interaction_id == matched_id), None
                )
                if matched_session:
                    logger.info(
                        "   📎 Matched to session: %s (%s)",
                        matched_id[:12],
                        matched_session.query[:50],
                    )
            else:
                # Fall back to most recent session
                previous_interaction_id = sessions[0].interaction_id
                logger.info(
                    "   📎 No semantic match, using most recent: %s (%s)",
                    sessions[0].interaction_id[:12],
                    sessions[0].query[:50],
                )

        response = await _research_followup(
            previous_interaction_id=previous_interaction_id,
            query=query,
            model=model,
        )

        lines = [
            "## Follow-up Response",
            "",
            response,
            "",
            "---",
            f"*Interaction ID: `{previous_interaction_id}`*",
        ]

        return "\n".join(lines)

    except Exception as e:
        logger.exception("research_followup failed: %s", e)
        return f"❌ Follow-up failed: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def list_research_sessions_tool(
    limit: Annotated[int, "Maximum number of sessions to return"] = 20,
    include_expired: Annotated[bool, "Include expired sessions"] = False,
) -> str:
    """
    List saved research sessions available for follow-up.

    Sessions are automatically saved when deep research completes successfully.
    Returns JSON for easy parsing by agents.

    Note: You don't need to extract interaction_ids manually.
    Just use research_followup with your question - it will automatically
    find the matching session.

    Returns:
        JSON array of research sessions with summaries
    """
    logger.info("📋 list_research_sessions: limit=%d, include_expired=%s", limit, include_expired)

    sessions = list_research_sessions(limit=limit, include_expired=include_expired)

    if not sessions:
        return json.dumps({"sessions": [], "message": "No research sessions found."})

    session_list = []
    for session in sessions:
        session_data: dict[str, str | int | float | None] = {
            "interaction_id": session.interaction_id,
            "query": session.query,
            "summary": session.summary,
            "status": session.status.value,
            "created_at": session.created_at_iso,
            "expires_in": session.time_remaining_human,
        }
        if session.title:
            session_data["title"] = session.title
        if session.duration_seconds:
            session_data["duration_seconds"] = session.duration_seconds
        if session.total_tokens:
            session_data["total_tokens"] = session.total_tokens

        session_list.append(session_data)

    # Count resumable sessions
    resumable_count = sum(1 for s in sessions if s.is_resumable)
    hint = "Use research_followup with your question - auto-matches session."
    if resumable_count > 0:
        hint = f"{resumable_count} session(s) can be resumed with resume_research tool."

    return json.dumps(
        {
            "sessions": session_list,
            "count": len(session_list),
            "resumable_count": resumable_count,
            "hint": hint,
        },
        indent=2,
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def resume_research(
    interaction_id: Annotated[
        str | None,
        "Optional: specific interaction_id to resume. If not provided, shows resumable sessions.",
    ] = None,
    ctx: Context[Any, Any, Any] | None = None,
) -> str:
    """
    Resume interrupted or in-progress research sessions.

    If VS Code disconnected or the research was interrupted, this tool can:
    1. List all resumable sessions (in_progress or interrupted)
    2. Check the status of a specific session and return results if completed

    Sessions are saved at the START of research, so even if VS Code loses connection,
    the research continues on Gemini's servers and can be retrieved later.

    Args:
        interaction_id: Optional specific session to resume/check

    Returns:
        Status of resumable sessions or completed research results
    """
    logger.info("🔄 resume_research: id=%s", interaction_id)

    try:
        # If no interaction_id, list all resumable sessions
        if not interaction_id:
            resumable = list_resumable_sessions(limit=10)
            if not resumable:
                return json.dumps({
                    "status": "no_resumable_sessions",
                    "message": "No interrupted or in-progress research sessions found.",
                    "hint": "All sessions are either completed or expired.",
                })

            session_list = []
            for s in resumable:
                session_list.append({
                    "interaction_id": s.interaction_id,
                    "query": s.query[:100],
                    "status": s.status.value,
                    "created_at": s.created_at_iso,
                    "expires_in": s.time_remaining_human,
                })

            return json.dumps({
                "status": "resumable_sessions_found",
                "count": len(session_list),
                "sessions": session_list,
                "hint": "Call resume_research with a specific interaction_id to check/resume.",
            }, indent=2)

        # Check specific session status
        session = get_research_session(interaction_id)
        if not session:
            return json.dumps({
                "status": "not_found",
                "message": f"Session not found: {interaction_id}",
            })

        # If already completed, return the report
        if session.status == ResearchStatus.COMPLETED:
            return json.dumps({
                "status": "already_completed",
                "message": "This research session is already completed.",
                "title": session.title,
                "summary": session.summary,
                "hint": "Use research_followup or export_research_session.",
            })

        # Check with Gemini API for current status
        if ctx:
            await ctx.info(f"Checking status of research: {session.query[:50]}...")

        try:
            result = await get_research_status(interaction_id)
            raw_status = "unknown"
            if result.raw_interaction:
                raw_status = getattr(result.raw_interaction, "status", "unknown")

            if raw_status == "completed":
                # Research completed on Gemini's side - update our records
                result = await process_citations(result, resolve_urls=True)

                total_tokens = None
                if result.usage and result.usage.total_tokens:
                    total_tokens = result.usage.total_tokens

                # Generate metadata
                metadata = await generate_session_metadata(
                    text=result.text or "",
                    query=session.query,
                )

                # Update session
                update_research_session(
                    interaction_id,
                    title=metadata.title or None,
                    summary=metadata.summary or None,
                    report_text=result.text,
                    total_tokens=total_tokens,
                    status=ResearchStatus.COMPLETED,
                )

                logger.info("   ✅ Research recovered and saved!")

                # Return the report
                lines = ["## Research Report (Resumed)"]
                if result.text:
                    lines.append(result.text)
                else:
                    lines.append("*No report text available.*")

                lines.extend([
                    "",
                    "---",
                    f"*Session recovered. Interaction ID: `{interaction_id}`*",
                ])

                return "\n".join(lines)

            elif raw_status in ("failed", "cancelled"):
                update_research_session(
                    interaction_id,
                    status=ResearchStatus.FAILED,
                )
                return json.dumps({
                    "status": raw_status,
                    "message": f"Research {raw_status} on Gemini's servers.",
                    "query": session.query,
                })

            else:
                # Still in progress
                return json.dumps({
                    "status": "still_in_progress",
                    "gemini_status": raw_status,
                    "message": "Research is still running on Gemini's servers.",
                    "query": session.query[:100],
                    "hint": "Try again in a few minutes.",
                })

        except Exception as api_error:
            logger.warning("Failed to check Gemini status: %s", api_error)
            # Mark as interrupted if we can't reach Gemini
            update_research_session(
                interaction_id,
                status=ResearchStatus.INTERRUPTED,
            )
            return json.dumps({
                "status": "api_error",
                "message": f"Could not check status: {api_error}",
                "hint": "The research may still be running. Try again later.",
            })

    except Exception as e:
        logger.exception("resume_research failed: %s", e)
        return json.dumps({"error": f"Resume failed: {e}"})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def export_research_session(
    interaction_id: Annotated[
        str | None,
        "Interaction ID of the session to export. If not provided, exports the most recent.",
    ] = None,
    format: Annotated[
        str,
        "Export format: 'markdown' (.md), 'json' (.json), or 'docx' (Word document)",
    ] = "markdown",
    query: Annotated[
        str | None,
        "Optional: search for a session by query text instead of interaction_id",
    ] = None,
) -> str | list[TextContent | EmbeddedResource]:
    """
    Export a research session to Markdown, JSON, or Word (DOCX) format.

    Similar to Google's Deep Research export feature, this creates professional
    documents suitable for sharing, archiving, or further editing.

    Supported formats:
    - **markdown**: Clean, readable .md file with full report and citations
    - **json**: Machine-readable with all metadata (for programmatic use)
    - **docx**: Professional Word document with proper formatting, headings, lists

    Args:
        interaction_id: Specific session ID to export (from list_research_sessions)
        format: Output format - markdown, json, or docx
        query: Search for session by query text (alternative to interaction_id)

    Returns:
        JSON with export details and resource URI for download
    """
    logger.info(
        "📤 export_research_session: id=%s, format=%s, query=%s",
        interaction_id,
        format,
        query[:30] if query else None,
    )

    try:
        session = None

        # Find session by interaction_id
        if interaction_id:
            session = get_research_session(interaction_id)
            if not session:
                return json.dumps({
                    "error": f"Session not found: {interaction_id}",
                    "hint": "Use list_research_sessions to see available sessions.",
                })

        # Find session by query using AI-powered semantic matching
        elif query:
            sessions = list_research_sessions(limit=20, include_expired=False)
            if not sessions:
                return json.dumps({
                    "error": "No research sessions found.",
                    "hint": "Complete a deep research first with research_deep.",
                })

            # Build session list for semantic matching (same as research_followup)
            session_dicts = [
                {
                    "id": s.interaction_id,
                    "query": s.query,
                    "summary": s.summary or s.query[:100],
                }
                for s in sessions
            ]

            matched_id = await semantic_match_session(query, session_dicts)
            if matched_id:
                session = next((s for s in sessions if s.interaction_id == matched_id), None)
                if session:
                    logger.info(
                        "   📎 Matched to session: %s (%s)",
                        matched_id[:12],
                        session.query[:50],
                    )
                else:
                    # Matched ID not found in sessions list - fall back to most recent
                    session = sessions[0]
                    logger.warning(
                        "   ⚠️ Matched ID %s not in sessions, using most recent",
                        matched_id[:12],
                    )
            else:
                # Fall back to most recent session
                session = sessions[0]
                logger.info(
                    "   📎 No semantic match, using most recent: %s (%s)",
                    session.interaction_id[:12],
                    session.query[:50],
                )

        # Default to most recent session
        else:
            sessions = list_research_sessions(limit=1)
            if not sessions:
                return json.dumps({
                    "error": "No research sessions found.",
                    "hint": "Complete a deep research first with research_deep.",
                })
            session = sessions[0]

        # Export
        result = export_session(session, format)

        # Return EmbeddedResource for all formats to enable VS Code "Save As" button
        # Following the ElevenLabs MCP pattern: put filename in URI for browser-like save dialog
        import base64

        # URI contains filename so clients extract it for "Save As" dialog (like browsers)
        resource_uri = f"research://{result.filename}"

        # For text formats (MD, JSON), use TextResourceContents
        # For binary formats (DOCX), use BlobResourceContents
        if result.format == ExportFormat.DOCX:
            resource_content: BlobResourceContents | TextResourceContents = BlobResourceContents(
                uri=AnyUrl(resource_uri),
                mimeType=result.mime_type,
                blob=base64.b64encode(result.content).decode("ascii"),
            )
        else:
            # Text formats - use TextResourceContents
            resource_content = TextResourceContents(
                uri=AnyUrl(resource_uri),
                mimeType=result.mime_type,
                text=result.content.decode("utf-8"),
            )

        embedded = EmbeddedResource(
            type="resource",
            resource=resource_content,
        )

        # Format-specific emoji and label
        format_info = {
            ExportFormat.DOCX: ("📄", "DOCX"),
            ExportFormat.MARKDOWN: ("📝", "Markdown"),
            ExportFormat.JSON: ("📋", "JSON"),
        }
        emoji, label = format_info.get(result.format, ("📁", result.format.value.upper()))

        # Return metadata as TextContent + file as EmbeddedResource
        # This enables VS Code's native "Save As" functionality for all formats
        metadata_text = (
            f"{emoji} **{label} Export Complete**\n\n"
            f"- **Filename:** {result.filename}\n"
            f"- **Size:** {result.size_human}\n"
            f"- **Session:** {session.query[:80]}\n"
            f"- **Resource URI:** {resource_uri}\n\n"
            f"The file is attached below. Use 'Save As' to download."
        )
        text_content = TextContent(
            type="text",
            text=metadata_text,
        )

        return [text_content, embedded]

    except ImportError as e:
        return json.dumps({
            "error": str(e),
            "hint": "Install skelmis-docx for DOCX export.",
        })
    except Exception as e:
        logger.exception("export_research_session failed: %s", e)
        return json.dumps({"error": f"Export failed: {e}"})


# =============================================================================
# Resources
# =============================================================================


@mcp.resource("research://models")
def get_research_models() -> str:
    """
    List available research models and their capabilities.

    Returns information about the models used by this server:
    - Quick research model (Gemini + Google Search grounding)
    - Deep Research Agent (autonomous multi-step research)
    """
    quick_model = get_model()
    deep_agent = get_deep_research_agent()

    return f"""# Available Research Models

## Quick Research (research_web)

**Model:** `{quick_model}`
- **Latency:** 5-30 seconds
- **API:** Gemini + Google Search grounding
- **Best for:** Fact-checking, current events, quick lookups, documentation
- **Features:** Real-time web search, thinking summaries

## Deep Research (research_deep)

**Agent:** `{deep_agent}`
- **Latency:** 3-20 minutes (can take up to 60 min for complex topics)
- **API:** Gemini Interactions API (Deep Research Agent)
- **Best for:** Research reports, competitive analysis, literature reviews
- **Features:**
  - Autonomous multi-step investigation
  - Built-in Google Search and URL analysis
  - Cited reports with sources
  - File search (RAG) with `file_search_store_names`
  - Format instructions for custom output structure

## Follow-up (research_followup)

**Model:** Configurable (default: gemini-3-pro-preview)
- **Latency:** 5-30 seconds
- **API:** Gemini Interactions API
- **Best for:** Clarification, elaboration, summarization of prior research
- **Requires:** `previous_interaction_id` from completed research
"""


@mcp.resource(
    "research://exports/{export_id}",
    name="Research Export",
    description="Download an exported research report. Use export_research_session tool first.",
)
def get_export_by_id(export_id: str) -> BlobResourceContents | TextResourceContents:
    """
    Retrieve an exported research report by its export ID.

    The export_research_session tool creates exports and returns resource URIs.
    This resource serves the content for download with proper MIME type.

    In VS Code Copilot, you can:
    - Click "Save" to download the file
    - Drag-and-drop from chat to your workspace

    Args:
        export_id: The unique export identifier from export_research_session

    Returns:
        Resource content with proper MIME type (Markdown, JSON, or DOCX)
    """
    import base64

    from pydantic import AnyUrl

    entry = _get_cached_export(export_id)
    if not entry:
        raise ValueError(f"Export not found or expired: {export_id}")

    logger.info("📥 Serving export %s (%s)", export_id, entry.result.filename)

    uri = AnyUrl(f"research://exports/{export_id}")
    mime_type = entry.result.mime_type

    # For text formats, return TextResourceContents
    if mime_type in ("text/markdown", "application/json"):
        return TextResourceContents(
            uri=uri,
            mimeType=mime_type,
            text=entry.result.content.decode("utf-8"),
        )

    # For binary formats (DOCX), return BlobResourceContents with base64
    return BlobResourceContents(
        uri=uri,
        mimeType=mime_type,
        blob=base64.b64encode(entry.result.content).decode("ascii"),
    )


@mcp.resource(
    "research://exports",
    name="Available Exports",
    description="List all currently cached exports ready for download.",
    mime_type="application/json",
)
def list_exports() -> str:
    """
    List all currently cached exports.

    Returns a JSON array of available exports with their metadata.
    Exports expire after 1 hour.
    """
    # Clean up expired entries first
    expired_keys = [k for k, v in _export_cache.items() if v.is_expired]
    for key in expired_keys:
        del _export_cache[key]

    exports = []
    for export_id, entry in _export_cache.items():
        remaining = (entry.created_at + timedelta(seconds=EXPORT_TTL_SECONDS)) - datetime.now(UTC)
        exports.append({
            "export_id": export_id,
            "uri": f"research://exports/{export_id}",
            "filename": entry.result.filename,
            "format": entry.result.format.value,
            "size": entry.result.size_human,
            "mime_type": entry.result.mime_type,
            "session_id": entry.session_id[:12] + "...",
            "expires_in": f"{max(0, int(remaining.total_seconds()))}s",
        })

    return json.dumps({"exports": exports, "count": len(exports)}, indent=2)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run the MCP server on stdio transport."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="gemini-research-mcp",
        description="Gemini Research MCP Server - AI-powered research tools",
    )
    parser.add_argument(
        "--api-key",
        metavar="KEY",
        help="Gemini API key (or set GEMINI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    # Set API key from CLI flag if provided (overrides env var)
    if args.api_key:
        os.environ["GEMINI_API_KEY"] = args.api_key

    logger.info("🚀 Starting Gemini Research MCP Server v%s (MCP SDK)", __version__)
    logger.info("   Transport: stdio")
    logger.info("   Task mode: enabled (MCP Tasks / SEP-1732)")

    mcp.run(transport="stdio")


# Export for use as module
__all__ = ["mcp", "main"]


if __name__ == "__main__":
    main()
