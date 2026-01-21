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
    TextContent,
    TextResourceContents,
    ToolAnnotations,
)
from pydantic import AnyUrl, BaseModel, Field

from gemini_research_mcp import __version__
from gemini_research_mcp.citations import process_citations
from gemini_research_mcp.config import LOGGER_NAME, get_deep_research_agent, get_model
from gemini_research_mcp.deep import deep_research_stream, get_research_status
from gemini_research_mcp.deep import research_followup as _research_followup
from gemini_research_mcp.export import (
    ExportFormat,
    ExportResult,
    export_session,
)
from gemini_research_mcp.quick import (
    generate_session_metadata,
    quick_research,
    semantic_match_session,
)
from gemini_research_mcp.storage import (
    get_research_session,
    list_research_sessions,
    save_research_session,
)
from gemini_research_mcp.types import DeepResearchError, DeepResearchResult

# Configure logging
logger = logging.getLogger(LOGGER_NAME)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
    """Initialize task support during server startup."""
    global _task_support

    # Enable experimental task support on the low-level server
    _task_support = app._mcp_server.experimental.enable_tasks()

    logger.info("✅ Experimental task support enabled")

    async with _task_support.run():
        yield


# =============================================================================
# Server Instance
# =============================================================================

mcp = FastMCP(
    name="Gemini Research",
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
- Sessions are saved with AI-generated summaries for later follow-up

## Follow-up (research_followup)
Continue conversation with any previous deep research session.
Use for: "elaborate", "clarify", "summarize", follow-up questions.
- Automatically finds the matching session based on your question
- No need to track interaction_ids manually
- Sessions last 55 days (paid tier)

## List Sessions (list_research_sessions_tool)
Returns JSON list of previous research sessions with queries and summaries.
Use to answer "what research did I do about X?" questions.

## Export (export_research_session)
Export completed research to Markdown, JSON, or Word (DOCX) format.
Use for: sharing reports, archiving research, creating deliverables.

**Workflow:**
- Simple questions → research_web
- Complex questions → research_deep
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

    Returns:
        Comprehensive research report with citations
    """
    logger.info("🔬 research_deep: %s", query[:100])
    if format_instructions:
        logger.info("   📝 Format: %s", format_instructions[:80])
    if file_search_store_names:
        logger.info("   📁 File search stores: %s", file_search_store_names)

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

        # Consume the stream to get interaction_id and track progress
        async for event in deep_research_stream(
            query=effective_query,
            format_instructions=format_instructions,
            file_search_store_names=file_search_store_names,
        ):
            if event.interaction_id:
                interaction_id = event.interaction_id
                logger.info("   📋 interaction_id: %s", interaction_id)

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

                # Auto-save session for later follow-up
                total_tokens = None
                if result.usage and result.usage.total_tokens:
                    total_tokens = result.usage.total_tokens

                # Generate title and summary in one call (~$0.0003/call)
                metadata = await generate_session_metadata(
                    text=result.text or "",
                    query=effective_query,
                )

                # Save session for later follow-up (guard against filesystem errors)
                try:
                    save_research_session(
                        interaction_id=interaction_id,
                        query=effective_query,
                        title=metadata.title or None,
                        summary=metadata.summary or None,
                        report_text=result.text,
                        format_instructions=format_instructions,
                        agent_name=get_deep_research_agent(),
                        duration_seconds=elapsed,
                        total_tokens=total_tokens,
                    )
                except Exception as save_error:
                    logger.warning(
                        "⚠️ Failed to save session (research succeeded): %s",
                        save_error,
                    )

                return _format_deep_research_report(result, interaction_id, elapsed)

            elif raw_status in ("failed", "cancelled"):
                logger.error("   ❌ Research %s after %s", raw_status, _format_duration(elapsed))
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

    return json.dumps(
        {
            "sessions": session_list,
            "count": len(session_list),
            "hint": "Use research_followup with your question - auto-matches session.",
        },
        indent=2,
    )


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
    logger.info("🚀 Starting Gemini Research MCP Server v%s (MCP SDK)", __version__)
    logger.info("   Transport: stdio")
    logger.info("   Task mode: enabled (MCP Tasks / SEP-1732)")

    mcp.run(transport="stdio")


# Export for use as module
__all__ = ["mcp", "main"]


if __name__ == "__main__":
    main()
