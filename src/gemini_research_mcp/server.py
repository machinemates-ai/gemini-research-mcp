"""
Gemini Research MCP Server

Provides AI-powered research tools via Gemini:
- research_web: Fast grounded web search (5-30 seconds) - Gemini + Google Search
- research_deep: Comprehensive multi-step research (3-20 minutes) - Deep Research Agent
- research_followup: Ask follow-up questions about completed research

Architecture:
- FastMCP with task=True for background task support (MCP Tasks / SEP-1732)
- Progress dependency for real-time progress reporting
"""

# NOTE: Do NOT use `from __future__ import annotations` with FastMCP/Pydantic
# as it breaks type resolution for Annotated parameters in tool functions

import asyncio
import contextlib
import logging
import time
from typing import Annotated

from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends, Progress
from fastmcp.server.sampling import SamplingTool
from fastmcp.server.tasks import TaskConfig
from pydantic import BaseModel, Field

from gemini_research_mcp import __version__
from gemini_research_mcp.citations import process_citations
from gemini_research_mcp.config import LOGGER_NAME, get_deep_research_agent, get_model
from gemini_research_mcp.deep import deep_research_stream, get_research_status, start_research_async
from gemini_research_mcp.deep import research_followup as _research_followup
from gemini_research_mcp.quick import quick_research
from gemini_research_mcp.types import DeepResearchError, DeepResearchResult

# Configure logging
logger = logging.getLogger(LOGGER_NAME)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# =============================================================================
# Optional Context Dependency (for background task compatibility)
# =============================================================================

from collections.abc import AsyncGenerator


@contextlib.asynccontextmanager
async def _optional_context() -> AsyncGenerator[Context | None, None]:
    """
    Provide Context if available, None otherwise.

    Unlike CurrentContext which raises RuntimeError when no context exists,
    this dependency gracefully returns None when running in background tasks.
    """
    try:
        from fastmcp.server.context import _current_context

        ctx = _current_context.get()
        yield ctx  # May be None
    except Exception:
        yield None


# Dependency marker for optional context
OptionalContext = _optional_context


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

## Follow-up (research_followup)
Continue conversation after deep research completes.
Use for: "elaborate", "clarify", "summarize", follow-up questions.

**Workflow:**
- Simple questions → research_web
- Complex questions → research_deep (handles everything automatically)
""",
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


@mcp.tool(annotations={"readOnlyHint": True})
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
# SEP-1577 Sampling with Tools - Clarification via ctx.sample()
# =============================================================================
#
# This implements the "Sampling with Tools" pattern from SEP-1577:
# - The LLM can call internal tools during sampling to gather information
# - Tools run within the sampling loop where Context is STILL available
# - This allows elicitation BEFORE background task spawns
#
# Requires: chat.mcp.serverSampling enabled for this server in VS Code settings


class AnalyzedQuery(BaseModel):
    """Result from sampling-based query analysis."""

    refined_query: str = Field(description="The refined research query to use")
    was_clarified: bool = Field(description="Whether clarifying questions were asked")
    summary: str = Field(description="Brief summary of what was refined")


# Context holder for sampling tools (passed via closure)
_clarification_context: Context | None = None


async def _ask_clarifying_questions(
    questions: list[str],
    original_query: str,
) -> str:
    """
    Internal sampling tool: Present clarifying questions to user via elicitation.

    This tool is called by the LLM during ctx.sample() when it determines
    that the query needs clarification. The Context is still available here
    because we're inside the sampling loop (foreground), not in a background task.

    Args:
        questions: List of clarifying questions to ask
        original_query: The original research query for context

    Returns:
        Concatenated answers from user, or empty string if skipped
    """
    global _clarification_context
    ctx = _clarification_context

    if ctx is None:
        logger.warning("No context available for clarification")
        return ""

    logger.info("   🎯 Sampling tool: asking %d clarifying questions", len(questions))

    # Build dynamic schema for elicitation
    field_definitions: dict[str, tuple] = {}
    for i, question in enumerate(questions[:5]):  # Cap at 5 questions
        field_name = f"answer_{i + 1}"
        field_definitions[field_name] = (
            str,
            Field(default="", description=question),
        )

    DynamicSchema = type(
        "ClarificationQuestions",
        (BaseModel,),
        {"__annotations__": {k: v[0] for k, v in field_definitions.items()}}
        | {k: v[1] for k, v in field_definitions.items()},
    )

    try:
        message = (
            f"To improve research quality for:\n\n**\"{original_query}\"**\n\n"
            f"Please answer these questions (optional - press 'Skip' to continue):"
        )

        result = await ctx.elicit(message=message, response_type=DynamicSchema)

        if result.action == "accept" and result.data:
            data = result.data if isinstance(result.data, dict) else result.data.model_dump()
            answers = [data.get(f"answer_{i + 1}", "") for i in range(len(questions))]
            non_empty = [a for a in answers if a.strip()]
            logger.info("   ✨ User provided %d/%d answers", len(non_empty), len(questions))
            return "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(questions, answers, strict=False) if a.strip())
        else:
            logger.info("   ⏭️ User skipped clarification")
            return ""

    except Exception as e:
        logger.warning("   ⚠️ Elicitation failed: %s", e)
        return ""


# Create SamplingTool for use with ctx.sample()
_clarify_tool = SamplingTool(
    name="ask_clarifying_questions",
    description=(
        "Ask the user clarifying questions to refine a vague research query. "
        "Call this when the query is ambiguous, lacks scope, or could benefit from context. "
        "Provide 2-5 focused questions that will help narrow down the research scope."
    ),
    parameters={
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of 2-5 clarifying questions to ask the user",
                "minItems": 1,
                "maxItems": 5,
            },
            "original_query": {
                "type": "string",
                "description": "The original research query being analyzed",
            },
        },
        "required": ["questions", "original_query"],
    },
    fn=_ask_clarifying_questions,
)


# System prompt for the sampling-based query analyzer
_ANALYZER_SYSTEM_PROMPT = """\
You are a research query analyzer. Analyze the query and decide if clarification is needed.

## Decision Criteria
- If the query is SPECIFIC (clear scope, timeframe, focus area) → return it as-is
- If the query is VAGUE (ambiguous terms, no scope, multiple interpretations) → use ask_clarifying_questions tool

## Examples of VAGUE queries needing clarification:
- "compare python frameworks" (Which frameworks? For what purpose?)
- "research AI" (What aspect? Applications? Research papers? Companies?)
- "best practices" (For what? What industry? What scale?)

## Examples of SPECIFIC queries (no clarification needed):
- "Compare FastAPI vs Django for building REST APIs in 2025"
- "Research the environmental impact of electric vehicles vs gasoline cars"
- "Analyze the top 5 JavaScript testing frameworks for React applications"

When calling ask_clarifying_questions, ask 2-4 focused questions that will meaningfully improve research quality.
"""


async def _clarify_query_via_sampling(
    query: str,
    ctx: Context,
) -> str:
    """
    Use ctx.sample() with tools to analyze and potentially clarify the query.

    This is the SEP-1577 pattern: the LLM can call the ask_clarifying_questions
    tool during sampling, which uses ctx.elicit() to interact with the user.
    All of this happens in FOREGROUND before any background task spawns.

    Args:
        query: The research query to analyze
        ctx: The MCP Context (must be available for this to work)

    Returns:
        The refined query (or original if no clarification needed/provided)
    """
    global _clarification_context

    logger.info("🔍 Analyzing query via sampling (SEP-1577)...")

    # Set context for the sampling tool
    _clarification_context = ctx

    try:
        result = await ctx.sample(
            messages=f"Analyze this research query and refine if needed:\n\n{query}",
            system_prompt=_ANALYZER_SYSTEM_PROMPT,
            tools=[_clarify_tool],
            result_type=AnalyzedQuery,
            max_tokens=1024,
        )

        analyzed = result.result
        if analyzed.was_clarified:
            logger.info("   ✨ Query clarified: %s", analyzed.summary[:80] if analyzed.summary else "")
            logger.info("   📝 Refined: %s", analyzed.refined_query[:100])
        else:
            logger.info("   ✅ Query was specific enough (no clarification needed)")

        return analyzed.refined_query

    except Exception as e:
        logger.warning("   ⚠️ Sampling-based clarification failed (%s), using original query", e)
        return query

    finally:
        # Clear the context after use
        _clarification_context = None


async def _maybe_clarify_query(
    query: str,
    ctx: Context | None,
) -> str:
    """
    Analyze query and optionally ask clarifying questions via ctx.sample().

    This implements SEP-1577 "Sampling with Tools" pattern:
    - Uses ctx.sample() with internal tools in FOREGROUND
    - LLM can call ask_clarifying_questions tool to elicit user input
    - All clarification happens BEFORE background task spawns

    Requires: chat.mcp.serverSampling enabled for gemini-research-mcp

    Args:
        query: The research query
        ctx: MCP Context (None when running in background task)

    Returns the refined query, or original if clarification was skipped/unavailable.
    """
    # Skip if no context (running in background task)
    if ctx is None:
        logger.info("🔍 Skipping clarification (background task mode)")
        return query

    # Use sampling-based clarification (SEP-1577 pattern)
    return await _clarify_query_via_sampling(query, ctx)


# =============================================================================
# Deep Research Tool (with integrated clarification)
# =============================================================================


@mcp.tool(task=TaskConfig(mode="optional"), annotations={"readOnlyHint": True})
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
    progress: Progress = Progress(),  # noqa: B008
    ctx: Context | None = Depends(OptionalContext),  # noqa: B008 - None in background tasks
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

    # Set progress total (100 = 100%)
    await progress.set_total(100)

    # ==========================================================================
    # Phase 1: Query Clarification (requires foreground Context)
    # ==========================================================================
    effective_query = query

    try:
        await progress.set_message("Analyzing query...")
        effective_query = await _maybe_clarify_query(query, ctx)
        if effective_query != query:
            logger.info("   ✨ Using refined query: %s", effective_query[:100])
    except Exception as e:
        # Clarification unavailable (background mode) - proceed with original
        logger.info("   ℹ️ Clarification skipped (%s), using original query", type(e).__name__)

    # ==========================================================================
    # Phase 2: Deep Research Execution
    # ==========================================================================
    await progress.set_message("Starting deep research...")

    try:
        await progress.set_message("Initiating research agent...")

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

            # Track events for progress - show thought/action CONTENT
            if event.event_type == "thought":
                thought_count += 1
                # Display thought content (truncated to 55 chars) for transparency
                content = event.content or ""
                short_thought = content[:55] + "..." if len(content) > 55 else content
                await progress.set_message(f"[{thought_count}] 🧠 {short_thought}")
                # Progress: cap at 50% during thinking phase
                await progress.increment(min(2, 50 - thought_count * 2))
            elif event.event_type == "action":
                action_count += 1
                # Display action content (e.g., search query) for transparency
                content = event.content or ""
                short_action = content[:55] + "..." if len(content) > 55 else content
                await progress.set_message(f"[{action_count}] 🔍 {short_action}")
            elif event.event_type == "start":
                await progress.set_message("🚀 Research agent autonomous investigation started")
            elif event.event_type == "error":
                logger.error("   Stream error: %s", event.content)

        if not interaction_id:
            raise ValueError("No interaction_id received from stream")

        logger.info("   📊 Stream consumed: %d thoughts, %d actions", thought_count, action_count)
        await progress.set_message("Waiting for research completion...")

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
                await progress.set_message(f"✅ Research complete ({_format_duration(elapsed)})")

                result = await process_citations(result, resolve_urls=True)

                return _format_deep_research_report(result, interaction_id, elapsed)

            elif raw_status in ("failed", "cancelled"):
                logger.error("   ❌ Research %s after %s", raw_status, _format_duration(elapsed))
                raise DeepResearchError(
                    code=f"RESEARCH_{raw_status.upper()}",
                    message=f"Research {raw_status} after {_format_duration(elapsed)}",
                )
            else:
                # Still working - update progress (cap at 90%)
                progress_pct = min(90, int(50 + (elapsed / max_wait) * 40))
                msg = f"⏳ Researching... ({_format_duration(elapsed)}, ~{progress_pct}%)"
                await progress.set_message(msg)

            # Wait before next poll
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


@mcp.tool(annotations={"readOnlyHint": True})
async def research_followup(
    previous_interaction_id: Annotated[
        str, "The interaction_id from a completed research_deep task"
    ],
    query: Annotated[
        str, "Follow-up question about the research (e.g., 'elaborate on the second point')"
    ],
    model: Annotated[
        str, "Model to use for follow-up. Default: gemini-3-pro-preview"
    ] = "gemini-3-pro-preview",
) -> str:
    """
    Continue conversation after deep research. Ask follow-up questions without restarting.

    Use for: "clarify", "elaborate", "summarize", "explain more", "what about",
    continue discussion, ask more questions about completed research results.

    Args:
        previous_interaction_id: The interaction_id from research_deep
        query: Your follow-up question
        model: Model to use (default: gemini-3-pro-preview)

    Returns:
        Response to the follow-up question
    """
    logger.info("💬 research_followup: %s -> %s", previous_interaction_id, query[:100])

    try:
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


# =============================================================================
# Async Research Tools (Non-blocking pattern)
# =============================================================================


@mcp.tool(annotations={"readOnlyHint": True})
async def start_research(
    query: Annotated[str, "Research question or complex topic requiring thorough investigation"],
    format_instructions: Annotated[
        str | None,
        (
            "Optional formatting instructions for the report output "
            "(e.g., 'Format as executive briefing', 'Include comparison table')"
        ),
    ] = None,
    file_search_store_names: Annotated[
        list[str] | None,
        (
            "Optional: Search your own data alongside web search. "
            "List of Gemini File Search store names (e.g., ['fileSearchStores/my-store'])."
        ),
    ] = None,
) -> str:
    """
    Start deep research WITHOUT waiting. Returns immediately with interaction_id.

    Use when: You want to kick off research and do other work while it runs.
    Research typically takes 3-20 minutes.

    **Workflow:**
    1. Call start_research() → get interaction_id
    2. Do other work or wait a few minutes
    3. Call check_research() with the interaction_id to get results

    For blocking research with progress updates, use research_deep instead.

    Args:
        query: Research question or complex topic
        format_instructions: Optional formatting instructions for the report output
        file_search_store_names: Optional file search store names for RAG

    Returns:
        interaction_id to use with check_research()
    """
    logger.info("🚀 start_research (async): %s", query[:100])

    try:
        interaction_id = await start_research_async(
            query=query,
            format_instructions=format_instructions,
            file_search_store_names=file_search_store_names,
        )

        return (
            f"✅ Research started!\n\n"
            f"**Interaction ID:** `{interaction_id}`\n\n"
            f"Research typically takes 3-20 minutes. "
            f"Use `check_research` with this ID to get results."
        )

    except Exception as e:
        logger.exception("start_research failed: %s", e)
        return f"❌ Failed to start research: {e}"


@mcp.tool(annotations={"readOnlyHint": True})
async def check_research(
    interaction_id: Annotated[str, "The interaction_id from start_research"],
) -> str:
    """
    Check status or get results of a research task started with start_research.

    Returns status ("in_progress", "completed", "failed") and the full report if complete.

    Args:
        interaction_id: The interaction_id from start_research()

    Returns:
        Status and results (if completed)
    """
    logger.info("🔍 check_research: %s", interaction_id)

    try:
        result = await get_research_status(interaction_id)

        raw_status = "unknown"
        if result.raw_interaction:
            raw_status = getattr(result.raw_interaction, "status", "unknown")

        if raw_status == "completed":
            result = await process_citations(result, resolve_urls=True)
            duration = result.duration_seconds or 0
            report = _format_deep_research_report(result, interaction_id, duration)
            return f"## ✅ Research Complete\n\n{report}"

        elif raw_status == "failed":
            error = getattr(result.raw_interaction, "error", "Unknown error")
            return f"## ❌ Research Failed\n\nError: {error}\n\nInteraction ID: `{interaction_id}`"

        elif raw_status == "cancelled":
            return f"## ⚠️ Research Cancelled\n\nInteraction ID: `{interaction_id}`"

        else:
            return (
                f"## ⏳ Research In Progress\n\n"
                f"Status: `{raw_status}`\n\n"
                f"Research is still running. Check again in a few minutes.\n\n"
                f"Interaction ID: `{interaction_id}`"
            )

    except Exception as e:
        logger.exception("check_research failed: %s", e)
        return f"❌ Failed to check research: {e}"


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

## Deep Research (research_deep, start_research)

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


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run the MCP server on stdio transport."""
    logger.info("🚀 Starting Gemini Research MCP Server v%s (FastMCP)", __version__)
    logger.info("   Transport: stdio")
    logger.info("   Task mode: enabled (MCP Tasks / SEP-1732)")

    mcp.run(transport="stdio")


# Export for use as module
__all__ = ["mcp", "main"]


if __name__ == "__main__":
    main()
