"""
Gemini Research MCP Server

Provides AI-powered research tools via Gemini:
- research_quick: Fast grounded search (5-30 seconds) - Gemini + Google Search
- research_deep: Comprehensive multi-step research (3-20 minutes) - Deep Research Agent
- research_status: Check status of deep research tasks
- research_followup: Ask follow-up questions about completed research

Architecture:
- FastMCP with task=True for background task support (MCP Tasks / SEP-1732)
- Progress dependency for real-time progress reporting
"""

# NOTE: Do NOT use `from __future__ import annotations` with FastMCP/Pydantic
# as it breaks type resolution for Annotated parameters in tool functions

import asyncio
import logging
import time
from typing import Annotated, Optional

from fastmcp import FastMCP
from fastmcp.dependencies import Progress
from fastmcp.server.tasks import TaskConfig

from gemini_research_mcp import __version__
from gemini_research_mcp.citations import process_citations
from gemini_research_mcp.deep import deep_research_stream, get_research_status, research_followup as _research_followup
from gemini_research_mcp.quick import quick_research
from gemini_research_mcp.types import DeepResearchError, DeepResearchResult

# Configure logging
logger = logging.getLogger("gemini-research-mcp")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# =============================================================================
# Server Instance
# =============================================================================

mcp = FastMCP(
    name="gemini-research",
    instructions="""
Gemini Research MCP Server - AI-powered research toolkit

This server provides four tools for different research needs:

## Quick Research (research_quick)
Fast web search with Gemini grounding (5-30 seconds)
- Uses: Gemini 2.5 Flash + Google Search grounding
- Best for: Quick lookups, fact-checking, current events, documentation
- Returns: Immediate results with citations

## Deep Research (research_deep)
Comprehensive autonomous research (3-20 minutes)
- Uses: Gemini Deep Research Agent (Interactions API)
- Best for: Complex questions, research reports, competitive analysis
- Runs as: Background task with progress updates
- Optional: Search your own data with file_search_store_names

## Status Check (research_status)
Check on ongoing or completed Deep Research tasks
- Provide the interaction_id from research_deep
- Returns: Current status and results if complete

## Follow-up (research_followup)
Continue the conversation after Deep Research completes
- Ask for clarification, elaboration, or summarization
- Avoids restarting the entire research task

**Workflow:** Use research_quick first for simple questions. If deeper 
investigation is needed, escalate to research_deep.
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


def _format_deep_research_report(result: DeepResearchResult, interaction_id: str, elapsed: float) -> str:
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
    lines.extend([
        "",
        "---",
        f"- Duration: {_format_duration(elapsed)}",
        f"- Interaction ID: `{interaction_id}`",
    ])

    return "\n".join(lines)


# =============================================================================
# Tools
# =============================================================================


@mcp.tool
async def research_quick(
    query: Annotated[str, "Research question or topic to search for"],
    include_thoughts: Annotated[bool, "Include thinking summary in response"] = False,
    thinking_budget: Annotated[str, "Thinking depth: 'minimal', 'low', 'medium' (default), 'high', 'max', 'dynamic'"] = "medium",
) -> str:
    """
    Fast web research using Gemini grounded search.

    Returns answer with citations in 5-30 seconds.
    Best for: fact-checking, quick lookups, current events, documentation searches.

    Args:
        query: Research question or topic to search for
        include_thoughts: Include thinking summary in response
        thinking_budget: Thinking depth level

    Returns:
        Research results with sources as markdown text
    """
    logger.info("🔎 research_quick: %s", query[:100])
    start = time.time()

    try:
        result = await quick_research(
            query=query,
            thinking_budget=thinking_budget,
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
        lines.extend([
            "",
            "---",
            f"*Completed in {_format_duration(elapsed)}*",
        ])

        return "\n".join(lines)

    except Exception as e:
        logger.exception("research_quick failed: %s", e)
        return f"❌ Research failed: {e}"


@mcp.tool(task=TaskConfig(mode="required"))
async def research_deep(
    query: Annotated[str, "Research question or complex topic requiring thorough investigation"],
    format_instructions: Annotated[Optional[str], "Optional instructions for report format (e.g., 'include comparison table')"] = None,
    file_search_store_names: Annotated[Optional[list[str]], "Optional list of Gemini File Search store names for RAG (e.g., ['fileSearchStores/my-store'])"] = None,
    progress: Progress = Progress(),
) -> str:
    """
    Comprehensive multi-step research using Gemini Deep Research Agent.

    Autonomously plans, searches multiple sources, reads pages, and synthesizes
    a detailed report. Takes 3-20 minutes.

    Best for: complex questions, research reports, comparative analysis.

    This tool runs as a background task. Progress updates will be sent during execution.

    Args:
        query: Research question or complex topic
        format_instructions: Optional formatting instructions for the report
        file_search_store_names: Optional list of file search store names for searching your own data
        progress: Progress tracker (injected by FastMCP)

    Returns:
        Comprehensive research report with citations
    """
    logger.info("🔬 research_deep (TASK): %s", query[:100])
    if file_search_store_names:
        logger.info("   📁 File search stores: %s", file_search_store_names)
    start = time.time()

    # Set progress total (100 = 100%)
    await progress.set_total(100)
    await progress.set_message("Starting deep research...")

    try:
        # Start the deep research stream
        await progress.set_message("Initiating research agent...")

        thought_count = 0
        action_count = 0
        interaction_id: str | None = None

        # Consume the stream to get interaction_id and track progress
        async for event in deep_research_stream(
            query=query,
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
                short_thought = (event.content[:55] + "...") if event.content and len(event.content) > 55 else (event.content or "")
                await progress.set_message(f"[{thought_count}] 🧠 {short_thought}")
                # Progress: cap at 50% during thinking phase
                await progress.increment(min(2, 50 - thought_count * 2))
            elif event.event_type == "action":
                action_count += 1
                # Display action content (e.g., search query) for transparency
                short_action = (event.content[:55] + "...") if event.content and len(event.content) > 55 else (event.content or "")
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
                raw_status = getattr(result.raw_interaction, 'status', 'unknown')

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
                await progress.set_message(f"⏳ Researching... ({_format_duration(elapsed)}, ~{progress_pct}%)")

            # Wait before next poll
            await asyncio.sleep(poll_interval)

        # Timeout
        elapsed = time.time() - start
        raise DeepResearchError(
            code="TIMEOUT",
            message=f"Research timed out after {_format_duration(elapsed)}. Check status with research_status.",
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


@mcp.tool
async def research_status(
    interaction_id: Annotated[str, "The interaction_id from a previous research_deep call"],
) -> str:
    """
    Check the status of a Deep Research task or retrieve results from a completed task.

    Use this to check on ongoing research or to re-fetch results from a previous session.

    Args:
        interaction_id: The interaction_id from research_deep

    Returns:
        Current status and results if complete
    """
    logger.info("📊 research_status: %s", interaction_id)

    try:
        result = await get_research_status(interaction_id)

        raw_status = "unknown"
        if result.raw_interaction:
            raw_status = getattr(result.raw_interaction, 'status', 'unknown')

        lines = [f"## Research Status: {raw_status.upper()}"]
        lines.append("")
        lines.append(f"**Interaction ID:** `{interaction_id}`")

        if raw_status == "completed":
            result = await process_citations(result, resolve_urls=True)

            if result.text:
                lines.extend(["", "## Report", result.text])

            if result.usage:
                lines.extend(["", "## Usage"])
                if result.usage.total_tokens:
                    lines.append(f"- Total tokens: {result.usage.total_tokens}")
                if result.usage.total_cost:
                    lines.append(f"- Estimated cost: ${result.usage.total_cost:.4f}")

            if result.duration_seconds:
                lines.append(f"- Duration: {_format_duration(result.duration_seconds)}")

        elif raw_status in ("failed", "cancelled"):
            lines.append(f"\n⚠️ Research {raw_status}. No results available.")

        else:
            lines.append("\n⏳ Research still in progress. Check again later.")

        return "\n".join(lines)

    except Exception as e:
        logger.exception("research_status failed: %s", e)
        return f"❌ Error checking status: {e}"


@mcp.tool
async def research_followup(
    previous_interaction_id: Annotated[str, "The interaction_id from a completed research_deep task"],
    query: Annotated[str, "Follow-up question about the research (e.g., 'elaborate on the second point')"],
    model: Annotated[str, "Model to use for follow-up. Default: gemini-3-pro-preview"] = "gemini-3-pro-preview",
) -> str:
    """
    Ask a follow-up question about completed Deep Research.

    Continue the conversation after research completes to get clarification,
    summarization, or elaboration on specific sections without restarting
    the entire research task.

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
