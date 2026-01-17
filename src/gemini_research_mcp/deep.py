"""
Deep research using Gemini Deep Research Agent.

Provides comprehensive multi-step research with real-time progress.
Takes 3-20 minutes typically.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from google import genai

from gemini_research_mcp.citations import process_citations
from gemini_research_mcp.config import (
    DEFAULT_TIMEOUT,
    LOGGER_NAME,
    MAX_INITIAL_RETRIES,
    MAX_POLL_TIME,
    RECONNECT_DELAY,
    STREAM_POLL_INTERVAL,
    get_api_key,
    get_deep_research_agent,
    is_retryable_error,
)
from gemini_research_mcp.types import (
    DeepResearchError,
    DeepResearchProgress,
    DeepResearchResult,
    DeepResearchUsage,
)

logger = logging.getLogger(LOGGER_NAME)


def _extract_usage(interaction: Any) -> DeepResearchUsage | None:
    """Extract usage/cost information from an interaction response."""
    usage_data = getattr(interaction, "usage_metadata", None)

    if usage_data is None:
        usage_data = getattr(interaction, "usage", None)

    if usage_data is None:
        return None

    prompt_tokens = getattr(usage_data, "prompt_token_count", None)
    if prompt_tokens is None:
        prompt_tokens = getattr(usage_data, "prompt_tokens", None)

    completion_tokens = getattr(usage_data, "candidates_token_count", None)
    if completion_tokens is None:
        completion_tokens = getattr(usage_data, "completion_tokens", None)

    total_tokens = getattr(usage_data, "total_token_count", None)
    if total_tokens is None:
        total_tokens = getattr(usage_data, "total_tokens", None)

    raw_usage: dict[str, Any] = {}
    if hasattr(usage_data, "__dict__"):
        raw_usage = vars(usage_data)
    elif hasattr(usage_data, "to_dict"):
        raw_usage = usage_data.to_dict()

    return DeepResearchUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        raw_usage=raw_usage,
    )


def _extract_text_from_interaction(interaction: Any) -> str | None:
    """Extract the final text output from an interaction."""
    outputs = getattr(interaction, "outputs", [])
    if outputs:
        last_output = outputs[-1]
        if hasattr(last_output, "text"):
            return last_output.text
        if hasattr(last_output, "content"):
            return last_output.content
    return None


async def deep_research_stream(
    query: str,
    *,
    format_instructions: str | None = None,
    file_search_store_names: list[str] | None = None,
    agent_name: str | None = None,
) -> AsyncIterator[DeepResearchProgress]:
    """
    Stream deep research with real-time progress updates.

    Uses stream=True to receive thinking summaries and text deltas as they happen.
    Implements automatic reconnection on network interruptions.

    Args:
        query: Research question or topic
        format_instructions: Optional formatting instructions for output
        file_search_store_names: Optional list of file search store names for RAG
        agent_name: Deep Research agent to use

    Yields:
        DeepResearchProgress events with type:
        - "start": Research started, includes interaction_id
        - "thought": Thinking summary from the agent
        - "text": Text delta from the final report
        - "complete": Research finished successfully
        - "error": Research failed
    """
    client = genai.Client(api_key=get_api_key())
    agent_name = agent_name or get_deep_research_agent()

    prompt = f"{query}\n\n{format_instructions}" if format_instructions else query

    tools = None
    if file_search_store_names:
        tools = [
            {
                "type": "file_search",
                "file_search_store_names": file_search_store_names,
            }
        ]

    create_kwargs: dict[str, Any] = {
        "input": prompt,
        "agent": agent_name,
        "background": True,
        "stream": True,
        "agent_config": {
            "type": "deep-research",
            "thinking_summaries": "auto",
        },
    }
    if tools:
        create_kwargs["tools"] = tools

    stream_start_time = time.time()

    logger.info("=" * 60)
    logger.info("🔬 DEEP RESEARCH AGENT")
    logger.info("   Agent: %s", agent_name)
    logger.info("   Query: %s", query[:100])
    logger.info("=" * 60)

    interaction_id: str | None = None
    last_event_id: str | None = None
    is_complete = False
    max_retries = 5
    retry_delay = RECONNECT_DELAY
    retry_count = 0
    disconnect_count = 0

    async def process_stream(stream: Any) -> AsyncIterator[DeepResearchProgress]:
        """Process events from a stream (initial or resumed)."""
        nonlocal interaction_id, last_event_id, is_complete

        chunk_count = 0
        async for chunk in stream:
            chunk_count += 1
            elapsed = time.time() - stream_start_time

            chunk_type = getattr(chunk, "event_type", "unknown")
            logger.debug("[%.1fs] 📦 CHUNK #%d: type=%s", elapsed, chunk_count, chunk_type)

            if chunk.event_type == "interaction.start":
                interaction_id = chunk.interaction.id
                logger.debug("[%.1fs] 🚀 interaction.start: id=%s", elapsed, interaction_id)
                yield DeepResearchProgress(
                    event_type="start",
                    interaction_id=interaction_id,
                    event_id=getattr(chunk, "event_id", None),
                )
                continue

            if hasattr(chunk, "event_id") and chunk.event_id:
                last_event_id = chunk.event_id

            if chunk.event_type == "content.delta":
                delta = chunk.delta
                if delta.type == "thought_summary":
                    content = delta.content
                    thought_text = content.text if hasattr(content, "text") else str(content)
                    logger.debug("[%.1fs] 🧠 thought_summary", elapsed)
                    yield DeepResearchProgress(
                        event_type="thought",
                        content=thought_text,
                        interaction_id=interaction_id,
                        event_id=last_event_id,
                    )
                elif delta.type == "text":
                    logger.debug("[%.1fs] 📝 text delta: %d chars", elapsed, len(delta.text))
                    yield DeepResearchProgress(
                        event_type="text",
                        content=delta.text,
                        interaction_id=interaction_id,
                        event_id=last_event_id,
                    )

            elif chunk.event_type == "interaction.complete":
                interaction = getattr(chunk, "interaction", None)
                interaction_status = getattr(interaction, "status", "unknown")
                logger.info(
                    "[%.1fs] ✅ interaction.complete (status=%s)", elapsed, interaction_status
                )

                if interaction_status == "completed":
                    is_complete = True
                    yield DeepResearchProgress(
                        event_type="complete",
                        interaction_id=interaction_id,
                        event_id=last_event_id,
                    )
                else:
                    logger.warning(
                        "[%.1fs] ⚠️ interaction.complete but status='%s'",
                        elapsed,
                        interaction_status,
                    )

            elif chunk.event_type == "error":
                is_complete = True
                error_msg = getattr(chunk, "error", "Unknown error")
                logger.error("[%.1fs] ❌ error: %s", elapsed, error_msg)
                yield DeepResearchProgress(
                    event_type="error",
                    content=str(error_msg),
                    interaction_id=interaction_id,
                    event_id=last_event_id,
                )

    # Initial connection
    initial_attempt = 0

    while initial_attempt < MAX_INITIAL_RETRIES:
        initial_attempt += 1
        try:
            logger.info("⏱️ [%.1fs] 🔌 Connecting...", time.time() - stream_start_time)

            stream = await client.aio.interactions.create(**create_kwargs)

            if stream is None:
                logger.warning("⏱️ [%.1fs] ⚠️ Stream returned None", time.time() - stream_start_time)
                await asyncio.sleep(RECONNECT_DELAY)
                continue

            logger.info("⏱️ [%.1fs] ✅ Stream connected", time.time() - stream_start_time)
            async for progress in process_stream(stream):
                yield progress
            break

        except TypeError as e:
            if "NoneType" in str(e) and "not iterable" in str(e):
                logger.warning("⏱️ [%.1fs] ⚠️ Stream returned None", time.time() - stream_start_time)
                await asyncio.sleep(RECONNECT_DELAY)
                continue
            disconnect_count += 1
            elapsed_t = time.time() - stream_start_time
            logger.warning("⏱️ [%.1fs] ❌ DISCONNECT #%d: %s", elapsed_t, disconnect_count, e)
            break

        except Exception as e:
            disconnect_count += 1
            elapsed_t = time.time() - stream_start_time
            logger.warning("⏱️ [%.1fs] ❌ DISCONNECT #%d: %s", elapsed_t, disconnect_count, e)
            break

    # Reconnection loop
    while not is_complete and interaction_id and retry_count < max_retries:
        retry_count += 1
        elapsed = time.time() - stream_start_time
        logger.info("⏱️ [%.1fs] RECONNECT attempt %d/%d", elapsed, retry_count, max_retries)

        await asyncio.sleep(retry_delay)

        try:
            resume_stream = await client.aio.interactions.get(
                id=interaction_id,
                stream=True,
                last_event_id=last_event_id,
            )
            logger.info("⏱️ [%.1fs] RECONNECTED", time.time() - stream_start_time)
            async for progress in process_stream(resume_stream):
                yield progress
                retry_count = 0
        except Exception as e:
            disconnect_count += 1
            elapsed_t = time.time() - stream_start_time
            logger.warning("⏱️ [%.1fs] DISCONNECT #%d: %s", elapsed_t, disconnect_count, e)
            retry_delay = min(retry_delay * 1.5, 30.0)

    if not is_complete:
        elapsed = time.time() - stream_start_time
        logger.error("⏱️ [%.1fs] FAILED after %d disconnects", elapsed, disconnect_count)
        yield DeepResearchProgress(
            event_type="error",
            content=f"Research interrupted after {elapsed:.0f}s ({disconnect_count} disconnects)",
            interaction_id=interaction_id,
            event_id=last_event_id,
        )


async def deep_research(
    query: str,
    *,
    format_instructions: str | None = None,
    file_search_store_names: list[str] | None = None,
    on_progress: Callable[[DeepResearchProgress], None | Awaitable[None]] | None = None,
    agent_name: str | None = None,
    resolve_citations: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
) -> DeepResearchResult:
    """
    Comprehensive multi-step research using Gemini Deep Research Agent.

    Uses streaming internally to receive real-time thinking summaries
    and progress updates. The agent autonomously plans, searches, reads,
    and synthesizes information to produce a detailed report.

    Takes 3-20 minutes typically.

    Args:
        query: Research question or topic
        format_instructions: Optional formatting instructions for output
        file_search_store_names: Optional list of file search store names for RAG
        on_progress: Callback for each progress event (sync or async)
        agent_name: Deep Research agent to use
        resolve_citations: Whether to extract and resolve citation URLs
        timeout: Maximum wait time in seconds

    Returns:
        DeepResearchResult with collected text, thinking summaries, usage, and citations

    Raises:
        DeepResearchError: On timeout, failure, or API errors
    """
    start_time = time.time()
    text_parts: list[str] = []
    thinking_summaries: list[str] = []
    interaction_id: str | None = None
    raw_interaction: Any = None

    async for progress in deep_research_stream(
        query,
        format_instructions=format_instructions,
        file_search_store_names=file_search_store_names,
        agent_name=agent_name,
    ):
        if on_progress:
            result = on_progress(progress)
            if inspect.isawaitable(result):
                await result

        if progress.event_type == "start":
            interaction_id = progress.interaction_id
        elif progress.event_type == "thought":
            if progress.content:
                thinking_summaries.append(progress.content)
        elif progress.event_type == "text":
            if progress.content:
                text_parts.append(progress.content)
        elif progress.event_type == "error":
            raise DeepResearchError(
                code="RESEARCH_FAILED",
                message=f"Deep Research failed: {progress.content}",
                details={"interaction_id": interaction_id},
            )

    final_text = "".join(text_parts)

    # Post-stream polling if we got no text but have interaction_id
    if not final_text.strip() and interaction_id:
        logger.info("🔄 POLLING: Stream ended without text...")
        client = genai.Client(api_key=get_api_key())
        poll_start = time.time()

        while time.time() - poll_start < MAX_POLL_TIME:
            try:
                final_interaction = await client.aio.interactions.get(id=interaction_id)
                status = getattr(final_interaction, "status", "unknown")

                if on_progress:
                    elapsed = time.time() - poll_start
                    prog = DeepResearchProgress(
                        event_type="status",
                        content=f"Waiting... ({status}, {elapsed:.0f}s)",
                        interaction_id=interaction_id,
                    )
                    result = on_progress(prog)
                    if inspect.isawaitable(result):
                        await result

                if status == "completed":
                    raw_interaction = final_interaction
                    outputs = getattr(final_interaction, "outputs", None)
                    if outputs and len(outputs) > 0:
                        final_text = getattr(outputs[-1], "text", "") or ""
                    break

                elif status == "failed":
                    error = getattr(final_interaction, "error", "Unknown error")
                    raise DeepResearchError(
                        code="RESEARCH_FAILED",
                        message=str(error),
                        details={"interaction_id": interaction_id},
                    )

                await asyncio.sleep(STREAM_POLL_INTERVAL)

            except DeepResearchError:
                raise
            except Exception as e:
                if is_retryable_error(str(e)):
                    await asyncio.sleep(STREAM_POLL_INTERVAL)
                else:
                    raise

        if not final_text.strip():
            raise DeepResearchError(
                code="TIMEOUT",
                message="Deep Research timed out",
                details={"interaction_id": interaction_id},
            )

    duration_seconds = time.time() - start_time
    usage = _extract_usage(raw_interaction) if raw_interaction else None

    result = DeepResearchResult(
        text=final_text,
        citations=[],
        thinking_summaries=thinking_summaries,
        interaction_id=interaction_id,
        usage=usage,
        duration_seconds=duration_seconds,
        raw_interaction=raw_interaction,
    )

    if resolve_citations and final_text:
        result = await process_citations(result, resolve_urls=True)

    return result


async def start_research_async(
    query: str,
    *,
    format_instructions: str | None = None,
    file_search_store_names: list[str] | None = None,
    agent_name: str | None = None,
) -> str:
    """
    Start a Deep Research task without waiting for completion.

    Use this for long-running research tasks where you want to:
    - Start research and do other work while it runs
    - Poll for status later with get_research_status()
    - Not block the MCP client

    Research typically takes 3-20 minutes to complete.

    Args:
        query: Research question or topic
        format_instructions: Optional formatting instructions for output
        file_search_store_names: Optional list of file search store names for RAG
        agent_name: Deep Research agent to use

    Returns:
        interaction_id: Use this to check status with get_research_status()
    """
    client = genai.Client(api_key=get_api_key())
    agent_name = agent_name or get_deep_research_agent()

    prompt = f"{query}\n\n{format_instructions}" if format_instructions else query

    tools = None
    if file_search_store_names:
        tools = [
            {
                "type": "file_search",
                "file_search_store_names": file_search_store_names,
            }
        ]

    create_kwargs: dict[str, Any] = {
        "input": prompt,
        "agent": agent_name,
        "background": True,  # Run in background mode
    }
    if tools:
        create_kwargs["tools"] = tools

    logger.info("🚀 Starting async deep research: %s", query[:100])
    interaction = await client.aio.interactions.create(**create_kwargs)
    logger.info("   📋 Interaction ID: %s", interaction.id)

    return interaction.id


async def get_research_status(interaction_id: str) -> DeepResearchResult:
    """
    Get the current status of a Deep Research task.

    Use this to check on research started with start_research_async().

    The returned DeepResearchResult includes:
    - raw_interaction.status: "in_progress", "completed", "failed", "cancelled"
    - text: The report text (only populated when completed)
    - usage: Token usage/cost info (when available)

    Args:
        interaction_id: The interaction ID from start_research_async()

    Returns:
        DeepResearchResult with current status and any available outputs
    """
    client = genai.Client(api_key=get_api_key())
    interaction = await client.aio.interactions.get(interaction_id)

    status = getattr(interaction, "status", "unknown")
    text = _extract_text_from_interaction(interaction) if status == "completed" else None
    usage = _extract_usage(interaction)

    return DeepResearchResult(
        text=text or "",
        citations=[],
        thinking_summaries=[],
        interaction_id=interaction_id,
        usage=usage,
        raw_interaction=interaction,
    )


async def research_followup(
    previous_interaction_id: str,
    query: str,
    *,
    model: str = "gemini-3-pro-preview",
) -> str:
    """
    Ask a follow-up question about a completed Deep Research task.

    This continues the conversation context from a previous research task,
    allowing clarification, summarization, or elaboration on specific sections
    without restarting the entire research.

    Args:
        previous_interaction_id: Interaction ID from a completed research task
                                 (available as result.interaction_id from research_deep)
        query: The follow-up question
        model: Model to use for the follow-up. Default: "gemini-3-pro-preview"

    Returns:
        The text response to the follow-up question

    Raises:
        DeepResearchError: On invalid interaction ID or API errors
    """
    logger.info("💬 Follow-up question for %s: %s", previous_interaction_id, query[:100])

    client = genai.Client(api_key=get_api_key())

    try:
        interaction = await client.aio.interactions.create(
            input=query,
            model=model,
            previous_interaction_id=previous_interaction_id,
        )

        # Extract text from the response
        text = _extract_text_from_interaction(interaction)

        if not text:
            # Try outputs directly
            outputs = getattr(interaction, "outputs", [])
            if outputs:
                text = str(outputs[-1])

        if not text:
            raise DeepResearchError(
                code="NO_RESPONSE",
                message="No response received from follow-up",
                details={"previous_interaction_id": previous_interaction_id},
            )

        logger.info("   ✅ Follow-up response received")
        return text

    except Exception as e:
        logger.exception("Follow-up question failed: %s", e)
        raise DeepResearchError(
            code="FOLLOWUP_FAILED",
            message=str(e),
            details={"previous_interaction_id": previous_interaction_id},
        ) from e
