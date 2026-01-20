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
from dataclasses import dataclass, field
from typing import Any

from google import genai

from gemini_research_mcp.citations import process_citations
from gemini_research_mcp.config import (
    CLIENT_MAX_AGE_SECONDS,
    CLIENT_MAX_REQUESTS,
    DEFAULT_TIMEOUT,
    INITIAL_RETRY_BACKOFF,
    LOGGER_NAME,
    MAX_INITIAL_RETRIES,
    MAX_INITIAL_RETRY_DELAY,
    MAX_POLL_TIME,
    MAX_STREAM_RETRIES,
    MAX_STREAM_RETRY_DELAY,
    RECONNECT_DELAY,
    STREAM_POLL_INTERVAL,
    STREAM_RETRY_BACKOFF,
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


# =============================================================================
# Client Health Management
# =============================================================================


@dataclass
class ClientHealth:
    """Track client health for long-running servers."""

    created_at: float = field(default_factory=time.time)
    request_count: int = 0
    last_request_at: float = field(default_factory=time.time)
    consecutive_failures: int = 0

    def record_request(self) -> None:
        """Record a successful request."""
        self.request_count += 1
        self.last_request_at = time.time()
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.consecutive_failures += 1

    def needs_refresh(self) -> bool:
        """Check if client should be refreshed."""
        age = time.time() - self.created_at
        idle_time = time.time() - self.last_request_at

        # Refresh if client is too old
        if age > CLIENT_MAX_AGE_SECONDS:
            logger.info(
                "🔄 Client needs refresh: age=%.0fs > max=%.0fs",
                age, CLIENT_MAX_AGE_SECONDS
            )
            return True

        # Refresh if too many requests (if enabled)
        if CLIENT_MAX_REQUESTS > 0 and self.request_count >= CLIENT_MAX_REQUESTS:
            logger.info(
                "🔄 Client needs refresh: requests=%d >= max=%d",
                self.request_count, CLIENT_MAX_REQUESTS
            )
            return True

        # Refresh if too many consecutive failures
        if self.consecutive_failures >= 3:
            logger.info(
                "🔄 Client needs refresh: consecutive_failures=%d",
                self.consecutive_failures
            )
            return True

        # Refresh if idle for too long (half of max age)
        if idle_time > CLIENT_MAX_AGE_SECONDS / 2:
            logger.info("🔄 Client needs refresh: idle_time=%.0fs", idle_time)
            return True

        return False


# Global client management
_client: genai.Client | None = None
_client_health: ClientHealth | None = None


def _get_healthy_client() -> genai.Client:
    """Get a healthy Gemini client, creating a new one if needed."""
    global _client, _client_health

    if _client is None or _client_health is None or _client_health.needs_refresh():
        logger.info("🔌 Creating new Gemini client")
        _client = genai.Client(api_key=get_api_key())
        _client_health = ClientHealth()

    return _client


def _record_client_success() -> None:
    """Record a successful client operation."""
    global _client_health
    if _client_health:
        _client_health.record_request()


def _record_client_failure() -> None:
    """Record a failed client operation."""
    global _client_health
    if _client_health:
        _client_health.record_failure()


def _force_client_refresh() -> None:
    """Force client refresh on next request."""
    global _client, _client_health
    logger.warning("⚠️ Forcing client refresh due to critical failure")
    _client = None
    _client_health = None


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
            text = last_output.text
            return str(text) if text is not None else None
        if hasattr(last_output, "content"):
            content = last_output.content
            return str(content) if content is not None else None
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
    Implements automatic reconnection on network interruptions with exponential backoff.

    Features:
    - Client health monitoring for long-running servers
    - Exponential backoff with configurable limits
    - Detailed logging for debugging null interaction_id issues
    - Automatic client refresh on consecutive failures

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
    # Use health-monitored client instead of creating new one each time
    client = _get_healthy_client()
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
    logger.info("   Max initial retries: %d", MAX_INITIAL_RETRIES)
    logger.info("   Max stream retries: %d", MAX_STREAM_RETRIES)
    logger.info("=" * 60)

    interaction_id: str | None = None
    last_event_id: str | None = None
    is_complete = False
    initial_retry_delay = RECONNECT_DELAY
    stream_retry_delay = RECONNECT_DELAY
    stream_retry_count = 0
    disconnect_count = 0
    received_any_event = False

    async def process_stream(stream: Any) -> AsyncIterator[DeepResearchProgress]:
        """Process events from a stream (initial or resumed)."""
        nonlocal interaction_id, last_event_id, is_complete, received_any_event

        chunk_count = 0
        async for chunk in stream:
            chunk_count += 1
            received_any_event = True
            elapsed = time.time() - stream_start_time

            chunk_type = getattr(chunk, "event_type", "unknown")
            logger.debug("[%.1fs] 📦 CHUNK #%d: type=%s", elapsed, chunk_count, chunk_type)

            if chunk.event_type == "interaction.start":
                interaction_id = chunk.interaction.id
                logger.info("[%.1fs] 🚀 interaction.start: id=%s", elapsed, interaction_id)
                _record_client_success()  # Record successful API interaction
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

    # ==========================================================================
    # Phase 1: Initial connection with exponential backoff
    # ==========================================================================
    initial_attempt = 0

    while initial_attempt < MAX_INITIAL_RETRIES:
        initial_attempt += 1
        elapsed_t = time.time() - stream_start_time

        # Refresh client on each retry attempt to pick up health-based refreshes
        client = _get_healthy_client()

        try:
            logger.info(
                "⏱️ [%.1fs] 🔌 Initial connection attempt %d/%d...",
                elapsed_t, initial_attempt, MAX_INITIAL_RETRIES
            )

            stream = await client.aio.interactions.create(**create_kwargs)

            if stream is None:
                _record_client_failure()
                logger.warning(
                    "⏱️ [%.1fs] ⚠️ Stream returned None (attempt %d/%d)",
                    time.time() - stream_start_time, initial_attempt, MAX_INITIAL_RETRIES
                )
                # Exponential backoff for retries
                backoff = INITIAL_RETRY_BACKOFF ** (initial_attempt - 1)
                wait_time = min(initial_retry_delay * backoff, MAX_INITIAL_RETRY_DELAY)
                logger.info("   ⏳ Waiting %.1fs before retry...", wait_time)
                await asyncio.sleep(wait_time)
                continue

            logger.info("⏱️ [%.1fs] ✅ Stream connected", time.time() - stream_start_time)
            async for progress in process_stream(stream):
                yield progress

            # If we got here without receiving interaction.start, log it
            if interaction_id is None and received_any_event:
                logger.warning(
                    "⏱️ [%.1fs] ⚠️ Stream ended but never received interaction.start event",
                    time.time() - stream_start_time
                )
            break

        except TypeError as e:
            if "NoneType" in str(e) and "not iterable" in str(e):
                _record_client_failure()
                logger.warning(
                    "⏱️ [%.1fs] ⚠️ Stream returned None (TypeError, attempt %d/%d): %s",
                    time.time() - stream_start_time, initial_attempt, MAX_INITIAL_RETRIES, e
                )
                backoff = INITIAL_RETRY_BACKOFF ** (initial_attempt - 1)
                wait_time = min(initial_retry_delay * backoff, MAX_INITIAL_RETRY_DELAY)
                logger.info("   ⏳ Waiting %.1fs before retry...", wait_time)
                await asyncio.sleep(wait_time)
                continue
            disconnect_count += 1
            _record_client_failure()
            elapsed_t = time.time() - stream_start_time
            logger.warning(
                "⏱️ [%.1fs] ❌ DISCONNECT #%d (TypeError): %s",
                elapsed_t, disconnect_count, e
            )
            break

        except Exception as e:
            disconnect_count += 1
            _record_client_failure()
            elapsed_t = time.time() - stream_start_time
            error_str = str(e)
            logger.warning(
                "⏱️ [%.1fs] ❌ DISCONNECT #%d: %s",
                elapsed_t, disconnect_count, error_str
            )

            # Check if this is a retryable error
            if is_retryable_error(error_str) and initial_attempt < MAX_INITIAL_RETRIES:
                backoff = INITIAL_RETRY_BACKOFF ** (initial_attempt - 1)
                wait_time = min(initial_retry_delay * backoff, MAX_INITIAL_RETRY_DELAY)
                logger.info("   🔄 Retryable error, waiting %.1fs before retry...", wait_time)
                await asyncio.sleep(wait_time)
                continue
            break

    # ==========================================================================
    # Phase 2: Check if we have interaction_id for reconnection
    # ==========================================================================
    if interaction_id is None and not is_complete:
        elapsed = time.time() - stream_start_time
        logger.error(
            "⏱️ [%.1fs] ❌ CRITICAL: No interaction_id received after %d initial attempts. "
            "This may indicate API issues or rate limiting. "
            "Server restart may be required if this persists.",
            elapsed, initial_attempt
        )
        # Force client refresh for next request
        _force_client_refresh()
        yield DeepResearchProgress(
            event_type="error",
            content=(
                f"Failed to start research after {initial_attempt} attempts ({elapsed:.0f}s). "
                f"No interaction_id received from API. This may be a temporary API issue. "
                f"Please try again in a few minutes."
            ),
            interaction_id=None,
            event_id=None,
        )
        return

    # ==========================================================================
    # Phase 3: Reconnection loop with exponential backoff (if stream interrupted)
    # ==========================================================================
    while not is_complete and interaction_id and stream_retry_count < MAX_STREAM_RETRIES:
        stream_retry_count += 1
        elapsed = time.time() - stream_start_time
        short_id = interaction_id[:16] + "..." if len(interaction_id) > 16 else interaction_id
        logger.info(
            "⏱️ [%.1fs] 🔄 RECONNECT attempt %d/%d (id=%s)",
            elapsed, stream_retry_count, MAX_STREAM_RETRIES, short_id
        )

        # Exponential backoff
        backoff = STREAM_RETRY_BACKOFF ** (stream_retry_count - 1)
        wait_time = min(stream_retry_delay * backoff, MAX_STREAM_RETRY_DELAY)
        logger.info("   ⏳ Waiting %.1fs before reconnect...", wait_time)
        await asyncio.sleep(wait_time)

        try:
            # Refresh client if needed before reconnection attempt
            client = _get_healthy_client()

            # last_event_id can be None on first reconnect
            get_kwargs: dict[str, Any] = {"id": interaction_id, "stream": True}
            if last_event_id is not None:
                get_kwargs["last_event_id"] = last_event_id

            resume_stream = await client.aio.interactions.get(**get_kwargs)

            # Validate stream before recording success (API may return None)
            if resume_stream is None:
                _record_client_failure()
                logger.warning(
                    "⏱️ [%.1fs] ⚠️ Reconnect returned None (attempt %d/%d)",
                    time.time() - stream_start_time, stream_retry_count, MAX_STREAM_RETRIES
                )
                continue

            logger.info(
                "⏱️ [%.1fs] ✅ RECONNECTED successfully",
                time.time() - stream_start_time
            )
            _record_client_success()

            async for progress in process_stream(resume_stream):
                yield progress
                # Reset retry count on successful event
                stream_retry_count = 0

        except Exception as e:
            disconnect_count += 1
            _record_client_failure()
            elapsed_t = time.time() - stream_start_time
            error_str = str(e)
            logger.warning(
                "⏱️ [%.1fs] ❌ RECONNECT FAILED #%d: %s",
                elapsed_t, disconnect_count, error_str
            )

            # Force client refresh after multiple failures
            if disconnect_count >= 3:
                _force_client_refresh()

    # ==========================================================================
    # Phase 4: Final status check
    # ==========================================================================
    if not is_complete:
        elapsed = time.time() - stream_start_time
        logger.error(
            "⏱️ [%.1fs] ❌ RESEARCH FAILED: disconnects=%d, retries=%d, id=%s",
            elapsed, disconnect_count, stream_retry_count, interaction_id
        )
        yield DeepResearchProgress(
            event_type="error",
            content=(
                f"Research interrupted after {elapsed:.0f}s "
                f"({disconnect_count} disconnects, {stream_retry_count} reconnect attempts). "
                f"Interaction ID: {interaction_id}"
            ),
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
            cb_result = on_progress(progress)
            if inspect.isawaitable(cb_result):
                await cb_result

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
        client = _get_healthy_client()  # Use health-monitored client
        poll_start = time.time()

        while time.time() - poll_start < MAX_POLL_TIME:
            try:
                final_interaction = await client.aio.interactions.get(id=interaction_id)
                _record_client_success()  # Keep client alive during polling
                status = getattr(final_interaction, "status", "unknown")

                if on_progress:
                    elapsed = time.time() - poll_start
                    prog = DeepResearchProgress(
                        event_type="status",
                        content=f"Waiting... ({status}, {elapsed:.0f}s)",
                        interaction_id=interaction_id,
                    )
                    poll_cb_result = on_progress(prog)
                    if inspect.isawaitable(poll_cb_result):
                        await poll_cb_result

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


async def get_research_status(interaction_id: str) -> DeepResearchResult:
    """
    Get the current status of a Deep Research task.

    Internal helper used by research_deep to poll for completion.

    Args:
        interaction_id: The interaction ID from a research task

    Returns:
        DeepResearchResult with current status and any available outputs
    """
    client = _get_healthy_client()  # Use health-monitored client
    interaction = await client.aio.interactions.get(interaction_id)
    _record_client_success()

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

    client = _get_healthy_client()  # Use health-monitored client

    try:
        interaction = await client.aio.interactions.create(
            input=query,
            model=model,
            previous_interaction_id=previous_interaction_id,
        )
        _record_client_success()

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
