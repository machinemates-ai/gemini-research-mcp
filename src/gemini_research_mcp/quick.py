"""
Quick research using Gemini grounded search.

Provides fast web research with citations in 5-30 seconds.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    ThinkingConfig,
    ThinkingLevel,
    Tool,
)

from gemini_research_mcp.config import (
    DEFAULT_THINKING_LEVEL,
    LOGGER_NAME,
    default_system_prompt,
    get_api_key,
    get_model,
    get_summary_model,
)
from gemini_research_mcp.types import ResearchResult, Source

if TYPE_CHECKING:
    from google.genai.types import GenerateContentResponse

logger = logging.getLogger(LOGGER_NAME)

# Map string levels to ThinkingLevel enum
THINKING_LEVEL_MAP = {
    "minimal": ThinkingLevel.MINIMAL,
    "low": ThinkingLevel.LOW,
    "medium": ThinkingLevel.MEDIUM,
    "high": ThinkingLevel.HIGH,
}


def _get_thinking_level(level: str) -> ThinkingLevel:
    """Convert string level to ThinkingLevel enum."""
    return THINKING_LEVEL_MAP.get(level.lower(), ThinkingLevel.HIGH)


def _extract_sources(response: GenerateContentResponse) -> tuple[list[Source], list[str]]:
    """Extract sources and queries from grounding metadata."""
    sources: list[Source] = []
    queries: list[str] = []

    if not response.candidates:
        return sources, queries

    candidate = response.candidates[0]
    if not candidate.grounding_metadata:
        return sources, queries

    gm = candidate.grounding_metadata

    # Extract search queries used
    if gm.web_search_queries:
        queries = list(gm.web_search_queries)

    # Extract sources from grounding chunks
    if gm.grounding_chunks:
        for chunk in gm.grounding_chunks:
            if hasattr(chunk, "web") and chunk.web:
                sources.append(
                    Source(
                        uri=chunk.web.uri or "",
                        title=chunk.web.title or "",
                    )
                )

    return sources, queries


async def quick_research(
    query: str,
    *,
    model: str | None = None,
    thinking_level: str = DEFAULT_THINKING_LEVEL,
    system_instruction: str | None = None,
    include_thoughts: bool = False,
) -> ResearchResult:
    """
    Fast grounded search using google_search tool.

    Returns response grounded in real-time web search results.
    Typically completes in 5-30 seconds.

    Args:
        query: Research question or topic
        model: Gemini model (default: gemini-3-flash-preview)
        thinking_level: Thinking depth: 'minimal', 'low', 'medium', 'high' (default)
        system_instruction: Optional system prompt
        include_thoughts: If True, include thinking summary in result

    Returns:
        ResearchResult with text, sources, queries, and optional thinking summary
    """
    client = genai.Client(api_key=get_api_key())
    model = model or get_model()
    level = _get_thinking_level(thinking_level)

    config = GenerateContentConfig(
        tools=[Tool(google_search=GoogleSearch())],
        thinking_config=ThinkingConfig(
            thinking_level=level,
            include_thoughts=include_thoughts,
        ),
        system_instruction=system_instruction or default_system_prompt(),
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=query,
        config=config,
    )

    sources, queries = _extract_sources(response)

    # Extract thinking summary if present
    thinking_summary = None
    if include_thoughts and response.candidates:
        content = response.candidates[0].content
        if content is not None and content.parts is not None:
            for part in content.parts:
                if hasattr(part, "thought") and part.thought:
                    thinking_summary = part.text
                    break

    return ResearchResult(
        text=response.text or "",
        sources=sources,
        queries=queries,
        thinking_summary=thinking_summary,
    )


async def generate_summary(
    text: str,
    query: str,
    *,
    max_chars: int = 300,
) -> str:
    """
    Generate a concise summary of research results using Gemini 3.0 Flash.

    Uses minimal thinking level for fastest, cheapest summarization.
    Cost: ~$0.0003 per summary (~100 output tokens).

    Args:
        text: The full research report text to summarize
        query: The original research query (for context)
        max_chars: Maximum characters for summary (default 300)

    Returns:
        A concise summary string
    """
    if not text:
        return ""

    client = genai.Client(api_key=get_api_key())
    model = get_summary_model()

    # Truncate input to first ~2000 chars to minimize tokens
    input_text = text[:2000]
    if len(text) > 2000:
        input_text += "..."

    prompt = f"""Summarize this research report in 2-3 sentences (max {max_chars} characters).
Focus on the key findings and main conclusions.

Original query: {query}

Report:
{input_text}

Summary:"""

    config = GenerateContentConfig(
        thinking_config=ThinkingConfig(
            thinking_level=ThinkingLevel.MINIMAL,
        ),
    )

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        summary = (response.text or "").strip()
        # Ensure we stay within limit
        if len(summary) > max_chars:
            summary = summary[: max_chars - 3] + "..."
        return summary
    except Exception as e:
        logger.warning("Summary generation failed: %s", e)
        return ""


async def semantic_match_session(
    user_query: str,
    sessions: list[dict[str, str]],
) -> str | None:
    """
    Find the research session that best matches a user's follow-up question.

    Uses Gemini 3.0 Flash with minimal thinking for fast semantic matching.
    Cost: ~$0.0003 per call.

    Args:
        user_query: The user's follow-up question
        sessions: List of dicts with 'id', 'query', and 'summary' keys

    Returns:
        The interaction_id of the best matching session, or None if no match
    """
    if not sessions:
        return None

    if len(sessions) == 1:
        # Only one session - return it directly
        return sessions[0]["id"]

    client = genai.Client(api_key=get_api_key())
    model = get_summary_model()

    # Build session list for prompt (truncate summaries to avoid context overflow)
    session_list = []
    for i, s in enumerate(sessions, 1):
        summary = s.get('summary', 'No summary')[:300]  # Limit summary length
        session_list.append(
            f"{i}. [{s['id']}] \"{s['query'][:100]}\"\n   Summary: {summary}"
        )

    prompt = f"""Given these research sessions and a user's follow-up question,
which session is the user most likely referring to?

Research Sessions:
{chr(10).join(session_list)}

User's follow-up question: "{user_query}"

Return ONLY the interaction_id (the value in brackets) of the best matching session.
If none of the sessions match the user's question, return exactly: NONE"""

    config = GenerateContentConfig(
        thinking_config=ThinkingConfig(
            thinking_level=ThinkingLevel.MINIMAL,
        ),
    )

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        result = (response.text or "").strip()

        # Handle no match
        if result.upper() == "NONE":
            return None

        # Validate the returned ID exists in our sessions
        valid_ids = {s["id"] for s in sessions}
        if result in valid_ids:
            return result

        # Try to extract ID if model added extra text
        for session_id in valid_ids:
            if session_id in result:
                return session_id

        logger.warning("semantic_match_session: invalid response '%s'", result)
        return None

    except Exception as e:
        logger.warning("semantic_match_session failed: %s", e)
        return None
