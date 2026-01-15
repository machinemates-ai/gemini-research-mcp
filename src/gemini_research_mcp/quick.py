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
    Tool,
)

from gemini_research_mcp.config import (
    default_system_prompt,
    get_api_key,
    get_model,
    get_thinking_budget,
)
from gemini_research_mcp.types import ResearchResult, Source

if TYPE_CHECKING:
    from google.genai.types import GenerateContentResponse

logger = logging.getLogger("deep-research-mcp")


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
    thinking_budget: str | int = "medium",
    system_instruction: str | None = None,
    include_thoughts: bool = False,
) -> ResearchResult:
    """
    Fast grounded search using google_search tool.

    Returns response grounded in real-time web search results.
    Typically completes in 5-30 seconds.

    Args:
        query: Research question or topic
        model: Gemini model (default: gemini-2.5-flash)
        thinking_budget: Token budget or level name
        system_instruction: Optional system prompt
        include_thoughts: If True, include thinking summary in result

    Returns:
        ResearchResult with text, sources, queries, and optional thinking summary
    """
    client = genai.Client(api_key=get_api_key())
    model = model or get_model()
    budget = get_thinking_budget(thinking_budget)

    config = GenerateContentConfig(
        tools=[Tool(google_search=GoogleSearch())],
        thinking_config=ThinkingConfig(
            thinking_budget=budget,
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
        for part in response.candidates[0].content.parts:
            if hasattr(part, "thought") and part.thought:
                thinking_summary = part.text
                break

    return ResearchResult(
        text=response.text or "",
        sources=sources,
        queries=queries,
        thinking_summary=thinking_summary,
    )
