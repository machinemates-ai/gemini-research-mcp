"""
Citation extraction and URL resolution for Deep Research reports.

Deep Research reports include citations in a **Sources:** section with
vertexaisearch redirect URLs. This module extracts and resolves them.
"""

from __future__ import annotations

import re

import httpx

from gemini_research_mcp.types import DeepResearchResult, ParsedCitation


async def resolve_redirect_url(
    redirect_url: str,
    timeout: float = 5.0,
) -> tuple[str | None, str | None]:
    """
    Follow a vertexaisearch redirect URL to get the real destination URL and page title.

    Args:
        redirect_url: The vertexaisearch redirect URL
        timeout: Request timeout in seconds

    Returns:
        Tuple of (resolved_url, page_title)
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            response = await client.get(redirect_url)
            resolved_url = str(response.url) if str(response.url) != redirect_url else None

            # Try to extract title from HTML
            title = None
            if resolved_url:
                try:
                    content = response.text[:32768]  # First 32KB should contain title
                    title_match = re.search(r"<title[^>]*>([^<]+)</title>", content, re.IGNORECASE)
                    if title_match:
                        title = title_match.group(1).strip()
                        # Clean up common HTML entities
                        for old, new in [
                            ("&amp;", "&"),
                            ("&lt;", "<"),
                            ("&gt;", ">"),
                            ("&#39;", "'"),
                            ("&quot;", '"'),
                            ("&#x27;", "'"),
                            ("&nbsp;", " "),
                        ]:
                            title = title.replace(old, new)
                except Exception:
                    pass

            return resolved_url, title

    except httpx.RequestError:
        return None, None


def is_blocked_page_title(title: str | None) -> bool:
    """Check if a page title indicates a blocked/error page (Cloudflare, etc.)."""
    if not title:
        return True
    blocked_indicators = [
        "attention required",
        "cloudflare",
        "access denied",
        "just a moment",
        "checking your browser",
        "security check",
        "403 forbidden",
        "404 not found",
        "not acceptable",
    ]
    title_lower = title.lower()
    return any(indicator in title_lower for indicator in blocked_indicators)


def extract_citations_from_text(text: str) -> tuple[str, list[ParsedCitation]]:
    """
    Extract citations from report text and return cleaned text + parsed citations.

    Deep Research reports typically end with a **Sources:** section containing
    numbered citations in markdown format.

    Returns:
        Tuple of (text_without_sources, list of ParsedCitation objects)
    """
    if not text:
        return text, []

    # Find the **Sources:** section (case-insensitive, handles markdown bold)
    sources_patterns = [
        r"\n\*\*Sources:\*\*\s*\n",  # **Sources:**
        r"\n## Sources\s*\n",  # ## Sources
        r"\n### Sources\s*\n",  # ### Sources
        r"\nSources:\s*\n",  # Sources:
    ]

    sources_start = -1
    for pattern in sources_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            sources_start = match.start()
            break

    if sources_start == -1:
        return text, []

    # Extract the sources section
    sources_section = text[sources_start:]
    text_without_sources = text[:sources_start].rstrip()

    # Parse individual citations: [number]. [domain](url)
    citation_pattern = r"(\d+)\.\s*\[([^\]]+)\]\(([^)]+)\)"
    matches = re.findall(citation_pattern, sources_section)

    parsed_citations = []
    for num_str, domain, url in matches:
        parsed_citations.append(
            ParsedCitation(
                number=int(num_str),
                domain=domain.strip(),
                redirect_url=url.strip(),
                url=None,
            )
        )

    return text_without_sources, parsed_citations


async def resolve_citation_urls(
    citations: list[ParsedCitation],
    timeout: float = 5.0,
) -> list[ParsedCitation]:
    """Resolve all redirect URLs in citations to get real destination URLs and page titles."""
    for citation in citations:
        if citation.redirect_url and "vertexaisearch" in citation.redirect_url:
            url, title = await resolve_redirect_url(citation.redirect_url, timeout)
            citation.url = url
            citation.title = citation.domain if is_blocked_page_title(title) else title

            if not citation.url:
                citation.url = f"https://{citation.domain}"
    return citations


def rebuild_sources_section(citations: list[ParsedCitation]) -> str:
    """Rebuild the sources section with resolved URLs."""
    if not citations:
        return ""

    lines = ["\n\n**Sources:**"]
    for cit in sorted(citations, key=lambda c: c.number):
        url = cit.url or cit.redirect_url or f"https://{cit.domain}"
        lines.append(f"{cit.number}. [{cit.domain}]({url})")

    return "\n".join(lines)


async def process_citations(
    result: DeepResearchResult,
    resolve_urls: bool = True,
    timeout: float = 5.0,
) -> DeepResearchResult:
    """
    Post-process a DeepResearchResult to extract and resolve citations.

    This function:
    1. Extracts citations from the **Sources:** section of the report text
    2. Optionally resolves vertexaisearch redirect URLs to real URLs
    3. Stores parsed citations in result.parsed_citations
    4. Stores the report text without sources in result.text_without_sources

    Args:
        result: The DeepResearchResult to process
        resolve_urls: Whether to follow redirects to get real URLs
        timeout: Timeout for each URL resolution request

    Returns:
        The same result object with parsed_citations and text_without_sources populated
    """
    if not result.text:
        return result

    # Step 1: Extract citations from text
    text_without_sources, parsed_citations = extract_citations_from_text(result.text)
    result.text_without_sources = text_without_sources

    # Step 2: Resolve redirect URLs if requested
    if resolve_urls and parsed_citations:
        parsed_citations = await resolve_citation_urls(parsed_citations, timeout)

    result.parsed_citations = parsed_citations

    # Step 3: Rebuild text with resolved URLs
    if parsed_citations:
        result.text = text_without_sources + rebuild_sources_section(parsed_citations)

    return result
