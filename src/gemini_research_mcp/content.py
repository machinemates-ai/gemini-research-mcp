"""
Web content extraction with SSRF protection.

Provides fetch_webpage functionality for extracting article content
from URLs as clean Markdown, using trafilatura for high-quality extraction.

Security: Blocks requests to private IPs, localhost, and cloud metadata endpoints.
"""

import ipaddress
import logging
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from gemini_research_mcp.config import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

# =============================================================================
# SSRF Protection
# =============================================================================

# Hosts that are always blocked (case-insensitive)
BLOCKED_HOSTS: frozenset[str] = frozenset([
    "localhost",
    "localhost.localdomain",
    "127.0.0.1",
    "::1",
    "[::1]",
    "0.0.0.0",
    "metadata.google.internal",
    "metadata.goog",
    "169.254.169.254",  # AWS/GCP/Azure metadata
])

# Private IP prefixes to block (SSRF protection)
BLOCKED_PREFIXES: tuple[str, ...] = (
    "10.",
    "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.",
    "172.28.", "172.29.", "172.30.", "172.31.",
    "192.168.",
    "127.",
    "169.254.",  # Link-local / cloud metadata
    "fd",  # IPv6 unique local (fd00::/8)
    "fe80:",  # IPv6 link-local
    "fc",  # IPv6 unique local (fc00::/7)
)


def is_private_ip(host: str) -> bool:
    """Check if a host resolves to a private IP address."""
    # Check blocked hosts first
    if host.lower() in BLOCKED_HOSTS:
        return True

    # Check blocked prefixes
    if any(host.lower().startswith(prefix.lower()) for prefix in BLOCKED_PREFIXES):
        return True

    # Try to resolve and check the IP
    try:
        # Get all IPs for the host
        addrs = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for addr_info in addrs:
            ip_str = addr_info[4][0]
            try:
                ip = ipaddress.ip_address(ip_str)
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                    return True
            except ValueError:
                continue
    except socket.gaierror:
        # DNS resolution failed - might be suspicious, but allow for now
        pass

    return False


def validate_url(url: str) -> tuple[bool, str]:
    """
    Validate a URL for SSRF safety.

    Returns (is_valid, error_message).
    """
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Invalid URL: {e}"

    # Must have scheme and host
    if not parsed.scheme or not parsed.netloc:
        return False, "URL must have scheme (http/https) and host"

    # Only allow http/https
    if parsed.scheme.lower() not in ("http", "https"):
        return False, f"Unsupported scheme: {parsed.scheme}. Only http/https allowed."

    # Extract hostname
    hostname = parsed.hostname
    if not hostname:
        return False, "URL must have a valid hostname"

    # Check for private IPs
    if is_private_ip(hostname):
        return False, f"SSRF blocked: {hostname} resolves to private/internal address"

    return True, ""


# =============================================================================
# Content Extraction
# =============================================================================

# HTTP client configuration
DEFAULT_TIMEOUT = 15.0
MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10MB max

# User agent for web requests
USER_AGENT = (
    "Mozilla/5.0 (compatible; GeminiResearchBot/1.0; +https://github.com/machinemates-ai/gemini-research-mcp)"
)


@dataclass
class FetchResult:
    """Result of fetching webpage content."""

    url: str
    title: str | None
    content: str
    word_count: int
    error: str | None = None


async def fetch_webpage(url: str) -> FetchResult:
    """
    Fetch and extract content from a webpage as Markdown.

    Uses trafilatura for high-quality content extraction (F1: 0.937).
    Falls back to basic HTML parsing if trafilatura is unavailable.

    Args:
        url: The URL to fetch

    Returns:
        FetchResult with extracted content or error message
    """
    logger.info("📥 Fetching: %s", url)

    # Validate URL for SSRF
    is_valid, error_msg = validate_url(url)
    if not is_valid:
        logger.warning("   ❌ SSRF blocked: %s", error_msg)
        return FetchResult(
            url=url,
            title=None,
            content="",
            word_count=0,
            error=error_msg,
        )

    # Fetch the page
    try:
        async with httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Check content length
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > MAX_RESPONSE_SIZE:
                size_mb = int(content_length) / 1024 / 1024
                return FetchResult(
                    url=url,
                    title=None,
                    content="",
                    word_count=0,
                    error=f"Content too large: {size_mb:.1f}MB (max: 10MB)",
                )

            html = response.text

    except httpx.TimeoutException:
        logger.warning("   ❌ Timeout fetching: %s", url)
        return FetchResult(
            url=url,
            title=None,
            content="",
            word_count=0,
            error=f"Timeout after {DEFAULT_TIMEOUT}s",
        )
    except httpx.HTTPStatusError as e:
        logger.warning("   ❌ HTTP error: %s", e)
        return FetchResult(
            url=url,
            title=None,
            content="",
            word_count=0,
            error=f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
        )
    except Exception as e:
        logger.exception("   ❌ Failed to fetch: %s", e)
        return FetchResult(
            url=url,
            title=None,
            content="",
            word_count=0,
            error=str(e),
        )

    # Extract content using trafilatura (preferred)
    try:
        import trafilatura
        from trafilatura.settings import DEFAULT_CONFIG

        # Configure for Markdown output
        extracted = trafilatura.extract(
            html,
            url=url,
            output_format="markdown",
            include_links=True,
            include_images=False,
            include_tables=True,
            include_comments=False,
            config=DEFAULT_CONFIG,
        )

        if extracted:
            # Extract title separately
            metadata = trafilatura.extract_metadata(html)
            title = metadata.title if metadata else None

            word_count = len(extracted.split())
            logger.info("   ✅ Extracted %d words via trafilatura", word_count)

            return FetchResult(
                url=url,
                title=title,
                content=extracted,
                word_count=word_count,
            )
        else:
            logger.warning("   ⚠️ trafilatura returned empty, falling back")

    except ImportError:
        logger.info("   ℹ️ trafilatura not installed, using basic extraction")
    except Exception as e:
        logger.warning("   ⚠️ trafilatura failed: %s, falling back", e)

    # Fallback: Basic HTML to text using built-in html.parser
    try:
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text_parts: list[str] = []
                self.title: str | None = None
                self._in_title = False
                self._skip_tags = {"script", "style", "nav", "header", "footer", "aside"}
                self._skip_depth = 0

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                if tag == "title":
                    self._in_title = True
                elif tag in self._skip_tags:
                    self._skip_depth += 1

            def handle_endtag(self, tag: str) -> None:
                if tag == "title":
                    self._in_title = False
                elif tag in self._skip_tags and self._skip_depth > 0:
                    self._skip_depth -= 1
                elif tag in ("p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li"):
                    self.text_parts.append("\n")

            def handle_data(self, data: str) -> None:
                text = data.strip()
                if not text:
                    return
                if self._in_title:
                    self.title = text
                elif self._skip_depth == 0:
                    self.text_parts.append(text + " ")

        parser = TextExtractor()
        parser.feed(html)
        content = "".join(parser.text_parts).strip()
        # Clean up multiple newlines
        while "\n\n\n" in content:
            content = content.replace("\n\n\n", "\n\n")

        word_count = len(content.split())
        logger.info("   ✅ Basic extraction: %d words", word_count)

        return FetchResult(
            url=url,
            title=parser.title,
            content=content,
            word_count=word_count,
        )

    except Exception as e:
        logger.exception("   ❌ Fallback extraction failed: %s", e)
        return FetchResult(
            url=url,
            title=None,
            content="",
            word_count=0,
            error=f"Content extraction failed: {e}",
        )


__all__ = ["fetch_webpage", "FetchResult", "validate_url", "is_private_ip"]
