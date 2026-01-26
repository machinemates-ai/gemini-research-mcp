"""
Data types for Gemini Research MCP Server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Exceptions
# =============================================================================


class DeepResearchError(Exception):
    """Base error for Deep Research operations.

    Provides structured error information with error codes for programmatic handling.
    """

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"{code}: {message}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# Source Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class Source:
    """A source/citation from grounded search."""

    uri: str
    title: str


@dataclass(slots=True)
class ParsedCitation:
    """A citation extracted from the report with resolved URL.

    Deep Research reports include citations with vertexaisearch redirect URLs.
    This class stores both the original redirect and the resolved real URL.
    """

    number: int
    domain: str
    url: str | None = None
    title: str | None = None
    redirect_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "number": self.number,
            "domain": self.domain,
            "url": self.url,
            "title": self.title,
            "redirect_url": self.redirect_url,
        }


# =============================================================================
# Usage Tracking
# =============================================================================


@dataclass(slots=True)
class DeepResearchUsage:
    """Token usage and cost information for a Deep Research task."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_cost: float | None = None
    completion_cost: float | None = None
    total_cost: float | None = None
    raw_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }


# =============================================================================
# Research Results
# =============================================================================


@dataclass(slots=True)
class ResearchResult:
    """Result from quick_research()."""

    text: str
    sources: list[Source] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    thinking_summary: str | None = None


@dataclass(slots=True)
class DeepResearchResult:
    """Result from deep_research()."""

    text: str
    text_without_sources: str | None = None
    citations: list[Source] = field(default_factory=list)
    parsed_citations: list[ParsedCitation] = field(default_factory=list)
    thinking_summaries: list[str] = field(default_factory=list)
    interaction_id: str | None = None
    usage: DeepResearchUsage | None = None
    duration_seconds: float | None = None
    raw_interaction: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.interaction_id,
            "text": self.text,
            "text_without_sources": self.text_without_sources,
            "citations": [c.to_dict() for c in self.parsed_citations] if self.parsed_citations else [],
            "thinking_summaries": self.thinking_summaries,
            "usage": self.usage.to_dict() if self.usage else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(slots=True)
class DeepResearchProgress:
    """Progress update from streaming deep research."""

    event_type: str  # "start", "thought", "text", "complete", "error", "status"
    content: str | None = None
    interaction_id: str | None = None
    event_id: str | None = None


# =============================================================================
# Vendor Docs
# =============================================================================


@dataclass(slots=True)
class VendorDocsResult:
    """Result from vendor_docs_external_api_async."""

    text: str
    sources: list[Source] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)


# =============================================================================
# File Search Store (RAG)
# =============================================================================


@dataclass(frozen=True, slots=True)
class FileSearchStore:
    """A file search store for RAG."""

    name: str
    display_name: str | None = None


@dataclass(frozen=True, slots=True)
class FileSearchDocument:
    """A document in a file search store."""

    name: str
    display_name: str | None = None
    file_name: str | None = None
