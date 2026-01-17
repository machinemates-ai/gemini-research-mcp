"""Gemini Research MCP Server

AI-powered research using Gemini:
- research_web: Fast grounded search (Gemini + Google Search)
- research_deep: Comprehensive research (Deep Research Agent, requires MCP Tasks)
- start_research / check_research: Async research pattern (non-blocking)
- research_followup: Continue conversation after research completes
"""

__version__ = "0.1.0"

from gemini_research_mcp.citations import process_citations
from gemini_research_mcp.deep import (
    deep_research,
    deep_research_stream,
    get_research_status,
    research_followup,
    start_research_async,
)
from gemini_research_mcp.quick import quick_research
from gemini_research_mcp.server import main, mcp
from gemini_research_mcp.types import (
    DeepResearchError,
    DeepResearchProgress,
    DeepResearchResult,
    DeepResearchUsage,
    ErrorCategory,
    ParsedCitation,
    ResearchResult,
    Source,
)

__all__ = [
    "__version__",
    "DeepResearchError",
    "DeepResearchProgress",
    "DeepResearchResult",
    "DeepResearchUsage",
    "ErrorCategory",
    "ParsedCitation",
    "ResearchResult",
    "Source",
    "deep_research",
    "deep_research_stream",
    "get_research_status",
    "main",
    "mcp",
    "process_citations",
    "quick_research",
    "research_followup",
    "start_research_async",
]
