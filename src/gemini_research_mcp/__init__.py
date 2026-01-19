"""Gemini Research MCP Server

AI-powered research using Gemini:
- research_web: Fast grounded search (Gemini + Google Search)
- research_deep: Comprehensive research (Deep Research Agent, requires MCP Tasks)
- research_followup: Continue conversation after research completes
"""

try:
    from importlib.metadata import PackageNotFoundError, version
except (ImportError, ModuleNotFoundError):
    __version__ = "0.0.0+unknown"
else:
    try:
        __version__ = version("gemini-research-mcp")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

from gemini_research_mcp.citations import process_citations
from gemini_research_mcp.deep import (
    deep_research,
    deep_research_stream,
    research_followup,
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
    "main",
    "mcp",
    "process_citations",
    "quick_research",
    "research_followup",
]
