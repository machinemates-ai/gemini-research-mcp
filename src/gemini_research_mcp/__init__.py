"""
Deep Research MCP Server

AI-powered research using Gemini grounding with MCP Tasks support.
"""

__version__ = "2.3.0"

from gemini_research_mcp.types import (
    DeepResearchError,
    DeepResearchProgress,
    DeepResearchResult,
    DeepResearchUsage,
    ParsedCitation,
    ResearchResult,
    Source,
    VendorDocsResult,
)
from gemini_research_mcp.quick import quick_research
from gemini_research_mcp.deep import (
    deep_research,
    deep_research_stream,
    get_research_status,
    research_followup,
)
from gemini_research_mcp.citations import process_citations
from gemini_research_mcp.server import main, mcp

__all__ = [
    "__version__",
    "DeepResearchError",
    "DeepResearchProgress",
    "DeepResearchResult",
    "DeepResearchUsage",
    "ParsedCitation",
    "ResearchResult",
    "Source",
    "VendorDocsResult",
    "quick_research",
    "deep_research",
    "deep_research_stream",
    "get_research_status",
    "research_followup",
    "process_citations",
    "main",
    "mcp",
]
