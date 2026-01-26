"""
Configuration management for Deep Research MCP Server.

All configuration is loaded from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from datetime import date

# =============================================================================
# Logging
# =============================================================================

LOGGER_NAME = "gemini-research-mcp"


# =============================================================================
# Model Configuration
# =============================================================================

# Default models - can be overridden via environment
DEFAULT_MODEL = "gemini-3-flash-preview"
# Interactions Deep Research agent name (preview; override via DEEP_RESEARCH_AGENT if needed)
DEFAULT_DEEP_RESEARCH_AGENT = "deep-research-pro-preview-12-2025"

# Thinking level for Gemini 3 models
# Values: "minimal", "low", "medium", "high"
# - minimal: minimize latency for chat/high-throughput
# - low: balance speed and quality
# - medium: good reasoning depth
# - high: maximum reasoning depth (recommended for research)
DEFAULT_THINKING_LEVEL = "high"


# =============================================================================
# Timeouts and Polling
# =============================================================================

# Deep Research configuration
STREAM_POLL_INTERVAL = 10.0  # seconds between polls
MAX_POLL_TIME = 3600.0  # 60 minutes max wait
DEFAULT_TIMEOUT = 3600.0  # 60 minutes default timeout
RECONNECT_DELAY = 2.0  # Initial delay before reconnection
MAX_INITIAL_RETRIES = 3  # Retries for initial stream creation

# Errors that should trigger reconnection
RETRYABLE_ERRORS = [
    "gateway_timeout",
    "deadline_expired",
    "timeout",
    "connection_reset",
    "closed",
    "aborted",
    "internal_error",
    "service_unavailable",
]


# =============================================================================
# Getters
# =============================================================================


def get_api_key() -> str:
    """Get Gemini API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    return api_key


def get_model() -> str:
    """Get model name with env override support."""
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def get_deep_research_agent() -> str:
    """Get Deep Research agent name with env override support."""
    return os.environ.get("DEEP_RESEARCH_AGENT", DEFAULT_DEEP_RESEARCH_AGENT)


def is_retryable_error(error_msg: str) -> bool:
    """Check if an error message indicates a retryable condition."""
    error_lower = str(error_msg).lower()
    return any(err in error_lower for err in RETRYABLE_ERRORS)


def default_system_prompt() -> str:
    """Default system prompt for research tasks."""
    today = date.today().strftime("%B %d, %Y")
    return f"""You are an expert research analyst. Today is {today}.

When answering questions:
1. Provide accurate, well-researched information grounded in your search results
2. Cite your sources by referring to the specific information from each source
3. If information is uncertain or sources conflict, acknowledge this
4. Structure complex answers with clear headings and bullet points
5. Prioritize recent and authoritative sources

Your goal is to provide comprehensive, factual answers that would satisfy
a professional researcher."""
