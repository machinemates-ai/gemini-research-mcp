"""
Configuration management for Deep Research MCP Server.

All configuration is loaded from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date


# =============================================================================
# Model Configuration
# =============================================================================

# Default models - can be overridden via environment
DEFAULT_MODEL = "gemini-2.5-flash"
# Interactions Deep Research agent name (preview; override via DEEP_RESEARCH_AGENT if needed)
DEFAULT_DEEP_RESEARCH_AGENT = "deep-research-pro-preview-12-2025"

# Thinking budget mapping (token counts for gemini-2.5-flash: 0-24576)
THINKING_BUDGETS = {
    "minimal": 0,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "max": 24576,
    "dynamic": -1,
}
DEFAULT_THINKING_BUDGET = 8192


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


def get_thinking_budget(level_or_budget: str | int) -> int:
    """Convert thinking level name or direct budget to token count."""
    if isinstance(level_or_budget, int):
        return level_or_budget
    return THINKING_BUDGETS.get(level_or_budget, DEFAULT_THINKING_BUDGET)


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


# =============================================================================
# ExternalApi Configuration (for Vendor Docs)
# =============================================================================


@dataclass(slots=True)
class ExternalApiConfig:
    """Configuration for ExternalApi grounding endpoint."""

    endpoint: str
    api_key: str
    vendor: str | None = None


def get_external_api_config() -> ExternalApiConfig | None:
    """Get ExternalApi configuration from environment."""
    endpoint = os.environ.get("EXTERNAL_API_ENDPOINT")
    api_key = os.environ.get("EXTERNAL_API_KEY")

    if not endpoint or not api_key:
        return None

    return ExternalApiConfig(
        endpoint=endpoint,
        api_key=api_key,
        vendor=os.environ.get("EXTERNAL_API_VENDOR"),
    )
