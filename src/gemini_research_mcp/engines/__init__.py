"""Research engine registry. Discovery is side-effect free."""

from __future__ import annotations

from gemini_research_mcp.engines.base import (
    ENGINE_IDS,
    ProgressCallback,
    ProviderIdCallback,
    ResearchEngine,
    SubmittingCallback,
    current_execution_identity,
    current_owner_scope,
    request_fingerprint,
    stable_job_id,
    validate_comparison_engines,
    validate_query,
)
from gemini_research_mcp.engines.gemini import GeminiEngine, describe_gemini
from gemini_research_mcp.engines.local import (
    LOCAL_ENGINE_ID,
    LocalDeepResearchEngine,
    describe_local,
)
from gemini_research_mcp.types import EngineDescriptor

__all__ = [
    "ENGINE_IDS",
    "ProgressCallback",
    "ProviderIdCallback",
    "ResearchEngine",
    "SubmittingCallback",
    "current_execution_identity",
    "current_owner_scope",
    "describe_engines",
    "get_engine",
    "request_fingerprint",
    "stable_job_id",
    "validate_comparison_engines",
    "validate_query",
]


def describe_engines() -> list[EngineDescriptor]:
    """Describe every selector without launching processes or calling providers."""
    return [
        describe_gemini("gemini"),
        describe_gemini("gemini_max"),
        describe_local(),
    ]


def get_engine(engine_id: str) -> ResearchEngine:
    """Build a new request-owned adapter. Unknown IDs fail before construction."""
    if engine_id in ("gemini", "gemini_max"):
        return GeminiEngine(engine_id)
    if engine_id == LOCAL_ENGINE_ID:
        return LocalDeepResearchEngine()
    raise ValueError(f"Unknown research engine: {engine_id!r}")
