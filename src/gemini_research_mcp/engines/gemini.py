"""Gemini-native Deep Research engine (Interactions API, background mode)."""

from __future__ import annotations

from gemini_research_mcp.config import get_api_key, get_deep_research_agent
from gemini_research_mcp.engines.base import (
    ProgressCallback,
    ProviderIdCallback,
    SubmittingCallback,
)
from gemini_research_mcp.types import (
    BACKEND_GEMINI,
    DeepResearchAgent,
    EngineDescriptor,
    EngineError,
    EngineReport,
    EngineRequest,
    EngineSnapshot,
)

_LIMITATIONS = (
    "Provider-side job: recoverable by provider interaction ID.",
    "Cancellation is only reported once the provider confirms it.",
)


def describe_gemini(engine_id: str) -> EngineDescriptor:
    """Descriptor for a Gemini selector; never creates a client or calls out."""
    reason: str | None = None
    model: str | None
    try:
        if engine_id == "gemini_max":
            model = DeepResearchAgent.DEEP_RESEARCH_MAX.value
        else:
            model = get_deep_research_agent().value
    except ValueError as exc:
        model = None
        reason = str(exc)
    if reason is None:
        try:
            get_api_key()
        except ValueError:
            reason = "GEMINI_API_KEY is not set."
    return EngineDescriptor(
        id=engine_id,
        label="Gemini Deep Research Max" if engine_id == "gemini_max" else "Gemini Deep Research",
        backend=BACKEND_GEMINI,
        execution_mode="background_interactions",
        recovery="provider_id",
        model=model,
        available=reason is None,
        availability_reason=reason,
        limitations=_LIMITATIONS,
    )


class GeminiEngine:
    """Request-owned Gemini adapter. Submission is implemented in Task 3."""

    def __init__(self, engine_id: str) -> None:
        self.engine_id = engine_id
        self.descriptor = describe_gemini(engine_id)

    async def preflight(self) -> None:
        if not self.descriptor.available:
            raise EngineError(
                "ENGINE_UNAVAILABLE",
                self.descriptor.availability_reason or "Gemini engine unavailable.",
            )

    async def run(
        self,
        request: EngineRequest,
        *,
        on_submitting: SubmittingCallback,
        on_provider_id: ProviderIdCallback,
        on_progress: ProgressCallback,
    ) -> EngineReport:
        raise EngineError("ENGINE_NOT_IMPLEMENTED", "Gemini submission lands in Task 3.")

    async def refresh(self, provider_id: str) -> EngineSnapshot:
        raise EngineError("ENGINE_NOT_IMPLEMENTED", "Gemini refresh lands in Task 3.")

    async def aclose(self) -> None:
        return None
