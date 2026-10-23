"""Optional Local Deep Research bridge (blocking stdio MCP)."""

from __future__ import annotations

from gemini_research_mcp.config import LDR_COMMAND_ENV_VAR, get_ldr_command
from gemini_research_mcp.engines.base import (
    ProgressCallback,
    ProviderIdCallback,
    SubmittingCallback,
)
from gemini_research_mcp.types import (
    BACKEND_LOCAL,
    EngineDescriptor,
    EngineError,
    EngineReport,
    EngineRequest,
    EngineSnapshot,
)

LOCAL_ENGINE_ID = "local_deep_research"

_LIMITATIONS = (
    "Blocking call with no provider ID: not recoverable after a restart.",
    "Cancellation is best-effort and local; there is no provider confirmation.",
    "Requires operator-installed Local Deep Research, Ollama on loopback and explicit search.",
)


def describe_local() -> EngineDescriptor:
    """Descriptor for the LDR bridge; never launches the command."""
    reason: str | None = None
    if get_ldr_command() is None:
        reason = f"{LDR_COMMAND_ENV_VAR} must be set to an absolute executable path."
    return EngineDescriptor(
        id=LOCAL_ENGINE_ID,
        label="Local Deep Research",
        backend=BACKEND_LOCAL,
        execution_mode="blocking_stdio",
        recovery="none",
        model=None,
        available=reason is None,
        availability_reason=reason,
        limitations=_LIMITATIONS,
    )


class LocalDeepResearchEngine:
    """Request-owned LDR adapter. The stdio bridge is implemented in Task 4."""

    def __init__(self) -> None:
        self.descriptor = describe_local()

    async def preflight(self) -> None:
        if not self.descriptor.available:
            raise EngineError(
                "ENGINE_UNAVAILABLE",
                self.descriptor.availability_reason or "Local engine unavailable.",
            )

    async def run(
        self,
        request: EngineRequest,
        *,
        on_submitting: SubmittingCallback,
        on_provider_id: ProviderIdCallback,
        on_progress: ProgressCallback,
    ) -> EngineReport:
        raise EngineError("ENGINE_NOT_IMPLEMENTED", "LDR bridge lands in Task 4.")

    async def refresh(self, provider_id: str) -> EngineSnapshot:
        raise EngineError("LOCAL_REFRESH_UNSUPPORTED", "Local runs cannot be refreshed.")

    async def aclose(self) -> None:
        return None
