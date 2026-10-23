"""Controllable fake engines shared by the research coordinator tests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from gemini_research_mcp.engines.base import (
    ProgressCallback,
    ProviderIdCallback,
    SubmittingCallback,
)
from gemini_research_mcp.types import (
    BACKEND_GEMINI,
    BACKEND_LOCAL,
    EngineDescriptor,
    EngineError,
    EngineReport,
    EngineRequest,
    EngineSnapshot,
    JobStatus,
)

LOCAL_ID = "local_deep_research"


def report(text: str = "# Report\nbody", **kwargs: Any) -> EngineReport:
    return EngineReport(text=text, **kwargs)


class FakeEngine:
    """Counts creates and lets a test gate, fail or inspect each phase."""

    def __init__(
        self,
        engine_id: str = "gemini",
        *,
        outcome: EngineReport | BaseException | None = None,
        provider_id: str | None = None,
        link: bool = True,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.engine_id = engine_id
        local = engine_id == LOCAL_ID
        self.local = local
        self.descriptor = EngineDescriptor(
            id=engine_id,
            label=engine_id,
            backend=BACKEND_LOCAL if local else BACKEND_GEMINI,
            execution_mode="blocking_stdio" if local else "background_interactions",
            recovery="none" if local else "provider_id",
            model="ldr-model" if local else "deep-research-preview-04-2026",
            available=True,
        )
        self.outcome: EngineReport | BaseException = outcome or report()
        self.provider_id = provider_id or f"int-{engine_id}"
        self.link = link and not local
        self.gate = gate
        self.started = asyncio.Event()
        self.creates = 0
        self.refresh_calls = 0
        self.closed = 0
        self.preflight_error: BaseException | None = None
        self.reserve_error: BaseException | None = None
        self.snapshot: EngineSnapshot | BaseException | None = None
        self.before_submit: Callable[[], Awaitable[None]] | None = None
        self.on_create: Callable[[], Awaitable[None]] | None = None
        self.events: list[str] = []
        self.log: list[str] | None = None

    def _note(self, event: str) -> None:
        self.events.append(event)
        if self.log is not None:
            self.log.append(f"{self.engine_id}:{event}")

    def reserve(self) -> None:
        self._note("reserve")
        if self.reserve_error is not None:
            raise self.reserve_error

    async def preflight(self) -> None:
        self._note("preflight")
        if self.preflight_error is not None:
            raise self.preflight_error

    async def run(
        self,
        request: EngineRequest,
        *,
        on_submitting: SubmittingCallback,
        on_provider_id: ProviderIdCallback,
        on_progress: ProgressCallback,
    ) -> EngineReport:
        if self.before_submit is not None:
            await self.before_submit()
        await on_submitting()
        self.creates += 1
        self._note("create")
        if self.on_create is not None:
            await self.on_create()
        if self.link:
            await on_provider_id(self.provider_id)
        await on_progress("Gemini research started.")
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    async def refresh(self, provider_id: str) -> EngineSnapshot:
        self.refresh_calls += 1
        if isinstance(self.snapshot, BaseException):
            raise self.snapshot
        if self.snapshot is None:
            return EngineSnapshot(provider_id=provider_id, status=JobStatus.RUNNING)
        return self.snapshot

    async def aclose(self) -> None:
        self.closed += 1


def boom(code: str) -> EngineError:
    return EngineError(code, f"{code} happened")


class Engines:
    """Factory returning one shared fake per selector."""

    def __init__(self, *engines: FakeEngine) -> None:
        self.by_id = {e.engine_id: e for e in engines}

    def __call__(self, engine_id: str) -> FakeEngine:
        if engine_id not in self.by_id:
            self.by_id[engine_id] = FakeEngine(engine_id)
        return self.by_id[engine_id]

    @property
    def creates(self) -> int:
        return sum(e.creates for e in self.by_id.values())
