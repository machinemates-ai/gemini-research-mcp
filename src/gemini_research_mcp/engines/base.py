"""Engine protocol, identity helpers and validation for research jobs."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from fastmcp_tasks.context import get_task_context, get_task_scope

from gemini_research_mcp.types import (
    EngineDescriptor,
    EngineError,
    EngineReport,
    EngineRequest,
    EngineSnapshot,
    ExecutionIdentity,
)

ProgressCallback = Callable[[str], Awaitable[None]]
SubmittingCallback = Callable[[], Awaitable[None]]
ProviderIdCallback = Callable[[str], Awaitable[None]]

ENGINE_IDS: tuple[str, ...] = ("gemini", "gemini_max", "local_deep_research")


class ResearchEngine(Protocol):
    """Request-owned adapter for one engine selector."""

    descriptor: EngineDescriptor

    async def preflight(self) -> None: ...

    async def run(
        self,
        request: EngineRequest,
        *,
        on_submitting: SubmittingCallback,
        on_provider_id: ProviderIdCallback,
        on_progress: ProgressCallback,
    ) -> EngineReport: ...

    async def refresh(self, provider_id: str) -> EngineSnapshot: ...

    async def aclose(self) -> None: ...


def validate_query(query: str) -> str:
    """Reject blank queries. The accepted query is returned unchanged."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be blank")
    return query


def validate_comparison_engines(engine_ids: Sequence[str]) -> tuple[str, str]:
    """Require exactly two distinct known engines, preserving caller order."""
    ids = list(engine_ids)
    if len(ids) != 2:
        raise ValueError("a comparison needs exactly two engines")
    unknown = [e for e in ids if e not in ENGINE_IDS]
    if unknown:
        raise ValueError(f"Unknown research engine: {unknown[0]!r}")
    if ids[0] == ids[1]:
        raise ValueError("a comparison needs two distinct engines")
    return ids[0], ids[1]


def _canonical(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


_JOB_NAMESPACE = uuid.UUID("5d0b6a58-6b6f-4e1c-9f3d-2f6f0d2a7c11")


def stable_job_id(scope: str | None, task_id: str, slot: str) -> str:
    """Deterministic ID for one (owner scope, Task, slot).

    Task IDs are server-generated and opaque, so a replay of the same Task maps
    to the same record while a new Task never reuses an earlier job.
    """
    digest = uuid.uuid5(_JOB_NAMESPACE, _canonical([scope, task_id, slot]))
    prefix = "cmp" if slot == "comparison" else "job"
    return f"{prefix}_{digest.hex}"


def request_fingerprint(query: str, engine_ids: Sequence[str]) -> str:
    """Hash of the request shape; never contains the query text."""
    return hashlib.sha256(_canonical([query, list(engine_ids)]).encode()).hexdigest()


def current_owner_scope() -> str | None:
    """Authenticated principal scope, resolved exactly as Task creation does."""
    return get_task_scope()


def current_execution_identity() -> ExecutionIdentity:
    """Identity of the running Task; fails closed outside a Task."""
    context = get_task_context()
    if context is None:
        raise EngineError("TASK_REQUIRED", "Research jobs must run inside a background Task.")
    redelivered = False
    try:
        from docket.dependencies import current_execution

        redelivered = bool(getattr(current_execution.get(), "redelivered", False))
    except LookupError:
        pass
    return ExecutionIdentity(
        task_id=context.task_id,
        task_scope=context.task_scope,
        redelivered=redelivered,
    )
