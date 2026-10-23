"""Single-owner coordination of research jobs and two-child comparisons.

One process owns submission for a data directory (OS lease). It persists a
submission-attempt marker before any create or blocking research call, links
the real provider ID only after the canonical session exists, and never
creates again for a job it has already attempted. State read back through
``ResearchJobStorage`` is authoritative; the in-memory views returned here are
built to match what a later ``inspect`` resolves.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from gemini_research_mcp.config import DEFAULT_TIMEOUT, get_tasks_backend_url
from gemini_research_mcp.engines import ENGINE_IDS, ResearchEngine, get_engine
from gemini_research_mcp.engines.base import (
    ProgressCallback,
    request_fingerprint,
    stable_job_id,
    validate_comparison_engines,
    validate_query,
)
from gemini_research_mcp.storage import (
    ResearchJobStorage,
    ResearchSession,
    ResearchStatus,
    derive_group_status,
    get_job_storage,
    get_storage_backend_url,
    get_ttl_seconds,
)
from gemini_research_mcp.types import (
    BACKEND_LOCAL,
    DeepResearchAgent,
    EngineError,
    EngineReport,
    EngineRequest,
    EngineSnapshot,
    ExecutionIdentity,
    JobStatus,
    ResearchComparisonRecord,
    ResearchJobRecord,
    ResearchRunSummary,
    ResolvedResearchComparison,
    ResolvedResearchJob,
)

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

logger = logging.getLogger(__name__)

CLEANUP_MARGIN_SECONDS = 900.0
SHIELD_TIMEOUT_SECONDS = 10.0
LEASE_FILENAME = ".submission.lock"
QUERY_PREVIEW_CHARS = 80

_PHASE_RE = re.compile(r"[A-Za-z0-9 _.\-]{1,60}")
_INTERRUPTED_CODES = frozenset(
    {
        "SUBMISSION_UNKNOWN",
        "POLL_TIMEOUT",
        "POLL_FAILED",
        "PROVIDER_REQUIRES_ACTION",
        "PROVIDER_LINK_FAILED",
        "OWNER_LOST",
        "PERSISTENCE_FAILED",
    }
)
_CANCELLED_CODES = frozenset({"PROVIDER_CANCELLED", "CANCELLED_BEFORE_DISPATCH"})
_SESSION_STATUS_FOR_JOB = {
    JobStatus.COMPLETED: ResearchStatus.COMPLETED,
    JobStatus.FAILED: ResearchStatus.FAILED,
    JobStatus.CANCELLED: ResearchStatus.CANCELLED,
}


class SubmissionLease:
    """Nonblocking OS lock proving this process is the only submitter for a data dir."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> None:
        if self._fd is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if sys.platform == "win32":
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise EngineError(
                "SUBMISSION_LEASE_HELD",
                "Another process owns research submission for this data directory.",
            ) from None
        self._fd = fd

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        with contextlib.suppress(OSError):
            if sys.platform == "win32":
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)


@dataclass(slots=True)
class _Run:
    """Live ownership of one job (or comparison) by this process."""

    record: ResearchJobRecord | None = None
    alive: bool = True
    attempted: bool = False
    provider_id: str | None = None


PersistImages = Callable[[str, Sequence[dict[str, Any]]], Awaitable[list[str]]]


def _timeout_of(engine: Any) -> float:
    for name in ("timeout", "research_timeout"):
        value = getattr(engine, name, None)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return DEFAULT_TIMEOUT


def _phase(text: str) -> str:
    return text.strip() if _PHASE_RE.fullmatch(text.strip()) else "working"


class ResearchJobService:
    """Run, compare, inspect and list research jobs for one process."""

    def __init__(
        self,
        storage: ResearchJobStorage | None = None,
        *,
        engine_factory: Callable[[str], ResearchEngine] = get_engine,
        persist_images: PersistImages | None = None,
        retention_seconds: Callable[[], float] = lambda: float(get_ttl_seconds()),
    ) -> None:
        self._storage = storage or get_job_storage()
        self._factory = engine_factory
        self._persist_images = persist_images
        self._retention = retention_seconds
        self._lease = SubmissionLease(self._storage._sessions.storage_dir / LEASE_FILENAME)
        self._nonce = uuid.uuid4().hex
        self._pid = os.getpid()
        self._owned: dict[str, _Run] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # -- ownership ---------------------------------------------------------------

    def is_owned(self, job_id: str) -> bool:
        run = self._owned.get(job_id)
        return run is not None and run.alive

    def revoke_ownership(self) -> None:
        """Mark every live run as lost; late engine results are then not persisted."""
        for run in self._owned.values():
            run.alive = False
        self._owned.clear()

    async def close(self) -> None:
        self.revoke_ownership()
        self._lease.release()

    def _claim(self, ids: Sequence[str]) -> dict[str, _Run]:
        if any(i in self._owned for i in ids):
            raise EngineError("JOB_BUSY", "This research request is already running.")
        claimed = {i: _Run() for i in ids}
        self._owned.update(claimed)
        return claimed

    def _release(self, ids: Sequence[str]) -> None:
        for i in ids:
            self._owned.pop(i, None)
            lock = self._locks.get(i)
            if lock is not None and not lock.locked():
                self._locks.pop(i, None)

    def _lock(self, job_id: str) -> asyncio.Lock:
        return self._locks.setdefault(job_id, asyncio.Lock())

    # -- admission ---------------------------------------------------------------

    def _admit(self) -> None:
        docket = get_tasks_backend_url()
        distributed = get_storage_backend_url() is not None or (
            docket is not None and urlsplit(docket).scheme.lower() != "memory"
        )
        if distributed:
            raise EngineError(
                "DISTRIBUTED_SUBMISSION_UNSUPPORTED",
                "Research jobs can only be submitted by a single process with local storage.",
            )
        self._lease.acquire()

    def _check_retention(self, engines: Sequence[Any]) -> None:
        required = max(_timeout_of(e) for e in engines) + CLEANUP_MARGIN_SECONDS
        if self._retention() < required:
            raise EngineError(
                "RETENTION_TOO_SHORT",
                f"Record retention must be at least {int(required)} seconds to cover "
                "execution and cleanup.",
            )

    async def _shielded(self, work: Awaitable[Any]) -> None:
        task = asyncio.ensure_future(asyncio.wait_for(work, SHIELD_TIMEOUT_SECONDS))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            with contextlib.suppress(BaseException):
                await task
            raise
        except Exception:
            logger.warning("Research cleanup step failed", exc_info=True)

    async def _close_engines(self, engines: Sequence[Any]) -> None:
        for engine in engines:
            await self._shielded(engine.aclose())

    @staticmethod
    def _reserve(engine: Any) -> None:
        reserve = getattr(engine, "reserve", None)
        if callable(reserve):
            reserve()

    # -- public API --------------------------------------------------------------

    async def run(
        self,
        query: str,
        engine_id: str,
        identity: ExecutionIdentity,
        *,
        on_progress: ProgressCallback,
    ) -> ResolvedResearchJob:
        validate_query(query)
        if engine_id not in ENGINE_IDS:
            raise ValueError(f"Unknown research engine: {engine_id!r}")
        self._admit()
        scope = identity.task_scope
        job_id = stable_job_id(scope, identity.task_id, "single")
        fingerprint = request_fingerprint(query, [engine_id])

        claimed = self._claim([job_id])
        try:
            existing = await self._storage.get_job(job_id, owner_scope=scope)
            if existing is None and identity.redelivered:
                raise EngineError(
                    "REPLAY_WITHOUT_RECORD",
                    "The record for this redelivered request is gone; it will not be recreated.",
                )
            if existing is None:
                return await self._run_new(
                    claimed[job_id], job_id, query, engine_id, identity, fingerprint, on_progress
                )
        finally:
            self._release([job_id])
        self._check_fingerprint(existing.request_fingerprint, fingerprint)
        return await self._replay_job(existing)

    async def compare(
        self,
        query: str,
        engine_ids: Sequence[str],
        identity: ExecutionIdentity,
        *,
        on_progress: ProgressCallback,
    ) -> ResolvedResearchComparison:
        validate_query(query)
        first, second = validate_comparison_engines(engine_ids)
        self._admit()
        scope = identity.task_scope
        comparison_id = stable_job_id(scope, identity.task_id, "comparison")
        child_ids = (
            stable_job_id(scope, identity.task_id, "child:0"),
            stable_job_id(scope, identity.task_id, "child:1"),
        )
        fingerprint = request_fingerprint(query, [first, second])

        ids = [comparison_id, *child_ids]
        claimed = self._claim(ids)
        try:
            existing = await self._storage.get_comparison(comparison_id, owner_scope=scope)
            if existing is None and identity.redelivered:
                raise EngineError(
                    "REPLAY_WITHOUT_RECORD",
                    "The record for this redelivered request is gone; it will not be recreated.",
                )
            if existing is None:
                return await self._compare_new(
                    claimed,
                    comparison_id,
                    child_ids,
                    (first, second),
                    query,
                    identity,
                    fingerprint,
                    on_progress,
                )
        finally:
            self._release(ids)
        self._check_fingerprint(existing.request_fingerprint, fingerprint)
        return await self._replay_comparison(existing)

    async def inspect(
        self, run_id: str, owner_scope: str | None, *, refresh: bool = False
    ) -> ResolvedResearchJob | ResolvedResearchComparison:
        if run_id.startswith("cmp_"):
            group = await self._storage.get_comparison(run_id, owner_scope=owner_scope)
            if group is None:
                raise self._not_found()
            return await self._resolve_comparison(group, refresh=refresh)
        job = (
            await self._storage.get_job(run_id, owner_scope=owner_scope)
            if run_id.startswith("job_")
            else None
        )
        if job is None:
            raise self._not_found()
        return await self._resolve_job(job, refresh=refresh)

    async def list_runs(
        self, owner_scope: str | None, *, limit: int = 20
    ) -> list[ResearchRunSummary]:
        limit = max(1, min(limit, 100))
        summaries: list[ResearchRunSummary] = []
        for job in await self._storage.list_jobs(limit, owner_scope=owner_scope):
            resolved = await self._storage.resolve_job(job, is_owned=self.is_owned)
            summaries.append(
                ResearchRunSummary(
                    run_id=job.job_id,
                    kind="job",
                    status=resolved.status.value,
                    engines=(job.engine_id,),
                    query_preview=self._preview(job.query),
                    created_at=job.created_at,
                    expires_at=job.expires_at,
                )
            )
        for group in await self._storage.list_comparisons(limit, owner_scope=owner_scope):
            resolved_group = await self._storage.resolve_comparison(group, is_owned=self.is_owned)
            summaries.append(
                ResearchRunSummary(
                    run_id=group.comparison_id,
                    kind="comparison",
                    status=resolved_group.status.value,
                    engines=group.engine_ids,
                    query_preview=self._preview(group.query),
                    created_at=group.created_at,
                    expires_at=group.expires_at,
                )
            )
        summaries.sort(key=lambda s: s.created_at, reverse=True)
        return summaries[:limit]

    async def mark_cancellation_requested(self, job_id: str) -> None:
        """Record cancellation intent; it never changes the provider-confirmed status."""
        run = self._owned.get(job_id)
        async with self._lock(job_id):
            job = run.record if run is not None and run.record is not None else None
            if job is None:
                job = await self._storage._load_job(job_id)
            if job is None:
                return
            job.cancellation_requested = True
            await self._storage.save_job(job)

    # -- helpers -----------------------------------------------------------------

    @staticmethod
    def _not_found() -> EngineError:
        return EngineError("RUN_NOT_FOUND", "No such research run.")

    @staticmethod
    def _preview(query: str) -> str:
        return " ".join(query.split())[:QUERY_PREVIEW_CHARS]

    @staticmethod
    def _check_fingerprint(stored: str, requested: str) -> None:
        if stored != requested:
            raise EngineError(
                "REQUEST_CONFLICT",
                "This Task already ran a different research request.",
            )

    def _new_record(
        self,
        engine: Any,
        *,
        job_id: str,
        query: str,
        fingerprint: str,
        identity: ExecutionIdentity,
        slot: str,
        created_at: float,
        expires_at: float,
        comparison_id: str | None = None,
    ) -> ResearchJobRecord:
        descriptor = engine.descriptor
        return ResearchJobRecord(
            job_id=job_id,
            engine_id=descriptor.id,
            backend=descriptor.backend,
            model=descriptor.model,
            query=query,
            request_fingerprint=fingerprint,
            task_id=identity.task_id,
            owner_scope=identity.task_scope,
            slot=slot,
            created_at=created_at,
            expires_at=expires_at,
            comparison_id=comparison_id,
        )

    # -- new single run ----------------------------------------------------------

    async def _run_new(
        self,
        run: _Run,
        job_id: str,
        query: str,
        engine_id: str,
        identity: ExecutionIdentity,
        fingerprint: str,
        on_progress: ProgressCallback,
    ) -> ResolvedResearchJob:
        engine = self._factory(engine_id)
        try:
            self._check_retention([engine])
            self._reserve(engine)
            await engine.preflight()
            now = time.time()
            record = self._new_record(
                engine,
                job_id=job_id,
                query=query,
                fingerprint=fingerprint,
                identity=identity,
                slot="single",
                created_at=now,
                expires_at=now + self._retention(),
            )
            run.record = record
            await self._storage.save_job(record)
            return await self._execute(run, engine, query, on_progress)
        finally:
            await self._close_engines([engine])

    # -- new comparison ----------------------------------------------------------

    async def _compare_new(
        self,
        claimed: dict[str, _Run],
        comparison_id: str,
        child_ids: tuple[str, str],
        engine_ids: tuple[str, str],
        query: str,
        identity: ExecutionIdentity,
        fingerprint: str,
        on_progress: ProgressCallback,
    ) -> ResolvedResearchComparison:
        engines = [self._factory(engine_ids[0]), self._factory(engine_ids[1])]
        try:
            self._check_retention(engines)
            for engine in engines:
                self._reserve(engine)
            for engine in engines:
                await engine.preflight()

            now = time.time()
            expires_at = now + self._retention()
            records = [
                self._new_record(
                    engine,
                    job_id=child_id,
                    query=query,
                    fingerprint=fingerprint,
                    identity=identity,
                    slot=f"child:{index}",
                    created_at=now,
                    expires_at=expires_at,
                    comparison_id=comparison_id,
                )
                for index, (engine, child_id) in enumerate(zip(engines, child_ids, strict=True))
            ]
            group = ResearchComparisonRecord(
                comparison_id=comparison_id,
                query=query,
                request_fingerprint=fingerprint,
                task_id=identity.task_id,
                owner_scope=identity.task_scope,
                engine_ids=engine_ids,
                child_job_ids=child_ids,
                created_at=now,
                expires_at=expires_at,
            )
            for record in records:
                claimed[record.job_id].record = record
                await self._storage.save_job(record)
            await self._storage.save_comparison(group)

            tasks = [
                asyncio.create_task(
                    self._execute(
                        claimed[child_id],
                        engine,
                        query,
                        on_progress,
                        comparison_id=comparison_id,
                    )
                )
                for engine, child_id in zip(engines, child_ids, strict=True)
            ]
            try:
                first, second = await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            return ResolvedResearchComparison(
                record=group,
                status=derive_group_status(first.status, second.status),
                children=(first, second),
            )
        finally:
            await self._close_engines(engines)

    # -- execution of one child ----------------------------------------------------

    async def _execute(
        self,
        run: _Run,
        engine: Any,
        query: str,
        on_progress: ProgressCallback,
        *,
        comparison_id: str | None = None,
    ) -> ResolvedResearchJob:
        record = run.record
        assert record is not None
        lock = self._lock(record.job_id)

        async def write(**changes: Any) -> None:
            async with lock:
                self._ensure_alive(run)
                for name, value in changes.items():
                    setattr(record, name, value)
                await self._storage.save_job(record)

        async def progress(phase: str) -> None:
            if not run.alive:
                return
            parts = [f"job={record.job_id}"]
            if comparison_id is not None:
                parts.append(f"comparison={comparison_id}")
            parts += [f"status={record.recorded_status.value}", f"phase={_phase(phase)}"]
            try:
                await on_progress(" ".join(parts))
            except Exception:  # noqa: BLE001 - progress must never fail research
                logger.debug("Progress delivery failed", exc_info=True)

        async def on_submitting() -> None:
            self._ensure_alive(run)
            if record.cancellation_requested:
                raise EngineError(
                    "CANCELLED_BEFORE_DISPATCH",
                    "Cancellation was requested before the research was submitted.",
                )
            await write(
                recorded_status=JobStatus.RUNNING,
                attempt_started_at=time.time(),
                owner_nonce=self._nonce,
                owner_pid=self._pid,
            )
            run.attempted = True
            await progress("submitting")

        async def on_provider_id(provider_id: str) -> None:
            run.provider_id = provider_id
            try:
                async with lock:
                    self._ensure_alive(run)
                    await self._save_linked_session(record, provider_id)
                    record.provider_id = provider_id
                    await self._storage.save_job(record)
            except EngineError:
                raise
            except Exception:
                logger.warning("Could not link provider job %s", record.job_id, exc_info=True)
                raise EngineError(
                    "PROVIDER_LINK_FAILED",
                    f"Research started at the provider but could not be linked to the job. "
                    f"Recover it manually with provider ID {provider_id}.",
                ) from None

        async def on_engine_progress(text: str) -> None:
            await progress(text)

        try:
            report = await engine.run(
                EngineRequest(query),
                on_submitting=on_submitting,
                on_provider_id=on_provider_id,
                on_progress=on_engine_progress,
            )
        except asyncio.CancelledError:
            await self._shielded(self._record_cancellation(run))
            raise
        except EngineError as exc:
            return await self._finish_error(run, exc)
        except Exception:
            logger.warning("Research engine failed unexpectedly", exc_info=True)
            return await self._finish_error(
                run, EngineError("ENGINE_ERROR", "The research engine failed unexpectedly.")
            )
        return await self._finish_report(run, report)

    @staticmethod
    def _ensure_alive(run: _Run) -> None:
        if not run.alive:
            raise EngineError("OWNER_LOST", "This process no longer owns the research job.")

    async def _save_linked_session(self, record: ResearchJobRecord, provider_id: str) -> None:
        try:
            agent = DeepResearchAgent(record.model) if record.model else None
        except ValueError:
            agent = None
        await self._storage._sessions.save_session_async(
            ResearchSession(
                interaction_id=provider_id,
                query=record.query,
                created_at=time.time(),
                agent_name=agent,
                status=ResearchStatus.IN_PROGRESS,
                expires_at=record.expires_at,
                job_id=record.job_id,
                owner_scope=record.owner_scope,
            )
        )

    async def _record_cancellation(self, run: _Run) -> None:
        record = run.record
        if record is None or not run.alive:
            return
        record.cancellation_requested = True
        if run.attempted:
            record.recorded_status = JobStatus.INTERRUPTED
            record.error_code = "CANCELLATION_REQUESTED"
            record.error_message = (
                "Cancellation was requested; the provider job may still be running."
            )
        else:
            record.recorded_status = JobStatus.CANCELLED
            record.error_code = "CANCELLED_BEFORE_DISPATCH"
            record.error_message = "Cancelled before anything was submitted."
        await self._storage.save_job(record)

    # -- terminal handling ----------------------------------------------------------

    def _view(
        self,
        run: _Run,
        status: JobStatus,
        *,
        report: EngineReport | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        warnings: tuple[str, ...] = (),
    ) -> ResolvedResearchJob:
        record = run.record
        assert record is not None
        return ResolvedResearchJob(
            record,
            status,
            report=report,
            provider_id=run.provider_id if record.backend != BACKEND_LOCAL else None,
            warnings=warnings,
            error_code=error_code,
            error_message=error_message,
        )

    async def _finish_report(self, run: _Run, report: EngineReport) -> ResolvedResearchJob:
        record = run.record
        assert record is not None
        if not run.alive:
            return self._lost_view(run)
        complete = report.status == "completed"
        status = JobStatus.COMPLETED if complete else JobStatus.FAILED
        code = None if complete else "PROVIDER_INCOMPLETE"
        message = None if complete else "The provider returned an incomplete report."
        warnings = report.warnings
        try:
            async with self._lock(record.job_id):
                self._ensure_alive(run)
                if record.backend != BACKEND_LOCAL and run.provider_id is not None:
                    warnings += await self._write_native_terminal(
                        run.provider_id, record, status, report
                    )
                else:
                    record.report = report
                record.recorded_status = status
                record.error_code = code
                record.error_message = message
                await self._storage.save_job(record)
        except EngineError as exc:
            if exc.code == "OWNER_LOST":
                return self._lost_view(run)
            raise
        except Exception:
            logger.warning("Could not persist terminal state for %s", record.job_id, exc_info=True)
            return self._view(
                run,
                JobStatus.INTERRUPTED,
                report=report,
                error_code="PERSISTENCE_FAILED",
                error_message="The report was produced but could not be stored; "
                "refresh the run to recover it.",
                warnings=warnings,
            )
        return self._view(
            run,
            status,
            report=report,
            error_code=code,
            error_message=message,
            warnings=warnings,
        )

    async def _write_native_terminal(
        self,
        provider_id: str,
        record: ResearchJobRecord,
        status: JobStatus,
        report: EngineReport | None,
        *,
        session_status: ResearchStatus | None = None,
    ) -> tuple[str, ...]:
        """Write the canonical native session. Returns warnings."""
        sessions = self._storage._sessions
        session = await sessions.get_session_async(provider_id)
        if session is None:
            await self._save_linked_session(record, provider_id)
            session = await sessions.get_session_async(provider_id)
        assert session is not None
        warnings: tuple[str, ...] = ()
        session.status = session_status or _SESSION_STATUS_FOR_JOB.get(
            status, ResearchStatus.INTERRUPTED
        )
        if report is not None:
            session.report_text = report.text
            session.sources = [s.to_dict() for s in report.sources]
            session.usage = report.usage.to_dict() if report.usage else None
            session.duration_seconds = report.duration_seconds
            if report.usage is not None:
                session.total_tokens = report.usage.total_tokens
            if report.images and self._persist_images is not None:
                try:
                    session.image_export_ids = await self._persist_images(
                        provider_id, list(report.images)
                    )
                except Exception:  # noqa: BLE001 - images are best effort
                    logger.warning("Could not persist report images", exc_info=True)
                    warnings = ("Report images could not be stored.",)
        await sessions.save_session_async(session)
        return warnings

    def _lost_view(self, run: _Run) -> ResolvedResearchJob:
        return self._view(
            run,
            JobStatus.INTERRUPTED,
            error_code="OWNER_LOST",
            error_message="This process no longer owns the job; refresh to recover the result.",
        )

    async def _finish_error(self, run: _Run, exc: EngineError) -> ResolvedResearchJob:
        record = run.record
        assert record is not None
        if exc.code == "OWNER_LOST" or not run.alive:
            return self._lost_view(run)

        code, message = exc.code, exc.message
        if code == "ENGINE_ERROR" and not run.attempted:
            code, message = "NOT_SUBMITTED", "The job was never submitted."
        if code in _CANCELLED_CODES:
            status = JobStatus.CANCELLED
        elif code in _INTERRUPTED_CODES or (code == "ENGINE_ERROR" and run.attempted):
            status = JobStatus.INTERRUPTED
        else:
            status = JobStatus.FAILED

        warnings: tuple[str, ...] = ()
        try:
            async with self._lock(record.job_id):
                self._ensure_alive(run)
                record.recorded_status = status
                record.error_code = code
                record.error_message = message
                if record.backend != BACKEND_LOCAL and record.provider_id is not None:
                    session_status = {
                        "PROVIDER_REQUIRES_ACTION": ResearchStatus.INTERRUPTED,
                    }.get(code)
                    if status in (JobStatus.FAILED, JobStatus.CANCELLED) or session_status:
                        await self._write_native_terminal(
                            record.provider_id, record, status, None, session_status=session_status
                        )
                await self._storage.save_job(record)
        except EngineError:
            return self._lost_view(run)
        except Exception:
            logger.warning("Could not persist failure for %s", record.job_id, exc_info=True)
            warnings = ("The outcome could not be stored.",)
        return self._view(run, status, error_code=code, error_message=message, warnings=warnings)

    # -- replay and refresh ---------------------------------------------------------

    async def _replay_job(self, job: ResearchJobRecord) -> ResolvedResearchJob:
        return await self._resolve_job(job, refresh=True)

    async def _replay_comparison(
        self, group: ResearchComparisonRecord
    ) -> ResolvedResearchComparison:
        return await self._resolve_comparison(group, refresh=True)

    async def _resolve_job(self, job: ResearchJobRecord, *, refresh: bool) -> ResolvedResearchJob:
        warnings = await self._refresh_job(job) if refresh else ()
        resolved = await self._storage.resolve_job(job, is_owned=self.is_owned)
        resolved.warnings += warnings
        return resolved

    async def _resolve_comparison(
        self, group: ResearchComparisonRecord, *, refresh: bool
    ) -> ResolvedResearchComparison:
        warnings: dict[str, tuple[str, ...]] = {}
        if refresh:
            if group.comparison_id in self._owned:
                raise EngineError("JOB_BUSY", "This comparison is still running.")
            for child_id in group.child_job_ids:
                child = await self._storage._load_job(child_id)
                if child is not None and child.owner_scope == group.owner_scope:
                    warnings[child_id] = await self._refresh_job(child)
        resolved = await self._storage.resolve_comparison(group, is_owned=self.is_owned)
        for view in resolved.children:
            view.warnings += warnings.get(view.job_id, ())
        return resolved

    async def _refresh_job(self, job: ResearchJobRecord) -> tuple[str, ...]:
        """Apply provider-confirmed state to an unowned interrupted native job."""
        if self.is_owned(job.job_id):
            raise EngineError("JOB_BUSY", "The job is still running in this process.")
        resolved = await self._storage.resolve_job(job, is_owned=self.is_owned)
        if resolved.status is not JobStatus.INTERRUPTED:
            return ()
        if job.backend == BACKEND_LOCAL:
            return ("Local Deep Research runs cannot be refreshed or resumed.",)
        if job.provider_id is None:
            return ("No provider job ID was recorded, so nothing can be refreshed.",)

        engine = self._factory(job.engine_id)
        try:
            async with self._lock(job.job_id):
                if self.is_owned(job.job_id):
                    raise EngineError("JOB_BUSY", "The job is still running in this process.")
                try:
                    snapshot = await engine.refresh(job.provider_id)
                except EngineError as exc:
                    return (f"Refresh failed ({exc.code}).",)
                return await self._apply_snapshot(job, snapshot)
        finally:
            await self._close_engines([engine])

    async def _apply_snapshot(
        self, job: ResearchJobRecord, snapshot: EngineSnapshot
    ) -> tuple[str, ...]:
        if snapshot.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            return ("The provider reports the job is still running.",)
        assert job.provider_id is not None
        session_status = (
            ResearchStatus.INTERRUPTED if snapshot.status is JobStatus.INTERRUPTED else None
        )
        warnings = await self._write_native_terminal(
            job.provider_id, job, snapshot.status, snapshot.report, session_status=session_status
        )
        job.recorded_status = snapshot.status
        job.error_code = snapshot.error_code
        job.error_message = snapshot.error_message
        await self._storage.save_job(job)
        return warnings + snapshot.warnings


_service: ResearchJobService | None = None
_service_storage: ResearchJobStorage | None = None


def get_job_service() -> ResearchJobService:
    """Process-wide service bound to the current global job storage."""
    global _service, _service_storage
    storage = get_job_storage()
    if _service is None or _service_storage is not storage:
        if _service is not None:
            _service.revoke_ownership()
            _service._lease.release()
        _service = ResearchJobService(storage)
        _service_storage = storage
    return _service
