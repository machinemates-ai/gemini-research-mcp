"""Single-job coordination: identity, replay, lease, lifecycle, failure mapping."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from research_fakes import LOCAL_ID, Engines, FakeEngine, boom, report

from gemini_research_mcp.jobs import ResearchJobService, SubmissionLease
from gemini_research_mcp.storage import (
    JOBS_COLLECTION,
    ResearchJobStorage,
    ResearchStatus,
    SessionStorage,
)
from gemini_research_mcp.types import (
    EngineError,
    EngineSnapshot,
    ExecutionIdentity,
    JobStatus,
)


@pytest.fixture(autouse=True)
def _single_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GEMINI_RESEARCH_STORAGE_URL",
        "FASTMCP_DOCKET_URL",
        "GEMINI_RESEARCH_TTL_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def sessions(tmp_path: Path) -> SessionStorage:
    return SessionStorage(storage_dir=tmp_path)


@pytest.fixture
def storage(sessions: SessionStorage) -> ResearchJobStorage:
    return ResearchJobStorage(sessions)


@pytest.fixture
def engines() -> Engines:
    return Engines()


@pytest.fixture
async def service(
    storage: ResearchJobStorage, engines: Engines
) -> AsyncIterator[ResearchJobService]:
    svc = ResearchJobService(storage, engine_factory=engines)
    yield svc
    await svc.close()


def ident(task_id: str = "t1", *, scope: str | None = None, redelivered: bool = False):  # type: ignore[no-untyped-def]
    return ExecutionIdentity(task_id=task_id, task_scope=scope, redelivered=redelivered)


async def noop(_message: str) -> None:
    return None


async def run(service: ResearchJobService, query: str = "why", engine: str = "gemini", **kw):  # type: ignore[no-untyped-def]
    return await service.run(query, engine, ident(**kw), on_progress=noop)


# --- completion and canonical storage ----------------------------------------------


async def test_native_run_completes_and_links_canonical_session(
    service: ResearchJobService,
    engines: Engines,
    sessions: SessionStorage,
    storage: ResearchJobStorage,
) -> None:
    engines.by_id["gemini"] = FakeEngine(
        "gemini",
        provider_id="int-1",
        outcome=report("# done", model="deep-research-preview-04-2026"),
    )
    result = await run(service, scope="alice")  # type: ignore[call-arg]

    assert result.status is JobStatus.COMPLETED
    assert result.report is not None and result.report.text == "# done"
    assert result.provider_id == "int-1"
    session = await sessions.get_session_async("int-1")
    assert session is not None
    assert session.status is ResearchStatus.COMPLETED
    assert session.job_id == result.job_id and session.owner_scope == "alice"
    assert session.report_text == "# done"
    record = await storage.get_job(result.job_id, owner_scope="alice")
    assert record is not None and record.provider_id == "int-1" and record.report is None
    assert record.attempt_started_at is not None and record.expires_at == session.expires_at
    assert engines.by_id["gemini"].closed == 1
    assert not service.is_owned(result.job_id)

    again = await service.inspect(result.job_id, "alice")
    assert again.status is JobStatus.COMPLETED  # type: ignore[union-attr]


async def test_marker_precedes_create_and_session_precedes_link(
    service: ResearchJobService,
    engines: Engines,
    sessions: SessionStorage,
    storage: ResearchJobStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = engines("gemini")
    observed: list[str] = []

    async def probe() -> None:
        job = (await storage.list_jobs(owner_scope=None))[0]
        observed.append("marker" if job.attempt_started_at else "no-marker")

    engine.on_create = probe
    original = storage.save_job

    async def spy(job):  # type: ignore[no-untyped-def]
        if job.provider_id is not None and not observed.count("session-first"):
            session = await sessions.get_session_async(job.provider_id)
            observed.append(
                "session-first" if session and session.job_id == job.job_id else "link-first"
            )
        await original(job)

    monkeypatch.setattr(storage, "save_job", spy)
    await run(service)
    assert observed[:2] == ["marker", "session-first"]


async def test_local_job_stores_report_and_never_exposes_provider_id(
    service: ResearchJobService,
    engines: Engines,
    sessions: SessionStorage,
    storage: ResearchJobStorage,
) -> None:
    engines.by_id[LOCAL_ID] = FakeEngine(LOCAL_ID, outcome=report("local text"))
    result = await run(service, engine=LOCAL_ID)
    assert result.status is JobStatus.COMPLETED and result.provider_id is None
    record = await storage.get_job(result.job_id, owner_scope=None)
    assert record is not None and record.report is not None and record.report.text == "local text"
    assert await sessions.list_sessions_async() == []


async def test_incomplete_report_keeps_partial_text_but_fails(
    service: ResearchJobService, engines: Engines
) -> None:
    engines.by_id["gemini"] = FakeEngine("gemini", outcome=report("partial", status="incomplete"))
    result = await run(service)
    assert result.status is JobStatus.FAILED
    assert result.report is not None and result.report.status == "incomplete"
    assert result.report.text == "partial"


# --- failure mapping -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "link", "status", "session"),
    [
        ("SUBMISSION_UNKNOWN", False, JobStatus.INTERRUPTED, None),
        ("SUBMISSION_REJECTED", False, JobStatus.FAILED, None),
        ("POLL_TIMEOUT", True, JobStatus.INTERRUPTED, ResearchStatus.IN_PROGRESS),
        ("POLL_FAILED", True, JobStatus.INTERRUPTED, ResearchStatus.IN_PROGRESS),
        ("PROVIDER_REQUIRES_ACTION", True, JobStatus.INTERRUPTED, ResearchStatus.INTERRUPTED),
        ("PROVIDER_FAILED", True, JobStatus.FAILED, ResearchStatus.FAILED),
        ("PROVIDER_NOT_FOUND", True, JobStatus.FAILED, ResearchStatus.FAILED),
        ("EMPTY_REPORT", True, JobStatus.FAILED, ResearchStatus.FAILED),
        ("PROVIDER_CANCELLED", True, JobStatus.CANCELLED, ResearchStatus.CANCELLED),
    ],
)
async def test_engine_error_codes_map_to_job_status(
    service: ResearchJobService,
    engines: Engines,
    sessions: SessionStorage,
    code: str,
    link: bool,
    status: JobStatus,
    session: ResearchStatus | None,
) -> None:
    engines.by_id["gemini"] = FakeEngine(
        "gemini", outcome=boom(code), link=link, provider_id="int-x"
    )
    result = await run(service)
    assert result.status is status
    assert result.error_code == code
    stored = await sessions.get_session_async("int-x")
    assert (stored.status if stored else None) is session
    inspected = await service.inspect(result.job_id, None)
    assert inspected.status is status  # type: ignore[union-attr]
    assert engines.by_id["gemini"].creates == 1


@pytest.mark.parametrize(
    "code",
    ["LOCAL_ENGINE_FAILED", "LOCAL_ENGINE_TIMEOUT", "LOCAL_EMPTY_REPORT", "LOCAL_ENGINE_BUSY"],
)
async def test_local_failures_are_conclusive(
    service: ResearchJobService, engines: Engines, code: str
) -> None:
    engines.by_id[LOCAL_ID] = FakeEngine(LOCAL_ID, outcome=boom(code))
    result = await run(service, engine=LOCAL_ID)
    assert result.status is JobStatus.FAILED and result.error_code == code


async def test_unexpected_exception_is_sanitized(
    service: ResearchJobService, engines: Engines
) -> None:
    engines.by_id["gemini"] = FakeEngine(
        "gemini", outcome=RuntimeError("secret prompt text"), link=False
    )
    result = await run(service)
    assert result.status is JobStatus.INTERRUPTED  # attempt started; outcome unknown
    assert result.error_code == "ENGINE_ERROR"
    assert "secret" not in (result.error_message or "")


# --- identity, replay and idempotency --------------------------------------------------


async def test_same_task_replays_without_creating_again(
    service: ResearchJobService, engines: Engines
) -> None:
    first = await run(service)
    second = await run(service, redelivered=True)
    assert second.job_id == first.job_id and second.status is JobStatus.COMPLETED
    assert engines.creates == 1


async def test_conflicting_replay_fails_before_dispatch(
    service: ResearchJobService, engines: Engines
) -> None:
    await run(service, query="one")
    with pytest.raises(EngineError) as exc:
        await run(service, query="two")
    assert exc.value.code == "REQUEST_CONFLICT"
    assert engines.creates == 1


async def test_distinct_tasks_get_distinct_jobs(
    service: ResearchJobService, engines: Engines
) -> None:
    a = await run(service, task_id="t1")
    b = await run(service, task_id="t2")
    assert a.job_id != b.job_id and engines.creates == 2


async def test_scope_is_part_of_identity(service: ResearchJobService) -> None:
    a = await run(service, scope="alice")  # type: ignore[call-arg]
    b = await run(service, scope="bob")  # type: ignore[call-arg]
    assert a.job_id != b.job_id


async def test_concurrent_same_task_creates_once(
    service: ResearchJobService, engines: Engines
) -> None:
    gate = asyncio.Event()
    engines.by_id["gemini"] = FakeEngine("gemini", gate=gate)
    first = asyncio.create_task(run(service))
    await engines.by_id["gemini"].started.wait()
    with pytest.raises(EngineError) as exc:
        await run(service)
    assert exc.value.code == "JOB_BUSY"
    gate.set()
    assert (await first).status is JobStatus.COMPLETED
    assert engines.creates == 1


async def test_redelivery_without_record_never_creates(
    service: ResearchJobService, engines: Engines
) -> None:
    with pytest.raises(EngineError) as exc:
        await run(service, redelivered=True)
    assert exc.value.code == "REPLAY_WITHOUT_RECORD"
    assert engines.creates == 0


async def test_replay_after_expiry_never_creates(
    service: ResearchJobService, engines: Engines, storage: ResearchJobStorage
) -> None:
    first = await run(service)
    await storage._drop(JOBS_COLLECTION, first.job_id)
    with pytest.raises(EngineError) as exc:
        await run(service, redelivered=True)
    assert exc.value.code == "REPLAY_WITHOUT_RECORD"
    assert engines.creates == 1


async def test_replay_of_interrupted_native_job_refreshes_not_creates(
    service: ResearchJobService, engines: Engines
) -> None:
    engine = FakeEngine("gemini", outcome=boom("POLL_TIMEOUT"), provider_id="int-r")
    engines.by_id["gemini"] = engine
    first = await run(service)
    assert first.status is JobStatus.INTERRUPTED

    engine.snapshot = EngineSnapshot(
        provider_id="int-r", status=JobStatus.COMPLETED, report=report("recovered")
    )
    replay = await run(service, redelivered=True)
    assert replay.status is JobStatus.COMPLETED
    assert replay.report is not None and replay.report.text == "recovered"
    assert engine.creates == 1 and engine.refresh_calls == 1

    again = await run(service, redelivered=True)
    assert again.status is JobStatus.COMPLETED
    assert engine.refresh_calls == 1  # terminal jobs are never refreshed


async def test_repeated_refresh_never_creates_and_does_not_renew_expiry(
    service: ResearchJobService, engines: Engines, storage: ResearchJobStorage
) -> None:
    engine = FakeEngine("gemini", outcome=boom("POLL_TIMEOUT"), provider_id="int-r")
    engines.by_id["gemini"] = engine
    first = await run(service)
    before = (await storage.get_job(first.job_id, owner_scope=None)).expires_at  # type: ignore[union-attr]
    for _ in range(3):
        view = await service.inspect(first.job_id, None, refresh=True)
        assert view.status is JobStatus.INTERRUPTED  # type: ignore[union-attr]
    assert engine.creates == 1 and engine.refresh_calls == 3
    after = (await storage.get_job(first.job_id, owner_scope=None)).expires_at  # type: ignore[union-attr]
    assert before == after


async def test_refresh_failure_becomes_a_warning(
    service: ResearchJobService, engines: Engines
) -> None:
    engine = FakeEngine("gemini", outcome=boom("POLL_TIMEOUT"))
    engines.by_id["gemini"] = engine
    first = await run(service)
    engine.snapshot = boom("POLL_FAILED")
    view = await service.inspect(first.job_id, None, refresh=True)
    assert view.status is JobStatus.INTERRUPTED  # type: ignore[union-attr]
    assert any("POLL_FAILED" in w for w in view.warnings)  # type: ignore[union-attr]


async def test_refresh_while_owner_is_live_is_busy(
    service: ResearchJobService, engines: Engines
) -> None:
    gate = asyncio.Event()
    engine = FakeEngine("gemini", gate=gate)
    engines.by_id["gemini"] = engine
    task = asyncio.create_task(run(service))
    await engine.started.wait()
    job = (await service._storage.list_jobs(owner_scope=None))[0]
    with pytest.raises(EngineError) as exc:
        await service.inspect(job.job_id, None, refresh=True)
    assert exc.value.code == "JOB_BUSY"
    assert engine.refresh_calls == 0
    gate.set()
    await task


async def test_local_job_refresh_is_unsupported_not_an_error(
    service: ResearchJobService, engines: Engines, storage: ResearchJobStorage
) -> None:
    done = await run(service, engine=LOCAL_ID)
    record = await storage.get_job(done.job_id, owner_scope=None)
    assert record is not None
    # An orphaned local record: attempt started, no owner, not terminal.
    record.recorded_status = JobStatus.RUNNING
    record.report = None
    await storage.save_job(record)
    view = await service.inspect(done.job_id, None, refresh=True)
    assert view.status is JobStatus.INTERRUPTED  # type: ignore[union-attr]
    assert view.error_code == "SUBMISSION_UNKNOWN"  # type: ignore[union-attr]
    assert any("cannot be refreshed" in w for w in view.warnings)  # type: ignore[union-attr]
    assert engines.by_id[LOCAL_ID].refresh_calls == 0


# --- admission gates -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "value"),
    [
        ("GEMINI_RESEARCH_STORAGE_URL", "redis://localhost:6379"),
        ("FASTMCP_DOCKET_URL", "redis://localhost:6379"),
    ],
)
async def test_distributed_backends_refuse_submission(
    service: ResearchJobService,
    engines: Engines,
    monkeypatch: pytest.MonkeyPatch,
    env: str,
    value: str,
) -> None:
    monkeypatch.setenv(env, value)
    with pytest.raises(EngineError) as exc:
        await run(service)
    assert exc.value.code == "DISTRIBUTED_SUBMISSION_UNSUPPORTED"
    assert engines.creates == 0
    assert await service.list_runs(None) == []


async def test_explicit_memory_docket_is_allowed(
    service: ResearchJobService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FASTMCP_DOCKET_URL", "memory://")
    assert (await run(service)).status is JobStatus.COMPLETED


async def test_retention_shorter_than_execution_is_rejected_before_dispatch(
    service: ResearchJobService, engines: Engines, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_RESEARCH_TTL_SECONDS", "60")
    with pytest.raises(EngineError) as exc:
        await run(service)
    assert exc.value.code == "RETENTION_TOO_SHORT"
    assert engines.creates == 0 and await service.list_runs(None) == []


async def test_second_service_on_same_data_dir_cannot_submit(
    service: ResearchJobService, storage: ResearchJobStorage
) -> None:
    other_engines = Engines()
    other = ResearchJobService(storage, engine_factory=other_engines)
    try:
        first = await run(service)
        with pytest.raises(EngineError) as exc:
            await other.run("why", "gemini", ident("t9"), on_progress=noop)
        assert exc.value.code == "SUBMISSION_LEASE_HELD"
        assert other_engines.creates == 0
        # Reads stay available to the non-owner.
        assert (await other.inspect(first.job_id, None)).status is JobStatus.COMPLETED  # type: ignore[union-attr]
    finally:
        await other.close()


async def test_lease_is_released_on_close_and_blocks_a_real_second_process(
    tmp_path: Path,
) -> None:
    lock = tmp_path / ".submission.lock"
    holder = SubmissionLease(lock)
    holder.acquire()
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from gemini_research_mcp.jobs import SubmissionLease
        from gemini_research_mcp.types import EngineError
        try:
            SubmissionLease(Path(sys.argv[1])).acquire()
        except EngineError as exc:
            print(exc.code)
        """
    )
    held = subprocess.run(
        [sys.executable, "-c", script, str(lock)], capture_output=True, text=True, check=True
    )
    assert held.stdout.strip() == "SUBMISSION_LEASE_HELD"
    holder.release()
    free = subprocess.run(
        [sys.executable, "-c", script, str(lock)], capture_output=True, text=True, check=True
    )
    assert free.stdout.strip() == ""


# --- zero dispatch ---------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   "])
async def test_blank_query_dispatches_nothing(
    service: ResearchJobService, engines: Engines, query: str
) -> None:
    with pytest.raises(ValueError):
        await run(service, query=query)
    assert engines.creates == 0


async def test_unknown_engine_dispatches_nothing(
    service: ResearchJobService, engines: Engines
) -> None:
    with pytest.raises(ValueError):
        await run(service, engine="nope")
    assert engines.creates == 0 and "nope" not in engines.by_id


async def test_preflight_failure_dispatches_nothing_and_leaves_no_record(
    service: ResearchJobService, engines: Engines
) -> None:
    engine = FakeEngine("gemini")
    engine.preflight_error = boom("ENGINE_UNAVAILABLE")
    engines.by_id["gemini"] = engine
    with pytest.raises(EngineError) as exc:
        await run(service)
    assert exc.value.code == "ENGINE_UNAVAILABLE"
    assert engine.creates == 0 and engine.closed == 1
    assert await service.list_runs(None) == []


async def test_storage_failure_before_submission_dispatches_nothing(
    service: ResearchJobService,
    engines: Engines,
    storage: ResearchJobStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(_job):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(storage, "save_job", fail)
    with pytest.raises(OSError):
        await run(service)
    assert engines.creates == 0 and engines.by_id["gemini"].closed == 1


async def test_marker_write_failure_prevents_the_create(
    service: ResearchJobService,
    engines: Engines,
    storage: ResearchJobStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = storage.save_job
    calls = 0

    async def flaky(job):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:  # the attempt-marker write
            raise OSError("disk full")
        await original(job)

    monkeypatch.setattr(storage, "save_job", flaky)
    result = await run(service)
    assert engines.creates == 0
    assert result.status is JobStatus.FAILED and result.error_code == "NOT_SUBMITTED"


# --- provider-link failure, races, owner loss, cancellation ----------------------------------


async def test_provider_link_failure_reports_the_real_id(
    service: ResearchJobService,
    engines: Engines,
    storage: ResearchJobStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engines.by_id["gemini"] = FakeEngine("gemini", provider_id="int-real-9")
    original = storage.save_job

    async def fail_link(job):  # type: ignore[no-untyped-def]
        if job.provider_id is not None:
            raise OSError("disk full")
        await original(job)

    monkeypatch.setattr(storage, "save_job", fail_link)
    result = await run(service)
    assert result.status is JobStatus.INTERRUPTED
    assert result.error_code == "PROVIDER_LINK_FAILED"
    assert result.provider_id == "int-real-9"
    assert "int-real-9" in (result.error_message or "")
    assert engines.creates == 1


async def test_owner_loss_prevents_stale_terminal_write(
    service: ResearchJobService, engines: Engines, sessions: SessionStorage
) -> None:
    gate = asyncio.Event()
    engine = FakeEngine("gemini", gate=gate, provider_id="int-lost", outcome=report("late"))
    engines.by_id["gemini"] = engine
    task = asyncio.create_task(run(service))
    await engine.started.wait()
    service.revoke_ownership()
    gate.set()
    result = await task
    assert result.status is JobStatus.INTERRUPTED and result.error_code == "OWNER_LOST"
    stored = await sessions.get_session_async("int-lost")
    assert stored is not None and stored.status is ResearchStatus.IN_PROGRESS
    assert stored.report_text is None
    assert engine.closed == 1


async def test_cancellation_marks_intent_and_stays_interrupted_until_confirmed(
    service: ResearchJobService, engines: Engines, storage: ResearchJobStorage
) -> None:
    gate = asyncio.Event()
    engine = FakeEngine("gemini", gate=gate, provider_id="int-c")
    engines.by_id["gemini"] = engine
    task = asyncio.create_task(run(service))
    await engine.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    job = (await storage.list_jobs(owner_scope=None))[0]
    assert job.cancellation_requested is True
    assert engine.closed == 1 and not service.is_owned(job.job_id)
    view = await service.inspect(job.job_id, None)
    assert view.status is JobStatus.INTERRUPTED  # type: ignore[union-attr]

    engine.snapshot = EngineSnapshot(
        provider_id="int-c", status=JobStatus.COMPLETED, report=report("finished anyway")
    )
    recovered = await service.inspect(job.job_id, None, refresh=True)
    assert recovered.status is JobStatus.COMPLETED  # type: ignore[union-attr]
    assert recovered.to_dict()["cancellation_requested"] is True  # type: ignore[union-attr]


async def test_cancellation_intent_before_dispatch_blocks_the_create(
    service: ResearchJobService, engines: Engines
) -> None:
    engine = FakeEngine("gemini")
    engines.by_id["gemini"] = engine

    async def intent() -> None:
        job = (await service._storage.list_jobs(owner_scope=None))[0]
        await service.mark_cancellation_requested(job.job_id)

    engine.before_submit = intent
    result = await run(service)
    assert engine.creates == 0
    assert result.status is JobStatus.CANCELLED
    assert result.error_code == "CANCELLED_BEFORE_DISPATCH"


async def test_progress_carries_ids_status_and_sanitized_phase_only(
    service: ResearchJobService, engines: Engines
) -> None:
    messages: list[str] = []

    async def capture(message: str) -> None:
        messages.append(message)

    engines.by_id["gemini"] = FakeEngine("gemini")
    result = await service.run("secret query text", "gemini", ident(), on_progress=capture)
    assert messages
    assert all(result.job_id in m and "status=" in m for m in messages)
    assert all("secret" not in m and "# Report" not in m for m in messages)


async def test_owner_loss_before_submission_blocks_the_create(
    service: ResearchJobService, engines: Engines
) -> None:
    engine = FakeEngine("gemini")
    engines.by_id["gemini"] = engine

    async def lose() -> None:
        service.revoke_ownership()

    engine.before_submit = lose
    result = await run(service)
    assert engine.creates == 0
    assert result.status is JobStatus.INTERRUPTED and result.error_code == "OWNER_LOST"
