"""Persistence and resolution tests for research jobs and comparisons."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from gemini_research_mcp.storage import (
    ResearchJobStorage,
    ResearchSession,
    ResearchStatus,
    SessionStorage,
)
from gemini_research_mcp.types import (
    BACKEND_GEMINI,
    BACKEND_LOCAL,
    EngineError,
    EngineReport,
    GroupStatus,
    JobStatus,
    ResearchComparisonRecord,
    ResearchJobRecord,
)

NOW = 1_000_000.0


@pytest.fixture(autouse=True)
def _local_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_RESEARCH_STORAGE_URL", raising=False)


@pytest.fixture
def sessions(tmp_path: Path) -> SessionStorage:
    return SessionStorage(storage_dir=tmp_path)


@pytest.fixture
def jobs(sessions: SessionStorage) -> ResearchJobStorage:
    return ResearchJobStorage(sessions)


def make_job(
    job_id: str = "job_a",
    *,
    engine_id: str = "gemini",
    scope: str | None = None,
    provider_id: str | None = None,
    status: JobStatus = JobStatus.RUNNING,
    expires_in: float = 3600.0,
    comparison_id: str | None = None,
    attempt: bool = True,
    report: EngineReport | None = None,
    slot: str = "single",
) -> ResearchJobRecord:
    created = time.time()
    local = engine_id == "local_deep_research"
    return ResearchJobRecord(
        job_id=job_id,
        engine_id=engine_id,
        backend=BACKEND_LOCAL if local else BACKEND_GEMINI,
        model=None if local else "deep-research-preview-04-2026",
        query="why is the sky blue",
        request_fingerprint="fp",
        task_id=f"task-{job_id}",
        owner_scope=scope,
        slot=slot,
        created_at=created,
        expires_at=created + expires_in,
        comparison_id=comparison_id,
        provider_id=provider_id,
        recorded_status=status,
        attempt_started_at=created if attempt else None,
        report=report,
    )


def make_session(
    interaction_id: str = "int-1",
    *,
    status: ResearchStatus = ResearchStatus.IN_PROGRESS,
    job_id: str | None = "job_a",
    scope: str | None = None,
    expires_at: float | None = None,
) -> ResearchSession:
    return ResearchSession(
        interaction_id=interaction_id,
        query="why is the sky blue",
        created_at=time.time(),
        status=status,
        job_id=job_id,
        owner_scope=scope,
        expires_at=expires_at,
    )


def owned(*ids: str):  # type: ignore[no-untyped-def]
    return lambda job_id: job_id in ids


# --- session compatibility ------------------------------------------------------


def test_legacy_session_dict_roundtrips_without_new_keys() -> None:
    legacy = {"interaction_id": "i", "query": "q", "created_at": 1.0}
    session = ResearchSession.from_dict(legacy)
    assert session.job_id is None and session.sources == [] and session.usage is None
    exported = session.to_dict()
    for key in ("job_id", "owner_scope", "sources", "usage"):
        assert key not in exported


def test_session_link_metadata_roundtrips() -> None:
    session = make_session(scope="s")
    session.sources = [{"uri": "https://e.com", "title": None, "provenance": "report_parsed"}]
    session.usage = {"total_tokens": 7}
    clone = ResearchSession.from_dict(session.to_dict())
    assert clone.job_id == "job_a" and clone.owner_scope == "s"
    assert clone.sources == session.sources and clone.usage == {"total_tokens": 7}


# --- job persistence ------------------------------------------------------------


async def test_job_roundtrip_and_scope_filtering(jobs: ResearchJobStorage) -> None:
    await jobs.save_job(make_job("job_a", scope="alice"))
    await jobs.save_job(make_job("job_b", scope="bob"))
    got = await jobs.get_job("job_a", owner_scope="alice")
    assert got is not None and got.owner_scope == "alice"
    assert await jobs.get_job("job_a", owner_scope="bob") is None
    assert await jobs.get_job("job_a", owner_scope=None) is None
    assert [j.job_id for j in await jobs.list_jobs(owner_scope="alice")] == ["job_a"]
    assert [j.job_id for j in await jobs.list_jobs(owner_scope="bob")] == ["job_b"]
    assert await jobs.list_jobs(owner_scope="carol") == []


async def test_anonymous_scope_only_sees_anonymous(jobs: ResearchJobStorage) -> None:
    await jobs.save_job(make_job("job_a", scope=None))
    await jobs.save_job(make_job("job_b", scope="bob"))
    assert [j.job_id for j in await jobs.list_jobs(owner_scope=None)] == ["job_a"]


async def test_list_hides_comparison_children_by_default(jobs: ResearchJobStorage) -> None:
    await jobs.save_job(make_job("job_s"))
    await jobs.save_job(make_job("job_c", comparison_id="cmp_x", slot="child:0"))
    assert [j.job_id for j in await jobs.list_jobs(owner_scope=None)] == ["job_s"]
    both = await jobs.list_jobs(owner_scope=None, include_children=True)
    assert {j.job_id for j in both} == {"job_s", "job_c"}


async def test_expiry_and_identity_cannot_be_changed_by_update(
    jobs: ResearchJobStorage,
) -> None:
    job = make_job("job_a")
    await jobs.save_job(job)
    job.recorded_status = JobStatus.COMPLETED
    await jobs.save_job(job)  # same identity/expiry: status update is allowed

    stretched = make_job("job_a")
    stretched.expires_at = job.expires_at + 999
    with pytest.raises(EngineError) as exc:
        await jobs.save_job(stretched)
    assert exc.value.code == "JOB_CONFLICT"

    other_scope = make_job("job_a", scope="mallory")
    other_scope.expires_at = job.expires_at
    with pytest.raises(EngineError):
        await jobs.save_job(other_scope)


async def test_expired_job_is_not_returned(jobs: ResearchJobStorage) -> None:
    await jobs.save_job(make_job("job_old", expires_in=-5.0))
    assert await jobs.get_job("job_old", owner_scope=None) is None
    assert await jobs.list_jobs(owner_scope=None) == []


# --- native resolution ----------------------------------------------------------


async def test_native_job_reads_through_to_the_canonical_session(
    sessions: SessionStorage, jobs: ResearchJobStorage
) -> None:
    job = make_job("job_a", provider_id="int-1")
    await sessions.save_session_async(make_session("int-1"))
    await jobs.save_job(job)

    running = await jobs.resolve_job(job, is_owned=owned("job_a"))
    assert running.status is JobStatus.RUNNING and running.report is None

    await sessions.update_session_async(
        "int-1",
        status=ResearchStatus.COMPLETED,
        report_text="the answer",
        total_tokens=11,
    )
    done = await jobs.resolve_job(job, is_owned=owned())
    assert done.status is JobStatus.COMPLETED
    assert done.report is not None and done.report.text == "the answer"
    assert done.provider_id == "int-1"
    stored = await jobs.get_job("job_a", owner_scope=None)
    assert stored is not None and stored.report is None
    assert stored.recorded_status is JobStatus.RUNNING


async def test_native_sources_and_usage_come_from_the_session(
    sessions: SessionStorage, jobs: ResearchJobStorage
) -> None:
    job = make_job("job_a", provider_id="int-1")
    session = make_session("int-1", status=ResearchStatus.COMPLETED)
    session.report_text = "text"
    session.sources = [{"uri": "https://e.com", "title": "E", "provenance": "report_parsed"}]
    session.usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    await sessions.save_session_async(session)
    resolved = await jobs.resolve_job(job, is_owned=owned())
    assert resolved.report is not None
    assert resolved.report.sources[0].uri == "https://e.com"
    assert resolved.report.usage is not None and resolved.report.usage.total_tokens == 3


async def test_missing_linked_session_is_expired_not_empty_success(
    jobs: ResearchJobStorage,
) -> None:
    resolved = await jobs.resolve_job(make_job("job_a", provider_id="gone"), is_owned=owned())
    assert resolved.status is JobStatus.EXPIRED
    assert resolved.report is None
    assert resolved.job_id == "job_a" and resolved.provider_id == "gone"


@pytest.mark.parametrize(
    ("session_status", "expected"),
    [
        (ResearchStatus.FAILED, JobStatus.FAILED),
        (ResearchStatus.CANCELLED, JobStatus.CANCELLED),
        (ResearchStatus.INTERRUPTED, JobStatus.INTERRUPTED),
    ],
)
async def test_terminal_session_status_wins(
    sessions: SessionStorage,
    jobs: ResearchJobStorage,
    session_status: ResearchStatus,
    expected: JobStatus,
) -> None:
    await sessions.save_session_async(make_session("int-1", status=session_status))
    resolved = await jobs.resolve_job(make_job("job_a", provider_id="int-1"), is_owned=owned())
    assert resolved.status is expected


async def test_awaiting_approval_is_interrupted_with_legacy_warning(
    sessions: SessionStorage, jobs: ResearchJobStorage
) -> None:
    await sessions.save_session_async(
        make_session("int-1", status=ResearchStatus.AWAITING_APPROVAL)
    )
    resolved = await jobs.resolve_job(
        make_job("job_a", provider_id="int-1"), is_owned=owned("job_a")
    )
    assert resolved.status is JobStatus.INTERRUPTED
    assert any("resume_research" in w for w in resolved.warnings)


async def test_unowned_in_progress_native_job_reads_as_interrupted(
    sessions: SessionStorage, jobs: ResearchJobStorage
) -> None:
    await sessions.save_session_async(make_session("int-1", status=ResearchStatus.EXECUTING))
    resolved = await jobs.resolve_job(make_job("job_a", provider_id="int-1"), is_owned=owned())
    assert resolved.status is JobStatus.INTERRUPTED
    assert resolved.error_code is None  # provider ID is linked; refresh can recover


async def test_unlinked_attempt_reads_interrupted_submission_unknown(
    jobs: ResearchJobStorage,
) -> None:
    resolved = await jobs.resolve_job(make_job("job_a", provider_id=None), is_owned=owned())
    assert resolved.status is JobStatus.INTERRUPTED
    assert resolved.error_code == "SUBMISSION_UNKNOWN"


async def test_unlinked_without_attempt_is_not_submitted(jobs: ResearchJobStorage) -> None:
    resolved = await jobs.resolve_job(
        make_job("job_a", status=JobStatus.QUEUED, attempt=False), is_owned=owned()
    )
    assert resolved.status is JobStatus.INTERRUPTED
    assert resolved.error_code == "NOT_SUBMITTED"


async def test_owned_unlinked_job_keeps_its_recorded_progress(jobs: ResearchJobStorage) -> None:
    resolved = await jobs.resolve_job(make_job("job_a"), is_owned=owned("job_a"))
    assert resolved.status is JobStatus.RUNNING


async def test_session_linked_to_another_job_or_scope_is_rejected(
    sessions: SessionStorage, jobs: ResearchJobStorage
) -> None:
    await sessions.save_session_async(
        make_session("int-1", status=ResearchStatus.COMPLETED, job_id="job_other")
    )
    resolved = await jobs.resolve_job(make_job("job_a", provider_id="int-1"), is_owned=owned())
    assert resolved.status is JobStatus.FAILED
    assert resolved.error_code == "LINK_MISMATCH"
    assert resolved.report is None

    await sessions.save_session_async(
        make_session("int-2", status=ResearchStatus.COMPLETED, scope="mallory")
    )
    resolved = await jobs.resolve_job(
        make_job("job_a", provider_id="int-2", scope="alice"), is_owned=owned()
    )
    assert resolved.error_code == "LINK_MISMATCH"


# --- local resolution -----------------------------------------------------------


async def test_local_job_resolves_from_its_own_record(jobs: ResearchJobStorage) -> None:
    report = EngineReport(text="local answer", provider_id="ldr-7")
    job = make_job(
        "job_l",
        engine_id="local_deep_research",
        status=JobStatus.COMPLETED,
        report=report,
    )
    resolved = await jobs.resolve_job(job, is_owned=owned())
    assert resolved.status is JobStatus.COMPLETED
    assert resolved.report == report
    assert resolved.provider_id is None  # a local ID never implies recovery


async def test_unowned_local_running_job_is_interrupted_without_recovery(
    jobs: ResearchJobStorage,
) -> None:
    job = make_job("job_l", engine_id="local_deep_research")
    resolved = await jobs.resolve_job(job, is_owned=owned())
    assert resolved.status is JobStatus.INTERRUPTED
    assert resolved.error_code == "SUBMISSION_UNKNOWN"


async def test_job_past_expiry_resolves_expired(jobs: ResearchJobStorage) -> None:
    job = make_job("job_a", expires_in=-1.0, status=JobStatus.COMPLETED)
    resolved = await jobs.resolve_job(job, is_owned=owned())
    assert resolved.status is JobStatus.EXPIRED


# --- comparisons ----------------------------------------------------------------


def make_comparison(
    scope: str | None = None, *, expires_in: float = 3600.0
) -> ResearchComparisonRecord:
    created = time.time()
    return ResearchComparisonRecord(
        comparison_id="cmp_x",
        query="why is the sky blue",
        request_fingerprint="fp",
        task_id="task-cmp",
        owner_scope=scope,
        engine_ids=("gemini", "local_deep_research"),
        child_job_ids=("job_0", "job_1"),
        created_at=created,
        expires_at=created + expires_in,
    )


async def test_comparison_is_immutable_and_scoped(jobs: ResearchJobStorage) -> None:
    group = make_comparison("alice")
    await jobs.save_comparison(group)
    await jobs.save_comparison(group)  # identical re-save is a no-op

    changed = make_comparison("alice")
    changed.expires_at = group.expires_at
    changed.child_job_ids = ("job_0", "job_9")
    with pytest.raises(EngineError) as exc:
        await jobs.save_comparison(changed)
    assert exc.value.code == "COMPARISON_CONFLICT"

    assert await jobs.get_comparison("cmp_x", owner_scope="alice") == group
    assert await jobs.get_comparison("cmp_x", owner_scope="bob") is None
    assert [g.comparison_id for g in await jobs.list_comparisons(owner_scope="alice")] == ["cmp_x"]
    assert await jobs.list_comparisons(owner_scope="bob") == []


async def _group(
    jobs: ResearchJobStorage,
    first: tuple[JobStatus, EngineReport | None],
    second: tuple[JobStatus, EngineReport | None],
):  # type: ignore[no-untyped-def]
    group = make_comparison()
    await jobs.save_comparison(group)
    for index, (status, report) in enumerate((first, second)):
        job = make_job(
            f"job_{index}",
            engine_id="local_deep_research",
            status=status,
            comparison_id="cmp_x",
            slot=f"child:{index}",
            report=report,
        )
        job.expires_at = group.expires_at
        await jobs.save_job(job)
    return group


OK = EngineReport(text="report")


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ((JobStatus.COMPLETED, OK), (JobStatus.COMPLETED, OK), GroupStatus.COMPLETED),
        ((JobStatus.COMPLETED, OK), (JobStatus.FAILED, None), GroupStatus.PARTIAL),
        ((JobStatus.CANCELLED, None), (JobStatus.COMPLETED, OK), GroupStatus.PARTIAL),
        ((JobStatus.COMPLETED, OK), (JobStatus.INTERRUPTED, None), GroupStatus.PARTIAL),
        ((JobStatus.FAILED, None), (JobStatus.CANCELLED, None), GroupStatus.FAILED),
        ((JobStatus.FAILED, None), (JobStatus.INTERRUPTED, None), GroupStatus.INTERRUPTED),
        ((JobStatus.INTERRUPTED, None), (JobStatus.INTERRUPTED, None), GroupStatus.INTERRUPTED),
        ((JobStatus.COMPLETED, OK), (JobStatus.RUNNING, None), GroupStatus.RUNNING),
        ((JobStatus.RUNNING, None), (JobStatus.RUNNING, None), GroupStatus.RUNNING),
        ((JobStatus.QUEUED, None), (JobStatus.QUEUED, None), GroupStatus.QUEUED),
        ((JobStatus.QUEUED, None), (JobStatus.COMPLETED, OK), GroupStatus.RUNNING),
    ],
)
async def test_group_status_is_derived_from_children(
    jobs: ResearchJobStorage,
    first: tuple[JobStatus, EngineReport | None],
    second: tuple[JobStatus, EngineReport | None],
    expected: GroupStatus,
) -> None:
    group = await _group(jobs, first, second)
    resolved = await jobs.resolve_comparison(group, is_owned=owned("job_0", "job_1"))
    assert resolved.status is expected
    assert [c.job_id for c in resolved.children] == ["job_0", "job_1"]


async def test_missing_child_surfaces_identity_and_expires_the_group(
    jobs: ResearchJobStorage,
) -> None:
    group = make_comparison()
    await jobs.save_comparison(group)
    resolved = await jobs.resolve_comparison(group, is_owned=owned())
    assert resolved.status is GroupStatus.EXPIRED
    assert [c.job_id for c in resolved.children] == ["job_0", "job_1"]
    assert [c.record.engine_id for c in resolved.children] == ["gemini", "local_deep_research"]
    assert all(c.status is JobStatus.EXPIRED for c in resolved.children)


async def test_children_and_group_share_the_exact_expiry(jobs: ResearchJobStorage) -> None:
    group = await _group(jobs, (JobStatus.RUNNING, None), (JobStatus.RUNNING, None))
    resolved = await jobs.resolve_comparison(group, is_owned=owned("job_0", "job_1"))
    assert {c.record.expires_at for c in resolved.children} == {group.expires_at}


# --- concurrency ----------------------------------------------------------------


async def test_concurrent_writers_stay_discoverable(tmp_path: Path) -> None:
    first_sessions = SessionStorage(storage_dir=tmp_path)
    second_sessions = SessionStorage(storage_dir=tmp_path)
    first = ResearchJobStorage(first_sessions)
    second = ResearchJobStorage(second_sessions)
    gate = asyncio.Event()

    async def write_job(storage: ResearchJobStorage, index: int) -> None:
        await gate.wait()
        await storage.save_job(make_job(f"job_{index}"))

    async def write_session(storage: SessionStorage, index: int) -> None:
        await gate.wait()
        await storage.save_session_async(make_session(f"int-{index}", job_id=None))

    tasks = [
        *(write_job(first if i % 2 else second, i) for i in range(12)),
        *(write_session(first_sessions if i % 2 else second_sessions, i) for i in range(12)),
    ]
    runners = [asyncio.create_task(t) for t in tasks]
    await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(*runners)

    listed = await first.list_jobs(owner_scope=None, limit=100)
    assert {j.job_id for j in listed} == {f"job_{i}" for i in range(12)}
    assert len(await second_sessions.list_sessions_async()) == 12


async def test_cleanup_removes_expired_jobs_and_comparisons(jobs: ResearchJobStorage) -> None:
    await jobs.save_job(make_job("job_old", expires_in=-5.0))
    await jobs.save_job(make_job("job_new"))
    await jobs.save_comparison(make_comparison(expires_in=-5.0))
    assert await jobs.cleanup_expired_async() == 2
    assert [j.job_id for j in await jobs.list_jobs(owner_scope=None)] == ["job_new"]
    assert await jobs.list_comparisons(owner_scope=None) == []
