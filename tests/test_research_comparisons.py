"""Two-child comparison coordination."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from research_fakes import LOCAL_ID, Engines, FakeEngine, boom, report

from gemini_research_mcp.jobs import ResearchJobService
from gemini_research_mcp.storage import ResearchJobStorage, SessionStorage
from gemini_research_mcp.types import (
    EngineError,
    EngineSnapshot,
    ExecutionIdentity,
    GroupStatus,
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
def storage(tmp_path: Path) -> ResearchJobStorage:
    return ResearchJobStorage(SessionStorage(storage_dir=tmp_path))


@pytest.fixture
def engines() -> Engines:
    return Engines(
        FakeEngine("gemini", outcome=report("gemini text")),
        FakeEngine("gemini_max", outcome=report("max text")),
    )


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


async def compare(service: ResearchJobService, ids=("gemini", "gemini_max"), query="q", **kw):  # type: ignore[no-untyped-def]
    return await service.compare(query, list(ids), ident(**kw), on_progress=noop)


async def test_both_children_complete_in_requested_order(
    service: ResearchJobService, engines: Engines, storage: ResearchJobStorage
) -> None:
    result = await compare(service, ("gemini_max", "gemini"))
    assert result.status is GroupStatus.COMPLETED
    assert [c.record.engine_id for c in result.children] == ["gemini_max", "gemini"]
    assert [c.report.text for c in result.children if c.report] == ["max text", "gemini text"]
    assert engines.by_id["gemini"].creates == 1 and engines.by_id["gemini_max"].creates == 1
    assert all(c.record.comparison_id == result.comparison_id for c in result.children)
    group = await storage.get_comparison(result.comparison_id, owner_scope=None)
    assert group is not None
    assert group.child_job_ids == tuple(c.job_id for c in result.children)
    assert len(await service.list_runs(None)) == 1
    assert result.to_dict()["synthesis"] == "caller_owned"
    assert all(e.closed == 1 for e in engines.by_id.values())

    inspected = await service.inspect(result.comparison_id, None)
    assert inspected.status is GroupStatus.COMPLETED  # type: ignore[union-attr]


async def test_children_run_concurrently(service: ResearchJobService, engines: Engines) -> None:
    gate = asyncio.Event()
    engines.by_id["gemini"].gate = gate
    engines.by_id["gemini_max"].gate = gate
    task = asyncio.create_task(compare(service))
    await asyncio.wait_for(
        asyncio.gather(
            engines.by_id["gemini"].started.wait(), engines.by_id["gemini_max"].started.wait()
        ),
        timeout=5,
    )
    gate.set()
    assert (await task).status is GroupStatus.COMPLETED


async def test_one_failure_keeps_the_successful_sibling(
    service: ResearchJobService, engines: Engines
) -> None:
    engines.by_id["gemini_max"].outcome = boom("PROVIDER_FAILED")
    result = await compare(service)
    assert result.status is GroupStatus.PARTIAL
    ok, bad = result.children
    assert ok.status is JobStatus.COMPLETED and ok.report is not None
    assert bad.status is JobStatus.FAILED and bad.error_code == "PROVIDER_FAILED"


async def test_all_failed(service: ResearchJobService, engines: Engines) -> None:
    engines.by_id["gemini"].outcome = boom("PROVIDER_FAILED")
    engines.by_id["gemini_max"].outcome = boom("SUBMISSION_REJECTED")
    assert (await compare(service)).status is GroupStatus.FAILED


async def test_mixed_local_and_native(service: ResearchJobService, engines: Engines) -> None:
    engines.by_id[LOCAL_ID] = FakeEngine(LOCAL_ID, outcome=report("local"))
    result = await compare(service, ("gemini", LOCAL_ID))
    assert result.status is GroupStatus.COMPLETED
    assert result.children[1].provider_id is None


@pytest.mark.parametrize(
    "ids",
    [("gemini",), ("gemini", "gemini"), ("gemini", "nope"), ("a", "b", "c")],
)
async def test_invalid_selection_dispatches_nothing(
    service: ResearchJobService, engines: Engines, ids: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        await compare(service, ids)
    assert engines.creates == 0 and await service.list_runs(None) == []


async def test_blank_query_dispatches_nothing(
    service: ResearchJobService, engines: Engines
) -> None:
    with pytest.raises(ValueError):
        await compare(service, query=" ")
    assert engines.creates == 0


async def test_second_preflight_failure_dispatches_neither(
    service: ResearchJobService, engines: Engines
) -> None:
    engines.by_id["gemini_max"].preflight_error = boom("ENGINE_UNAVAILABLE")
    with pytest.raises(EngineError) as exc:
        await compare(service)
    assert exc.value.code == "ENGINE_UNAVAILABLE"
    assert engines.creates == 0 and await service.list_runs(None) == []
    assert all(e.closed == 1 for e in engines.by_id.values())


async def test_local_slot_is_reserved_before_any_native_dispatch(
    service: ResearchJobService, engines: Engines
) -> None:
    local = FakeEngine(LOCAL_ID)
    local.reserve_error = boom("LOCAL_ENGINE_BUSY")
    engines.by_id[LOCAL_ID] = local
    with pytest.raises(EngineError) as exc:
        await compare(service, ("gemini", LOCAL_ID))
    assert exc.value.code == "LOCAL_ENGINE_BUSY"
    assert engines.creates == 0
    assert await service.list_runs(None) == []


async def test_reserve_and_preflight_precede_every_create(
    service: ResearchJobService, engines: Engines
) -> None:
    log: list[str] = []
    local = FakeEngine(LOCAL_ID)
    engines.by_id[LOCAL_ID] = local
    for engine in (local, engines.by_id["gemini"]):
        engine.log = log
    await compare(service, (LOCAL_ID, "gemini"))
    first_create = min(i for i, e in enumerate(log) if e.endswith(":create"))
    assert {
        "local_deep_research:reserve",
        "local_deep_research:preflight",
        "gemini:preflight",
    } <= set(log[:first_create])
    assert log.index("local_deep_research:reserve") < log.index("local_deep_research:preflight")


async def test_storage_failure_persisting_children_dispatches_neither(
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
        if calls == 2:
            raise OSError("disk full")
        await original(job)

    monkeypatch.setattr(storage, "save_job", flaky)
    with pytest.raises(OSError):
        await compare(service)
    assert engines.creates == 0
    assert all(e.closed == 1 for e in engines.by_id.values())


async def test_replay_never_creates_and_conflict_fails(
    service: ResearchJobService, engines: Engines
) -> None:
    first = await compare(service)
    again = await compare(service, redelivered=True)
    assert again.comparison_id == first.comparison_id
    assert again.status is GroupStatus.COMPLETED
    assert engines.creates == 2
    with pytest.raises(EngineError) as exc:
        await compare(service, query="different")
    assert exc.value.code == "REQUEST_CONFLICT"
    assert engines.creates == 2


async def test_redelivery_without_record_never_creates(
    service: ResearchJobService, engines: Engines
) -> None:
    with pytest.raises(EngineError) as exc:
        await compare(service, redelivered=True)
    assert exc.value.code == "REPLAY_WITHOUT_RECORD"
    assert engines.creates == 0


async def test_concurrent_same_task_creates_once_per_child(
    service: ResearchJobService, engines: Engines
) -> None:
    gate = asyncio.Event()
    for engine in engines.by_id.values():
        engine.gate = gate
    first = asyncio.create_task(compare(service))
    await engines.by_id["gemini"].started.wait()
    await engines.by_id["gemini_max"].started.wait()
    with pytest.raises(EngineError) as exc:
        await compare(service)
    assert exc.value.code == "JOB_BUSY"
    gate.set()
    await first
    assert engines.creates == 2


async def test_external_cancellation_cleans_up_both_children(
    service: ResearchJobService, engines: Engines, storage: ResearchJobStorage
) -> None:
    gate = asyncio.Event()
    for engine in engines.by_id.values():
        engine.gate = gate
    task = asyncio.create_task(compare(service))
    await engines.by_id["gemini"].started.wait()
    await engines.by_id["gemini_max"].started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(e.closed == 1 for e in engines.by_id.values())
    jobs = await storage.list_jobs(owner_scope=None, include_children=True)
    assert len(jobs) == 2 and all(j.cancellation_requested for j in jobs)
    view = await service.inspect(jobs[0].comparison_id, None)  # type: ignore[arg-type]
    assert view.status is GroupStatus.INTERRUPTED  # type: ignore[union-attr]


async def test_group_refresh_recovers_native_children_and_skips_local(
    service: ResearchJobService, engines: Engines
) -> None:
    local = FakeEngine(LOCAL_ID, outcome=boom("LOCAL_ENGINE_FAILED"))
    engines.by_id[LOCAL_ID] = local
    engines.by_id["gemini"].outcome = boom("POLL_TIMEOUT")
    first = await compare(service, ("gemini", LOCAL_ID))
    assert first.status is GroupStatus.FAILED or first.status is GroupStatus.INTERRUPTED
    engines.by_id["gemini"].snapshot = EngineSnapshot(
        provider_id="int-gemini", status=JobStatus.COMPLETED, report=report("recovered")
    )
    view = await service.inspect(first.comparison_id, None, refresh=True)
    assert view.children[0].status is JobStatus.COMPLETED  # type: ignore[union-attr]
    assert view.status is GroupStatus.PARTIAL  # type: ignore[union-attr]
    assert local.refresh_calls == 0 and engines.by_id["gemini"].refresh_calls == 1


async def test_group_refresh_while_a_child_is_live_is_busy(
    service: ResearchJobService, engines: Engines
) -> None:
    gate = asyncio.Event()
    engines.by_id["gemini"].gate = gate
    task = asyncio.create_task(compare(service))
    await engines.by_id["gemini"].started.wait()
    group = (await service._storage.list_comparisons(owner_scope=None))[0]
    with pytest.raises(EngineError) as exc:
        await service.inspect(group.comparison_id, None, refresh=True)
    assert exc.value.code == "JOB_BUSY"
    gate.set()
    await task


async def test_runs_are_scoped_to_their_owner(service: ResearchJobService) -> None:
    mine = await compare(service, scope="alice")  # type: ignore[call-arg]
    assert [r.run_id for r in await service.list_runs("alice")] == [mine.comparison_id]
    assert await service.list_runs("bob") == []
    assert await service.list_runs(None) == []
    with pytest.raises(EngineError) as exc:
        await service.inspect(mine.comparison_id, "bob")
    assert exc.value.code == "RUN_NOT_FOUND"
    child = mine.children[0].job_id
    with pytest.raises(EngineError):
        await service.inspect(child, "bob")


async def test_list_runs_merges_jobs_and_comparisons_without_report_bodies(
    service: ResearchJobService,
) -> None:
    await service.run("single", "gemini", ident("a"), on_progress=noop)
    await compare(service, task_id="b")
    runs = await service.list_runs(None, limit=10)
    assert sorted(r.kind for r in runs) == ["comparison", "job"]
    assert all("text" not in r.to_dict() and "report" not in r.to_dict() for r in runs)
    assert len(await service.list_runs(None, limit=1)) == 1


async def test_unknown_run_id_is_not_found(service: ResearchJobService) -> None:
    for run_id in ("job_missing", "cmp_missing", "weird"):
        with pytest.raises(EngineError) as exc:
            await service.inspect(run_id, None)
        assert exc.value.code == "RUN_NOT_FOUND"
