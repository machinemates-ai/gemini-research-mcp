"""Research job facade over a real FastMCP Client, with fake engines."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.tools.base import ToolResult
from fastmcp_tasks import call_tool_task
from research_fakes import LOCAL_ID, Engines, FakeEngine, boom, report

from gemini_research_mcp import job_tools, server, storage
from gemini_research_mcp.export import render_research_bundle
from gemini_research_mcp.jobs import ResearchJobService
from gemini_research_mcp.storage import (
    ExportArtifactStore,
    ResearchStatus,
    SessionStorage,
    get_job_storage,
)
from gemini_research_mcp.types import ExecutionIdentity

_ = ToolResult  # imported at module level for Pydantic resolution in FastMCP


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for name in (
        "GEMINI_RESEARCH_STORAGE_URL",
        "FASTMCP_DOCKET_URL",
        "GEMINI_RESEARCH_TTL_SECONDS",
        "GEMINI_RESEARCH_LDR_ALLOW_HTTP",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_RESEARCH_EXPORT_DIR", str(tmp_path / "exports"))
    (tmp_path / "exports").mkdir()
    monkeypatch.setattr(storage, "_storage", SessionStorage(storage_dir=tmp_path / "store"))
    monkeypatch.setattr(storage, "_export_store", ExportArtifactStore(tmp_path / "store"))
    monkeypatch.setattr(storage, "_job_storage", None)
    monkeypatch.setattr(storage, "_job_storage_base", None)
    job_tools.set_http_transport(False)
    yield
    job_tools.set_http_transport(False)


@pytest.fixture
def engines() -> Engines:
    return Engines(FakeEngine("gemini"), FakeEngine("gemini_max"), FakeEngine(LOCAL_ID))


@pytest.fixture
async def service(
    engines: Engines, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[ResearchJobService]:
    svc = ResearchJobService(
        get_job_storage(), engine_factory=engines, persist_images=server._persist_job_images
    )
    monkeypatch.setattr(server, "_job_service", lambda: svc)
    yield svc
    await svc.close()


@pytest.fixture
async def client(service: ResearchJobService) -> AsyncIterator[Client]:
    async with Client(server.mcp, timeout=30) as c:
        yield c


async def submit(client: Client, tool: str, args: dict[str, Any]) -> Any:
    handle = await call_tool_task(client, tool, args)
    await handle.wait(timeout=20)
    return await handle.result()


def as_scope(monkeypatch: pytest.MonkeyPatch, scope: str | None, *, http: bool = False) -> None:
    """Pretend the caller is authenticated as ``scope`` (Task identity and reads)."""
    real = job_tools.current_execution_identity

    def identity() -> ExecutionIdentity:
        base = real()
        return ExecutionIdentity(base.task_id, scope, base.redelivered)

    monkeypatch.setattr(job_tools, "current_execution_identity", identity)
    monkeypatch.setattr(job_tools, "current_owner_scope", lambda: scope)
    monkeypatch.setattr(server, "get_task_scope", lambda: scope)
    job_tools.set_http_transport(http)


async def call(client: Client, tool: str, args: dict[str, Any] | None = None) -> Any:
    return await client.call_tool(tool, args or {}, raise_on_error=False)


# -- registration ---------------------------------------------------------------


async def test_six_tools_are_registered_and_submissions_require_tasks() -> None:
    names = {t.name for t in await server.mcp._list_tools()}
    assert {
        "research_run",
        "research_compare",
        "list_research_engines",
        "list_research_runs",
        "get_research_run",
        "export_research_bundle",
    } <= names
    visible = {t.name for t in await server.mcp.list_tools()}
    assert {"research_run", "research_compare"} <= visible
    assert visible.isdisjoint({"get_research_run", "export_research_bundle"})


# -- submissions ------------------------------------------------------------------


async def test_research_run_returns_report_and_ids(client: Client, engines: Engines) -> None:
    engines.by_id["gemini"].outcome = report(
        "# Answer\nbody", model="deep-research-preview-04-2026"
    )
    result = await submit(client, "research_run", {"query": "why"})
    assert result.is_error is False
    data = result.structured_content
    assert data["schema_version"] == 1
    assert data["result"] == result.content[0].text
    run = data["run"]
    assert run["run_id"].startswith("job_") and run["status"] == "completed"
    assert run["provider_id"] == "int-gemini" and run["run_id"] != run["provider_id"]
    assert "# Answer" in result.content[0].text and run["report"] == "# Answer\nbody"
    assert "owner_scope" not in json.dumps(data) and "task_id" not in json.dumps(data)


async def test_failed_run_is_an_error_carrying_the_id_and_code(
    client: Client, engines: Engines
) -> None:
    engines.by_id["gemini"].outcome = boom("PROVIDER_FAILED")
    with pytest.raises(ToolError, match=r"(?s)job_[0-9a-f]+.*PROVIDER_FAILED"):
        await submit(client, "research_run", {"query": "why"})


@pytest.mark.parametrize(
    "args",
    [
        {"query": "   "},
        {"query": "q", "engine": "nope"},
    ],
)
async def test_invalid_run_requests_never_dispatch(
    client: Client, engines: Engines, args: dict[str, Any]
) -> None:
    with pytest.raises(ToolError, match="INVALID_REQUEST"):
        await submit(client, "research_run", args)
    assert engines.creates == 0


async def test_compare_returns_attributed_reports_without_synthesis(
    client: Client, engines: Engines
) -> None:
    engines.by_id["gemini"].outcome = report("native report")
    engines.by_id[LOCAL_ID].outcome = report("local report")
    result = await submit(
        client, "research_compare", {"query": "q", "engines": ["gemini", LOCAL_ID]}
    )
    assert result.is_error is False
    run = result.structured_content["run"]
    assert run["kind"] == "comparison" and run["synthesis"] == "caller_owned"
    assert [c["engine"] for c in run["children"]] == ["gemini", LOCAL_ID]
    assert [c["report"] for c in run["children"]] == ["native report", "local report"]
    text = result.content[0].text
    assert "No synthesis was performed" in text
    assert text.index("native report") < text.index("local report")


async def test_partial_comparison_keeps_the_usable_report(client: Client, engines: Engines) -> None:
    engines.by_id["gemini"].outcome = report("native report")
    engines.by_id["gemini_max"].outcome = boom("PROVIDER_FAILED")
    result = await submit(
        client, "research_compare", {"query": "q", "engines": ["gemini", "gemini_max"]}
    )
    assert result.is_error is False
    run = result.structured_content["run"]
    assert run["status"] == "partial"
    assert run["children"][0]["status"] == "completed"
    assert run["children"][1]["error"]["code"] == "PROVIDER_FAILED"


async def test_all_failed_comparison_is_an_error(client: Client, engines: Engines) -> None:
    for engine in ("gemini", "gemini_max"):
        engines.by_id[engine].outcome = boom("PROVIDER_FAILED")
    with pytest.raises(ToolError, match=r"cmp_[0-9a-f]+"):
        await submit(
            client, "research_compare", {"query": "q", "engines": ["gemini", "gemini_max"]}
        )


@pytest.mark.parametrize(
    "selection",
    [["gemini"], ["gemini", "gemini"], ["gemini", "gemini_max", LOCAL_ID], ["gemini", "nope"]],
)
async def test_invalid_comparison_selection_never_dispatches(
    client: Client, engines: Engines, selection: list[str]
) -> None:
    with pytest.raises(ToolError, match="INVALID_REQUEST"):
        await submit(client, "research_compare", {"query": "q", "engines": selection})
    assert engines.creates == 0


# -- transport restrictions ---------------------------------------------------------


async def test_http_without_a_principal_is_refused_before_dispatch(
    client: Client, engines: Engines, monkeypatch: pytest.MonkeyPatch
) -> None:
    as_scope(monkeypatch, None, http=True)
    with pytest.raises(ToolError, match="AUTH_REQUIRED"):
        await submit(client, "research_run", {"query": "q"})
    result = await call(client, "list_research_runs")
    assert result.is_error and "AUTH_REQUIRED" in result.content[0].text
    assert engines.creates == 0


async def test_local_engine_over_http_needs_operator_opt_in(
    client: Client, engines: Engines, monkeypatch: pytest.MonkeyPatch
) -> None:
    as_scope(monkeypatch, "alice", http=True)
    with pytest.raises(ToolError, match="LOCAL_ENGINE_HTTP_DISABLED"):
        await submit(client, "research_run", {"query": "q", "engine": LOCAL_ID})
    with pytest.raises(ToolError, match="LOCAL_ENGINE_HTTP_DISABLED"):
        await submit(client, "research_compare", {"query": "q", "engines": ["gemini", LOCAL_ID]})
    assert engines.creates == 0

    listing = (await call(client, "list_research_engines")).structured_content["engines"]
    local = next(e for e in listing if e["id"] == LOCAL_ID)
    assert local["available"] is False and "HTTP" in local["availability_reason"]

    monkeypatch.setenv("GEMINI_RESEARCH_LDR_ALLOW_HTTP", "true")
    result = await submit(client, "research_run", {"query": "q", "engine": LOCAL_ID})
    assert result.structured_content["run"]["status"] == "completed"


# -- utilities ------------------------------------------------------------------------


async def test_list_engines_reads_configuration_only(client: Client, engines: Engines) -> None:
    result = await call(client, "list_research_engines")
    ids = [e["id"] for e in result.structured_content["engines"]]
    assert ids == ["gemini", "gemini_max", LOCAL_ID]
    assert engines.creates == 0 and all(e.refresh_calls == 0 for e in engines.by_id.values())


async def test_list_runs_has_metadata_only_and_validates_limit(client: Client) -> None:
    await submit(client, "research_run", {"query": "why is the sky blue"})
    result = await call(client, "list_research_runs", {"limit": 5})
    runs = result.structured_content["runs"]
    assert len(runs) == 1 and runs[0]["status"] == "completed"
    assert "report" not in runs[0] and "body" not in json.dumps(runs)
    for bad in (0, 101):
        error = await call(client, "list_research_runs", {"limit": bad})
        assert error.is_error and "INVALID_REQUEST" in error.content[0].text


async def test_get_run_resolves_by_logical_id_only(client: Client, engines: Engines) -> None:
    run = (await submit(client, "research_run", {"query": "q"})).structured_content["run"]
    got = await call(client, "get_research_run", {"run_id": run["run_id"]})
    assert got.structured_content["run"]["report"] == "# Report\nbody"
    for unknown in (run["provider_id"], "job_missing", "cmp_missing", ""):
        missing = await call(client, "get_research_run", {"run_id": unknown})
        assert missing.is_error and "RUN_NOT_FOUND" in missing.content[0].text
    assert engines.by_id["gemini"].refresh_calls == 0


async def test_export_bundle_is_attributed_and_makes_no_provider_call(
    client: Client, engines: Engines, tmp_path: Path
) -> None:
    engines.by_id["gemini"].outcome = report("native report")
    engines.by_id[LOCAL_ID].outcome = report("local report")
    run = (
        await submit(client, "research_compare", {"query": "q", "engines": ["gemini", LOCAL_ID]})
    ).structured_content["run"]
    creates = engines.creates

    exported = await call(
        client, "export_research_bundle", {"run_id": run["run_id"], "format": "json"}
    )
    bundle = exported.structured_content["bundle"]
    on_disk = json.loads(Path(bundle["path"]).read_text(encoding="utf-8"))
    assert on_disk["synthesis"] == "caller_owned" and on_disk["kind"] == "comparison"
    assert [e["engine"] for e in on_disk["engines"]] == ["gemini", LOCAL_ID]
    assert [e["report"] for e in on_disk["engines"]] == ["native report", "local report"]
    assert on_disk["engines"][1]["recovery"] == "none"
    artifact = await storage.get_export_store().get_async(bundle["resource_uri"].rsplit("/", 1)[1])
    assert artifact is not None and artifact.scoped

    md = await call(
        client, "export_research_bundle", {"run_id": run["run_id"], "format": "markdown"}
    )
    assert "## Engine: gemini" in Path(md.structured_content["bundle"]["path"]).read_text()
    bad = await call(client, "export_research_bundle", {"run_id": run["run_id"], "format": "docx"})
    assert bad.is_error
    assert engines.creates == creates
    assert all(e.refresh_calls == 0 for e in engines.by_id.values())


# -- ownership --------------------------------------------------------------------------


async def test_runs_sessions_and_artifacts_are_invisible_to_other_principals(
    client: Client, engines: Engines, monkeypatch: pytest.MonkeyPatch
) -> None:
    as_scope(monkeypatch, "alice")
    run = (await submit(client, "research_run", {"query": "q"})).structured_content["run"]
    exported = await call(client, "export_research_bundle", {"run_id": run["run_id"]})
    export_uri = exported.structured_content["bundle"]["resource_uri"]
    session_id = run["provider_id"]
    legacy = await call(
        client, "export_research_session", {"interaction_id": session_id, "format": "docx"}
    )
    assert "Saved to" in legacy.content[0].text

    assert server.get_research_session(session_id) is not None
    export_id = export_uri.rsplit("/", 1)[1]
    assert (await server.get_export_by_id(export_id)).text

    as_scope(monkeypatch, "bob")
    assert (await call(client, "list_research_runs")).structured_content["runs"] == []
    for tool, args in (
        ("get_research_run", {"run_id": run["run_id"]}),
        ("export_research_bundle", {"run_id": run["run_id"]}),
    ):
        denied = await call(client, tool, args)
        assert denied.is_error and "RUN_NOT_FOUND" in denied.content[0].text
    assert server.get_research_session(session_id) is None
    assert session_id not in {s.interaction_id for s in server._list_sessions()}
    with pytest.raises(ValueError, match="not found"):
        await server.get_export_by_id(export_id)
    assert json.loads(await server.list_exports())["count"] == 0
    denied_export = await call(
        client, "export_research_session", {"interaction_id": session_id, "format": "json"}
    )
    assert "not found" in denied_export.content[0].text.lower()


async def test_legacy_sessions_without_a_link_stay_visible_to_everyone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage.save_research_session("legacy-1", "old question")
    as_scope(monkeypatch, "bob")
    assert server.get_research_session("legacy-1") is not None
    assert "legacy-1" in {s.interaction_id for s in server._list_sessions()}


async def test_legacy_resume_is_refused_while_the_job_is_running(
    client: Client, service: ResearchJobService, engines: Engines, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = (await submit(client, "research_run", {"query": "q"})).structured_content["run"]
    session = storage.get_research_session(run["provider_id"])
    assert session is not None and session.job_id == run["run_id"]
    monkeypatch.setattr(service, "is_owned", lambda job_id: job_id == run["run_id"])
    result = await call(client, "resume_research", {"interaction_id": run["provider_id"]})
    assert json.loads(result.content[0].text)["status"] == "busy"


async def test_legacy_resume_completion_is_visible_through_the_job(
    client: Client, engines: Engines
) -> None:
    engines.by_id["gemini"].outcome = boom("POLL_TIMEOUT")
    with pytest.raises(ToolError):
        await submit(client, "research_run", {"query": "q"})
    jobs = await get_job_storage().list_jobs(5, owner_scope=None)
    job = jobs[0]
    assert job.provider_id
    got = await call(client, "get_research_run", {"run_id": job.job_id})
    assert got.structured_content["run"]["status"] == "interrupted"

    storage.update_research_session(
        job.provider_id, status=ResearchStatus.COMPLETED, report_text="# recovered"
    )
    again = await call(client, "get_research_run", {"run_id": job.job_id})
    assert again.structured_content["run"]["status"] == "completed"
    assert again.structured_content["run"]["report"] == "# recovered"
    expiry = storage.get_research_session(job.provider_id).expires_at  # type: ignore[union-attr]
    assert expiry == job.expires_at


# -- bundle rendering --------------------------------------------------------------------


async def test_bundle_rendering_rejects_unknown_formats_and_hides_internals(
    client: Client,
) -> None:
    run = (await submit(client, "research_run", {"query": "q"})).structured_content["run"]
    resolved = await server._job_service().inspect(run["run_id"], None)
    json_bundle = render_research_bundle(resolved, format="json")
    assert b"owner_scope" not in json_bundle.content and b"task_id" not in json_bundle.content
    assert render_research_bundle(resolved, format="md").mime_type == "text/markdown"
    with pytest.raises(ValueError):
        render_research_bundle(resolved, format="docx")


async def test_startup_cleanup_failure_does_not_block_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Broken:
        async def cleanup_expired_async(self) -> int:
            calls.append("cleanup")
            raise RuntimeError("store offline")

    monkeypatch.setattr(server, "get_job_storage", lambda: Broken())
    async with server.lifespan(server.mcp):
        pass
    assert calls == ["cleanup"]
