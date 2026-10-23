"""Contract tests for the engine-neutral research job layer."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastmcp.tools.base import ToolResult

from gemini_research_mcp.engines import (
    ENGINE_IDS,
    describe_engines,
    get_engine,
    request_fingerprint,
    stable_job_id,
    validate_comparison_engines,
    validate_query,
)
from gemini_research_mcp.types import (
    EngineError,
    EngineReport,
    JobStatus,
    ResearchSource,
)


@pytest.mark.parametrize(
    "ids",
    [
        [],
        ["gemini"],
        ["gemini", "gemini"],
        ["gemini", "unknown"],
        ["gemini", "gemini_max", "local_deep_research"],
    ],
)
def test_invalid_comparison_selection(ids: list[str]) -> None:
    with pytest.raises(ValueError):
        validate_comparison_engines(ids)


def test_comparison_selection_preserves_order() -> None:
    assert validate_comparison_engines(["local_deep_research", "gemini"]) == (
        "local_deep_research",
        "gemini",
    )
    assert validate_comparison_engines(["gemini", "gemini_max"]) == ("gemini", "gemini_max")


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_is_rejected(query: str) -> None:
    with pytest.raises(ValueError):
        validate_query(query)


def test_query_is_not_rewritten() -> None:
    assert validate_query("  keep surrounding space  ") == "  keep surrounding space  "


def test_replay_identity_is_stable_but_task_identity_is_distinct() -> None:
    first = stable_job_id("scope-a", "opaque-task-a", "single")
    assert first == stable_job_id("scope-a", "opaque-task-a", "single")
    assert first != stable_job_id("scope-a", "opaque-task-b", "single")
    assert first != stable_job_id("scope-b", "opaque-task-a", "single")
    assert first != stable_job_id(None, "opaque-task-a", "single")
    assert first.startswith("job_")


def test_slots_and_comparison_identity_are_distinct() -> None:
    comparison = stable_job_id(None, "t", "comparison")
    children = {stable_job_id(None, "t", "child:0"), stable_job_id(None, "t", "child:1")}
    assert comparison.startswith("cmp_")
    assert len(children) == 2
    assert all(child.startswith("job_") for child in children)
    assert comparison not in children


def test_identity_does_not_leak_or_collide_on_separator_tricks() -> None:
    assert stable_job_id("a|b", "c", "single") != stable_job_id("a", "b|c", "single")
    assert "scope" not in stable_job_id("scope", "task", "single")


def test_fingerprint_is_canonical_and_query_sensitive() -> None:
    base = request_fingerprint("q", ["gemini"])
    assert base == request_fingerprint("q", ["gemini"])
    assert base != request_fingerprint("q2", ["gemini"])
    assert base != request_fingerprint("q", ["gemini_max"])
    assert base != request_fingerprint("q", ["gemini", "gemini_max"])
    assert "q" not in base


def test_registry_exposes_exactly_three_selectors() -> None:
    assert ENGINE_IDS == ("gemini", "gemini_max", "local_deep_research")


def test_unknown_engine_is_rejected_before_any_construction() -> None:
    with pytest.raises(ValueError, match="Unknown research engine"):
        get_engine("not-an-engine")


def test_discovery_is_local_only_and_needs_no_gemini_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_RESEARCH_LDR_COMMAND", raising=False)
    descriptors = {d.id: d for d in describe_engines()}
    assert set(descriptors) == set(ENGINE_IDS)
    assert descriptors["gemini"].available is False
    assert "GEMINI_API_KEY" in (descriptors["gemini"].availability_reason or "")
    assert descriptors["local_deep_research"].available is False
    assert "GEMINI_RESEARCH_LDR_COMMAND" in (
        descriptors["local_deep_research"].availability_reason or ""
    )
    assert descriptors["local_deep_research"].recovery == "none"
    assert descriptors["gemini"].recovery == "provider_id"
    json.dumps([d.to_dict() for d in descriptors.values()])


def test_discovery_reports_gemini_available_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-not-a-key")
    monkeypatch.delenv("DEEP_RESEARCH_AGENT", raising=False)
    descriptors = {d.id: d for d in describe_engines()}
    assert descriptors["gemini"].available is True
    assert descriptors["gemini"].model == "deep-research-preview-04-2026"
    assert descriptors["gemini_max"].model == "deep-research-max-preview-04-2026"


def test_legacy_agent_override_is_labelled_for_standard_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-not-a-key")
    monkeypatch.setenv("DEEP_RESEARCH_AGENT", "pro")
    descriptors = {d.id: d for d in describe_engines()}
    assert descriptors["gemini"].model == "deep-research-pro-preview-12-2025"
    assert descriptors["gemini_max"].model == "deep-research-max-preview-04-2026"


def test_factories_return_request_owned_adapters() -> None:
    assert get_engine("gemini") is not get_engine("gemini")


def test_report_and_source_roundtrip_without_loss() -> None:
    report = EngineReport(
        text="  exact text\n",
        status="completed",
        sources=(
            ResearchSource(uri="https://example.com", title="Example", provenance="report_parsed"),
        ),
        usage=None,
        model=None,
        provider_id="int-1",
        warnings=("w",),
    )
    clone = EngineReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert clone == report
    assert clone.usage is None and clone.model is None


def test_job_status_values_match_public_contract() -> None:
    assert {s.value for s in JobStatus} == {
        "queued",
        "running",
        "completed",
        "failed",
        "interrupted",
        "cancelled",
        "expired",
    }


def test_engine_error_is_structured() -> None:
    error = EngineError("LOCAL_ENGINE_FAILED", "synthetic")
    assert error.code == "LOCAL_ENGINE_FAILED"
    assert error.to_dict()["code"] == "LOCAL_ENGINE_FAILED"


# --- FastMCP Task prerequisites -------------------------------------------------


async def test_task_tool_result_propagates_structured_content_and_scope() -> None:
    from fastmcp import Client, FastMCP
    from fastmcp.utilities.tasks import TaskConfig
    from fastmcp_tasks import TasksExtension, call_tool_task
    from mcp.types import TextContent

    from gemini_research_mcp.engines import current_execution_identity

    server = FastMCP("probe")
    server.add_extension(TasksExtension(name="probe-task-contract"))

    @server.tool(task=TaskConfig(mode="required"))
    async def probe(query: str) -> ToolResult:
        identity = current_execution_identity()
        return ToolResult(
            content=[TextContent(type="text", text="human text")],
            structured_content={
                "result": "human text",
                "payload": {
                    "has_task_id": bool(identity.task_id),
                    "scope": identity.task_scope,
                    "redelivered": identity.redelivered,
                },
            },
        )

    async with Client(server, timeout=15) as client:
        handle = await call_tool_task(client, "probe", {"query": "q"})
        await handle.wait(timeout=10)
        result = await handle.result()
    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["result"] == "human text"
    assert result.structured_content["payload"] == {
        "has_task_id": True,
        "scope": None,
        "redelivered": False,
    }


async def test_task_error_result_surfaces_as_tool_error() -> None:
    from fastmcp import Client, FastMCP
    from fastmcp.exceptions import ToolError
    from fastmcp.utilities.tasks import TaskConfig
    from fastmcp_tasks import TasksExtension, call_tool_task
    from mcp.types import TextContent

    server = FastMCP("probe")
    server.add_extension(TasksExtension(name="probe-task-error"))

    @server.tool(task=TaskConfig(mode="required"))
    async def failing(query: str) -> ToolResult:
        return ToolResult(
            content=[TextContent(type="text", text="job_x FAILED: synthetic")],
            structured_content={"result": "job_x FAILED: synthetic"},
            is_error=True,
        )

    async with Client(server, timeout=15) as client:
        handle = await call_tool_task(client, "failing", {"query": "q"})
        await handle.wait(timeout=10)
        with pytest.raises(ToolError, match="job_x FAILED"):
            await handle.result()


def test_owner_scope_uses_the_same_resolver_as_task_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastmcp_tasks.context import get_task_scope

    from gemini_research_mcp.engines import current_owner_scope

    assert current_owner_scope() is None
    token = SimpleNamespace(client_id="client-1", claims={"sub": "user-9"})
    monkeypatch.setattr("fastmcp.server.dependencies.get_access_token", lambda: token)
    assert current_owner_scope() == get_task_scope() == "client-1|user-9"


def test_identity_reports_redelivery_from_docket_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from docket.dependencies import current_execution

    import gemini_research_mcp.engines.base as base

    monkeypatch.setattr(
        base,
        "get_task_context",
        lambda: SimpleNamespace(task_id="opaque-id", task_scope="s"),
    )
    marker = current_execution.set(SimpleNamespace(redelivered=True, attempt=2))
    try:
        identity = base.current_execution_identity()
    finally:
        current_execution.reset(marker)
    assert (identity.task_id, identity.task_scope, identity.redelivered) == (
        "opaque-id",
        "s",
        True,
    )


def test_identity_requires_a_task_context(monkeypatch: pytest.MonkeyPatch) -> None:
    import gemini_research_mcp.engines.base as base

    monkeypatch.setattr(base, "get_task_context", lambda: None)
    with pytest.raises(EngineError) as exc:
        base.current_execution_identity()
    assert exc.value.code == "TASK_REQUIRED"


@pytest.mark.parametrize("command", ["ldr-mcp", "./ldr-mcp", "  "])
def test_local_engine_requires_absolute_command(
    monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    monkeypatch.setenv("GEMINI_RESEARCH_LDR_COMMAND", command)
    descriptor = {d.id: d for d in describe_engines()}["local_deep_research"]
    assert descriptor.available is False


def test_local_engine_is_available_with_absolute_command_without_launching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_RESEARCH_LDR_COMMAND", "/opt/ldr/bin/ldr-mcp")
    monkeypatch.setenv("LDR_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LDR_LLM_MODEL", "llama3:8b")
    monkeypatch.setenv("LDR_LLM_OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("LDR_SEARCH_TOOL", "searxng")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    descriptor = {d.id: d for d in describe_engines()}["local_deep_research"]
    assert descriptor.available is True
