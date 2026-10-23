"""Exercise memory Task loss and persisted job state across real MCP processes."""

import os

import pytest
from fastmcp_tasks import ToolTask, call_tool_task
from mcp.shared.exceptions import MCPError
from test_research_live_e2e import (
    QUERY,
    Acceptance,
    checked,
)
from test_research_live_e2e import (
    test_00_http_principal_and_local_opt_in as check_http_guards,
)


async def test_memory_restart_loses_task_but_retains_interrupted_job(tmp_path) -> None:
    acceptance = Acceptance(tmp_path, {**os.environ, "GEMINI_API_KEY": "synthetic-not-a-key"})
    async with acceptance.stdio("restart", synthetic=True) as client:
        handle = await call_tool_task(client, "research_run", {"query": QUERY})
        run = await acceptance.wait_job(client, linked=True)
        creation = handle.create_result
        assert run["status"] == "running"
        acceptance.kill_server("restart")

    async with acceptance.stdio("restart", synthetic=True) as client:
        with pytest.raises(MCPError, match=r"(?i)task.*not found"):
            await ToolTask(client, "research_run", creation).status()
        restored = checked(
            await client.call_tool("get_research_run", {"run_id": run["run_id"]})
        ).structured_content["run"]
        assert restored["status"] == "interrupted"
        assert restored["provider_id"] == run["provider_id"]
        assert restored["expires_at"] == run["expires_at"]
        assert acceptance.events("restart") == []


async def test_real_http_guards_before_provider_dispatch(tmp_path) -> None:
    acceptance = Acceptance(tmp_path, {**os.environ, "GEMINI_API_KEY": "synthetic-not-a-key"})
    await check_http_guards(acceptance)
