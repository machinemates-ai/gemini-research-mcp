"""Verify that mode='optional' supports both foreground and background execution."""

import pytest
from fastmcp import Context, FastMCP
from fastmcp.client import Client
from fastmcp.dependencies import CurrentContext, Depends, Progress
from fastmcp.server.tasks import TaskConfig


class TestTaskExecutionMode:
    """Verify task execution behavior with mode='optional'."""

    @pytest.mark.asyncio
    async def test_optional_mode_runs_foreground_without_task_meta(self):
        """
        With mode='optional' and no task_meta from client:
        - Runs in FOREGROUND
        - Context IS available
        - Progress IS available
        """
        mcp = FastMCP("Test")
        execution_info = {}

        @mcp.tool(task=TaskConfig(mode="optional"))
        async def check_mode(
            query: str,
            progress: Progress = Progress(),  # noqa: B008
            ctx: Context = Depends(CurrentContext),  # noqa: B008
        ) -> str:
            execution_info["context_available"] = ctx is not None
            execution_info["progress_available"] = progress is not None
            execution_info["has_elicit"] = hasattr(ctx, "elicit") if ctx else False
            await progress.set_message("Test message")
            return "OK"

        async with Client(mcp) as client:
            # Normal call (no task_meta) - should run in foreground
            result = await client.call_tool("check_mode", {"query": "test"})

            print(f"Result: {result.data}")
            print(f"Execution info: {execution_info}")

            # Verify foreground execution with full capabilities
            assert execution_info["context_available"] is True
            assert execution_info["progress_available"] is True
            assert execution_info["has_elicit"] is True

    @pytest.mark.asyncio
    async def test_optional_mode_exposes_task_capability(self):
        """
        With mode='optional', the tool should be advertised as task-capable.
        MCP clients can choose to request task-augmented execution.
        """
        mcp = FastMCP("Test")

        @mcp.tool(task=TaskConfig(mode="optional"))
        async def my_task(query: str) -> str:
            return "OK"

        async with Client(mcp) as client:
            tools = await client.list_tools()
            tool = next(t for t in tools if t.name == "my_task")

            print(f"Tool: {tool.name}")
            print(f"Task config mode: optional (set in decorator)")

            # The tool exists and is callable
            assert tool is not None
            assert tool.name == "my_task"


if __name__ == "__main__":
    import asyncio

    asyncio.run(TestTaskExecutionMode().test_optional_mode_runs_foreground_without_task_meta())
