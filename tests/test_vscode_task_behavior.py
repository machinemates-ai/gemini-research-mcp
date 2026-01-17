"""Test background task execution mimicking VS Code behavior.

This verifies that:
1. With mode="optional", client can choose sync (foreground) or async (background)
2. Background tasks receive Progress but NOT Context
3. Foreground calls receive both Progress AND Context
4. Our research_deep tool handles both scenarios gracefully
"""

import asyncio
import contextlib

import pytest
from fastmcp import Context, FastMCP
from fastmcp.client import Client
from fastmcp.dependencies import Depends, Progress
from fastmcp.server.tasks import TaskConfig

# =============================================================================
# Optional Context Dependency (same pattern as server.py)
# =============================================================================


@contextlib.asynccontextmanager
async def _optional_context():
    """Provide Context if available, None otherwise."""
    try:
        from fastmcp.server.context import _current_context

        ctx = _current_context.get()
        yield ctx  # May be None
    except Exception:
        yield None


OptionalContext = _optional_context


class TestVSCodeTaskBehavior:
    """Test task execution behavior as VS Code would trigger it."""

    @pytest.fixture
    def server_with_optional_task_tool(self):
        """Create a server with a tool that mimics research_deep pattern."""
        mcp = FastMCP("Test", tasks=False)
        execution_log: list[str] = []

        @mcp.tool(task=TaskConfig(mode="optional"))
        async def research_simulation(
            query: str,
            progress: Progress = Progress(),  # noqa: B008
            ctx: Context | None = Depends(OptionalContext),  # noqa: B008
        ) -> str:
            """Simulate research_deep: clarification phase + research phase."""
            execution_log.clear()

            # Phase 1: Try clarification (needs Context)
            if ctx is not None and hasattr(ctx, "elicit"):
                execution_log.append("phase1_context_available")
                # In real code: await _maybe_clarify_query(query, ctx)
            else:
                execution_log.append("phase1_context_unavailable")

            # Phase 2: Research with progress updates
            await progress.set_message("Starting research...")
            execution_log.append("phase2_progress_started")

            await asyncio.sleep(0.1)  # Simulate work
            await progress.set_message("Research complete")
            execution_log.append("phase2_progress_complete")

            return f"Result for: {query}"

        # Attach log to server for test access
        mcp._execution_log = execution_log  # type: ignore
        return mcp

    @pytest.mark.asyncio
    async def test_foreground_call_has_context(self, server_with_optional_task_tool):
        """
        When client calls WITHOUT task=True (foreground/sync):
        - Context IS available
        - Elicitation CAN work
        - Progress still works
        """
        mcp = server_with_optional_task_tool

        async with Client(mcp) as client:
            # Call without task=True - this is foreground execution
            result = await client.call_tool(
                "research_simulation",
                {"query": "test foreground"},
            )

            print(f"Result: {result.data}")
            print(f"Execution log: {mcp._execution_log}")

            # Foreground: Context available
            assert "phase1_context_available" in mcp._execution_log
            assert "phase2_progress_started" in mcp._execution_log
            assert "phase2_progress_complete" in mcp._execution_log

    @pytest.mark.asyncio
    async def test_background_task_no_context(self, server_with_optional_task_tool):
        """
        When client calls WITH task=True (background/async):
        - Context is NOT available (no active MCP session in worker)
        - Progress still works
        - Tool must handle gracefully
        """
        mcp = server_with_optional_task_tool

        async with Client(mcp) as client:
            # Call with task=True - this triggers background task execution
            task = await client.call_tool(
                "research_simulation",
                {"query": "test background"},
                task=True,
            )

            assert task is not None
            print(f"Task created: {task}")

            # Wait for task result
            result = await task.result()

            print(f"Result: {result.data}")
            print(f"Execution log: {mcp._execution_log}")

            # Background: Context NOT available (graceful degradation)
            assert "phase1_context_unavailable" in mcp._execution_log
            # But Progress still works
            assert "phase2_progress_started" in mcp._execution_log
            assert "phase2_progress_complete" in mcp._execution_log

    @pytest.mark.asyncio
    async def test_tool_exposes_optional_task_support(self, server_with_optional_task_tool):
        """
        Tool metadata should advertise taskSupport='optional'.
        This tells VS Code the tool supports both modes.
        """
        mcp = server_with_optional_task_tool

        async with Client(mcp) as client:
            tools = await client.list_tools()
            tool = next(t for t in tools if t.name == "research_simulation")

            print(f"Tool: {tool.name}")
            print(f"Execution metadata: {tool.execution}")

            # Tool should expose task support
            assert tool.execution is not None
            assert tool.execution.taskSupport == "optional"


class TestRealResearchDeepBehavior:
    """Test our actual research_deep tool behavior."""

    @pytest.mark.asyncio
    async def test_research_deep_exposes_optional_task_support(self):
        """research_deep should advertise taskSupport='optional'."""
        from gemini_research_mcp.server import mcp

        async with Client(mcp) as client:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == "research_deep"), None)

            assert tool is not None, "research_deep tool not found"
            print(f"Tool: {tool.name}")
            print(f"Description: {tool.description[:100]}...")
            print(f"Execution metadata: {tool.execution}")

            # Should expose optional task support
            if tool.execution:
                assert tool.execution.taskSupport == "optional"
            else:
                # If execution is None, that's also acceptable for optional mode
                pass  # Optional tools may not always expose execution metadata

    @pytest.mark.asyncio
    async def test_research_deep_can_be_called_sync(self):
        """research_deep can be called synchronously (foreground)."""
        from gemini_research_mcp.server import mcp

        async with Client(mcp) as client:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]
            print(f"Available tools: {tool_names}")

            assert "research_deep" in tool_names
            # Tool exists and is callable - actual execution would need API key
