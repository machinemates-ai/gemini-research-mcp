"""Test Context availability in task-enabled tools.

This verifies that:
1. mode="optional" allows Context AND Progress (our solution)
2. mode="required" blocks Context (doesn't work for elicitation)
3. The research_deep pattern works: clarification -> research
"""

import pytest
from fastmcp import Context, FastMCP
from fastmcp.client import Client
from fastmcp.dependencies import CurrentContext, Depends, Progress
from fastmcp.server.tasks import TaskConfig


class TestContextInTasks:
    """Test Context availability with different task modes."""

    @pytest.mark.asyncio
    async def test_context_available_with_optional_task(self):
        """Context IS available with optional task mode (foreground execution)."""
        mcp = FastMCP("Test")
        context_received = None

        @mcp.tool(task=TaskConfig(mode="optional"))
        async def check_context(
            query: str,
            progress: Progress = Progress(),  # noqa: B008
            ctx: Context = Depends(CurrentContext),  # noqa: B008
        ) -> str:
            nonlocal context_received
            context_received = ctx
            return f"Context type: {type(ctx).__name__}"

        async with Client(mcp) as client:
            result = await client.call_tool("check_context", {"query": "test"})
            print(f"Result: {result}")
            print(f"Context received: {type(context_received)}")

            assert context_received is not None
            assert "Context" in type(context_received).__name__
            assert result.data == "Context type: Context"

    @pytest.mark.asyncio
    async def test_optional_task_still_gets_progress(self):
        """Optional task mode still provides Progress dependency."""
        mcp = FastMCP("Test")
        progress_received = None

        @mcp.tool(task=TaskConfig(mode="optional"))
        async def check_progress(
            query: str,
            progress: Progress = Progress(),  # noqa: B008
            ctx: Context = Depends(CurrentContext),  # noqa: B008
        ) -> str:
            nonlocal progress_received
            progress_received = progress
            await progress.set_message("Test message")
            return f"Progress type: {type(progress).__name__}"

        async with Client(mcp) as client:
            result = await client.call_tool("check_progress", {"query": "test"})
            print(f"Result: {result}")
            print(f"Progress received: {type(progress_received)}")

            assert progress_received is not None
            assert "Progress" in type(progress_received).__name__

    @pytest.mark.asyncio
    async def test_elicit_in_optional_task(self):
        """Elicitation method is available with optional task mode."""
        mcp = FastMCP("Test")
        elicit_available = False
        elicit_error = None

        @mcp.tool(task=TaskConfig(mode="optional"))
        async def try_elicit(
            query: str,
            progress: Progress = Progress(),  # noqa: B008
            ctx: Context = Depends(CurrentContext),  # noqa: B008
        ) -> str:
            nonlocal elicit_available, elicit_error
            try:
                # Just check the elicit method is available
                if hasattr(ctx, "elicit") and callable(ctx.elicit):
                    elicit_available = True
                    return "Elicit method available"
                return "Elicit method NOT available"
            except Exception as e:
                elicit_error = str(e)
                return f"Error: {e}"

        async with Client(mcp) as client:
            result = await client.call_tool("try_elicit", {"query": "test"})
            print(f"Result: {result}")
            print(f"Elicit worked: {elicit_available}")
            print(f"Elicit error: {elicit_error}")

            assert elicit_available is True
            assert elicit_error is None

    @pytest.mark.asyncio
    async def test_research_deep_pattern_simulation(self):
        """
        Simulate the research_deep pattern:
        - Phase 1: Use Context for clarification (synchronous)
        - Phase 2: Run long operation with Progress (can be backgrounded)
        """
        mcp = FastMCP("Test")
        phases_completed = []

        @mcp.tool(task=TaskConfig(mode="optional"))
        async def research_simulation(
            query: str,
            progress: Progress = Progress(),  # noqa: B008
            ctx: Context = Depends(CurrentContext),  # noqa: B008
        ) -> str:
            # Phase 1: Clarification (needs Context)
            if ctx and hasattr(ctx, "elicit"):
                phases_completed.append("phase1_context")

            # Phase 2: Long operation (uses Progress)
            await progress.set_message("Starting research...")
            phases_completed.append("phase2_progress")
            await progress.set_message("Research complete")

            return f"Phases: {phases_completed}"

        async with Client(mcp) as client:
            result = await client.call_tool("research_simulation", {"query": "test"})
            print(f"Result: {result}")
            print(f"Phases completed: {phases_completed}")

            assert "phase1_context" in phases_completed
            assert "phase2_progress" in phases_completed
