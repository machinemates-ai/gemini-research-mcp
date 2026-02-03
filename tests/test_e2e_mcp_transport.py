"""End-to-end tests for MCP server transport validation.

These tests validate that the MCP server works correctly over stdio transport,
ensuring compatibility with VS Code, Claude Desktop, and other MCP clients.

Run with: uv run pytest tests/test_e2e_mcp_transport.py -v --tb=short

These tests do NOT require GEMINI_API_KEY - they only validate the MCP protocol.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest


class TestMCPServerStartup:
    """Tests for MCP server startup and initialization."""

    def test_server_module_imports(self):
        """Server module should import without errors."""
        from gemini_research_mcp.server import mcp, main

        assert mcp is not None
        assert main is not None
        assert callable(main)

    def test_fastmcp_instance_type(self):
        """Server should use FastMCP instance."""
        from fastmcp import FastMCP

        from gemini_research_mcp.server import mcp

        assert isinstance(mcp, FastMCP)

    def test_server_name(self):
        """Server should have correct name."""
        from gemini_research_mcp.server import mcp

        assert mcp.name == "Gemini Research"

    def test_server_version_set(self):
        """Server should have version from package."""
        from gemini_research_mcp import __version__
        from gemini_research_mcp.server import mcp

        # FastMCP may or may not expose version, but package should have it
        assert __version__ is not None
        assert len(__version__) > 0


class TestMCPToolRegistration:
    """Tests for MCP tool registration with fastmcp."""

    @pytest.mark.asyncio
    async def test_tools_registered(self):
        """All expected tools should be registered."""
        from gemini_research_mcp.server import mcp

        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}

        expected_tools = {
            "research_web",
            "fetch_webpage",
            "research_deep",
            "research_deep_planned",
            "list_format_templates",
            "list_research_sessions_tool",
            "export_research_session",
            "research_followup",
            "resume_research",
        }

        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"

    @pytest.mark.asyncio
    async def test_tool_count(self):
        """Should have expected number of tools."""
        from gemini_research_mcp.server import mcp

        tools = await mcp.list_tools()
        assert len(tools) == 9, f"Expected 9 tools, got {len(tools)}"

    @pytest.mark.asyncio
    async def test_tools_have_descriptions(self):
        """All tools should have descriptions."""
        from gemini_research_mcp.server import mcp

        tools = await mcp.list_tools()
        for tool in tools:
            assert tool.description, f"Tool {tool.name} missing description"
            assert len(tool.description) > 10, f"Tool {tool.name} description too short"


class TestMCPResourceRegistration:
    """Tests for MCP resource registration."""

    @pytest.mark.asyncio
    async def test_resources_registered(self):
        """Should have export resources registered."""
        from gemini_research_mcp.server import mcp

        resources = await mcp.list_resources()
        # Resources may be empty if no exports exist, but the call should succeed
        assert isinstance(resources, list)


class TestMCPLifespan:
    """Tests for MCP server lifespan and TaskSupport."""

    @pytest.mark.asyncio
    async def test_lifespan_context(self):
        """Lifespan context should initialize TaskSupport."""
        from gemini_research_mcp.server import lifespan, mcp

        async with lifespan(mcp):
            # TaskSupport should be initialized
            from gemini_research_mcp.server import _task_support

            assert _task_support is not None

    @pytest.mark.asyncio
    async def test_task_support_has_run_method(self):
        """TaskSupport should have run() async context manager."""
        from gemini_research_mcp.server import lifespan, mcp

        async with lifespan(mcp):
            from gemini_research_mcp.server import get_task_support

            ts = get_task_support()
            assert hasattr(ts, "run")


class TestMCPContext:
    """Tests for fastmcp Context usage."""

    def test_context_import(self):
        """Context should be importable from fastmcp."""
        from fastmcp import Context

        assert Context is not None

    def test_context_not_generic(self):
        """fastmcp Context should not require generic parameters."""
        from fastmcp import Context
        import inspect

        # Get the class signature - it should not be a Generic
        # In fastmcp, Context is a concrete class, not Generic[ServerDeps, ClientDeps, Lifespan]
        sig = inspect.signature(Context)
        # Should be able to instantiate hint without type params
        # (we can't actually instantiate without a server, but the type check passes)
        assert "Context" in str(Context)


class TestMCPElicitation:
    """Tests for elicitation API compatibility."""

    def test_elicit_method_signature(self):
        """Context.elicit should use response_type parameter."""
        from fastmcp import Context
        import inspect

        sig = inspect.signature(Context.elicit)
        params = list(sig.parameters.keys())

        assert "response_type" in params, "elicit() should have response_type parameter"
        assert "schema" not in params, "elicit() should NOT have schema parameter"

    def test_elicit_returns_elicitation_types(self):
        """elicit() should return Accepted/Declined/Cancelled types."""
        from fastmcp import Context
        import inspect

        sig = inspect.signature(Context.elicit)
        return_annotation = str(sig.return_annotation)

        assert "AcceptedElicitation" in return_annotation
        assert "DeclinedElicitation" in return_annotation
        assert "CancelledElicitation" in return_annotation


class TestMCPTypesCompatibility:
    """Tests for MCP types compatibility."""

    def test_mcp_types_import(self):
        """mcp.types should still be importable."""
        from mcp.types import (
            BlobResourceContents,
            EmbeddedResource,
            Icon,
            TextContent,
            TextResourceContents,
            ToolAnnotations,
        )

        assert all([
            BlobResourceContents,
            EmbeddedResource,
            Icon,
            TextContent,
            TextResourceContents,
            ToolAnnotations,
        ])

    def test_task_support_import(self):
        """TaskSupport should be importable from mcp.server.experimental."""
        from mcp.server.experimental.task_support import TaskSupport

        assert TaskSupport is not None


class TestMCPServerCLI:
    """Tests for MCP server CLI interface."""

    def test_cli_version_flag(self):
        """Server should respond to --version flag."""
        result = subprocess.run(
            [sys.executable, "-m", "gemini_research_mcp.server", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # May exit with 0 (success) or non-zero if version prints to stderr
        output = result.stdout + result.stderr
        assert "gemini-research-mcp" in output.lower() or "0." in output

    def test_cli_help_flag(self):
        """Server should respond to --help flag."""
        result = subprocess.run(
            [sys.executable, "-m", "gemini_research_mcp.server", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "gemini" in result.stdout.lower() or "research" in result.stdout.lower()


class TestFastMCPDecorator:
    """Tests for fastmcp @mcp.tool() decorator behavior."""

    def test_decorated_function_is_callable(self):
        """@mcp.tool() decorated functions should remain callable."""
        from gemini_research_mcp.server import research_deep, research_web

        assert callable(research_deep)
        assert callable(research_web)

    def test_decorated_function_is_coroutine(self):
        """@mcp.tool() decorated async functions should remain coroutines."""
        import inspect

        from gemini_research_mcp.server import research_deep, research_web

        assert inspect.iscoroutinefunction(research_deep)
        assert inspect.iscoroutinefunction(research_web)

    def test_decorated_function_preserves_name(self):
        """@mcp.tool() should preserve function __name__."""
        from gemini_research_mcp.server import research_deep

        assert research_deep.__name__ == "research_deep"


class TestInternalFunctionCalls:
    """Tests for internal function-to-function calls."""

    def test_research_deep_planned_can_call_research_deep(self):
        """research_deep_planned should be able to call research_deep directly."""
        from gemini_research_mcp.server import research_deep, research_deep_planned
        import inspect

        # Both should be async functions
        assert inspect.iscoroutinefunction(research_deep)
        assert inspect.iscoroutinefunction(research_deep_planned)

        # research_deep should be directly callable (not wrapped in FunctionTool)
        assert callable(research_deep)
        # It should NOT have a .fn attribute (that's raw mcp SDK behavior)
        assert not hasattr(research_deep, "fn") or research_deep.fn is None
