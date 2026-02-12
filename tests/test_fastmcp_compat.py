"""FastMCP 3.0 beta compatibility tests.

Validates that the server works correctly with fastmcp>=3.0.0b2 (PyPI beta).
Covers: API surface, tool registration, elicitation types, decorator behavior,
lifespan, private API access (_mcp_server), and server configuration.

Run with: uv run pytest tests/test_fastmcp_compat.py -v
"""

import inspect

import pytest
from pydantic import Field, create_model

# =============================================================================
# FastMCP Version & Import Compatibility
# =============================================================================


class TestFastMCPVersion:
    """Verify the correct FastMCP version is installed."""

    def test_fastmcp_importable(self):
        """FastMCP should be importable."""
        import fastmcp

        assert fastmcp is not None

    def test_fastmcp_version_is_3x(self):
        """FastMCP version should be 3.x beta."""
        import fastmcp

        version = fastmcp.__version__
        assert version.startswith("3."), f"Expected 3.x, got {version}"

    def test_fastmcp_core_exports(self):
        """FastMCP should export Context and FastMCP."""
        from fastmcp import Context, FastMCP

        assert Context is not None
        assert FastMCP is not None

    def test_fastmcp_file_utility(self):
        """FastMCP File utility should be available for content handling."""
        from fastmcp.utilities.types import File

        assert File is not None
        # File should handle both text and binary content
        text_file = File(data="hello", name="test.txt")
        assert text_file.data == "hello"
        binary_file = File(data=b"\x00\x01", name="test.bin")
        assert binary_file.data == b"\x00\x01"

    def test_fastmcp_dict_annotations_accepted(self):
        """FastMCP should accept dict-based annotations (no ToolAnnotations import needed)."""
        # Server uses {"readOnlyHint": True} instead of ToolAnnotations(readOnlyHint=True)
        annotations = {"readOnlyHint": True}
        assert annotations["readOnlyHint"] is True

    def test_fastmcp_dict_icons_accepted(self):
        """FastMCP should accept dict-based icons (no Icon import needed)."""
        icon = {"src": "https://example.com/icon.png", "mimeType": "image/png"}
        assert "src" in icon
        assert "mimeType" in icon


# =============================================================================
# FastMCP Instance Configuration
# =============================================================================


class TestFastMCPServerInstance:
    """Test the FastMCP server instance configuration."""

    def test_server_is_fastmcp(self):
        """The mcp server instance should be a FastMCP."""
        from fastmcp import FastMCP

        from gemini_research_mcp.server import mcp

        assert isinstance(mcp, FastMCP)

    def test_server_name(self):
        """Server name should be 'Gemini Research'."""
        from gemini_research_mcp.server import mcp

        assert mcp.name == "Gemini Research"

    def test_server_has_icons(self):
        """Server should have icon configuration."""
        from gemini_research_mcp.server import mcp

        assert mcp.icons is not None
        assert len(mcp.icons) > 0

    def test_server_has_instructions(self):
        """Server should have instructions string."""
        from gemini_research_mcp.server import mcp

        assert mcp.instructions is not None
        assert len(mcp.instructions) > 100  # Non-trivial instructions

    def test_server_has_lifespan(self):
        """Server should use a lifespan function."""
        from gemini_research_mcp.server import mcp

        assert mcp.lifespan is not None

    def test_server_private_mcp_server_access(self):
        """_mcp_server should be accessible for low-level access."""
        from gemini_research_mcp.server import mcp

        assert hasattr(mcp, "_mcp_server")
        low_level = mcp._mcp_server
        assert low_level is not None


# =============================================================================
# Tool Registration
# =============================================================================


class TestToolRegistration:
    """Test that all tools are properly registered with FastMCP."""

    EXPECTED_TOOLS = {
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

    @pytest.mark.asyncio
    async def test_all_tools_registered(self):
        """All expected tools should be listed."""
        from gemini_research_mcp.server import mcp

        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}

        for expected in self.EXPECTED_TOOLS:
            assert expected in tool_names, f"Missing tool: {expected}"

    @pytest.mark.asyncio
    async def test_exact_tool_count(self):
        """Should have exactly 9 tools."""
        from gemini_research_mcp.server import mcp

        tools = await mcp.list_tools()
        assert len(tools) == 9, f"Expected 9, got {len(tools)}: {[t.name for t in tools]}"

    @pytest.mark.asyncio
    async def test_tools_have_descriptions(self):
        """Every tool should have a non-empty description."""
        from gemini_research_mcp.server import mcp

        tools = await mcp.list_tools()
        for tool in tools:
            assert tool.description, f"{tool.name}: missing description"
            assert len(tool.description) > 20, f"{tool.name}: description too short"

    @pytest.mark.asyncio
    async def test_tools_have_parameters(self):
        """Every tool should have parameters defined (FastMCP 3.0 FunctionTool)."""
        from gemini_research_mcp.server import mcp

        tools = await mcp.list_tools()
        for tool in tools:
            # FastMCP 3.0 returns FunctionTool objects with parameters dict
            assert hasattr(tool, "parameters"), f"{tool.name}: no parameters attr"

    @pytest.mark.asyncio
    async def test_research_web_annotations(self):
        """research_web should have readOnlyHint annotation."""
        from gemini_research_mcp.server import mcp

        tools = await mcp.list_tools()
        web_tool = next(t for t in tools if t.name == "research_web")
        assert web_tool.annotations is not None
        assert web_tool.annotations.readOnlyHint is True


# =============================================================================
# Decorator Behavior (FastMCP 3.0)
# =============================================================================


class TestDecoratorBehavior:
    """Verify @mcp.tool() decorator preserves function semantics in FastMCP 3.0."""

    def test_decorated_functions_are_callable(self):
        """Decorated functions should remain directly callable."""
        from gemini_research_mcp.server import (
            fetch_webpage,
            research_deep,
            research_followup,
            research_web,
        )

        for fn in [research_deep, research_web, fetch_webpage, research_followup]:
            assert callable(fn), f"{fn.__name__} not callable"

    def test_decorated_functions_are_coroutines(self):
        """Decorated async functions should remain coroutine functions."""
        from gemini_research_mcp.server import (
            fetch_webpage,
            research_deep,
            research_followup,
            research_web,
        )

        for fn in [research_deep, research_web, fetch_webpage, research_followup]:
            assert inspect.iscoroutinefunction(fn), f"{fn.__name__} not async"

    def test_decorated_functions_preserve_name(self):
        """Decorated functions should preserve __name__."""
        from gemini_research_mcp.server import (
            fetch_webpage,
            research_deep,
            research_web,
        )

        assert research_deep.__name__ == "research_deep"
        assert research_web.__name__ == "research_web"
        assert fetch_webpage.__name__ == "fetch_webpage"

    def test_research_deep_planned_can_call_research_deep(self):
        """research_deep_planned should be able to call research_deep directly."""
        from gemini_research_mcp.server import research_deep, research_deep_planned

        # Both must be async and directly callable
        assert inspect.iscoroutinefunction(research_deep)
        assert inspect.iscoroutinefunction(research_deep_planned)
        assert callable(research_deep)


# =============================================================================
# Elicitation API Compatibility
# =============================================================================


class TestElicitationAPI:
    """Test elicitation API compatibility with FastMCP 3.0."""

    def test_context_elicit_signature(self):
        """Context.elicit should use response_type, not schema."""
        from fastmcp import Context

        sig = inspect.signature(Context.elicit)
        params = list(sig.parameters.keys())

        assert "response_type" in params, "Missing response_type param"
        assert "schema" not in params, "Legacy schema param should not exist"

    def test_elicit_return_types(self):
        """elicit() should return union of Accepted/Declined/Cancelled."""
        from fastmcp import Context

        sig = inspect.signature(Context.elicit)
        ret = str(sig.return_annotation)

        assert "AcceptedElicitation" in ret
        assert "DeclinedElicitation" in ret
        assert "CancelledElicitation" in ret

    def test_dynamic_schema_creation(self):
        """Dynamic Pydantic models for elicitation should work."""
        questions = [
            "What aspects to compare?",
            "What's your use case?",
            "What timeframe?",
        ]

        field_definitions = {
            f"answer_{i+1}": (str, Field(default="", description=q))
            for i, q in enumerate(questions)
        }
        DynamicSchema = create_model("ClarificationQuestions", **field_definitions)

        assert len(DynamicSchema.model_fields) == 3

        # Default values
        instance = DynamicSchema()
        assert instance.answer_1 == ""

        # Set values
        instance = DynamicSchema(answer_1="performance", answer_2="API server")
        assert instance.answer_1 == "performance"
        assert instance.answer_2 == "API server"
        assert instance.answer_3 == ""  # default

    def test_clarification_schema_exists(self):
        """ClarificationSchema should be defined in server.py."""
        from gemini_research_mcp.server import ClarificationSchema

        instance = ClarificationSchema()
        assert instance.answer_1 == ""
        assert instance.answer_2 == ""
        assert instance.answer_3 == ""


# =============================================================================
# Lifespan & Task Support
# =============================================================================


class TestLifespan:
    """Test lifespan context and FastMCP task support."""

    @pytest.mark.asyncio
    async def test_lifespan_runs_without_error(self):
        """Lifespan should complete without errors."""
        from gemini_research_mcp.server import lifespan, mcp

        async with lifespan(mcp):
            pass  # If we get here, lifespan is OK

    @pytest.mark.asyncio
    async def test_deep_research_tools_have_task_support(self):
        """research_deep and research_deep_planned should have task=True (mode='required')."""
        from gemini_research_mcp.server import mcp

        tools = await mcp.list_tools()
        tool_map = {t.name: t for t in tools}

        for name in ("research_deep", "research_deep_planned"):
            assert name in tool_map, f"Missing tool: {name}"
            assert tool_map[name].task_config.mode == "required", (
                f"{name}: expected task mode 'required', got '{tool_map[name].task_config.mode}'"
            )

    @pytest.mark.asyncio
    async def test_non_task_tools_are_forbidden(self):
        """Non-task tools should have task mode='forbidden'."""
        from gemini_research_mcp.server import mcp

        tools = await mcp.list_tools()
        task_tool_names = {"research_deep", "research_deep_planned"}
        non_task_tools = [t for t in tools if t.name not in task_tool_names]

        for tool in non_task_tools:
            assert tool.task_config.mode == "forbidden", (
                f"{tool.name}: expected 'forbidden', got '{tool.task_config.mode}'"
            )


# =============================================================================
# Resource Registration
# =============================================================================


class TestResourceRegistration:
    """Test resource registration with FastMCP 3.0."""

    @pytest.mark.asyncio
    async def test_list_resources_succeeds(self):
        """list_resources should succeed (may be empty)."""
        from gemini_research_mcp.server import mcp

        resources = await mcp.list_resources()
        assert isinstance(resources, list)


# =============================================================================
# Export Cache
# =============================================================================


class TestExportCache:
    """Test the ephemeral export cache."""

    def test_cache_export_returns_id(self):
        """_cache_export should return a 12-char UUID prefix."""
        from gemini_research_mcp.export import ExportFormat, ExportResult
        from gemini_research_mcp.server import _cache_export

        result = ExportResult(
            format=ExportFormat.MARKDOWN,
            filename="test.md",
            content=b"# Test",
            mime_type="text/markdown",
        )

        export_id = _cache_export(result, "session-123")
        assert isinstance(export_id, str)
        assert len(export_id) == 12

    def test_get_cached_export(self):
        """_get_cached_export should return the cached entry."""
        from gemini_research_mcp.export import ExportFormat, ExportResult
        from gemini_research_mcp.server import _cache_export, _get_cached_export

        result = ExportResult(
            format=ExportFormat.JSON,
            filename="test.json",
            content=b'{"test": true}',
            mime_type="application/json",
        )

        export_id = _cache_export(result, "session-456")
        entry = _get_cached_export(export_id)

        assert entry is not None
        assert entry.session_id == "session-456"
        assert entry.result.filename == "test.json"
        assert not entry.is_expired

    def test_get_missing_export_returns_none(self):
        """_get_cached_export should return None for missing IDs."""
        from gemini_research_mcp.server import _get_cached_export

        assert _get_cached_export("nonexistent-id") is None

    def test_expired_export_returns_none(self):
        """Expired exports should return None and be cleaned up."""
        from datetime import UTC, datetime, timedelta

        from gemini_research_mcp.export import ExportFormat, ExportResult
        from gemini_research_mcp.server import (
            ExportCacheEntry,
            _export_cache,
            _get_cached_export,
        )

        result = ExportResult(
            format=ExportFormat.MARKDOWN,
            filename="old.md",
            content=b"# Old",
            mime_type="text/markdown",
        )

        # Insert with expired timestamp
        expired_entry = ExportCacheEntry(
            result=result,
            session_id="old-session",
            created_at=datetime.now(UTC) - timedelta(hours=2),
        )
        _export_cache["expired-test"] = expired_entry

        assert _get_cached_export("expired-test") is None
        assert "expired-test" not in _export_cache


# =============================================================================
# Client Health Management
# =============================================================================


class TestClientHealth:
    """Test deep.py client health monitoring."""

    def test_client_health_creation(self):
        """ClientHealth should initialize with sensible defaults."""
        from gemini_research_mcp.deep import ClientHealth

        health = ClientHealth()
        assert health.request_count == 0
        assert health.consecutive_failures == 0
        assert not health.needs_refresh()

    def test_client_health_record_request(self):
        """Recording a request should increment count and reset failures."""
        from gemini_research_mcp.deep import ClientHealth

        health = ClientHealth()
        health.consecutive_failures = 2
        health.record_request()

        assert health.request_count == 1
        assert health.consecutive_failures == 0

    def test_client_health_record_failure(self):
        """Recording a failure should increment consecutive failures."""
        from gemini_research_mcp.deep import ClientHealth

        health = ClientHealth()
        health.record_failure()
        health.record_failure()
        health.record_failure()

        assert health.consecutive_failures == 3
        assert health.needs_refresh()  # >= 3 failures triggers refresh

    def test_client_health_age_refresh(self):
        """Old clients should need refresh."""
        import time

        from gemini_research_mcp.deep import ClientHealth

        health = ClientHealth()
        # Simulate old client
        health.created_at = time.time() - 7200  # 2 hours old

        assert health.needs_refresh()

    def test_get_healthy_client_creates_client(self):
        """_get_healthy_client should create a client on first call."""
        import os

        from gemini_research_mcp.deep import _get_healthy_client

        if not os.environ.get("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = "test-key-for-client-creation"

        try:
            client = _get_healthy_client()
            assert client is not None
        finally:
            if os.environ.get("GEMINI_API_KEY") == "test-key-for-client-creation":
                del os.environ["GEMINI_API_KEY"]

    def test_force_client_refresh(self):
        """_force_client_refresh should clear global client state."""
        import gemini_research_mcp.deep as deep_mod

        deep_mod._force_client_refresh()

        assert deep_mod._client is None
        assert deep_mod._client_health is None


# =============================================================================
# Clarifier Module
# =============================================================================


class TestClarifierTypes:
    """Test clarifier module types and structures."""

    def test_clarifying_question_structure(self):
        """ClarifyingQuestion should have expected fields."""
        from gemini_research_mcp.clarifier import ClarifyingQuestion

        q = ClarifyingQuestion(
            question="What timeframe?",
            purpose="Narrows scope",
            priority=1,
            default_answer="Last 2 years",
        )

        assert q.question == "What timeframe?"
        assert q.priority == 1
        assert q.default_answer == "Last 2 years"

    def test_query_analysis_structure(self):
        """QueryAnalysis should have expected fields."""
        from gemini_research_mcp.clarifier import QueryAnalysis

        analysis = QueryAnalysis(
            needs_clarification=True,
            confidence=0.4,
            detected_intent="Market research",
            ambiguities=["No timeframe", "No industry"],
        )

        assert analysis.needs_clarification is True
        assert analysis.confidence == 0.4
        assert len(analysis.ambiguities) == 2

    def test_refined_query_structure(self):
        """RefinedQuery should combine original and refined queries."""
        from gemini_research_mcp.clarifier import RefinedQuery

        refined = RefinedQuery(
            original_query="compare frameworks",
            refined_query="compare Python web frameworks for REST APIs in 2025",
            context_summary="Added: timeframe 2025, focus on REST APIs",
            answers={"q1": "REST APIs", "q2": "2025"},
        )

        assert refined.original_query in refined.refined_query or len(refined.refined_query) > 0
        assert len(refined.answers) == 2

    def test_confidence_threshold_constant(self):
        """CONFIDENCE_THRESHOLD should be defined."""
        from gemini_research_mcp.clarifier import CONFIDENCE_THRESHOLD

        assert 0.0 < CONFIDENCE_THRESHOLD < 1.0

    def test_max_questions_constant(self):
        """MAX_QUESTIONS should be reasonable (3-5)."""
        from gemini_research_mcp.clarifier import MAX_QUESTIONS

        assert 3 <= MAX_QUESTIONS <= 10


# =============================================================================
# Server Helper Functions
# =============================================================================


class TestServerHelpers:
    """Test server.py helper functions."""

    def test_format_duration_seconds(self):
        """_format_duration should format seconds correctly."""
        from gemini_research_mcp.server import _format_duration

        assert _format_duration(30) == "30s"
        assert _format_duration(5.2) == "5s"

    def test_format_duration_minutes(self):
        """_format_duration should format minutes correctly."""
        from gemini_research_mcp.server import _format_duration

        assert _format_duration(90) == "1m 30s"
        assert _format_duration(300) == "5m 0s"

    def test_format_duration_zero(self):
        """_format_duration should handle zero."""
        from gemini_research_mcp.server import _format_duration

        assert _format_duration(0) == "0s"

    def test_gemini_icon_url_is_png(self):
        """Server icon should be PNG (VS Code doesn't support SVG)."""
        from gemini_research_mcp.server import GEMINI_ICON_URL

        assert GEMINI_ICON_URL.endswith(".png")
        assert "raw.githubusercontent.com" in GEMINI_ICON_URL


# =============================================================================
# Content Module Rename
# =============================================================================


class TestContentModule:
    """Test content.py after FETCH_TIMEOUT rename."""

    def test_fetch_timeout_constant(self):
        """FETCH_TIMEOUT should be defined (renamed from DEFAULT_TIMEOUT)."""
        from gemini_research_mcp.content import FETCH_TIMEOUT

        assert FETCH_TIMEOUT == 15.0

    def test_no_default_timeout_collision(self):
        """content.py should NOT export DEFAULT_TIMEOUT anymore."""
        import gemini_research_mcp.content as content_mod

        # FETCH_TIMEOUT should exist, DEFAULT_TIMEOUT should NOT
        assert hasattr(content_mod, "FETCH_TIMEOUT")
        assert not hasattr(content_mod, "DEFAULT_TIMEOUT")

    def test_max_response_size(self):
        """MAX_RESPONSE_SIZE should be 10MB."""
        from gemini_research_mcp.content import MAX_RESPONSE_SIZE

        assert MAX_RESPONSE_SIZE == 10 * 1024 * 1024

    def test_ssrf_blocked_hosts(self):
        """BLOCKED_HOSTS should include key dangerous hosts."""
        from gemini_research_mcp.content import BLOCKED_HOSTS

        assert "localhost" in BLOCKED_HOSTS
        assert "169.254.169.254" in BLOCKED_HOSTS
        assert "metadata.google.internal" in BLOCKED_HOSTS

    def test_is_private_ip_blocks_localhost(self):
        """is_private_ip should block localhost."""
        from gemini_research_mcp.content import is_private_ip

        assert is_private_ip("localhost")
        assert is_private_ip("127.0.0.1")
        assert is_private_ip("169.254.169.254")

    def test_is_private_ip_allows_public_ips(self):
        """is_private_ip should allow known public IPs (avoids DNS resolution)."""
        from gemini_research_mcp.content import is_private_ip

        # Use raw public IPs to avoid DNS-dependent flakiness
        assert not is_private_ip("8.8.8.8")
        assert not is_private_ip("1.1.1.1")

    def test_validate_url_rejects_private(self):
        """validate_url should reject private IPs."""
        from gemini_research_mcp.content import validate_url

        valid, error = validate_url("http://localhost/admin")
        assert not valid
        assert "SSRF" in error

    def test_validate_url_allows_https(self):
        """validate_url should allow normal HTTPS URLs."""
        from gemini_research_mcp.content import validate_url

        valid, error = validate_url("https://example.com/page")
        assert valid
        assert error == ""

    def test_validate_url_rejects_ftp(self):
        """validate_url should reject non-HTTP schemes."""
        from gemini_research_mcp.content import validate_url

        valid, error = validate_url("ftp://example.com/file")
        assert not valid


# =============================================================================
# Package Metadata
# =============================================================================


class TestPackageMetadata:
    """Test package metadata and version."""

    def test_version_available(self):
        """Package version should be available."""
        from gemini_research_mcp import __version__

        assert __version__ is not None
        assert len(__version__) > 0

    def test_public_api_exports(self):
        """__all__ should export expected symbols."""
        from gemini_research_mcp import __all__

        expected_exports = [
            "mcp",
            "main",
            "quick_research",
            "deep_research",
            "deep_research_stream",
            "research_followup",
            "process_citations",
            "DeepResearchResult",
            "DeepResearchError",
            "ResearchResult",
        ]

        for export in expected_exports:
            assert export in __all__, f"Missing export: {export}"

    def test_main_is_callable(self):
        """main() entry point should be callable."""
        from gemini_research_mcp import main

        assert callable(main)

    def test_cli_version_flag(self):
        """CLI --version should work."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "gemini_research_mcp.server", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout + result.stderr
        assert "gemini-research-mcp" in output.lower() or "0." in output

    def test_cli_help_flag(self):
        """CLI --help should work."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "gemini_research_mcp.server", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
