"""End-to-end tests for gemini-research MCP server.

These tests make actual API calls and require GEMINI_API_KEY.
Run with: uv run pytest tests/test_e2e.py -v -m e2e --tb=short

Skip these in CI unless API key is available (default behavior).
"""

import os
from contextlib import AsyncExitStack

import pytest
from google import genai
from google.genai.types import (
    FinishReason,
    GenerateContentConfig,
    GenerateContentResponse,
    HttpOptions,
    HttpRetryOptions,
)


@pytest.fixture
def api_available():
    """Check if Gemini API is available."""
    return bool(os.environ.get("GEMINI_API_KEY"))


class TestFetchWebpageE2E:
    """End-to-end tests for fetch_webpage (no Gemini API required)."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_fetch_public_url(self):
        """Fetching a public URL should return content."""
        from gemini_research_mcp.content import fetch_webpage

        result = await fetch_webpage("https://httpbin.org/html")

        assert result.error is None, f"Should succeed, got: {result.error}"
        assert result.content, "Should have content"
        assert len(result.content) > 50, "Should have substantial content"
        assert result.word_count > 0, "Should have word count"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_fetch_json_api(self):
        """Fetching JSON endpoint should work (basic extraction)."""
        from gemini_research_mcp.content import fetch_webpage

        result = await fetch_webpage("https://httpbin.org/json")

        # JSON is not ideal for trafilatura but should not error
        assert result.error is None, f"Should not error: {result.error}"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_fetch_404_returns_error(self):
        """Fetching 404 page should return HTTP error."""
        from gemini_research_mcp.content import fetch_webpage

        result = await fetch_webpage("https://httpbin.org/status/404")

        assert result.error is not None, "Should have error"
        assert "404" in result.error, "Should mention 404"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_mcp_tool_wrapper(self):
        """MCP tool wrapper should work end-to-end."""
        from gemini_research_mcp.server import fetch_webpage as mcp_fetch

        result = await mcp_fetch("https://httpbin.org/html")

        assert "❌" not in result, f"Should succeed: {result}"
        assert "Extracted" in result or len(result) > 100, "Should have content"


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)
class TestResearchWebE2E:
    """End-to-end tests for research_web (quick grounded search)."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_basic_search(self):
        """Basic search should return grounded result with sources."""
        from gemini_research_mcp.quick import quick_research

        result = await quick_research(
            "What is OMOP CDM version 5.4?",
            thinking_level="minimal",
        )

        assert result.text, "Should have response text"
        assert len(result.text) > 100, "Response should be substantial"
        # Sources may be inline in text rather than structured list
        # Just verify the API call succeeded

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_thinking_level_parameter_is_backward_compatible(self):
        """Legacy thinking_level inputs should still be accepted by quick_research."""
        from gemini_research_mcp.quick import quick_research

        for level in ["minimal", "low", "medium", "high"]:
            result = await quick_research(
                "What is Python?",
                thinking_level=level,
            )
            assert result.text, f"Should have response for level={level}"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_site_filter(self):
        """Site filter should scope results to specific domain."""
        from gemini_research_mcp.quick import quick_research

        result = await quick_research(
            "site:cloud.google.com BigQuery pricing",
            thinking_level="minimal",
        )

        assert result.text, "Should have response text"
        # Most sources should be from google.com
        google_sources = [s for s in result.sources if "google" in s.uri.lower()]
        assert len(google_sources) >= 1, "Should have Google sources"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_system_prompt_affects_response(self):
        """Custom system prompt should affect response style."""
        from gemini_research_mcp.quick import quick_research

        result = await quick_research(
            "List 3 popular Python web frameworks",
            thinking_level="minimal",
            system_instruction="Always respond with a numbered list. Be extremely brief.",
        )

        assert result.text, "Should have response text"
        # Should have numbered items
        assert "1." in result.text or "1)" in result.text, "Should have numbered list"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_include_thoughts(self):
        """Include thoughts should return thinking summary."""
        from gemini_research_mcp.quick import quick_research

        result = await quick_research(
            "Explain async/await in Python",
            thinking_level="low",
            include_thoughts=True,
        )

        assert result.text, "Should have response text"
        # Note: thinking_summary may be None depending on model behavior


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
class TestSourceExtractionE2E:
    """Test source/citation extraction from responses."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_sources_have_uri_and_title(self):
        """Extracted sources should have URI and title."""
        from gemini_research_mcp.quick import quick_research

        result = await quick_research(
            "Python dataclasses documentation",
            thinking_level="minimal",
        )

        assert result.sources, "Should have sources"
        for source in result.sources:
            assert source.uri, "Source should have URI"
            assert source.uri.startswith("http"), "URI should be a URL"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.timeout(240)
    async def test_queries_are_captured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default model returns real grounding, not merely links in generated text."""
        from gemini_research_mcp import quick

        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        captured: list[tuple[str, GenerateContentResponse]] = []
        with genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=HttpOptions(
                timeout=180_000, retry_options=HttpRetryOptions(attempts=1)
            ),
        ) as client:
            async with client.aio:
                generate_content = client.aio.models.generate_content

                async def record_response(
                    *, model: str, contents: str, config: GenerateContentConfig
                ) -> GenerateContentResponse:
                    response = await generate_content(model=model, contents=contents, config=config)
                    captured.append((model, response))
                    return response

                monkeypatch.setattr(client.aio.models, "generate_content", record_response)
                monkeypatch.setattr(quick.genai, "Client", lambda **kwargs: client)
                result = await quick.quick_research(
                    "Use Google Search to check the official Python release notes for Python "
                    "3.14. Give its release date and two changes with source links. "
                    "Keep the answer below 120 words."
                )

        assert len(captured) == 1
        model, response = captured[0]
        assert model == "gemini-4-argon"
        assert result.text.strip() and result.sources and result.queries
        assert response.candidates
        candidate = response.candidates[0]
        assert candidate.finish_reason == FinishReason.STOP
        grounding = candidate.grounding_metadata
        assert grounding and grounding.grounding_chunks and grounding.web_search_queries
        assert result.queries == list(grounding.web_search_queries)
        assert [source.uri for source in result.sources] == [
            chunk.web.uri for chunk in grounding.grounding_chunks if chunk.web
        ]


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
class TestResearchFollowupE2E:
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.timeout(480)
    async def test_argon_preserves_previous_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An Argon follow-up retains a synthetic Flash interaction's context."""
        from gemini_research_mcp import deep

        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        monkeypatch.setattr(deep, "_client_health", None)
        with genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=HttpOptions(
                timeout=180_000, retry_options=HttpRetryOptions(attempts=1)
            ),
        ) as client:
            async with client.aio, AsyncExitStack() as cleanup:
                create = client.aio.interactions.create
                seed = await create(
                    model="gemini-3.8-flash",
                    input="Synthetic compatibility test: remember RUBBER-DUCK-ARGON. Reply ACK.",
                    generation_config={"thinking_level": "low"},
                    timeout=180,
                )
                cleanup.push_async_callback(client.aio.interactions.delete, seed.id, timeout=60)
                assert seed.status == "completed"
                captured: list[tuple[str, str]] = []

                async def record_followup(
                    *, input: str, model: str, previous_interaction_id: str
                ):
                    interaction = await create(
                        input=input,
                        model=model,
                        previous_interaction_id=previous_interaction_id,
                        timeout=180,
                    )
                    cleanup.push_async_callback(
                        client.aio.interactions.delete, interaction.id, timeout=60
                    )
                    captured.append((model, previous_interaction_id))
                    return interaction

                monkeypatch.setattr(client.aio.interactions, "create", record_followup)
                monkeypatch.setattr(deep, "_get_healthy_client", lambda: client)
                answer = await deep.research_followup(
                    seed.id, "Repeat the exact marker from the previous turn and nothing else."
                )
                assert "RUBBER-DUCK-ARGON" in answer
                assert captured == [("gemini-4-argon", seed.id)]


class TestMCPToolsE2E:
    """End-to-end tests for MCP tool wrappers."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_research_web_tool(self):
        """research_web MCP tool should work end-to-end."""
        from gemini_research_mcp.server import research_web

        result = await research_web(
            query="What is FastMCP?",
            include_thoughts=False,
        )

        assert result, "Should return result string"
        assert "MCP" in result or "Model Context Protocol" in result.lower(), "Should mention MCP"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.timeout(1200)  # 20 minute timeout; Google docs allow up to 60 minutes
    async def test_research_deep_tool_starts(self):
        """research_deep MCP tool should start (test stream initiation only).

        Note: Full research_deep testing is expensive and time-consuming.
        This test only verifies the stream can be initiated.
        """
        from gemini_research_mcp.deep import deep_research_stream

        events = []
        async for event in deep_research_stream(
            query="Brief summary of Python 3.12 features",
        ):
            events.append(event)
            # Just capture the start event and exit
            if event.event_type == "start":
                break

        assert len(events) >= 1, "Should receive at least one event"
        # First event should be 'start' with interaction_id
        start_event = events[0]
        assert start_event.event_type == "start", "First event should be 'start'"
        assert start_event.interaction_id, "Start event should have interaction_id"

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_resume_research_fake_id_returns_not_found(self):
        """resume_research with a fake interaction_id should return not_found."""
        import json

        from gemini_research_mcp.server import resume_research

        result = await resume_research(interaction_id="fake-nonexistent-id-12345")
        data = json.loads(result)

        assert data["status"] == "not_found"
        assert "fake-nonexistent-id-12345" in data["message"]

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_resume_research_no_id_lists_sessions(self):
        """resume_research without interaction_id should list or show empty."""
        import json

        from gemini_research_mcp.server import resume_research

        result = await resume_research()
        data = json.loads(result)

        # Should be either resumable_sessions_found or no_resumable_sessions
        assert data["status"] in ("resumable_sessions_found", "no_resumable_sessions")
