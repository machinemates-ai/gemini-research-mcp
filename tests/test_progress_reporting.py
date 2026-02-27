import types
from unittest.mock import AsyncMock, MagicMock
from typing import Any, AsyncIterator

import pytest


@pytest.mark.asyncio
async def test_research_deep_emits_progress_for_thought(monkeypatch: pytest.MonkeyPatch) -> None:
    from gemini_research_mcp.types import DeepResearchProgress, DeepResearchResult

    import gemini_research_mcp.server as server

    async def fake_stream(
        *,
        query: str,
        format_instructions: str | None = None,
        file_search_store_names: list[str] | None = None,
    ) -> AsyncIterator[DeepResearchProgress]:
        yield DeepResearchProgress(event_type="start", interaction_id="test-interaction")
        yield DeepResearchProgress(
            event_type="thought",
            interaction_id="test-interaction",
            content="Thinking about the answer",
        )

    async def fake_status(interaction_id: str) -> DeepResearchResult:
        return DeepResearchResult(
            text="Hello world",
            citations=[],
            thinking_summaries=[],
            interaction_id=interaction_id,
            usage=None,
            raw_interaction=types.SimpleNamespace(status="completed"),
        )

    async def passthrough_citations(result: DeepResearchResult, resolve_urls: bool) -> DeepResearchResult:
        return result

    async def fake_generate_title_from_query(query: str) -> str | None:
        return None

    async def fake_generate_session_metadata(text: str, query: str) -> Any:
        return types.SimpleNamespace(title=None, summary=None)

    def fake_save_research_session(**kwargs: Any) -> None:
        return None

    def fake_update_research_session(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(server, "deep_research_stream", fake_stream)
    monkeypatch.setattr(server, "get_research_status", fake_status)
    monkeypatch.setattr(server, "process_citations", passthrough_citations)
    monkeypatch.setattr(server, "generate_title_from_query", fake_generate_title_from_query)
    monkeypatch.setattr(server, "generate_session_metadata", fake_generate_session_metadata)
    monkeypatch.setattr(server, "save_research_session", fake_save_research_session)
    monkeypatch.setattr(server, "update_research_session", fake_update_research_session)

    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.report_progress = AsyncMock()
    ctx.elicit = AsyncMock()

    result = await server.research_deep(query="test", ctx=ctx)

    assert "## Research Report" in result

    ctx.report_progress.assert_any_await(
        progress=5,
        total=100,
        message="[1] 🧠 Thinking about the answer",
    )

    # v0.10.4 behavior: thought steps are progress-only in foreground execution.
    assert not any(
        (call.args and call.args[0] == "[1] 🧠 Thinking about the answer")
        for call in ctx.info.await_args_list
    )
