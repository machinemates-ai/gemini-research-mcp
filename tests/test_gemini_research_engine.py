"""Gemini adapter: no create retry, callback ordering, status mapping, refresh."""

from __future__ import annotations

import httpx
import pytest

from gemini_research_mcp.engines import gemini as gemini_module
from gemini_research_mcp.engines import get_engine
from gemini_research_mcp.engines.gemini import GeminiEngine
from gemini_research_mcp.types import EngineError, EngineRequest, JobStatus

STANDARD = "deep-research-preview-04-2026"
MAX = "deep-research-max-preview-04-2026"


class FakeInteractions:
    def __init__(self, create_result=None, gets=()):
        self.create_result = create_result
        self.gets = list(gets)
        self.create_calls: list[dict] = []
        self.get_calls: list[str] = []
        self.events: list[str] = []

    async def create(self, **kwargs):
        self.events.append("create")
        self.create_calls.append(kwargs)
        if isinstance(self.create_result, BaseException):
            raise self.create_result
        return self.create_result

    async def get(self, *, id):
        self.events.append("get")
        self.get_calls.append(id)
        item = self.gets.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    def __init__(self, interactions):
        self.aio = type("Aio", (), {})()
        self.aio.interactions = interactions
        self.closed = False

        async def aclose():
            self.closed = True

        self.aio.aclose = aclose


def interaction(status="in_progress", text=None, usage=None, id="int-1", steps=None):
    data = {"id": id, "status": status}
    if text is not None:
        data["output_text"] = text
    if usage is not None:
        data["usage"] = usage
    if steps is not None:
        data["steps"] = steps
    return data


def make_engine(interactions, engine_id="gemini"):
    client = FakeClient(interactions)
    engine = GeminiEngine(engine_id, client_factory=lambda: client)
    engine.poll_interval = 0
    return engine, client


class Recorder:
    def __init__(self, events):
        self.events = events

    async def submitting(self):
        self.events.append("submitting")

    async def provider_id(self, value):
        self.events.append(f"provider_id:{value}")

    async def progress(self, message):
        self.events.append("progress")


async def run(engine, interactions, query="topic"):
    rec = Recorder(interactions.events)
    return await engine.run(
        EngineRequest(query=query),
        on_submitting=rec.submitting,
        on_provider_id=rec.provider_id,
        on_progress=rec.progress,
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-not-a-key")
    monkeypatch.delenv("DEEP_RESEARCH_AGENT", raising=False)


async def test_callback_order_and_create_arguments():
    inter = FakeInteractions(
        create_result=interaction(),
        gets=[interaction(), interaction("completed", text="Report body")],
    )
    engine, _ = make_engine(inter)
    report = await run(engine, inter, query="  verbatim query ")

    assert inter.events[:3] == ["submitting", "create", "provider_id:int-1"]
    assert inter.events.index("provider_id:int-1") < inter.events.index("get")
    (call,) = inter.create_calls
    assert call["input"] == "  verbatim query "
    assert call["agent"] == STANDARD
    assert call["background"] is True
    assert "stream" not in call
    assert call["agent_config"]["collaborative_planning"] is False
    assert report.text == "Report body"
    assert report.status == "completed"
    assert report.provider_id == "int-1"
    assert len(inter.create_calls) == 1


@pytest.mark.parametrize(
    ("engine_id", "env", "expected"),
    [
        ("gemini", None, STANDARD),
        ("gemini", "pro", "deep-research-pro-preview-12-2025"),
        ("gemini_max", "pro", MAX),
    ],
)
async def test_agent_selection(monkeypatch, engine_id, env, expected):
    if env:
        monkeypatch.setenv("DEEP_RESEARCH_AGENT", env)
    inter = FakeInteractions(create_result=interaction("completed", text="x"))
    engine, _ = make_engine(inter, engine_id)
    await run(engine, inter)
    assert inter.create_calls[0]["agent"] == expected


async def test_report_keeps_text_sources_and_zero_usage():
    text = (
        "Findings.\n\n**Sources:**\n"
        "1. [example.com](https://vertexaisearch.cloud.google.com/redirect/abc)\n"
    )
    inter = FakeInteractions(
        create_result=interaction(
            "completed",
            text=text,
            usage={"total_input_tokens": 0, "total_output_tokens": None, "total_tokens": 7},
            steps=[
                {
                    "type": "model_output",
                    "content": [{"type": "image", "data": "AAAA", "mime_type": "image/png"}],
                }
            ],
        )
    )
    engine, _ = make_engine(inter)
    report = await run(engine, inter)

    assert report.text == text
    assert [(s.uri, s.title, s.provenance) for s in report.sources] == [
        ("https://vertexaisearch.cloud.google.com/redirect/abc", "example.com", "report_parsed")
    ]
    assert report.usage is not None
    assert report.usage.prompt_tokens == 0
    assert report.usage.completion_tokens is None
    assert report.usage.total_tokens == 7
    assert report.images[0]["mime_type"] == "image/png"
    assert report.model == STANDARD


async def test_no_usage_stays_none():
    inter = FakeInteractions(create_result=interaction("completed", text="x"))
    engine, _ = make_engine(inter)
    assert (await run(engine, inter)).usage is None


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (httpx.ReadTimeout("synthetic"), "SUBMISSION_UNKNOWN"),
        (RuntimeError("boom"), "SUBMISSION_UNKNOWN"),
        (type("E", (Exception,), {"status_code": 503})(), "SUBMISSION_UNKNOWN"),
        (type("E", (Exception,), {"status_code": 400})(), "SUBMISSION_REJECTED"),
        (type("E", (Exception,), {"status_code": 429})(), "SUBMISSION_REJECTED"),
    ],
)
async def test_create_failure_is_classified_and_never_retried(exc, code):
    inter = FakeInteractions(create_result=exc)
    engine, _ = make_engine(inter)
    with pytest.raises(EngineError) as info:
        await run(engine, inter)
    assert info.value.code == code
    assert inter.events == ["submitting", "create"]


async def test_create_without_interaction_id_is_unknown():
    inter = FakeInteractions(create_result={"status": "in_progress"})
    engine, _ = make_engine(inter)
    with pytest.raises(EngineError) as info:
        await run(engine, inter)
    assert info.value.code == "SUBMISSION_UNKNOWN"
    assert not any(e.startswith("provider_id") for e in inter.events)


@pytest.mark.parametrize(
    ("status", "code"),
    [
        ("failed", "PROVIDER_FAILED"),
        ("cancelled", "PROVIDER_CANCELLED"),
        ("canceled", "PROVIDER_CANCELLED"),
        ("incomplete", "PROVIDER_INCOMPLETE"),
        ("requires_action", "PROVIDER_REQUIRES_ACTION"),
    ],
)
async def test_non_success_terminal_states_raise(status, code):
    inter = FakeInteractions(create_result=interaction(), gets=[interaction(status)])
    engine, _ = make_engine(inter)
    with pytest.raises(EngineError) as info:
        await run(engine, inter)
    assert info.value.code == code


async def test_completed_without_text_is_never_empty_success():
    inter = FakeInteractions(create_result=interaction("completed", text="   "))
    engine, _ = make_engine(inter)
    with pytest.raises(EngineError) as info:
        await run(engine, inter)
    assert info.value.code == "EMPTY_REPORT"


async def test_incomplete_text_is_kept_as_incomplete_report():
    inter = FakeInteractions(create_result=interaction("incomplete", text="Partial findings"))
    engine, _ = make_engine(inter)
    report = await run(engine, inter)
    assert report.status == "incomplete"
    assert report.text == "Partial findings"
    assert report.warnings == ()


async def test_poll_timeout_keeps_job_recoverable():
    inter = FakeInteractions(create_result=interaction(), gets=[interaction()] * 50)
    engine, _ = make_engine(inter)
    engine.timeout = 0
    with pytest.raises(EngineError) as info:
        await run(engine, inter)
    assert info.value.code == "POLL_TIMEOUT"
    assert "create" in inter.events and inter.events.count("create") == 1


async def test_get_retries_are_bounded_and_get_only(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(gemini_module.asyncio, "sleep", no_sleep)
    inter = FakeInteractions(
        create_result=interaction(),
        gets=[RuntimeError("blip"), RuntimeError("blip"), interaction("completed", text="ok")],
    )
    engine, _ = make_engine(inter)
    report = await run(engine, inter)
    assert report.text == "ok"
    assert inter.events.count("create") == 1
    assert inter.events.count("get") == 3


async def test_get_failure_after_bound_raises_poll_failed(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(gemini_module.asyncio, "sleep", no_sleep)
    inter = FakeInteractions(create_result=interaction(), gets=[RuntimeError("x")] * 3)
    engine, _ = make_engine(inter)
    with pytest.raises(EngineError) as info:
        await run(engine, inter)
    assert info.value.code == "POLL_FAILED"
    assert inter.events.count("get") == 3


async def test_refresh_only_gets_and_never_creates():
    inter = FakeInteractions(gets=[interaction("completed", text="Done", id="int-9")])
    engine, _ = make_engine(inter)
    snap = await engine.refresh("int-9")
    assert inter.events == ["get"]
    assert inter.get_calls == ["int-9"]
    assert snap.status is JobStatus.COMPLETED
    assert snap.report is not None and snap.report.text == "Done"
    assert snap.report.provider_id == "int-9"


@pytest.mark.parametrize(
    ("status", "expected", "code"),
    [
        ("in_progress", JobStatus.RUNNING, None),
        ("failed", JobStatus.FAILED, "PROVIDER_FAILED"),
        ("cancelled", JobStatus.CANCELLED, "PROVIDER_CANCELLED"),
        ("requires_action", JobStatus.INTERRUPTED, "PROVIDER_REQUIRES_ACTION"),
        ("incomplete", JobStatus.FAILED, "PROVIDER_INCOMPLETE"),
    ],
)
async def test_refresh_status_mapping(status, expected, code):
    inter = FakeInteractions(gets=[interaction(status)])
    engine, _ = make_engine(inter)
    snap = await engine.refresh("int-1")
    assert snap.status is expected
    assert snap.error_code == code
    assert snap.report is None


async def test_refresh_not_found_is_typed():
    inter = FakeInteractions(gets=[type("E", (Exception,), {"status_code": 404})()])
    engine, _ = make_engine(inter)
    with pytest.raises(EngineError) as info:
        await engine.refresh("gone")
    assert info.value.code == "PROVIDER_NOT_FOUND"
    assert inter.events == ["get"]


async def test_aclose_closes_client_once():
    inter = FakeInteractions(create_result=interaction("completed", text="x"))
    engine, client = make_engine(inter)
    await run(engine, inter)
    await engine.aclose()
    await engine.aclose()
    assert client.closed


async def test_unavailable_without_api_key_makes_no_calls(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    inter = FakeInteractions(create_result=interaction())
    engine, _ = make_engine(inter)
    with pytest.raises(EngineError) as info:
        await run(engine, inter)
    assert info.value.code == "ENGINE_UNAVAILABLE"
    assert inter.events == []


@pytest.mark.parametrize("status_code", [429, 503])
async def test_real_sdk_client_sends_exactly_one_create_post(monkeypatch, status_code):
    posts: list[str] = []

    async def fake_send(self, request, *args, **kwargs):
        if request.method == "POST":
            posts.append(str(request.url))
        return httpx.Response(
            status_code,
            request=request,
            json={"error": {"code": status_code, "message": "synthetic", "status": "X"}},
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    engine = get_engine("gemini")
    assert isinstance(engine, GeminiEngine)
    events: list[str] = []
    rec = Recorder(events)
    try:
        with pytest.raises(EngineError) as info:
            await engine.run(
                EngineRequest(query="synthetic"),
                on_submitting=rec.submitting,
                on_provider_id=rec.provider_id,
                on_progress=rec.progress,
            )
    finally:
        await engine.aclose()
    assert len(posts) == 1
    assert events == ["submitting"]
    assert info.value.code == (
        "SUBMISSION_REJECTED" if status_code == 429 else "SUBMISSION_UNKNOWN"
    )
