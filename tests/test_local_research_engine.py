"""Local Deep Research bridge: config validation, result normalization, stdio child."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP

from gemini_research_mcp.engines import describe_engines, get_engine
from gemini_research_mcp.engines import local as local_module
from gemini_research_mcp.engines.local import (
    LocalDeepResearchEngine,
    load_ldr_settings,
    local_slot_busy,
    normalize_ldr_result,
)
from gemini_research_mcp.types import EngineError, EngineRequest

MODEL = "llama3:8b"
OLLAMA = "http://127.0.0.1:11434"
SEARCH = "searxng"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("LDR_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("LDR_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LDR_LLM_MODEL", MODEL)
    monkeypatch.setenv("LDR_LLM_OLLAMA_URL", OLLAMA)
    monkeypatch.setenv("LDR_SEARCH_TOOL", SEARCH)


# --- configuration -----------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "https://[::1]:11434",
        "http://127.5.5.5",
    ],
)
def test_loopback_ollama_urls_are_accepted(monkeypatch, url):
    monkeypatch.setenv("LDR_LLM_OLLAMA_URL", url)
    assert load_ldr_settings().ollama_url == url


@pytest.mark.parametrize(
    "url",
    [
        "",
        "http://ollama.example.com:11434",
        "http://10.0.0.5:11434",
        "http://0.0.0.0:11434",
        "http://user:pw@localhost:11434",
        "ftp://localhost",
        "localhost:11434",
        "http://localhost.evil.test",
    ],
)
def test_non_loopback_ollama_urls_are_refused(monkeypatch, url):
    monkeypatch.setenv("LDR_LLM_OLLAMA_URL", url)
    with pytest.raises(EngineError) as info:
        load_ldr_settings()
    assert info.value.code == "LOCAL_CONFIG_INVALID"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LDR_LLM_PROVIDER", "openai"),
        ("LDR_LLM_PROVIDER", ""),
        ("LDR_LLM_MODEL", ""),
        ("LDR_SEARCH_TOOL", ""),
        ("LDR_SEARCH_TOOL", "auto"),
        ("LDR_SEARCH_TOOL", "local_all"),
        ("LDR_SEARCH_TOOL", "library"),
        ("LDR_SEARCH_TOOL", "collection_abc"),
        ("LDR_SEARCH_ENGINE_WEB_SEARXNG_DEFAULT_PARAMS_INSTANCE_URL", "not a url"),
    ],
)
def test_unverifiable_or_unsafe_configuration_is_refused(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(EngineError) as info:
        load_ldr_settings()
    assert info.value.code == "LOCAL_CONFIG_INVALID"


def test_child_env_is_allowlisted(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini")
    monkeypatch.setenv("LDR_LLM_OPENAI_API_KEY", "secret-openai")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-aws")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg")
    env = load_ldr_settings().child_env()
    assert env["LDR_LLM_PROVIDER"] == "ollama"
    assert env["LDR_LLM_MODEL"] == MODEL
    assert env["XDG_CONFIG_HOME"] == "/tmp/xdg"
    assert not any("secret" in value for value in env.values())
    assert set(env) <= {
        "LDR_LLM_PROVIDER",
        "LDR_LLM_MODEL",
        "LDR_LLM_OLLAMA_URL",
        "LDR_SEARCH_TOOL",
        "LDR_SEARCH_ENGINE_WEB_SEARXNG_DEFAULT_PARAMS_INSTANCE_URL",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "XDG_STATE_HOME",
    }


def test_descriptor_needs_command_and_valid_config_without_launching(monkeypatch):
    monkeypatch.setenv("GEMINI_RESEARCH_LDR_COMMAND", "/opt/ldr/bin/ldr-mcp")
    descriptor = {d.id: d for d in describe_engines()}["local_deep_research"]
    assert descriptor.available and descriptor.model == MODEL
    monkeypatch.setenv("LDR_LLM_OLLAMA_URL", "http://example.com")
    descriptor = {d.id: d for d in describe_engines()}["local_deep_research"]
    assert not descriptor.available
    assert "loopback" in (descriptor.availability_reason or "")


# --- normalization -----------------------------------------------------------------


def test_sdk_wrapped_structured_payload_is_unwrapped():
    """The real LDR server (MCP SDK) returns dicts as {"result": {...}}."""
    payload = {"status": "success", "config": {}}
    wrapped = type("Result", (), {"structured_content": {"result": payload}, "content": []})
    plain = type("Result", (), {"structured_content": payload, "content": []})
    assert local_module._extract_payload(wrapped) == payload
    assert local_module._extract_payload(plain) == payload


def test_ldr_error_payload_is_not_success():
    with pytest.raises(EngineError) as exc:
        normalize_ldr_result(
            {"status": "error", "error": "synthetic failure", "summary": "must not be a report"}
        )
    assert exc.value.code == "LOCAL_ENGINE_FAILED"
    assert "synthetic failure" not in exc.value.message


def test_error_type_is_kept_only_when_it_is_a_safe_identifier():
    with pytest.raises(EngineError) as exc:
        normalize_ldr_result({"status": "error", "error_type": "validation_error"})
    assert "validation_error" in exc.value.message
    with pytest.raises(EngineError) as exc:
        normalize_ldr_result({"status": "error", "error_type": "SECRET key=abc\nline"})
    assert "SECRET" not in exc.value.message


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "text",
        [],
        {"status": "success"},
        {"status": "success", "formatted_findings": "  ", "summary": ""},
        {"summary": "no status"},
    ],
)
def test_unusable_payloads_are_rejected(payload):
    with pytest.raises(EngineError) as exc:
        normalize_ldr_result(payload)
    assert exc.value.code in {"LOCAL_ENGINE_FAILED", "LOCAL_EMPTY_REPORT"}


def test_report_prefers_formatted_findings_and_keeps_sources_attributed():
    report = normalize_ldr_result(
        {
            "status": "success",
            "research_id": "abc",
            "summary": "short",
            "formatted_findings": "# Full findings\nBody",
            "sources": [
                "https://a.example/x",
                {"url": "https://b.example/y", "title": "B"},
                {"title": "No url"},
                42,
            ],
        }
    )
    assert report.text == "# Full findings\nBody"
    assert report.status == "completed"
    assert report.usage is None
    assert report.provider_id is None
    assert [(s.uri, s.title, s.provenance) for s in report.sources] == [
        ("https://a.example/x", None, "engine_reported"),
        ("https://b.example/y", "B", "engine_reported"),
        (None, "No url", "engine_reported"),
    ]


def test_summary_is_the_fallback_text():
    report = normalize_ldr_result({"status": "success", "summary": "Only a summary"})
    assert report.text == "Only a summary"


# --- in-process MCP contract -----------------------------------------------------


def fake_server(*, provider="ollama", model=MODEL, engine=SEARCH, payload=None, tools=True):
    server = FastMCP("fake-ldr")
    result = payload or {
        "status": "success",
        "summary": "S",
        "formatted_findings": "Findings",
        "sources": ["https://a.example"],
    }

    @server.tool
    def get_configuration() -> dict:
        return {
            "status": "success",
            "config": {
                "llm": {"provider": provider, "model": model},
                "search": {"default_engine": engine},
            },
        }

    if tools:

        @server.tool
        def detailed_research(query: str, iterations: int | None = None) -> dict:
            return dict(result, query=query)

    return server


def patch_client(monkeypatch, server):
    monkeypatch.setenv("GEMINI_RESEARCH_LDR_COMMAND", "/opt/ldr/bin/ldr-mcp")
    monkeypatch.setattr(
        LocalDeepResearchEngine, "_make_client", lambda self, command, settings: Client(server)
    )


class Recorder:
    def __init__(self):
        self.events: list[str] = []

    async def submitting(self):
        self.events.append("submitting")

    async def provider_id(self, value):
        self.events.append("provider_id")

    async def progress(self, message):
        self.events.append("progress")

    async def run(self, engine, query="topic"):
        return await engine.run(
            EngineRequest(query=query),
            on_submitting=self.submitting,
            on_provider_id=self.provider_id,
            on_progress=self.progress,
        )


async def test_run_through_in_memory_server_never_reports_a_provider_id(monkeypatch):
    patch_client(monkeypatch, fake_server())
    engine = get_engine("local_deep_research")
    rec = Recorder()
    try:
        report = await rec.run(engine, "my query")
    finally:
        await engine.aclose()
    assert report.text == "Findings"
    assert report.model == MODEL
    assert report.provider_id is None
    assert "provider_id" not in rec.events
    assert rec.events.index("submitting") < rec.events.index("progress")
    assert not local_slot_busy()


async def test_error_payload_through_server_is_a_typed_failure(monkeypatch):
    patch_client(
        monkeypatch,
        fake_server(payload={"status": "error", "error": "boom", "error_type": "llm_error"}),
    )
    engine = get_engine("local_deep_research")
    try:
        with pytest.raises(EngineError) as info:
            await Recorder().run(engine)
    finally:
        await engine.aclose()
    assert info.value.code == "LOCAL_ENGINE_FAILED"
    assert "boom" not in info.value.message


@pytest.mark.parametrize(
    "kwargs",
    [
        {"provider": "openai"},
        {"model": "other:1b"},
        {"engine": "wikipedia"},
        {"tools": False},
    ],
)
async def test_preflight_refuses_unverifiable_servers(monkeypatch, kwargs):
    patch_client(monkeypatch, fake_server(**kwargs))
    engine = get_engine("local_deep_research")
    rec = Recorder()
    try:
        with pytest.raises(EngineError) as info:
            await rec.run(engine)
    finally:
        await engine.aclose()
    assert info.value.code == "LOCAL_CONFIG_INVALID"
    assert "submitting" not in rec.events


async def test_one_local_slot_per_process(monkeypatch):
    patch_client(monkeypatch, fake_server())
    first = get_engine("local_deep_research")
    second = get_engine("local_deep_research")
    assert isinstance(first, LocalDeepResearchEngine)
    assert isinstance(second, LocalDeepResearchEngine)
    first.reserve()
    try:
        with pytest.raises(EngineError) as info:
            await Recorder().run(second)
        assert info.value.code == "LOCAL_ENGINE_BUSY"
    finally:
        await first.aclose()
        await second.aclose()
    assert not local_slot_busy()
    second.reserve()
    await second.aclose()


async def test_refresh_is_unsupported():
    engine = get_engine("local_deep_research")
    with pytest.raises(EngineError) as info:
        await engine.refresh("anything")
    assert info.value.code == "LOCAL_REFRESH_UNSUPPORTED"


# --- real stdio child ---------------------------------------------------------------

CHILD = textwrap.dedent(
    """\
    import json, os, time
    from fastmcp import FastMCP

    with open({pid_file!r}, "w") as fh:
        fh.write(str(os.getpid()))
    with open({env_file!r}, "w") as fh:
        json.dump(sorted(os.environ), fh)

    mcp = FastMCP("synthetic-ldr")

    @mcp.tool
    def get_configuration() -> dict:
        return {{
            "status": "success",
            "config": {{
                "llm": {{"provider": os.environ.get("LDR_LLM_PROVIDER"),
                        "model": os.environ.get("LDR_LLM_MODEL")}},
                "search": {{"default_engine": os.environ.get("LDR_SEARCH_TOOL")}},
            }},
        }}

    @mcp.tool
    def detailed_research(query: str) -> dict:
        open({started_file!r}, "w").write("1")
        if {mode!r} == "slow":
            time.sleep(120)
        if {mode!r} == "error":
            return {{"status": "error", "error": "synthetic", "error_type": "llm_error"}}
        return {{"status": "success", "formatted_findings": "Child findings: " + query,
                "sources": ["https://child.example"]}}

    mcp.run(transport="stdio", show_banner=False)
    """
)


def make_child(tmp_path: Path, mode: str) -> dict[str, Path]:
    paths = {
        "script": tmp_path / "ldr-mcp",
        "pid": tmp_path / "pid",
        "env": tmp_path / "env.json",
        "started": tmp_path / "started",
    }
    child_py = tmp_path / "ldr_child.py"
    child_py.write_text(
        CHILD.format(
            pid_file=str(paths["pid"]),
            env_file=str(paths["env"]),
            started_file=str(paths["started"]),
            mode=mode,
        )
    )
    # A shell shim avoids shebang limits on interpreter paths containing spaces.
    paths["script"].write_text(f"#!/bin/sh\nexec '{sys.executable}' '{child_py}'\n")
    paths["script"].chmod(paths["script"].stat().st_mode | stat.S_IXUSR)
    return paths


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def wait_for(predicate, timeout=20.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.1)
    return False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shebang child")
async def test_real_stdio_child_success_filters_environment(monkeypatch, tmp_path):
    paths = make_child(tmp_path, "success")
    monkeypatch.setenv("GEMINI_RESEARCH_LDR_COMMAND", str(paths["script"]))
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini")
    monkeypatch.setenv("LDR_LLM_OPENAI_API_KEY", "secret-openai")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-aws")
    engine = get_engine("local_deep_research")
    try:
        report = await Recorder().run(engine, "child query")
    finally:
        await engine.aclose()
    assert report.text == "Child findings: child query"
    assert report.sources[0].uri == "https://child.example"
    names = json.loads(paths["env"].read_text())
    assert "LDR_LLM_PROVIDER" in names and "PATH" in names
    for forbidden in ("GEMINI_API_KEY", "LDR_LLM_OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY"):
        assert forbidden not in names
    pid = int(paths["pid"].read_text())
    assert await wait_for(lambda: not pid_alive(pid))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shebang child")
async def test_real_stdio_child_error_payload(monkeypatch, tmp_path):
    paths = make_child(tmp_path, "error")
    monkeypatch.setenv("GEMINI_RESEARCH_LDR_COMMAND", str(paths["script"]))
    engine = get_engine("local_deep_research")
    try:
        with pytest.raises(EngineError) as info:
            await Recorder().run(engine)
    finally:
        await engine.aclose()
    assert info.value.code == "LOCAL_ENGINE_FAILED"
    assert "llm_error" in info.value.message


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shebang child")
async def test_real_stdio_child_timeout_closes_child(monkeypatch, tmp_path):
    paths = make_child(tmp_path, "slow")
    monkeypatch.setenv("GEMINI_RESEARCH_LDR_COMMAND", str(paths["script"]))
    engine = get_engine("local_deep_research")
    assert isinstance(engine, LocalDeepResearchEngine)
    engine.research_timeout = 1.0
    try:
        with pytest.raises(EngineError) as info:
            await Recorder().run(engine)
    finally:
        await engine.aclose()
    assert info.value.code == "LOCAL_ENGINE_TIMEOUT"
    pid = int(paths["pid"].read_text())
    assert await wait_for(lambda: not pid_alive(pid), timeout=15)
    assert not local_slot_busy()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shebang child")
async def test_real_stdio_child_cancellation_closes_child(monkeypatch, tmp_path):
    paths = make_child(tmp_path, "slow")
    monkeypatch.setenv("GEMINI_RESEARCH_LDR_COMMAND", str(paths["script"]))
    engine = get_engine("local_deep_research")
    task = asyncio.create_task(Recorder().run(engine))
    try:
        assert await wait_for(paths["started"].exists)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await engine.aclose()
    pid = int(paths["pid"].read_text())
    assert await wait_for(lambda: not pid_alive(pid), timeout=15)
    assert not local_slot_busy()


async def test_discovery_launches_no_subprocess(monkeypatch):
    monkeypatch.setenv("GEMINI_RESEARCH_LDR_COMMAND", "/nonexistent/ldr-mcp")

    def boom(*args, **kwargs):
        raise AssertionError("discovery must not launch a process")

    monkeypatch.setattr(local_module, "StdioTransport", boom)
    descriptor = {d.id: d for d in describe_engines()}["local_deep_research"]
    assert descriptor.available


async def test_unlaunchable_command_is_a_sanitized_unavailable_error(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_RESEARCH_LDR_COMMAND", str(tmp_path / "missing" / "ldr-mcp"))
    engine = get_engine("local_deep_research")
    try:
        with pytest.raises(EngineError) as info:
            await Recorder().run(engine)
    finally:
        await engine.aclose()
    assert info.value.code == "LOCAL_ENGINE_UNAVAILABLE"
    assert str(tmp_path) not in info.value.message
