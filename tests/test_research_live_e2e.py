"""Explicit, bounded release acceptance over real MCP transports.

Run only with GEMINI_RESEARCH_RUN_RELEASE_E2E=1 and
GEMINI_RESEARCH_E2E_DIR pointing outside the checkout. Reports remain there;
normal CI never submits these paid workloads.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import signal
import socket
import sys
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError
from fastmcp_tasks import ToolTask, call_tool_task
from mcp.shared.exceptions import MCPError
from mcp.types import TextResourceContents

pytestmark = pytest.mark.e2e
REPO = Path(__file__).resolve().parent.parent
QUERY = (
    "Using public sources, compare operational transformation and conflict-free replicated "
    "data types for collaborative editing. Explain two differences, cite source URLs, and "
    "keep the report under 500 words. Do not ask clarifying questions."
)

# Materialized outside the checkout. Requests/responses are passed through;
# only content-free counters are written. The research-create budget survives
# a process restart and stops an unexpected second POST before it is sent.
LAUNCHER = r"""
import asyncio
import json
import logging
import os
from pathlib import Path

root = Path(os.environ["E2E_SCENARIO_DIR"])
root.mkdir(parents=True, exist_ok=True)
(root / "server.pid").write_text(str(os.getpid()))
audit = root / "audit.jsonl"

def record(kind, **fields):
    with audit.open("a") as stream:
        stream.write(json.dumps({"kind": kind, **fields}) + "\n")

def observe(request):
    if request.url.host != "generativelanguage.googleapis.com":
        return
    path = request.url.path
    if request.method == "POST" and path.endswith("/interactions"):
        body = json.loads(request.content)
        agent = body.get("agent")
        kind = "research_create" if agent else "followup_create"
        allowance = int(os.environ.get("E2E_RESEARCH_ALLOWANCE", "0"))
        if kind == "research_create":
            used = sum(json.loads(line)["kind"] == kind for line in
                       audit.read_text().splitlines()) if audit.exists() else 0
            if used >= allowance:
                record("budget_blocked", agent=agent)
                raise RuntimeError("Research submission allowance exhausted")
        record(kind, agent=agent, model=body.get("model"))
    elif request.method == "POST" and ":generateContent" in path:
        record("generate_content", model=path.split("/models/")[-1].split(":")[0])
    else:
        record("provider_observation", method=request.method)

seen = set()
for module_name in ("httpx", "httpx2"):
    module = __import__(module_name)
    cls = module.AsyncClient
    if cls in seen:
        continue
    seen.add(cls)
    original_send = cls.send
    def wrap(send):
        async def observed(self, request, *args, **kwargs):
            observe(request)
            return await send(self, request, *args, **kwargs)
        return observed
    cls.send = wrap(original_send)

from google import genai
original_init = genai.Client.__init__
def client_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    generate = self.aio.models.generate_content
    async def observed_generate(*args, **kwargs):
        result = await generate(*args, **kwargs)
        config = kwargs.get("config")
        thinking = getattr(getattr(config, "thinking_config", None), "thinking_level", None)
        candidates = result.candidates or []
        grounding = candidates[0].grounding_metadata if candidates else None
        record("generation_result", model=kwargs.get("model"),
               thinking=str(thinking),
               sources=len(grounding.grounding_chunks or []) if grounding else 0,
               queries=len(grounding.web_search_queries or []) if grounding else 0)
        return result
    self.aio.models.generate_content = observed_generate
genai.Client.__init__ = client_init

from gemini_research_mcp import server
from gemini_research_mcp.engines.local import LocalDeepResearchEngine
original_local_run = LocalDeepResearchEngine.run
async def local_run(self, *args, **kwargs):
    record("local_dispatch")
    return await original_local_run(self, *args, **kwargs)
LocalDeepResearchEngine.run = local_run

if os.environ.get("E2E_SYNTHETIC") == "1":
    from research_fakes import Engines, FakeEngine
    from gemini_research_mcp.jobs import ResearchJobService
    from gemini_research_mcp.storage import get_job_storage
    engines = Engines(FakeEngine("gemini", gate=asyncio.Event()))
    service = ResearchJobService(get_job_storage(), engine_factory=engines)
    server._job_service = lambda: service

tokens = json.loads(os.environ.get("E2E_HTTP_TOKENS", "{}"))
if tokens:
    from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
    server.mcp.auth = StaticTokenVerifier(tokens={
        token: {"client_id": principal, "scopes": []}
        for token, principal in tokens.items()
    })

# Legacy progress and third-party diagnostics may contain entire reports.
logging.disable(logging.CRITICAL)
server.main()
"""


def text(result: Any) -> str:
    return "\n".join(getattr(item, "text", "") for item in result.content)


def checked(result: Any) -> Any:
    if result.is_error:
        match = re.search(r"Error \[([A-Z_]+)\]", text(result))
        pytest.fail(f"MCP tool failed: {match[1] if match else 'unclassified error'}")
    return result


class Acceptance:
    def __init__(self, root: Path, env: dict[str, str]) -> None:
        self.root = root
        self.env = env
        root.mkdir(parents=True, exist_ok=True)
        self.launcher = root / "mcp_launcher.py"
        self.launcher.write_text(LAUNCHER, encoding="utf-8")

    def directory(self, scenario: str) -> Path:
        directory = self.root / scenario
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def events(self, scenario: str) -> list[dict[str, Any]]:
        path = self.directory(scenario) / "audit.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def local_events(self) -> list[dict[str, Any]]:
        path = Path(self.env["XDG_STATE_HOME"]) / "ldr-observer" / "audit.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    async def assert_local_teardown(self, since: int) -> None:
        pids = {
            event["pid"] for event in self.local_events()[since:] if event["event"] == "child_start"
        }
        assert pids, "Real LDR child start must be observed"
        async with asyncio.timeout(30):
            while pids:
                for pid in tuple(pids):
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        pids.remove(pid)
                if pids:
                    await asyncio.sleep(0.2)

    def assert_local_requests(self, since: int) -> None:
        successful = {
            event.get("category")
            for event in self.local_events()[since:]
            if event["event"] == "request_end" and event.get("status") == 200
        }
        assert {"ollama_inference", "wikipedia_search"} <= successful

    def save(self, name: str, value: Any) -> None:
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        path.chmod(0o600)

    def load(self, name: str) -> Any:
        return json.loads((self.root / f"{name}.json").read_text())

    def server_env(self, scenario: str, *, allowance: int = 0) -> dict[str, str]:
        directory = self.directory(scenario)
        store = directory / "store"
        store.mkdir(exist_ok=True)
        exports = directory / "exports"
        exports.mkdir(exist_ok=True)
        env = {
            **self.env,
            "E2E_SCENARIO_DIR": str(directory),
            "E2E_RESEARCH_ALLOWANCE": str(allowance),
            "GEMINI_RESEARCH_STORAGE_PATH": str(store),
            "GEMINI_RESEARCH_EXPORT_DIR": str(exports),
            "PYTHONPATH": os.pathsep.join([str(REPO / "tests"), str(REPO / "src")]),
        }
        for key in (
            "GEMINI_RESEARCH_STORAGE_URL",
            "FASTMCP_DOCKET_URL",
            "GEMINI_RESEARCH_HTTP_BEARER_TOKEN",
            "GEMINI_RESEARCH_TRANSPORT",
            "GEMINI_MODEL",
            "DEEP_RESEARCH_AGENT",
        ):
            env.pop(key, None)
        return env

    @asynccontextmanager
    async def stdio(
        self, scenario: str, *, allowance: int = 0, synthetic: bool = False
    ) -> AsyncIterator[Client]:
        env = self.server_env(scenario, allowance=allowance)
        if synthetic:
            env["E2E_SYNTHETIC"] = "1"
        with open(os.devnull, "w") as sink:
            transport = StdioTransport(
                command=sys.executable,
                args=[str(self.launcher)],
                env=env,
                cwd=str(REPO),
                keep_alive=False,
                log_file=sink,
            )
            async with Client(transport, timeout=3700) as client:
                yield client

    @asynccontextmanager
    async def http(
        self, scenario: str, *, tokens: dict[str, str], local: bool = False, allowance: int = 0
    ) -> AsyncIterator[str]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = self.server_env(scenario, allowance=allowance)
        env["E2E_HTTP_TOKENS"] = json.dumps(tokens)
        env["GEMINI_RESEARCH_LDR_ALLOW_HTTP"] = str(local).lower()
        url = f"http://127.0.0.1:{port}/mcp"
        with open(os.devnull, "w") as sink:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(self.launcher),
                "--transport",
                "streamable-http",
                "--port",
                str(port),
                env=env,
                cwd=REPO,
                stdout=sink,
                stderr=sink,
            )
            try:
                async with httpx.AsyncClient() as http:
                    async with asyncio.timeout(30):
                        while True:
                            try:
                                response = await http.get(url)
                                if response.status_code in (401, 405, 406):
                                    break
                            except httpx.ConnectError:
                                pass
                            assert process.returncode is None, "HTTP MCP process exited"
                            await asyncio.sleep(0.2)
                yield url
            finally:
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), 15)
                    except TimeoutError:
                        process.kill()
                        await process.wait()

    def kill_server(self, scenario: str) -> None:
        pid = int((self.directory(scenario) / "server.pid").read_text())
        os.kill(pid, signal.SIGKILL)

    async def wait_job(self, client: Client, *, linked: bool = False) -> dict[str, Any]:
        async with asyncio.timeout(180):
            while True:
                listing = checked(await client.call_tool("list_research_runs"))
                runs = listing.structured_content["runs"]
                if runs:
                    run = checked(
                        await client.call_tool("get_research_run", {"run_id": runs[0]["run_id"]})
                    ).structured_content["run"]
                    if not linked or run["provider_id"]:
                        return run
                await asyncio.sleep(0.25)


@pytest.fixture(scope="module")
def acceptance() -> Acceptance:
    if os.environ.get("GEMINI_RESEARCH_RUN_RELEASE_E2E") != "1":
        pytest.fail("Release E2E requires explicit GEMINI_RESEARCH_RUN_RELEASE_E2E=1")
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    assert os.environ.get("GEMINI_API_KEY"), "Gemini key is required for requested acceptance"
    raw = os.environ.get("GEMINI_RESEARCH_E2E_DIR")
    assert raw, "An external acceptance directory is required"
    root = Path(raw).resolve()
    assert not root.is_relative_to(REPO), "Live reports must stay outside the checkout"
    env = dict(os.environ)
    setup = root / "ldr-setup-config.json"
    if setup.exists():
        env.update(
            {
                key: value
                for key, value in json.loads(setup.read_text())["launch"].items()
                if key.startswith(("LDR_", "XDG_", "GEMINI_RESEARCH_LDR_"))
            }
        )
    return Acceptance(root, env)


async def submit(client: Client, name: str, args: dict[str, Any]) -> Any:
    handle = await call_tool_task(client, name, args)
    await handle.wait(timeout=3600)
    return checked(await handle.result())


async def test_00_http_principal_and_local_opt_in(acceptance: Acceptance) -> None:
    import secrets

    async with acceptance.http("http-anonymous", tokens={}) as url, Client(url) as client:
        denied = await client.call_tool("list_research_runs", raise_on_error=False)
        assert denied.is_error and "AUTH_REQUIRED" in text(denied)
        with pytest.raises(ToolError, match="AUTH_REQUIRED"):
            await submit(client, "research_run", {"query": QUERY})
    token = secrets.token_urlsafe(32)
    async with (
        acceptance.http("http-local-disabled", tokens={token: "alice"}) as url,
        Client(url, auth=token) as client,
    ):
        with pytest.raises(ToolError, match="LOCAL_ENGINE_HTTP_DISABLED"):
            await submit(client, "research_run", {"query": QUERY, "engine": "local_deep_research"})
    assert acceptance.events("http-anonymous") == []
    assert acceptance.events("http-local-disabled") == []


async def test_01_grounded_web_and_fetch(acceptance: Acceptance) -> None:
    async with acceptance.stdio("web") as client:
        result = checked(
            await client.call_tool(
                "research_web",
                {
                    "query": "Use Google Search to find Python 3.14's official release date and "
                    "two documented changes, with official source links. Answer under 150 words."
                },
            )
        )
        assert "Sources" in text(result)
        listing = checked(await client.call_tool("search_tools", {"query": "fetch webpage"}))
        assert "fetch_webpage" in text(listing)
        fetched = checked(
            await client.call_tool(
                "call_tool",
                {
                    "name": "fetch_webpage",
                    "arguments": {"url": "https://docs.python.org/3/whatsnew/3.14.html"},
                },
            )
        )
        assert "Python" in text(fetched)
        missing = await client.call_tool(
            "fetch_webpage", {"url": "https://httpbin.org/status/404"}, raise_on_error=False
        )
        assert "404" in text(missing)
        acceptance.save("web-result", result.structured_content)
    grounded = [e for e in acceptance.events("web") if e["kind"] == "generation_result"]
    assert len(grounded) == 1
    assert grounded[0]["model"] == "gemini-4-argon"
    assert "HIGH" in grounded[0]["thinking"].upper()
    assert grounded[0]["sources"] > 0 and grounded[0]["queries"] > 0


@pytest.mark.parametrize(
    "tool,agent",
    [
        ("research_deep", "deep-research-preview-04-2026"),
        ("research_deep_max", "deep-research-max-preview-04-2026"),
    ],
)
async def test_02_legacy_deep(acceptance: Acceptance, tool: str, agent: str) -> None:
    async with acceptance.stdio(tool, allowance=1) as client:
        handle = await call_tool_task(
            client,
            tool,
            {
                "query": QUERY,
                "format_instructions": "A concise Markdown comparison with source URLs.",
            },
        )
        async with asyncio.timeout(180):
            while True:
                data = json.loads(text(checked(await client.call_tool("list_research_sessions"))))
                if data["sessions"]:
                    break
                await asyncio.sleep(0.5)
        session_id = data["sessions"][0]["interaction_id"]
        acceptance.save(
            f"{tool}-started",
            {
                "task_id": handle.task_id,
                "provider_id": session_id,
            },
        )
        assert handle.task_id != session_id
        await handle.wait(timeout=3600)
        result = checked(await handle.result())
        assert text(result).strip() and re.search(r"https?://", text(result))
        sessions = checked(await client.call_tool("list_research_sessions"))
        data = json.loads(text(sessions))
        assert data["count"] == 1
        assert data["sessions"][0]["status"] == "completed"
        assert data["sessions"][0]["interaction_id"] == session_id
        acceptance.save(tool, {"provider_id": session_id, "text": text(result)})
        if tool == "research_deep":
            followup = checked(
                await client.call_tool(
                    "research_followup",
                    {
                        "interaction_id": session_id,
                        "query": "Summarize two differences using only the previous report.",
                    },
                )
            )
            assert "## Follow-up Response" in text(followup)
            for format in ("markdown", "json", "docx"):
                exported = checked(
                    await client.call_tool(
                        "export_research_session",
                        {
                            "interaction_id": session_id,
                            "format": format,
                        },
                    )
                )
                uri = re.search(r"research://exports/[\w-]+", text(exported))
                assert uri, "Export must expose a downloadable resource URI"
                contents = await client.read_resource(uri[0])
                assert len(contents) == 1
                content = contents[0]
                embedded = next(
                    item.resource for item in exported.content if item.type == "resource"
                )
                assert content.mime_type == embedded.mime_type
                if format == "docx":
                    raw = base64.b64decode(content.blob)
                    assert raw == base64.b64decode(embedded.blob)
                    assert content.mime_type == (
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                        assert "word/document.xml" in archive.namelist()
                else:
                    assert isinstance(content, TextResourceContents) and content.text.strip()
                    assert content.text == embedded.text
                    assert (
                        content.mime_type
                        == {"markdown": "text/markdown", "json": "application/json"}[format]
                    )
    creates = [e for e in acceptance.events(tool) if e["kind"] == "research_create"]
    assert len(creates) == 1 and creates[0]["agent"] == agent


async def test_03_native_process_recovery(acceptance: Acceptance) -> None:
    scenario = "native-recovery"
    async with acceptance.stdio(scenario, allowance=1) as client:
        handle = await call_tool_task(client, "research_run", {"query": QUERY})
        run = await acceptance.wait_job(client, linked=True)
        assert run["status"] == "running" and run["run_id"] != run["provider_id"]
        task_result = handle.create_result
        acceptance.save("native-recovery-start", run)
        acceptance.kill_server(scenario)
    async with acceptance.stdio(scenario, allowance=1) as client:
        with pytest.raises(MCPError, match=r"(?i)task.*not found"):
            await ToolTask(client, "research_run", task_result).status()
        inspected = checked(
            await client.call_tool("get_research_run", {"run_id": run["run_id"]})
        ).structured_content["run"]
        assert inspected["status"] == "interrupted"
        checked(
            await client.call_tool("get_research_run", {"run_id": run["run_id"], "refresh": True})
        )
        async with asyncio.timeout(3600):
            while True:
                checked(
                    await client.call_tool(
                        "resume_research", {"interaction_id": run["provider_id"]}
                    )
                )
                final = checked(
                    await client.call_tool("get_research_run", {"run_id": run["run_id"]})
                ).structured_content["run"]
                if final["status"] == "completed":
                    break
                assert final["status"] not in ("failed", "expired", "cancelled")
                await asyncio.sleep(10)
        assert final["provider_id"] == run["provider_id"] and final["report"].strip()
        assert final["expires_at"] == run["expires_at"]
        acceptance.save("native-recovery-final", final)
    assert sum(e["kind"] == "research_create" for e in acceptance.events(scenario)) == 1


async def test_04_real_local_and_interruption(acceptance: Acceptance) -> None:
    assert acceptance.env.get("GEMINI_RESEARCH_LDR_COMMAND"), "Real LDR is required"
    baseline = len(acceptance.local_events())
    async with acceptance.stdio("local-complete") as client:
        result = await submit(
            client, "research_run", {"query": QUERY, "engine": "local_deep_research"}
        )
        run = result.structured_content["run"]
        assert run["status"] == "completed" and run["provider_id"] is None
        assert run["report"].strip() and run["sources"]
        assert run["backend"] == "local-deep-research" and run["recovery"] == "none"
        assert run["model"] == acceptance.env["LDR_LLM_MODEL"]
        acceptance.save("local-complete", run)
    acceptance.assert_local_requests(baseline)
    await acceptance.assert_local_teardown(baseline)
    scenario = "local-interruption"
    baseline = len(acceptance.local_events())
    async with acceptance.stdio(scenario) as client:
        handle = await call_tool_task(
            client, "research_run", {"query": QUERY, "engine": "local_deep_research"}
        )
        run = await acceptance.wait_job(client)
        async with asyncio.timeout(180):
            while not any(
                e["event"] == "request_start" and e.get("category") == "ollama_inference"
                for e in acceptance.local_events()[baseline:]
            ):
                await asyncio.sleep(0.25)
        await handle.cancel()
        await handle.wait(timeout=30)
        await acceptance.assert_local_teardown(baseline)
    async with acceptance.stdio(scenario) as client:
        final = checked(
            await client.call_tool("get_research_run", {"run_id": run["run_id"], "refresh": True})
        ).structured_content["run"]
        assert final["status"] == "interrupted" and final["provider_id"] is None
        assert final["recovery"] == "none" and final["cancellation_requested"]
        acceptance.save(scenario, final)
    assert sum(e["kind"] == "local_dispatch" for e in acceptance.events(scenario)) == 1


async def test_05_http_comparison_and_bundle(acceptance: Acceptance) -> None:
    import secrets

    alice, bob = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    scenario = "http-comparison"
    baseline = len(acceptance.local_events())
    async with acceptance.http(
        scenario,
        tokens={alice: "acceptance-alice", bob: "acceptance-bob"},
        local=True,
        allowance=1,
    ) as url:
        async with Client(url, auth=alice, timeout=3700) as client:
            result = await submit(
                client,
                "research_compare",
                {"query": QUERY, "engines": ["gemini_max", "local_deep_research"]},
            )
            run = result.structured_content["run"]
            acceptance.save(scenario, run)
            assert run["status"] == "completed" and run["synthesis"] == "caller_owned"
            assert [child["engine"] for child in run["children"]] == [
                "gemini_max",
                "local_deep_research",
            ]
            assert all(c["report"].strip() and c["sources"] for c in run["children"])
            acceptance.assert_local_requests(baseline)
            before = acceptance.events(scenario)
            local_before = acceptance.local_events()
            uris = []
            for format in ("json", "markdown"):
                exported = checked(
                    await client.call_tool(
                        "export_research_bundle",
                        {
                            "run_id": run["run_id"],
                            "format": format,
                        },
                    )
                )
                uri = exported.structured_content["bundle"]["resource_uri"]
                uris.append(uri)
                content = (await client.read_resource(uri))[0]
                assert isinstance(content, TextResourceContents)
                assert "owner_scope" not in content.text and "task_id" not in content.text
                if format == "json":
                    bundle = json.loads(content.text)
                    assert [e["report"] for e in bundle["engines"]] == [
                        c["report"] for c in run["children"]
                    ]
                else:
                    assert all(c["report"] in content.text for c in run["children"])
            assert acceptance.events(scenario) == before
            assert acceptance.local_events() == local_before
        native_id = run["children"][0]["provider_id"]
        async with Client(url, auth=bob) as client:
            assert (
                checked(await client.call_tool("list_research_runs")).structured_content["runs"]
                == []
            )
            assert (
                json.loads(text(checked(await client.call_tool("list_research_sessions"))))[
                    "sessions"
                ]
                == []
            )
            denied = await client.call_tool(
                "get_research_run", {"run_id": run["run_id"]}, raise_on_error=False
            )
            assert denied.is_error and "RUN_NOT_FOUND" in text(denied)
            resumed = checked(
                await client.call_tool("resume_research", {"interaction_id": native_id})
            )
            assert json.loads(text(resumed))["status"] == "not_found"
            exported = checked(
                await client.call_tool(
                    "export_research_session", {"interaction_id": native_id, "format": "docx"}
                )
            )
            assert "Session not found" in json.loads(text(exported))["error"]
            for uri in uris:
                with pytest.raises(MCPError, match="not found"):
                    await client.read_resource(uri)
            assert (
                json.loads((await client.read_resource("research://exports"))[0].text)["count"] == 0
            )
        creates = [e for e in acceptance.events(scenario) if e["kind"] == "research_create"]
        assert len(creates) == 1
        assert creates[0]["agent"] == "deep-research-max-preview-04-2026"
    await acceptance.assert_local_teardown(baseline)
