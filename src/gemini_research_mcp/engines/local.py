"""Optional Local Deep Research (LDR) bridge over a blocking stdio MCP child.

Contract verified against ``LearningCircuit/local-deep-research`` at
``362d65652156732b08a704f0d3faa77996b4f0de`` (``docs/mcp-server.md``,
``src/local_deep_research/mcp/server.py``): ``detailed_research`` blocks until
done and there is no job ID, poll or resume. Hence ``recovery="none"``.

The operator installs LDR, Ollama and the search backend; nothing is installed
or started here. The child is launched only from an absolute operator-supplied
path, with an allowlisted environment and its stderr discarded.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import os
import re
import time
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from typing import IO, Any
from urllib.parse import urlparse

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from gemini_research_mcp.config import DEFAULT_TIMEOUT, LDR_COMMAND_ENV_VAR, get_ldr_command
from gemini_research_mcp.engines.base import (
    ProgressCallback,
    ProviderIdCallback,
    SubmittingCallback,
)
from gemini_research_mcp.types import (
    BACKEND_LOCAL,
    PROVENANCE_ENGINE,
    EngineDescriptor,
    EngineError,
    EngineReport,
    EngineRequest,
    EngineSnapshot,
    ResearchSource,
)

LOCAL_ENGINE_ID = "local_deep_research"
PREFLIGHT_TIMEOUT_SECONDS = 30.0
# The outer asyncio timeout must win so the failure is classified as a timeout.
_CLIENT_TIMEOUT_MARGIN = 5.0

LDR_PROVIDER_ENV = "LDR_LLM_PROVIDER"
LDR_MODEL_ENV = "LDR_LLM_MODEL"
LDR_OLLAMA_URL_ENV = "LDR_LLM_OLLAMA_URL"
LDR_SEARCH_TOOL_ENV = "LDR_SEARCH_TOOL"
LDR_SEARXNG_URL_ENV = "LDR_SEARCH_ENGINE_WEB_SEARXNG_DEFAULT_PARAMS_INSTANCE_URL"

# Names forwarded from the operator environment besides the validated LDR_* ones.
_RUNTIME_ENV = (
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "XDG_STATE_HOME",
)
# Search tools that read local files or the LDR library instead of the web.
_FORBIDDEN_SEARCH_TOOLS = frozenset({"auto", "local_all", "library", "local"})
_FORBIDDEN_SEARCH_PREFIXES = ("collection_", "local_")

_LIMITATIONS = (
    "Blocking call with no provider ID: not recoverable after a restart.",
    "Cancellation is best-effort and local; there is no provider confirmation.",
    "Requires operator-installed Local Deep Research, Ollama on loopback and explicit search.",
    "Child process filesystem access is not sandboxed.",
)


@dataclass(frozen=True, slots=True)
class LdrSettings:
    """Validated operator configuration for the LDR child."""

    model: str
    ollama_url: str
    search_tool: str
    searxng_url: str | None = None

    def child_env(self, environ: Mapping[str, str] | None = None) -> dict[str, str]:
        """Allowlisted child environment; never a copy of the parent's."""
        source = os.environ if environ is None else environ
        env = {
            LDR_PROVIDER_ENV: "ollama",
            LDR_MODEL_ENV: self.model,
            LDR_OLLAMA_URL_ENV: self.ollama_url,
            LDR_SEARCH_TOOL_ENV: self.search_tool,
        }
        if self.searxng_url:
            env[LDR_SEARXNG_URL_ENV] = self.searxng_url
        for name in _RUNTIME_ENV:
            value = source.get(name)
            if value:
                env[name] = value
        return env


def _is_loopback_url(raw: str) -> bool:
    try:
        parsed = urlparse(raw)
        host = parsed.hostname
        _ = parsed.port
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_http_url(raw: str) -> bool:
    try:
        parsed = urlparse(raw)
        _ = parsed.port
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname) and not parsed.username


def load_ldr_settings(environ: Mapping[str, str] | None = None) -> LdrSettings:
    """Validate the operator configuration; raises ``LOCAL_CONFIG_INVALID``."""
    source = os.environ if environ is None else environ

    def get(name: str) -> str:
        return (source.get(name) or "").strip()

    if get(LDR_PROVIDER_ENV).lower() != "ollama":
        raise EngineError("LOCAL_CONFIG_INVALID", f"{LDR_PROVIDER_ENV} must be 'ollama'.")
    model = get(LDR_MODEL_ENV)
    if not model:
        raise EngineError("LOCAL_CONFIG_INVALID", f"{LDR_MODEL_ENV} must pin an Ollama model.")
    ollama_url = get(LDR_OLLAMA_URL_ENV)
    if not _is_loopback_url(ollama_url):
        raise EngineError(
            "LOCAL_CONFIG_INVALID", f"{LDR_OLLAMA_URL_ENV} must be an http(s) loopback URL."
        )
    search_tool = get(LDR_SEARCH_TOOL_ENV)
    lowered = search_tool.lower()
    if (
        not search_tool
        or lowered in _FORBIDDEN_SEARCH_TOOLS
        or lowered.startswith(_FORBIDDEN_SEARCH_PREFIXES)
    ):
        raise EngineError(
            "LOCAL_CONFIG_INVALID",
            f"{LDR_SEARCH_TOOL_ENV} must name an explicit web search engine (e.g. searxng).",
        )
    searxng_url = get(LDR_SEARXNG_URL_ENV) or None
    if searxng_url is not None and not _is_http_url(searxng_url):
        raise EngineError("LOCAL_CONFIG_INVALID", f"{LDR_SEARXNG_URL_ENV} must be an http(s) URL.")
    return LdrSettings(model, ollama_url, search_tool, searxng_url)


def describe_local() -> EngineDescriptor:
    """Descriptor for the LDR bridge; never launches the command."""
    reason: str | None = None
    model: str | None = None
    if get_ldr_command() is None:
        reason = f"{LDR_COMMAND_ENV_VAR} must be set to an absolute executable path."
    else:
        try:
            model = load_ldr_settings().model
        except EngineError as exc:
            reason = exc.message
    return EngineDescriptor(
        id=LOCAL_ENGINE_ID,
        label="Local Deep Research",
        backend=BACKEND_LOCAL,
        execution_mode="blocking_stdio",
        recovery="none",
        model=model,
        available=reason is None,
        availability_reason=reason,
        limitations=_LIMITATIONS,
    )


_ERROR_TYPE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


def _payload_status_error(payload: Mapping[str, Any]) -> EngineError:
    kind = payload.get("error_type")
    suffix = f" ({kind})" if isinstance(kind, str) and _ERROR_TYPE.match(kind) else ""
    return EngineError("LOCAL_ENGINE_FAILED", f"Local Deep Research reported an error{suffix}.")


def _as_sources(raw: object) -> tuple[ResearchSource, ...]:
    if not isinstance(raw, list):
        return ()
    sources: list[ResearchSource] = []
    for item in raw:
        uri: str | None = None
        title: str | None = None
        if isinstance(item, str):
            uri = item.strip() or None
        elif isinstance(item, Mapping):
            for key in ("url", "link", "uri"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    uri = value.strip()
                    break
            raw_title = item.get("title")
            if isinstance(raw_title, str) and raw_title.strip():
                title = raw_title.strip()
        if uri or title:
            sources.append(ResearchSource(uri=uri, title=title, provenance=PROVENANCE_ENGINE))
    return tuple(sources)


def normalize_ldr_result(payload: object) -> EngineReport:
    """Turn a ``detailed_research`` payload into a report or a typed failure."""
    if not isinstance(payload, Mapping):
        raise EngineError("LOCAL_ENGINE_FAILED", "Local Deep Research returned no object payload.")
    if payload.get("status") != "success":
        raise _payload_status_error(payload)
    text: str | None = None
    for key in ("formatted_findings", "summary"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            text = value
            break
    if text is None:
        raise EngineError("LOCAL_EMPTY_REPORT", "Local Deep Research returned no report text.")
    return EngineReport(text=text, sources=_as_sources(payload.get("sources")))


def _extract_payload(result: Any) -> object:
    """Structured content first, otherwise one unambiguous JSON object."""
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, Mapping):
        return structured
    texts = [
        block.text
        for block in getattr(result, "content", None) or []
        if isinstance(getattr(block, "text", None), str)
    ]
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except ValueError:
            pass
    raise EngineError("LOCAL_ENGINE_FAILED", "Local Deep Research returned an unreadable result.")


class _LocalSlot:
    """One local research run per process."""

    def __init__(self) -> None:
        self._held = False

    def try_acquire(self) -> bool:
        if self._held:
            return False
        self._held = True
        return True

    def release(self) -> None:
        self._held = False

    @property
    def held(self) -> bool:
        return self._held


_SLOT = _LocalSlot()


def local_slot_busy() -> bool:
    return _SLOT.held


class LocalDeepResearchEngine:
    """Request-owned LDR adapter holding one stdio client from preflight to close."""

    def __init__(self) -> None:
        self.engine_id = LOCAL_ENGINE_ID
        self.descriptor = describe_local()
        self.research_timeout: float = DEFAULT_TIMEOUT
        self._stack: AsyncExitStack | None = None
        self._client: Any = None
        self._sink: IO[str] | None = None
        self._reserved = False
        self._preflighted = False

    def reserve(self) -> None:
        """Take the process-wide local slot; a comparison does this before native dispatch."""
        if self._reserved:
            return
        if not _SLOT.try_acquire():
            raise EngineError("LOCAL_ENGINE_BUSY", "Another local research run is in progress.")
        self._reserved = True

    def _make_client(self, command: str, settings: LdrSettings) -> Any:
        self._sink = open(os.devnull, "w")  # noqa: SIM115 - closed in aclose
        transport = StdioTransport(
            command=command,
            args=[],
            env=settings.child_env(),
            keep_alive=False,
            log_file=self._sink,
        )
        return Client(transport, timeout=self.research_timeout + _CLIENT_TIMEOUT_MARGIN)

    async def preflight(self) -> None:
        if not self.descriptor.available:
            raise EngineError(
                "ENGINE_UNAVAILABLE",
                self.descriptor.availability_reason or "Local engine unavailable.",
            )
        if self._preflighted:
            return
        command = get_ldr_command()
        if command is None:  # pragma: no cover - descriptor guards this
            raise EngineError("ENGINE_UNAVAILABLE", "LDR command is not configured.")
        settings = load_ldr_settings()
        try:
            async with asyncio.timeout(PREFLIGHT_TIMEOUT_SECONDS):
                self._stack = AsyncExitStack()
                self._client = self._make_client(command, settings)
                await self._stack.enter_async_context(self._client)
                tools = {tool.name: tool for tool in await self._client.list_tools()}
                self._check_schemas(tools)
                result = await self._client.call_tool("get_configuration", raise_on_error=False)
                self._check_configuration(result, settings)
        except EngineError:
            await self._teardown()
            raise
        except TimeoutError:
            await self._teardown()
            raise EngineError(
                "LOCAL_ENGINE_UNAVAILABLE", "Local Deep Research preflight timed out."
            ) from None
        except asyncio.CancelledError:
            await self._teardown()
            raise
        except Exception as exc:
            await self._teardown()
            raise EngineError(
                "LOCAL_ENGINE_UNAVAILABLE",
                "Could not start or initialize Local Deep Research.",
            ) from exc
        self._preflighted = True

    @staticmethod
    def _check_schemas(tools: Mapping[str, Any]) -> None:
        for name in ("detailed_research", "get_configuration"):
            if name not in tools:
                raise EngineError(
                    "LOCAL_CONFIG_INVALID", f"Local Deep Research does not expose {name}."
                )
        tool = tools["detailed_research"]
        schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
        properties = schema.get("properties") or {}
        if "query" not in properties or "query" not in (schema.get("required") or []):
            raise EngineError(
                "LOCAL_CONFIG_INVALID", "detailed_research has an unexpected input schema."
            )

    @staticmethod
    def _check_configuration(result: Any, settings: LdrSettings) -> None:
        if getattr(result, "is_error", False):
            raise EngineError("LOCAL_CONFIG_INVALID", "Could not read LDR configuration.")
        payload = _extract_payload(result)
        if not isinstance(payload, Mapping) or payload.get("status") != "success":
            raise EngineError("LOCAL_CONFIG_INVALID", "Could not read LDR configuration.")
        config = payload.get("config")
        llm = config.get("llm") if isinstance(config, Mapping) else None
        search = config.get("search") if isinstance(config, Mapping) else None
        if not isinstance(llm, Mapping) or not isinstance(search, Mapping):
            raise EngineError("LOCAL_CONFIG_INVALID", "LDR configuration is unreadable.")
        if str(llm.get("provider", "")).lower() != "ollama":
            raise EngineError("LOCAL_CONFIG_INVALID", "LDR is not using the Ollama provider.")
        if llm.get("model") != settings.model:
            raise EngineError("LOCAL_CONFIG_INVALID", "LDR is not using the pinned model.")
        if search.get("default_engine") != settings.search_tool:
            raise EngineError("LOCAL_CONFIG_INVALID", "LDR is not using the explicit search tool.")

    async def run(
        self,
        request: EngineRequest,
        *,
        on_submitting: SubmittingCallback,
        on_provider_id: ProviderIdCallback,
        on_progress: ProgressCallback,
    ) -> EngineReport:
        self.reserve()
        await self.preflight()
        started = time.monotonic()
        await on_submitting()
        await on_progress("Local Deep Research is running (blocking, not resumable).")
        try:
            async with asyncio.timeout(self.research_timeout):
                result = await self._client.call_tool(
                    "detailed_research",
                    {"query": request.query},
                    timeout=self.research_timeout + _CLIENT_TIMEOUT_MARGIN,
                    raise_on_error=False,
                )
        except TimeoutError:
            raise EngineError(
                "LOCAL_ENGINE_TIMEOUT", "Local Deep Research did not finish in time."
            ) from None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise EngineError("LOCAL_ENGINE_FAILED", "Local Deep Research call failed.") from exc
        if getattr(result, "is_error", False):
            raise EngineError("LOCAL_ENGINE_FAILED", "Local Deep Research reported an error.")
        report = normalize_ldr_result(_extract_payload(result))
        return replace(
            report,
            model=self.descriptor.model,
            duration_seconds=time.monotonic() - started,
        )

    async def refresh(self, provider_id: str) -> EngineSnapshot:
        raise EngineError("LOCAL_REFRESH_UNSUPPORTED", "Local runs cannot be refreshed.")

    async def _teardown(self) -> None:
        stack, self._stack = self._stack, None
        self._client = None
        self._preflighted = False
        try:
            if stack is not None:
                with contextlib.suppress(Exception):
                    await stack.aclose()
        finally:
            if self._sink is not None:
                with contextlib.suppress(OSError):
                    self._sink.close()
                self._sink = None

    async def aclose(self) -> None:
        try:
            await self._teardown()
        finally:
            if self._reserved:
                self._reserved = False
                _SLOT.release()
