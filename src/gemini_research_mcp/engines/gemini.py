"""Gemini-native Deep Research engine (Interactions API, background mode)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from google import genai
from google.genai import types as genai_types

from gemini_research_mcp.citations import extract_citations_from_text
from gemini_research_mcp.config import (
    DEFAULT_TIMEOUT,
    STREAM_POLL_INTERVAL,
    get_api_key,
    get_deep_research_agent,
)
from gemini_research_mcp.deep import (
    _extract_images_from_interaction,
    _extract_text_from_interaction,
    _extract_usage,
    _get_interaction_field,
)
from gemini_research_mcp.engines.base import (
    ProgressCallback,
    ProviderIdCallback,
    SubmittingCallback,
)
from gemini_research_mcp.types import (
    BACKEND_GEMINI,
    PROVENANCE_REPORT,
    DeepResearchAgent,
    EngineDescriptor,
    EngineError,
    EngineReport,
    EngineRequest,
    EngineSnapshot,
    EngineUsage,
    JobStatus,
    ResearchSource,
)

logger = logging.getLogger(__name__)

_LIMITATIONS = (
    "Provider-side job: recoverable by provider interaction ID.",
    "Cancellation is only reported once the provider confirms it.",
)


def describe_gemini(engine_id: str) -> EngineDescriptor:
    """Descriptor for a Gemini selector; never creates a client or calls out."""
    reason: str | None = None
    model: str | None
    try:
        if engine_id == "gemini_max":
            model = DeepResearchAgent.DEEP_RESEARCH_MAX.value
        else:
            model = get_deep_research_agent().value
    except ValueError as exc:
        model = None
        reason = str(exc)
    if reason is None:
        try:
            get_api_key()
        except ValueError:
            reason = "GEMINI_API_KEY is not set."
    return EngineDescriptor(
        id=engine_id,
        label="Gemini Deep Research Max" if engine_id == "gemini_max" else "Gemini Deep Research",
        backend=BACKEND_GEMINI,
        execution_mode="background_interactions",
        recovery="provider_id",
        model=model,
        available=reason is None,
        availability_reason=reason,
        limitations=_LIMITATIONS,
    )


_AGENT_CONFIG: dict[str, Any] = {
    "type": "deep-research",
    "thinking_summaries": "auto",
    "visualization": "off",
    "collaborative_planning": False,
}
_GET_ATTEMPTS = 3
_GET_BACKOFF_SECONDS = 1.0
_CREATE_TIMEOUT_SECONDS = 120.0

_STATUS_MAP: dict[str, JobStatus] = {
    "completed": JobStatus.COMPLETED,
    "failed": JobStatus.FAILED,
    "cancelled": JobStatus.CANCELLED,
    "canceled": JobStatus.CANCELLED,
    "incomplete": JobStatus.FAILED,
    "requires_action": JobStatus.INTERRUPTED,
}


def _http_status(exc: BaseException) -> int | None:
    for name in ("status_code", "code"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
    return None


def _usage(interaction: Any) -> EngineUsage | None:
    raw = _extract_usage(interaction)
    if raw is None:
        return None
    return EngineUsage(
        prompt_tokens=raw.prompt_tokens,
        completion_tokens=raw.completion_tokens,
        total_tokens=raw.total_tokens,
    )


def _parsed_sources(text: str) -> tuple[ResearchSource, ...]:
    _, citations = extract_citations_from_text(text)
    return tuple(
        ResearchSource(uri=c.redirect_url, title=c.domain, provenance=PROVENANCE_REPORT)
        for c in citations
    )


class GeminiEngine:
    """Request-owned Gemini adapter over the Interactions API (background mode).

    The create call is never retried: the SDK retry surface is disabled on a
    dedicated client, so an ambiguous create failure surfaces as
    ``SUBMISSION_UNKNOWN`` instead of risking a duplicate paid job.
    """

    poll_interval: float = STREAM_POLL_INTERVAL
    timeout: float = DEFAULT_TIMEOUT

    def __init__(
        self,
        engine_id: str,
        *,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.engine_id = engine_id
        self.descriptor = describe_gemini(engine_id)
        self._client_factory = client_factory or self._default_client
        self._client: Any = None

    @staticmethod
    def _default_client() -> Any:
        client = genai.Client(
            api_key=get_api_key(),
            http_options=genai_types.HttpOptions(
                retry_options=genai_types.HttpRetryOptions(attempts=1)
            ),
        )
        # HttpRetryOptions alone still leaves one generated-SDK retry on create.
        client.aio.interactions.sdk_configuration.retry_config = None
        return client

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def _agent(self) -> str:
        if self.engine_id == "gemini_max":
            return DeepResearchAgent.DEEP_RESEARCH_MAX.value
        return get_deep_research_agent().value

    async def preflight(self) -> None:
        if not self.descriptor.available:
            raise EngineError(
                "ENGINE_UNAVAILABLE",
                self.descriptor.availability_reason or "Gemini engine unavailable.",
            )

    async def run(
        self,
        request: EngineRequest,
        *,
        on_submitting: SubmittingCallback,
        on_provider_id: ProviderIdCallback,
        on_progress: ProgressCallback,
    ) -> EngineReport:
        await self.preflight()
        client = self._get_client()
        started = time.monotonic()
        await on_submitting()
        try:
            interaction = await asyncio.wait_for(
                client.aio.interactions.create(
                    input=request.query,
                    agent=self._agent(),
                    background=True,
                    agent_config=dict(_AGENT_CONFIG),
                ),
                timeout=_CREATE_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            status = _http_status(exc)
            if status is not None and 400 <= status < 500:
                raise EngineError(
                    "SUBMISSION_REJECTED", f"Gemini rejected the request (HTTP {status})."
                ) from exc
            raise EngineError(
                "SUBMISSION_UNKNOWN",
                "Gemini submission outcome is unknown; the job was not retried.",
            ) from exc

        provider_id = _get_interaction_field(interaction, "id")
        if not isinstance(provider_id, str) or not provider_id:
            raise EngineError("SUBMISSION_UNKNOWN", "Gemini returned no interaction ID.")
        await on_provider_id(provider_id)
        await on_progress("Gemini research started.")

        snapshot = self._interpret(interaction, provider_id)
        last_status: str | None = None
        while not snapshot.status.is_terminal:
            if time.monotonic() - started >= self.timeout:
                raise EngineError(
                    "POLL_TIMEOUT",
                    "Gemini research is still running; refresh it by job ID later.",
                )
            await asyncio.sleep(self.poll_interval)
            interaction = await self._get(provider_id)
            snapshot = self._interpret(interaction, provider_id)
            raw_status = str(_get_interaction_field(interaction, "status") or "unknown")
            if raw_status != last_status:
                last_status = raw_status
                await on_progress(f"Gemini research {raw_status}.")

        elapsed = time.monotonic() - started
        if snapshot.report is not None:
            return replace(snapshot.report, duration_seconds=elapsed)
        raise EngineError(
            snapshot.error_code or "PROVIDER_FAILED",
            snapshot.error_message or "Gemini research did not complete.",
        )

    async def refresh(self, provider_id: str) -> EngineSnapshot:
        await self.preflight()
        interaction = await self._get(provider_id)
        return self._interpret(interaction, provider_id)

    async def _get(self, provider_id: str) -> Any:
        client = self._get_client()
        last_exc: Exception | None = None
        for attempt in range(_GET_ATTEMPTS):
            try:
                return await client.aio.interactions.get(id=provider_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exc = exc
                status = _http_status(exc)
                if status is not None and 400 <= status < 500 and status != 429:
                    break
                if attempt + 1 < _GET_ATTEMPTS:
                    await asyncio.sleep(_GET_BACKOFF_SECONDS * (attempt + 1))
        status = _http_status(last_exc) if last_exc else None
        if status == 404:
            raise EngineError("PROVIDER_NOT_FOUND", "Gemini no longer has this interaction.")
        raise EngineError("POLL_FAILED", "Could not read the Gemini interaction status.")

    def _interpret(self, interaction: Any, provider_id: str) -> EngineSnapshot:
        raw_status = str(_get_interaction_field(interaction, "status") or "unknown").lower()
        status = _STATUS_MAP.get(raw_status, JobStatus.RUNNING)
        if not status.is_terminal:
            return EngineSnapshot(provider_id=provider_id, status=JobStatus.RUNNING)

        text = _extract_text_from_interaction(interaction)
        has_text = bool(text and text.strip())
        report: EngineReport | None = None
        warnings: tuple[str, ...] = ()
        error_code: str | None = None
        error_message: str | None = None

        if raw_status == "completed":
            if has_text:
                assert text is not None
                report = self._report(interaction, provider_id, text, "completed")
            else:
                status = JobStatus.FAILED
                error_code = "EMPTY_REPORT"
                error_message = "Gemini completed without report text."
        elif raw_status == "incomplete":
            if has_text:
                assert text is not None
                report = self._report(interaction, provider_id, text, "incomplete")
                warnings = ("Gemini returned an incomplete report.",)
            else:
                error_code = "PROVIDER_INCOMPLETE"
                error_message = "Gemini ended incomplete without report text."
        elif raw_status == "requires_action":
            error_code = "PROVIDER_REQUIRES_ACTION"
            error_message = "Gemini requires action; use resume_research with the interaction."
            warnings = (error_message,)
        elif status is JobStatus.CANCELLED:
            error_code = "PROVIDER_CANCELLED"
            error_message = "Gemini reported the job as cancelled."
        else:
            error_code = "PROVIDER_FAILED"
            error_message = "Gemini reported the job as failed."

        return EngineSnapshot(
            provider_id=provider_id,
            status=status,
            report=report,
            warnings=warnings,
            error_code=error_code,
            error_message=error_message,
        )

    def _report(self, interaction: Any, provider_id: str, text: str, status: str) -> EngineReport:
        return EngineReport(
            text=text,
            status=status,
            sources=_parsed_sources(text),
            usage=_usage(interaction),
            model=self._agent(),
            provider_id=provider_id,
            images=tuple(_extract_images_from_interaction(interaction)),
        )

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        try:
            await client.aio.aclose()
        except Exception:  # noqa: BLE001
            logger.debug("Gemini client close failed", exc_info=True)
