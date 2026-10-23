"""
Data types for Gemini Research MCP Server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# =============================================================================
# Error Categories (inspired by DanDaDaDanDan/mcp-gemini)
# =============================================================================


class ErrorCategory(StrEnum):
    """Categorized error types for programmatic handling."""

    AUTH_ERROR = "AUTH_ERROR"  # API key invalid or missing
    RATE_LIMIT = "RATE_LIMIT"  # Quota exceeded, retry later
    CONTENT_BLOCKED = "CONTENT_BLOCKED"  # Content policy violation
    SAFETY_BLOCK = "SAFETY_BLOCK"  # Safety filter triggered
    TIMEOUT = "TIMEOUT"  # Research exceeded max time
    NOT_FOUND = "NOT_FOUND"  # Interaction ID not found
    RESEARCH_FAILED = "RESEARCH_FAILED"  # Research task failed
    RESEARCH_CANCELLED = "RESEARCH_CANCELLED"  # Research was cancelled
    UNSUPPORTED_FEATURE = "UNSUPPORTED_FEATURE"  # Feature disabled or unavailable
    INTERNAL_ERROR = "INTERNAL_ERROR"  # Unexpected internal error
    API_ERROR = "API_ERROR"  # Other API errors


class DeepResearchAgent(StrEnum):
    """Supported agent for deep research."""

    DEEP_RESEARCH = "deep-research-preview-04-2026"
    DEEP_RESEARCH_MAX = "deep-research-max-preview-04-2026"
    DEEP_RESEARCH_PRO = "deep-research-pro-preview-12-2025"


def _categorize_error_message(message: str) -> ErrorCategory:
    """Infer error category from error message."""
    msg_lower = message.lower()

    if "api key" in msg_lower or "unauthorized" in msg_lower or "401" in msg_lower:
        return ErrorCategory.AUTH_ERROR
    if "rate" in msg_lower or "quota" in msg_lower or "429" in msg_lower:
        return ErrorCategory.RATE_LIMIT
    if "safety" in msg_lower:
        return ErrorCategory.SAFETY_BLOCK
    if "blocked" in msg_lower or "content policy" in msg_lower:
        return ErrorCategory.CONTENT_BLOCKED
    if "timeout" in msg_lower or "timed out" in msg_lower:
        return ErrorCategory.TIMEOUT
    if "not found" in msg_lower or "404" in msg_lower:
        return ErrorCategory.NOT_FOUND
    if "cancelled" in msg_lower or "canceled" in msg_lower:
        return ErrorCategory.RESEARCH_CANCELLED
    if "failed" in msg_lower:
        return ErrorCategory.RESEARCH_FAILED

    return ErrorCategory.API_ERROR


# =============================================================================
# Exceptions
# =============================================================================


class DeepResearchError(Exception):
    """Base error for Deep Research operations.

    Provides structured error information with error codes for programmatic handling.
    """

    def __init__(
        self,
        code: str,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        category: ErrorCategory | None = None,
    ):
        """Initialize a structured Deep Research error.

        `message` is optional for compatibility with runtime/task wrappers that may
        only provide an error code. We normalize to a deterministic fallback message
        so error construction itself never masks the original failure path.
        """
        resolved_message = (message or "").strip()
        if not resolved_message and details:
            details_message = details.get("message")
            if isinstance(details_message, str) and details_message.strip():
                resolved_message = details_message.strip()
        if not resolved_message:
            resolved_message = "No error message provided."

        self.code = code
        self.message = resolved_message
        self.details = details or {}
        # Auto-categorize if not provided
        self.category = category or _categorize_error_message(resolved_message)
        super().__init__(f"{code}: {resolved_message}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "code": self.code,
            "category": self.category.value if self.category else None,
            "message": self.message,
            "details": self.details,
        }

    @property
    def is_retryable(self) -> bool:
        """Whether this error might succeed on retry."""
        return self.category in (
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.TIMEOUT,
            ErrorCategory.API_ERROR,
        )


# =============================================================================
# Source Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class Source:
    """A source/citation from grounded search."""

    uri: str
    title: str


@dataclass(slots=True)
class ParsedCitation:
    """A citation extracted from the report with resolved URL.

    Deep Research reports include citations with vertexaisearch redirect URLs.
    This class stores both the original redirect and the resolved real URL.
    """

    number: int
    domain: str
    url: str | None = None
    title: str | None = None
    redirect_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "number": self.number,
            "domain": self.domain,
            "url": self.url,
            "title": self.title,
            "redirect_url": self.redirect_url,
        }


# =============================================================================
# Usage Tracking
# =============================================================================


@dataclass(slots=True)
class DeepResearchUsage:
    """Token usage and cost information for a Deep Research task."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_cost: float | None = None
    completion_cost: float | None = None
    total_cost: float | None = None
    raw_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }


# =============================================================================
# Research Results
# =============================================================================


@dataclass(slots=True)
class ResearchResult:
    """Result from quick_research()."""

    text: str
    sources: list[Source] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    thinking_summary: str | None = None


@dataclass(slots=True)
class DeepResearchResult:
    """Result from deep_research()."""

    text: str
    text_without_sources: str | None = None
    citations: list[Source] = field(default_factory=list)
    parsed_citations: list[ParsedCitation] = field(default_factory=list)
    thinking_summaries: list[str] = field(default_factory=list)
    interaction_id: str | None = None
    usage: DeepResearchUsage | None = None
    duration_seconds: float | None = None
    raw_interaction: Any = None
    images: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        citations = [c.to_dict() for c in self.parsed_citations] if self.parsed_citations else []
        return {
            "id": self.interaction_id,
            "text": self.text,
            "text_without_sources": self.text_without_sources,
            "citations": citations,
            "thinking_summaries": self.thinking_summaries,
            "usage": self.usage.to_dict() if self.usage else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(slots=True)
class DeepResearchProgress:
    """Progress update from streaming deep research."""

    # "start", "thought", "text", "image", "action", "complete", "plan_ready",
    # "error", "status". "image" is distinct from "text" so image content/deltas
    # are never concatenated into the report text. "plan_ready" is emitted when
    # collaborative_planning=True and the agent is waiting for plan approval
    # (Interactions API status "requires_action").
    event_type: str
    content: str | None = None  # For "image": base64-encoded data, when provided inline.
    interaction_id: str | None = None
    event_id: str | None = None  # For stream resumption after disconnection
    image_mime_type: str | None = None  # For "image" events: e.g. "image/png".
    image_uri: str | None = None  # For "image" events: hosted URI, when data isn't inlined.


# =============================================================================
# File Search Store (RAG)
# TODO: These types are defined for future use with file search capabilities
# =============================================================================


@dataclass(frozen=True, slots=True)
class FileSearchStore:
    """A file search store for RAG (future use)."""

    name: str
    display_name: str | None = None


@dataclass(frozen=True, slots=True)
class FileSearchDocument:
    """A document in a file search store (future use)."""

    name: str
    display_name: str | None = None
    file_name: str | None = None


# =============================================================================
# Engine-neutral research jobs (v0.17)
#
# Product version, research job, research engine and underlying model are
# independent axes. A job is one query run by one engine selector; a comparison
# is an immutable pair of such jobs. Nothing here merges or synthesizes reports.
# =============================================================================

RECORD_VERSION = 1

BACKEND_GEMINI = "gemini-native"
BACKEND_LOCAL = "local-deep-research"

PROVENANCE_PROVIDER = "provider_reported"
PROVENANCE_REPORT = "report_parsed"
PROVENANCE_ENGINE = "engine_reported"


class JobStatus(StrEnum):
    """Caller-visible lifecycle of one research job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

    @property
    def is_terminal(self) -> bool:
        return self not in (JobStatus.QUEUED, JobStatus.RUNNING)


class GroupStatus(StrEnum):
    """Derived status of a two-engine comparison. Partial is not consensus."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    EXPIRED = "expired"


class EngineError(DeepResearchError):
    """Typed failure of a research engine or of the job layer.

    Codes are stable, sanitized identifiers; ``message`` never contains prompt
    or report text, credentials, or raw child-process output.
    """


@dataclass(frozen=True, slots=True)
class ResearchSource:
    """One source with explicit provenance; never merged across engines."""

    uri: str | None
    title: str | None
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {"uri": self.uri, "title": self.title, "provenance": self.provenance}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchSource:
        return cls(
            uri=data.get("uri"),
            title=data.get("title"),
            provenance=str(data.get("provenance", PROVENANCE_ENGINE)),
        )


@dataclass(frozen=True, slots=True)
class EngineUsage:
    """Known token usage only. Unknown values stay None, never fabricated."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineUsage:
        return cls(
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            total_tokens=data.get("total_tokens"),
        )


@dataclass(frozen=True, slots=True)
class EngineDescriptor:
    """Static, side-effect-free description of a selectable engine profile."""

    id: str
    label: str
    backend: str
    execution_mode: str  # "background_interactions" | "blocking_stdio"
    recovery: str  # "provider_id" | "none"
    model: str | None
    available: bool
    availability_reason: str | None = None
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "backend": self.backend,
            "execution_mode": self.execution_mode,
            "recovery": self.recovery,
            "model": self.model,
            "available": self.available,
            "availability_reason": self.availability_reason,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class EngineRequest:
    """Input to an engine. The query is sent verbatim."""

    query: str


@dataclass(frozen=True, slots=True)
class EngineReport:
    """Unchanged report text plus typed provenance.

    ``images`` carries transient provider image dicts so the coordinator can
    persist them as separate artifacts; they are never serialized here.
    """

    text: str
    status: str = "completed"  # "completed" | "incomplete"
    sources: tuple[ResearchSource, ...] = ()
    usage: EngineUsage | None = None
    model: str | None = None
    provider_id: str | None = None
    warnings: tuple[str, ...] = ()
    duration_seconds: float | None = None
    images: tuple[dict[str, Any], ...] = field(default=(), compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "status": self.status,
            "sources": [s.to_dict() for s in self.sources],
            "usage": self.usage.to_dict() if self.usage else None,
            "model": self.model,
            "provider_id": self.provider_id,
            "warnings": list(self.warnings),
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineReport:
        usage = data.get("usage")
        return cls(
            text=data.get("text", ""),
            status=data.get("status", "completed"),
            sources=tuple(ResearchSource.from_dict(s) for s in data.get("sources", [])),
            usage=EngineUsage.from_dict(usage) if usage else None,
            model=data.get("model"),
            provider_id=data.get("provider_id"),
            warnings=tuple(data.get("warnings", [])),
            duration_seconds=data.get("duration_seconds"),
        )


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    """Read-only observation of a native provider job."""

    provider_id: str
    status: JobStatus
    report: EngineReport | None = None
    warnings: tuple[str, ...] = ()
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionIdentity:
    """Trusted identity of the current Task execution."""

    task_id: str
    task_scope: str | None
    redelivered: bool = False


@dataclass(slots=True)
class ResearchJobRecord:
    """Durable job metadata. Native jobs never copy the provider report."""

    job_id: str
    engine_id: str
    backend: str
    model: str | None
    query: str
    request_fingerprint: str
    task_id: str
    owner_scope: str | None
    slot: str
    created_at: float
    expires_at: float
    comparison_id: str | None = None
    provider_id: str | None = None
    recorded_status: JobStatus = JobStatus.QUEUED
    attempt_started_at: float | None = None
    owner_nonce: str | None = None
    owner_pid: int | None = None
    cancellation_requested: bool = False
    error_code: str | None = None
    error_message: str | None = None
    report: EngineReport | None = None
    version: int = RECORD_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "engine_id": self.engine_id,
            "backend": self.backend,
            "model": self.model,
            "query": self.query,
            "request_fingerprint": self.request_fingerprint,
            "task_id": self.task_id,
            "owner_scope": self.owner_scope,
            "slot": self.slot,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "comparison_id": self.comparison_id,
            "provider_id": self.provider_id,
            "recorded_status": self.recorded_status.value,
            "attempt_started_at": self.attempt_started_at,
            "owner_nonce": self.owner_nonce,
            "owner_pid": self.owner_pid,
            "cancellation_requested": self.cancellation_requested,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "report": self.report.to_dict() if self.report else None,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchJobRecord:
        report = data.get("report")
        return cls(
            job_id=data["job_id"],
            engine_id=data["engine_id"],
            backend=data["backend"],
            model=data.get("model"),
            query=data["query"],
            request_fingerprint=data["request_fingerprint"],
            task_id=data["task_id"],
            owner_scope=data.get("owner_scope"),
            slot=data["slot"],
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            comparison_id=data.get("comparison_id"),
            provider_id=data.get("provider_id"),
            recorded_status=JobStatus(data.get("recorded_status", "queued")),
            attempt_started_at=data.get("attempt_started_at"),
            owner_nonce=data.get("owner_nonce"),
            owner_pid=data.get("owner_pid"),
            cancellation_requested=bool(data.get("cancellation_requested", False)),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
            report=EngineReport.from_dict(report) if report else None,
            version=data.get("version", RECORD_VERSION),
        )


@dataclass(slots=True)
class ResearchComparisonRecord:
    """Immutable pair of child jobs. No report copies, no mutable status."""

    comparison_id: str
    query: str
    request_fingerprint: str
    task_id: str
    owner_scope: str | None
    engine_ids: tuple[str, str]
    child_job_ids: tuple[str, str]
    created_at: float
    expires_at: float
    version: int = RECORD_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "query": self.query,
            "request_fingerprint": self.request_fingerprint,
            "task_id": self.task_id,
            "owner_scope": self.owner_scope,
            "engine_ids": list(self.engine_ids),
            "child_job_ids": list(self.child_job_ids),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchComparisonRecord:
        engines = data["engine_ids"]
        children = data["child_job_ids"]
        return cls(
            comparison_id=data["comparison_id"],
            query=data["query"],
            request_fingerprint=data["request_fingerprint"],
            task_id=data["task_id"],
            owner_scope=data.get("owner_scope"),
            engine_ids=(engines[0], engines[1]),
            child_job_ids=(children[0], children[1]),
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            version=data.get("version", RECORD_VERSION),
        )


@dataclass(slots=True)
class ResolvedResearchJob:
    """A job as callers see it: authoritative status/report resolved on read."""

    record: ResearchJobRecord
    status: JobStatus
    report: EngineReport | None = None
    provider_id: str | None = None
    warnings: tuple[str, ...] = ()
    error_code: str | None = None
    error_message: str | None = None

    @property
    def job_id(self) -> str:
        return self.record.job_id

    def to_dict(self, *, include_report: bool = False) -> dict[str, Any]:
        record = self.record
        data: dict[str, Any] = {
            "run_id": record.job_id,
            "kind": "job",
            "engine": record.engine_id,
            "backend": record.backend,
            "model": record.model,
            "status": self.status.value,
            "provider_id": self.provider_id,
            "comparison_id": record.comparison_id,
            "created_at": record.created_at,
            "expires_at": record.expires_at,
            "cancellation_requested": record.cancellation_requested,
            "recovery": "none" if record.backend == BACKEND_LOCAL else "provider_id",
            "warnings": list(self.warnings),
            "error": (
                {"code": self.error_code, "message": self.error_message}
                if self.error_code
                else None
            ),
            "has_report": bool(self.report and self.report.text.strip()),
        }
        if self.report is not None:
            data["report_status"] = self.report.status
            data["sources"] = [s.to_dict() for s in self.report.sources]
            data["usage"] = self.report.usage.to_dict() if self.report.usage else None
            if include_report:
                data["report"] = self.report.text
        return data


@dataclass(slots=True)
class ResolvedResearchComparison:
    """Derived view over exactly two child jobs."""

    record: ResearchComparisonRecord
    status: GroupStatus
    children: tuple[ResolvedResearchJob, ResolvedResearchJob]

    @property
    def comparison_id(self) -> str:
        return self.record.comparison_id

    def to_dict(self, *, include_report: bool = False) -> dict[str, Any]:
        return {
            "run_id": self.record.comparison_id,
            "kind": "comparison",
            "status": self.status.value,
            "engines": list(self.record.engine_ids),
            "created_at": self.record.created_at,
            "expires_at": self.record.expires_at,
            "children": [c.to_dict(include_report=include_report) for c in self.children],
            "synthesis": "caller_owned",
        }


@dataclass(frozen=True, slots=True)
class ResearchRunSummary:
    """List entry. Metadata only; never a report body."""

    run_id: str
    kind: str  # "job" | "comparison"
    status: str
    engines: tuple[str, ...]
    query_preview: str
    created_at: float
    expires_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "status": self.status,
            "engines": list(self.engines),
            "query_preview": self.query_preview,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }
