"""
Persistent storage for research sessions.

Stores interaction metadata for later retrieval and follow-up conversations.
Uses py-key-value-aio DiskStore for persistent storage with automatic TTL cleanup.

Storage Location (XDG-compliant via platformdirs):
- macOS: ~/Library/Application Support/gemini-research-mcp/
- Linux: ~/.local/share/gemini-research-mcp/
- Windows: %APPDATA%\\gemini-research-mcp\\

Gemini Interaction Retention:
- Paid tier: 55 days
- Free tier: 24 hours
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Coroutine
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import platformdirs
from key_value.aio.stores.disk import DiskStore

from gemini_research_mcp.config import LOGGER_NAME
from gemini_research_mcp.types import DeepResearchAgent

logger = logging.getLogger(LOGGER_NAME)

T = TypeVar("T")

# =============================================================================
# Configuration
# =============================================================================

# Default TTL matches Gemini paid tier (55 days in seconds)
DEFAULT_TTL_SECONDS = 55 * 24 * 60 * 60  # 55 days

# Free tier TTL (24 hours)
FREE_TIER_TTL_SECONDS = 24 * 60 * 60  # 24 hours

# Application name for platformdirs
APP_NAME = "gemini-research-mcp"

# Storage collection name (DiskStore uses this as filename prefix)
SESSIONS_COLLECTION = "sessions"


def get_storage_dir() -> Path:
    """Get storage directory from env or XDG-compliant default.

    If GEMINI_RESEARCH_STORAGE_PATH is set:
    - If it's a directory path (exists or ends with /), use it directly
    - Otherwise, treat it as a file path and use its parent
    """
    custom_path = os.environ.get("GEMINI_RESEARCH_STORAGE_PATH")
    if custom_path:
        # Expand ~ and resolve path
        expanded = Path(custom_path).expanduser().resolve()
        # If path exists and is a directory, or ends with /, use it directly
        if expanded.is_dir() or custom_path.endswith(os.sep) or custom_path.endswith("/"):
            return expanded
        # Otherwise treat as file path and use parent
        return expanded.parent
    return Path(platformdirs.user_data_dir(APP_NAME))


def get_ttl_seconds() -> int:
    """Get TTL from env or default (55 days for paid tier)."""
    custom_ttl = os.environ.get("GEMINI_RESEARCH_TTL_SECONDS")
    if custom_ttl:
        try:
            return int(custom_ttl)
        except ValueError:
            logger.warning("Invalid GEMINI_RESEARCH_TTL_SECONDS: %s", custom_ttl)
    return DEFAULT_TTL_SECONDS


# =============================================================================
# Data Types
# =============================================================================


class ResearchStatus(str, Enum):
    """Status of a research session for resume functionality."""

    IN_PROGRESS = "in_progress"  # Research started, not yet completed
    COMPLETED = "completed"  # Research finished successfully
    FAILED = "failed"  # Research failed with error
    INTERRUPTED = "interrupted"  # Research interrupted (VS Code disconnected, etc.)


@dataclass
class ResearchSession:
    """A stored research session with metadata."""

    interaction_id: str
    query: str
    created_at: float  # Unix timestamp
    title: str | None = None  # Short descriptive title
    summary: str | None = None  # AI-generated synopsis for discovery
    report_text: str | None = None  # Full research report
    format_instructions: str | None = None
    agent_name: DeepResearchAgent | None = None
    duration_seconds: float | None = None
    total_tokens: int | None = None
    expires_at: float | None = None  # Unix timestamp
    tags: list[str] = field(default_factory=list)
    notes: str | None = None  # User-added notes
    status: ResearchStatus = ResearchStatus.COMPLETED  # For resume functionality

    def __post_init__(self) -> None:
        """Set expiration if not provided."""
        if self.expires_at is None:
            self.expires_at = self.created_at + get_ttl_seconds()

    @property
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def created_at_iso(self) -> str:
        """Return created_at as ISO format string."""
        return datetime.fromtimestamp(self.created_at, tz=UTC).isoformat()

    @property
    def expires_at_iso(self) -> str | None:
        """Return expires_at as ISO format string."""
        if self.expires_at is None:
            return None
        return datetime.fromtimestamp(self.expires_at, tz=UTC).isoformat()

    @property
    def time_remaining(self) -> float | None:
        """Seconds remaining until expiration."""
        if self.expires_at is None:
            return None
        return max(0, self.expires_at - time.time())

    @property
    def time_remaining_human(self) -> str | None:
        """Human-readable time remaining."""
        remaining = self.time_remaining
        if remaining is None:
            return None
        if remaining <= 0:
            return "expired"

        days = int(remaining // (24 * 60 * 60))
        hours = int((remaining % (24 * 60 * 60)) // (60 * 60))

        if days > 0:
            return f"{days}d {hours}h"
        minutes = int((remaining % (60 * 60)) // 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert enums to string for JSON serialization
        result["status"] = self.status.value
        if self.agent_name is not None:
            result["agent_name"] = self.agent_name.value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchSession:
        """Create from dictionary.

        Handles missing optional fields gracefully.

        Raises:
            KeyError: If required fields (interaction_id, query, created_at) are missing.
        """
        # Validate required fields explicitly for better error messages
        required = ["interaction_id", "query", "created_at"]
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Missing required fields: {', '.join(missing)}")

        # Handle agent_name enum
        agent_name_raw = data.get("agent_name")
        agent_name = (
            DeepResearchAgent(agent_name_raw) if agent_name_raw is not None else None
        )

        return cls(
            interaction_id=data["interaction_id"],
            query=data["query"],
            created_at=data["created_at"],
            title=data.get("title"),
            summary=data.get("summary"),
            report_text=data.get("report_text"),
            format_instructions=data.get("format_instructions"),
            agent_name=agent_name,
            duration_seconds=data.get("duration_seconds"),
            total_tokens=data.get("total_tokens"),
            expires_at=data.get("expires_at"),
            tags=data.get("tags", []),
            notes=data.get("notes"),
            status=ResearchStatus(data.get("status", "completed")),
        )

    @property
    def is_resumable(self) -> bool:
        """Check if the session can be resumed (in_progress or interrupted)."""
        return self.status in (ResearchStatus.IN_PROGRESS, ResearchStatus.INTERRUPTED)

    def short_description(self) -> str:
        """Return a short description for listing."""
        title = self.title or self.query[:50]
        remaining = self.time_remaining_human or "unknown"
        return f"[{self.interaction_id[:12]}...] {title} (expires: {remaining})"


# =============================================================================
# Storage Operations
# =============================================================================


class SessionStorage:
    """
    Persistent storage for research sessions using py-key-value-aio DiskStore.

    Features:
    - XDG-compliant storage paths via platformdirs
    - Automatic TTL-based expiration (handled by diskcache)
    - Async-first API with sync wrappers for convenience
    - Industry-standard key-value interface (used by FastMCP, agentpool, etc.)
    """

    def __init__(self, storage_dir: Path | None = None):
        """Initialize storage with DiskStore backend."""
        self.storage_dir = storage_dir or get_storage_dir()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._store = DiskStore(directory=str(self.storage_dir))
        logger.debug("💾 Storage initialized at %s", self.storage_dir)

    # -------------------------------------------------------------------------
    # Key Enumeration (via underlying diskcache)
    # -------------------------------------------------------------------------

    def _iter_session_keys(self) -> list[str]:
        """
        Iterate all session keys using diskcache's native iterkeys().

        DiskStore uses format: "{collection}::{key}" internally.
        This is more elegant than maintaining a separate index.
        """
        prefix = f"{SESSIONS_COLLECTION}::"
        keys: list[str] = []
        for raw_key in self._store._cache.iterkeys():
            if isinstance(raw_key, str) and raw_key.startswith(prefix):
                keys.append(raw_key[len(prefix):])
        return keys

    # -------------------------------------------------------------------------
    # Async Core Operations
    # -------------------------------------------------------------------------

    async def save_session_async(self, session: ResearchSession) -> None:
        """Save a research session (async)."""
        # Calculate TTL in seconds from now
        ttl: float | None = None
        if session.expires_at is not None:
            ttl = max(1.0, session.expires_at - time.time())

        await self._store.put(
            session.interaction_id,
            session.to_dict(),
            ttl=ttl,
            collection=SESSIONS_COLLECTION,
        )
        logger.info(
            "💾 Saved session: %s (expires: %s)",
            session.interaction_id[:16],
            session.time_remaining_human,
        )

    async def get_session_async(self, interaction_id: str) -> ResearchSession | None:
        """Get a session by interaction_id (async)."""
        data = await self._store.get(interaction_id, collection=SESSIONS_COLLECTION)
        if data is None:
            return None

        session = ResearchSession.from_dict(data)
        # DiskStore handles TTL, but double-check for edge cases
        if session.is_expired:
            logger.debug("Session %s has expired", interaction_id[:16])
            await self._store.delete(interaction_id, collection=SESSIONS_COLLECTION)
            return None
        return session

    async def list_sessions_async(
        self,
        *,
        include_expired: bool = False,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ResearchSession]:
        """
        List all sessions, optionally filtered (async).

        Args:
            include_expired: Include expired sessions
            tags: Filter by tags (any match)
            limit: Maximum number of sessions to return

        Returns:
            List of sessions, sorted by created_at (newest first)
        """
        sessions: list[ResearchSession] = []

        for interaction_id in self._iter_session_keys():
            data = await self._store.get(interaction_id, collection=SESSIONS_COLLECTION)
            if data is None:
                continue

            try:
                session = ResearchSession.from_dict(data)
            except KeyError as e:
                logger.warning("Skipping corrupted session %s: %s", interaction_id[:16], e)
                continue

            if not include_expired and session.is_expired:
                continue

            if tags and not any(tag in session.tags for tag in tags):
                continue

            sessions.append(session)

        # Sort by created_at, newest first
        sessions.sort(key=lambda s: s.created_at, reverse=True)

        # Apply limit if positive
        if limit is not None and limit > 0:
            sessions = sessions[:limit]

        return sessions

    async def delete_session_async(self, interaction_id: str) -> bool:
        """Delete a session (async)."""
        exists = await self._store.get(interaction_id, collection=SESSIONS_COLLECTION)
        if exists is None:
            return False
        await self._store.delete(interaction_id, collection=SESSIONS_COLLECTION)
        logger.info("🗑️ Deleted session: %s", interaction_id[:16])
        return True

    async def update_session_async(
        self,
        interaction_id: str,
        *,
        title: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        status: ResearchStatus | None = None,
        summary: str | None = None,
        report_text: str | None = None,
        duration_seconds: float | None = None,
        total_tokens: int | None = None,
    ) -> ResearchSession | None:
        """Update session metadata (async)."""
        session = await self.get_session_async(interaction_id)
        if session is None:
            return None

        if title is not None:
            session.title = title
        if tags is not None:
            session.tags = tags
        if notes is not None:
            session.notes = notes
        if status is not None:
            session.status = status
        if summary is not None:
            session.summary = summary
        if report_text is not None:
            session.report_text = report_text
        if duration_seconds is not None:
            session.duration_seconds = duration_seconds
        if total_tokens is not None:
            session.total_tokens = total_tokens

        await self.save_session_async(session)
        return session

    async def cleanup_expired_async(self) -> int:
        """Remove all expired sessions (async). Returns count of removed sessions."""
        expired_ids: list[str] = []

        for interaction_id in self._iter_session_keys():
            data = await self._store.get(interaction_id, collection=SESSIONS_COLLECTION)
            if data is None:
                # Already expired/removed by TTL
                continue
            session = ResearchSession.from_dict(data)
            if session.is_expired:
                expired_ids.append(interaction_id)

        for interaction_id in expired_ids:
            await self._store.delete(interaction_id, collection=SESSIONS_COLLECTION)

        if expired_ids:
            logger.info("🧹 Cleaned up %d expired sessions", len(expired_ids))

        return len(expired_ids)

    async def search_async(self, query: str, limit: int = 10) -> list[ResearchSession]:
        """Search sessions by query text (searches query and title) (async)."""
        query_lower = query.lower()
        sessions = await self.list_sessions_async()

        matches = []
        for session in sessions:
            in_query = query_lower in session.query.lower()
            in_title = session.title and query_lower in session.title.lower()
            if in_query or in_title:
                matches.append(session)

        return matches[:limit]

    # -------------------------------------------------------------------------
    # Sync Wrappers (for convenience)
    # -------------------------------------------------------------------------

    def _run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run async coroutine in sync context."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(coro)
        else:
            # Running inside an async context - schedule and wait
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                result: T = future.result()
                return result

    def save_session(self, session: ResearchSession) -> None:
        """Save a research session (sync wrapper)."""
        self._run_async(self.save_session_async(session))

    def get_session(self, interaction_id: str) -> ResearchSession | None:
        """Get a session by interaction_id (sync wrapper)."""
        return self._run_async(self.get_session_async(interaction_id))

    def list_sessions(
        self,
        *,
        include_expired: bool = False,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ResearchSession]:
        """List all sessions (sync wrapper)."""
        return self._run_async(
            self.list_sessions_async(
                include_expired=include_expired, tags=tags, limit=limit
            )
        )

    def delete_session(self, interaction_id: str) -> bool:
        """Delete a session (sync wrapper)."""
        return self._run_async(self.delete_session_async(interaction_id))

    def update_session(
        self,
        interaction_id: str,
        *,
        title: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        status: ResearchStatus | None = None,
        summary: str | None = None,
        report_text: str | None = None,
        duration_seconds: float | None = None,
        total_tokens: int | None = None,
    ) -> ResearchSession | None:
        """Update session metadata (sync wrapper)."""
        return self._run_async(
            self.update_session_async(
                interaction_id,
                title=title,
                tags=tags,
                notes=notes,
                status=status,
                summary=summary,
                report_text=report_text,
                duration_seconds=duration_seconds,
                total_tokens=total_tokens,
            )
        )

    def cleanup_expired(self) -> int:
        """Remove all expired sessions (sync wrapper)."""
        return self._run_async(self.cleanup_expired_async())

    def search(self, query: str, limit: int = 10) -> list[ResearchSession]:
        """Search sessions (sync wrapper)."""
        return self._run_async(self.search_async(query, limit=limit))


# =============================================================================
# Global Storage Instance
# =============================================================================

_storage: SessionStorage | None = None


def get_storage() -> SessionStorage:
    """Get the global storage instance."""
    global _storage
    if _storage is None:
        _storage = SessionStorage()
    return _storage


# =============================================================================
# Convenience Functions
# =============================================================================


def save_research_session(
    interaction_id: str,
    query: str,
    *,
    title: str | None = None,
    summary: str | None = None,
    report_text: str | None = None,
    format_instructions: str | None = None,
    agent_name: DeepResearchAgent | None = None,
    duration_seconds: float | None = None,
    total_tokens: int | None = None,
    tags: list[str] | None = None,
    status: ResearchStatus = ResearchStatus.COMPLETED,
) -> ResearchSession:
    """
    Save a research session for later follow-up.

    Args:
        interaction_id: The Gemini interaction ID
        query: The research query
        title: Optional short title (defaults to query[:50])
        summary: Optional AI-generated synopsis for discovery
        report_text: Optional full research report
        format_instructions: Optional format instructions used
        agent_name: Optional agent name used
        duration_seconds: Optional research duration
        total_tokens: Optional total tokens used
        tags: Optional tags for filtering
        status: Session status (default: COMPLETED)

    Returns:
        The saved ResearchSession
    """
    session = ResearchSession(
        interaction_id=interaction_id,
        query=query,
        created_at=time.time(),
        title=title,
        summary=summary,
        report_text=report_text,
        format_instructions=format_instructions,
        agent_name=agent_name,
        duration_seconds=duration_seconds,
        total_tokens=total_tokens,
        tags=tags or [],
        status=status,
    )
    get_storage().save_session(session)
    return session


def update_research_session(
    interaction_id: str,
    *,
    title: str | None = None,
    summary: str | None = None,
    report_text: str | None = None,
    duration_seconds: float | None = None,
    total_tokens: int | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    status: ResearchStatus | None = None,
) -> ResearchSession | None:
    """
    Update an existing research session.

    Args:
        interaction_id: The Gemini interaction ID
        title: Optional new title
        summary: Optional new summary
        report_text: Optional new report text
        duration_seconds: Optional research duration
        total_tokens: Optional total tokens used
        tags: Optional new tags
        notes: Optional user notes
        status: Optional new status

    Returns:
        The updated ResearchSession or None if not found
    """
    return get_storage().update_session(
        interaction_id,
        title=title,
        summary=summary,
        report_text=report_text,
        duration_seconds=duration_seconds,
        total_tokens=total_tokens,
        tags=tags,
        notes=notes,
        status=status,
    )


def list_resumable_sessions(limit: int = 10) -> list[ResearchSession]:
    """
    List sessions that can be resumed (in_progress or interrupted).

    Returns:
        List of resumable sessions, sorted by created_at (newest first)
    """
    sessions = get_storage().list_sessions(include_expired=False, limit=None)
    resumable = [s for s in sessions if s.is_resumable]
    return resumable[:limit]


def get_research_session(interaction_id: str) -> ResearchSession | None:
    """Get a research session by interaction_id."""
    return get_storage().get_session(interaction_id)


def list_research_sessions(
    *,
    include_expired: bool = False,
    tags: list[str] | None = None,
    limit: int | None = None,
) -> list[ResearchSession]:
    """List research sessions."""
    return get_storage().list_sessions(
        include_expired=include_expired,
        tags=tags,
        limit=limit,
    )
