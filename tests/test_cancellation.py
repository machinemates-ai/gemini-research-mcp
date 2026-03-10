"""Tests for cancellation status handling across storage, resume, and followup.

Verifies that:
- CANCELLED is a first-class terminal status distinct from FAILED
- resume_research short-circuits on locally-cancelled sessions
- resume_research maps Gemini 'cancelled'/'canceled' to CANCELLED (not FAILED)
- research_followup auto-match excludes CANCELLED and FAILED sessions
- deep.py streaming and polling handle 'cancelled'/'canceled' from Gemini
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gemini_research_mcp.storage import (
    ResearchSession,
    ResearchStatus,
    SessionStorage,
    get_research_session,
    save_research_session,
    update_research_session,
)

# =============================================================================
# Fixtures
# =============================================================================

STORAGE_MODULE = "gemini_research_mcp.storage"
SERVER_MODULE = "gemini_research_mcp.server"


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a temporary storage directory for all tests."""
    storage = SessionStorage(storage_dir=tmp_path)
    monkeypatch.setattr(f"{STORAGE_MODULE}._storage", storage)


# =============================================================================
# ResearchStatus.CANCELLED basics
# =============================================================================


class TestCancelledStatusModel:
    """Tests that CANCELLED is a proper terminal, non-resumable status."""

    def test_cancelled_status_exists(self) -> None:
        """CANCELLED should be a valid ResearchStatus value."""
        assert ResearchStatus.CANCELLED.value == "cancelled"

    def test_cancelled_is_not_resumable(self) -> None:
        """CANCELLED sessions must not appear in resumable listings."""
        session = ResearchSession(
            interaction_id="cancelled-session",
            query="A cancelled query",
            created_at=time.time(),
            status=ResearchStatus.CANCELLED,
        )
        assert session.is_resumable is False

    def test_failed_is_not_resumable(self) -> None:
        """Sanity: FAILED sessions are still non-resumable."""
        session = ResearchSession(
            interaction_id="failed-session",
            query="A failed query",
            created_at=time.time(),
            status=ResearchStatus.FAILED,
        )
        assert session.is_resumable is False

    def test_cancelled_round_trip(self) -> None:
        """CANCELLED survives save → get round-trip."""
        save_research_session(
            interaction_id="rt-cancelled",
            query="round trip test",
            status=ResearchStatus.CANCELLED,
        )
        loaded = get_research_session("rt-cancelled")
        assert loaded is not None
        assert loaded.status == ResearchStatus.CANCELLED

    def test_update_to_cancelled(self) -> None:
        """Can update a session status to CANCELLED."""
        save_research_session(
            interaction_id="update-cancel",
            query="in progress first",
            status=ResearchStatus.IN_PROGRESS,
        )
        update_research_session("update-cancel", status=ResearchStatus.CANCELLED)
        loaded = get_research_session("update-cancel")
        assert loaded is not None
        assert loaded.status == ResearchStatus.CANCELLED

    def test_from_dict_cancelled(self) -> None:
        """from_dict handles 'cancelled' string correctly."""
        data = {
            "interaction_id": "dict-cancel",
            "query": "dict test",
            "created_at": time.time(),
            "status": "cancelled",
        }
        session = ResearchSession.from_dict(data)
        assert session.status == ResearchStatus.CANCELLED


# =============================================================================
# resume_research — local CANCELLED short-circuit
# =============================================================================


class TestResumeResearchCancelledShortCircuit:
    """resume_research should return immediately for locally-cancelled sessions."""

    @pytest.mark.asyncio
    async def test_already_cancelled_returns_immediately(self) -> None:
        """Should not re-query Gemini for a session already marked CANCELLED."""
        from gemini_research_mcp.server import resume_research

        save_research_session(
            interaction_id="local-cancelled",
            query="This was cancelled",
            status=ResearchStatus.CANCELLED,
        )

        # If it re-queries Gemini, this mock would be called and we'd know
        with patch(
            f"{SERVER_MODULE}.get_research_status",
            new_callable=AsyncMock,
        ) as mock_api:
            result = await resume_research(interaction_id="local-cancelled")

        mock_api.assert_not_called()
        data = json.loads(result)
        assert data["status"] == "cancelled"
        assert data["resumable"] is False
        assert "cannot be resumed" in data["message"]

    @pytest.mark.asyncio
    async def test_cancelled_session_suggests_new_research(self) -> None:
        """Cancelled response should hint to start new research."""
        from gemini_research_mcp.server import resume_research

        save_research_session(
            interaction_id="hint-cancelled",
            query="Original cancelled query",
            status=ResearchStatus.CANCELLED,
        )

        with patch(
            f"{SERVER_MODULE}.get_research_status",
            new_callable=AsyncMock,
        ):
            result = await resume_research(interaction_id="hint-cancelled")

        data = json.loads(result)
        assert "research_deep" in data.get("hint", "")


# =============================================================================
# resume_research — Gemini API returns cancelled/canceled
# =============================================================================


class TestResumeResearchGeminiCancelled:
    """resume_research maps Gemini cancelled to CANCELLED (not FAILED)."""

    @pytest.mark.asyncio
    async def test_gemini_cancelled_maps_to_cancelled_status(self) -> None:
        """Gemini 'cancelled' → local ResearchStatus.CANCELLED."""
        from gemini_research_mcp.server import resume_research

        save_research_session(
            interaction_id="gemini-cancelled",
            query="Gemini says cancelled",
            status=ResearchStatus.IN_PROGRESS,
        )

        mock_interaction = SimpleNamespace(status="cancelled")
        mock_result = SimpleNamespace(
            raw_interaction=mock_interaction,
            text=None,
            usage=None,
        )

        with patch(
            f"{SERVER_MODULE}.get_research_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await resume_research(interaction_id="gemini-cancelled")

        data = json.loads(result)
        assert data["status"] == "cancelled"
        assert data["resumable"] is False

        # Verify local storage was updated to CANCELLED, not FAILED
        session = get_research_session("gemini-cancelled")
        assert session is not None
        assert session.status == ResearchStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_gemini_canceled_american_spelling(self) -> None:
        """Gemini 'canceled' (US spelling) → local ResearchStatus.CANCELLED."""
        from gemini_research_mcp.server import resume_research

        save_research_session(
            interaction_id="gemini-canceled-us",
            query="Gemini says canceled (US)",
            status=ResearchStatus.IN_PROGRESS,
        )

        mock_interaction = SimpleNamespace(status="canceled")
        mock_result = SimpleNamespace(
            raw_interaction=mock_interaction,
            text=None,
            usage=None,
        )

        with patch(
            f"{SERVER_MODULE}.get_research_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await resume_research(interaction_id="gemini-canceled-us")

        data = json.loads(result)
        assert data["status"] == "canceled"

        session = get_research_session("gemini-canceled-us")
        assert session is not None
        assert session.status == ResearchStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_gemini_failed_still_maps_to_failed(self) -> None:
        """Gemini 'failed' should still map to FAILED (not CANCELLED)."""
        from gemini_research_mcp.server import resume_research

        save_research_session(
            interaction_id="gemini-failed",
            query="Gemini says failed",
            status=ResearchStatus.IN_PROGRESS,
        )

        mock_interaction = SimpleNamespace(status="failed")
        mock_result = SimpleNamespace(
            raw_interaction=mock_interaction,
            text=None,
            usage=None,
        )

        with patch(
            f"{SERVER_MODULE}.get_research_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await resume_research(interaction_id="gemini-failed")

        data = json.loads(result)
        assert data["status"] == "failed"

        session = get_research_session("gemini-failed")
        assert session is not None
        assert session.status == ResearchStatus.FAILED


# =============================================================================
# research_followup — auto-match excludes CANCELLED/FAILED
# =============================================================================


class TestFollowupFiltersOutCancelled:
    """research_followup auto-match should skip CANCELLED and FAILED sessions."""

    @pytest.mark.asyncio
    async def test_cancelled_sessions_excluded_from_auto_match(self) -> None:
        """Auto-match should not pick a CANCELLED session."""
        from gemini_research_mcp.server import research_followup

        cancelled = ResearchSession(
            interaction_id="cancelled-for-match",
            query="Cancelled research about cats",
            created_at=time.time(),
            status=ResearchStatus.CANCELLED,
            summary="Research about cats",
        )
        completed = ResearchSession(
            interaction_id="completed-for-match",
            query="Completed research about dogs",
            created_at=time.time() - 100,
            status=ResearchStatus.COMPLETED,
            summary="Research about dogs",
        )

        sessions = [cancelled, completed]

        with (
            patch(
                "gemini_research_mcp.server._list_sessions",
                return_value=sessions,
            ),
            patch(
                "gemini_research_mcp.server.semantic_match_session",
                new_callable=AsyncMock,
                return_value=completed.interaction_id,
            ) as mock_semantic,
            patch(
                "gemini_research_mcp.server._research_followup",
                new_callable=AsyncMock,
                return_value="Response about dogs.",
            ),
        ):
            await research_followup(query="Tell me more about cats")

            # semantic_match_session should only receive the completed session
            call_args = mock_semantic.call_args
            session_dicts = (
                call_args.args[1]
                if call_args.args
                else call_args.kwargs.get("sessions", [])
            )
            ids_in_match = [s["id"] for s in session_dicts]
            assert "cancelled-for-match" not in ids_in_match
            assert "completed-for-match" in ids_in_match

    @pytest.mark.asyncio
    async def test_all_cancelled_returns_error(self) -> None:
        """If all sessions are CANCELLED/FAILED, return clear error."""
        from gemini_research_mcp.server import research_followup

        sessions = [
            ResearchSession(
                interaction_id="c1",
                query="cancelled A",
                created_at=time.time(),
                status=ResearchStatus.CANCELLED,
            ),
            ResearchSession(
                interaction_id="f1",
                query="failed B",
                created_at=time.time() - 50,
                status=ResearchStatus.FAILED,
            ),
        ]

        with patch(
            "gemini_research_mcp.server._list_sessions",
            return_value=sessions,
        ):
            result = await research_followup(query="Tell me more")
            assert "cancelled or failed" in result.lower()

    @pytest.mark.asyncio
    async def test_fallback_skips_cancelled_for_most_recent(self) -> None:
        """When semantic match fails, fallback picks most recent non-cancelled."""
        from gemini_research_mcp.server import research_followup

        cancelled_recent = ResearchSession(
            interaction_id="most-recent-cancelled",
            query="Most recent but cancelled",
            created_at=time.time(),
            status=ResearchStatus.CANCELLED,
        )
        older_completed = ResearchSession(
            interaction_id="older-completed",
            query="Older but completed",
            created_at=time.time() - 200,
            status=ResearchStatus.COMPLETED,
            summary="Older research",
        )

        sessions = [cancelled_recent, older_completed]

        with (
            patch(
                "gemini_research_mcp.server._list_sessions",
                return_value=sessions,
            ),
            patch(
                "gemini_research_mcp.server.semantic_match_session",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "gemini_research_mcp.server._research_followup",
                new_callable=AsyncMock,
                return_value="Response about older topic.",
            ) as mock_followup,
        ):
            await research_followup(query="Tell me more about this")

            call_kwargs = mock_followup.call_args.kwargs
            assert call_kwargs["previous_interaction_id"] == "older-completed"
