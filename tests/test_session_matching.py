"""Tests for session matching in research_followup and export_research_session."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from gemini_research_mcp.storage import ResearchSession

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def quantum_session() -> ResearchSession:
    """Create a quantum computing research session."""
    return ResearchSession(
        interaction_id="session-quantum-12345",
        query="What are the latest advances in quantum computing?",
        created_at=time.time() - 100,  # 100 seconds ago
        title="Quantum Computing Advances",
        summary="Research on quantum computing advances including superconducting qubits.",
        report_text="## Quantum Computing Report\n\nContent here...",
    )


@pytest.fixture
def climate_session() -> ResearchSession:
    """Create a climate research session."""
    return ResearchSession(
        interaction_id="session-climate-67890",
        query="How is climate change affecting polar ice caps?",
        created_at=time.time() - 50,  # 50 seconds ago (more recent)
        title="Climate Change and Polar Ice",
        summary="Research on climate change impact on Arctic and Antarctic ice.",
        report_text="## Climate Report\n\nContent here...",
    )


@pytest.fixture
def recent_session() -> ResearchSession:
    """Create the most recent session."""
    return ResearchSession(
        interaction_id="session-recent-99999",
        query="What is the state of AI regulation in Europe?",
        created_at=time.time(),  # Now (most recent)
        title="AI Regulation in Europe",
        summary="Overview of EU AI Act and related regulations.",
        report_text="## AI Regulation Report\n\nContent here...",
    )


# =============================================================================
# research_followup Session Matching Tests
# =============================================================================


class TestResearchFollowupSessionMatching:
    """Tests for session matching in research_followup."""

    @pytest.mark.asyncio
    async def test_explicit_interaction_id_used(
        self, quantum_session: ResearchSession
    ) -> None:
        """Test that explicit interaction_id is used directly."""
        from gemini_research_mcp.server import research_followup

        with (
            patch(
                "gemini_research_mcp.server._list_sessions"
            ) as mock_list,
            patch(
                "gemini_research_mcp.server._research_followup",
                new_callable=AsyncMock,
            ) as mock_followup,
        ):
            mock_followup.return_value = "Follow-up response about quantum computing."

            await research_followup(
                query="Tell me more about qubits",
                interaction_id="explicit-session-id",
            )

            # Should NOT call list_research_sessions when id is provided
            mock_list.assert_not_called()
            # Should use the explicit ID
            mock_followup.assert_called_once()
            call_kwargs = mock_followup.call_args.kwargs
            assert call_kwargs["previous_interaction_id"] == "explicit-session-id"

    @pytest.mark.asyncio
    async def test_semantic_match_found(
        self,
        quantum_session: ResearchSession,
        climate_session: ResearchSession,
        recent_session: ResearchSession,
    ) -> None:
        """Test that semantic matching finds the right session."""
        from gemini_research_mcp.server import research_followup

        sessions = [recent_session, climate_session, quantum_session]

        with (
            patch(
                "gemini_research_mcp.server._list_sessions",
                return_value=sessions,
            ),
            patch(
                "gemini_research_mcp.server.semantic_match_session",
                new_callable=AsyncMock,
                return_value=quantum_session.interaction_id,
            ) as mock_semantic,
            patch(
                "gemini_research_mcp.server._research_followup",
                new_callable=AsyncMock,
                return_value="Details about superconducting qubits...",
            ) as mock_followup,
        ):
            await research_followup(
                query="Explain more about superconducting qubits",
            )

            # Should call semantic matching
            mock_semantic.assert_called_once()
            # Should use the matched quantum session
            call_kwargs = mock_followup.call_args.kwargs
            assert call_kwargs["previous_interaction_id"] == quantum_session.interaction_id

    @pytest.mark.asyncio
    async def test_fallback_to_most_recent_when_no_match(
        self,
        quantum_session: ResearchSession,
        climate_session: ResearchSession,
        recent_session: ResearchSession,
    ) -> None:
        """Test fallback to most recent session when semantic match fails."""
        from gemini_research_mcp.server import research_followup

        # Sessions ordered by recency (most recent first)
        sessions = [recent_session, climate_session, quantum_session]

        with (
            patch(
                "gemini_research_mcp.server._list_sessions",
                return_value=sessions,
            ),
            patch(
                "gemini_research_mcp.server.semantic_match_session",
                new_callable=AsyncMock,
                return_value=None,  # No match found
            ),
            patch(
                "gemini_research_mcp.server._research_followup",
                new_callable=AsyncMock,
                return_value="Elaborating on the topic...",
            ) as mock_followup,
        ):
            result = await research_followup(
                query="Tell me more",  # Generic query that won't match
            )

            # Should fall back to most recent session (first in list)
            call_kwargs = mock_followup.call_args.kwargs
            assert call_kwargs["previous_interaction_id"] == recent_session.interaction_id
            assert "Elaborating" in result

    @pytest.mark.asyncio
    async def test_no_sessions_available(self) -> None:
        """Test error when no sessions exist."""
        from gemini_research_mcp.server import research_followup

        with patch(
            "gemini_research_mcp.server._list_sessions",
            return_value=[],
        ):
            result = await research_followup(query="Elaborate on this")

            assert "No research sessions found" in result


# =============================================================================
# export_research_session Session Matching Tests
# =============================================================================


class TestExportSessionMatching:
    """Tests for session matching in export_research_session."""

    @pytest.mark.asyncio
    async def test_explicit_interaction_id_used(
        self, quantum_session: ResearchSession
    ) -> None:
        """Test that explicit interaction_id is used directly."""
        from gemini_research_mcp.server import export_research_session

        with patch(
            "gemini_research_mcp.server.get_research_session",
            return_value=quantum_session,
        ) as mock_get:
            result = await export_research_session(
                interaction_id="explicit-session-id",
                format="json",
            )

            mock_get.assert_called_once_with("explicit-session-id")
            # Result is now [TextContent, EmbeddedResource] for successful exports
            assert isinstance(result, list)
            assert len(result) == 2
            # Extract JSON content from EmbeddedResource's TextResourceContents
            embedded = result[1]
            data = json.loads(embedded.resource.text)
            assert data["interaction_id"] == quantum_session.interaction_id

    @pytest.mark.asyncio
    async def test_semantic_match_found(
        self,
        quantum_session: ResearchSession,
        climate_session: ResearchSession,
        recent_session: ResearchSession,
    ) -> None:
        """Test that semantic matching finds the right session for export."""
        from gemini_research_mcp.server import export_research_session

        sessions = [recent_session, climate_session, quantum_session]

        with (
            patch(
                "gemini_research_mcp.server._list_sessions",
                return_value=sessions,
            ),
            patch(
                "gemini_research_mcp.server.semantic_match_session",
                new_callable=AsyncMock,
                return_value=climate_session.interaction_id,
            ) as mock_semantic,
        ):
            result = await export_research_session(
                query="Export the climate research",
                format="json",
            )

            mock_semantic.assert_called_once()
            # Result is now [TextContent, EmbeddedResource] for successful exports
            assert isinstance(result, list)
            embedded = result[1]
            data = json.loads(embedded.resource.text)
            assert data["interaction_id"] == climate_session.interaction_id

    @pytest.mark.asyncio
    async def test_fallback_to_most_recent_when_no_match(
        self,
        quantum_session: ResearchSession,
        climate_session: ResearchSession,
        recent_session: ResearchSession,
    ) -> None:
        """Test fallback to most recent session when semantic match fails."""
        from gemini_research_mcp.server import export_research_session

        # Sessions ordered by recency (most recent first)
        sessions = [recent_session, climate_session, quantum_session]

        with (
            patch(
                "gemini_research_mcp.server._list_sessions",
                return_value=sessions,
            ),
            patch(
                "gemini_research_mcp.server.semantic_match_session",
                new_callable=AsyncMock,
                return_value=None,  # No match found
            ),
        ):
            result = await export_research_session(
                query="Export this research",  # Generic query
                format="json",
            )

            # Result is now [TextContent, EmbeddedResource] for successful exports
            assert isinstance(result, list)
            embedded = result[1]
            data = json.loads(embedded.resource.text)
            # Should fall back to most recent session
            assert data["interaction_id"] == recent_session.interaction_id

    @pytest.mark.asyncio
    async def test_default_to_most_recent_no_query(
        self,
        quantum_session: ResearchSession,
        recent_session: ResearchSession,
    ) -> None:
        """Test that no query defaults to most recent session."""
        from gemini_research_mcp.server import export_research_session

        sessions = [recent_session, quantum_session]

        with patch(
            "gemini_research_mcp.server._list_sessions",
            return_value=sessions,
        ):
            result = await export_research_session(format="json")

            # Result is now [TextContent, EmbeddedResource] for successful exports
            assert isinstance(result, list)
            embedded = result[1]
            data = json.loads(embedded.resource.text)
            assert data["interaction_id"] == recent_session.interaction_id

    @pytest.mark.asyncio
    async def test_no_sessions_available(self) -> None:
        """Test error when no sessions exist."""
        from gemini_research_mcp.server import export_research_session

        with patch(
            "gemini_research_mcp.server._list_sessions",
            return_value=[],
        ):
            result = await export_research_session(format="json")

            data = json.loads(result)
            assert "error" in data
            assert "No research sessions found" in data["error"]

    @pytest.mark.asyncio
    async def test_session_not_found_by_id(self) -> None:
        """Test error when explicit interaction_id not found."""
        from gemini_research_mcp.server import export_research_session

        with patch(
            "gemini_research_mcp.server.get_research_session",
            return_value=None,
        ):
            result = await export_research_session(
                interaction_id="nonexistent-id",
                format="json",
            )

            data = json.loads(result)
            assert "error" in data
            assert "Session not found" in data["error"]


# =============================================================================
# Consistency Tests
# =============================================================================


class TestSessionMatchingConsistency:
    """Tests to ensure consistent behavior between tools."""

    @pytest.mark.asyncio
    async def test_both_tools_use_same_session_dicts_structure(
        self,
        quantum_session: ResearchSession,
        climate_session: ResearchSession,
    ) -> None:
        """Verify both tools build session_dicts with same structure."""
        from gemini_research_mcp.server import (
            export_research_session,
            research_followup,
        )

        sessions = [climate_session, quantum_session]
        captured_dicts: list[list[dict]] = []

        async def capture_dicts(query: str, session_dicts: list[dict]) -> str | None:
            captured_dicts.append(session_dicts)
            return quantum_session.interaction_id

        with (
            patch(
                "gemini_research_mcp.server._list_sessions",
                return_value=sessions,
            ),
            patch(
                "gemini_research_mcp.server.semantic_match_session",
                side_effect=capture_dicts,
            ),
            patch(
                "gemini_research_mcp.server._research_followup",
                new_callable=AsyncMock,
                return_value="Response",
            ),
        ):
            await research_followup(query="Test query")
            await export_research_session(query="Test query", format="json")

            # Both should have captured session_dicts
            assert len(captured_dicts) == 2

            # Verify structure is identical
            for session_dict_list in captured_dicts:
                assert len(session_dict_list) == 2
                for d in session_dict_list:
                    assert "id" in d
                    assert "query" in d
                    assert "summary" in d


# =============================================================================
# export_research_session output_path Tests
# =============================================================================


class TestExportSessionOutputPath:
    """Tests for the output_path parameter of export_research_session.

    Rationale: headless MCP clients (Copilot CLI, CI) silently drop the binary
    EmbeddedResource blob, so DOCX (and other) bytes are unreachable without a
    server-side filesystem write. These tests pin down the contract.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("fmt", "ext", "magic"),
        [
            ("markdown", "md", b"# "),
            ("json", "json", b"{"),
            ("docx", "docx", b"PK\x03\x04"),
        ],
    )
    async def test_output_path_writes_file(
        self,
        tmp_path,
        quantum_session: ResearchSession,
        fmt: str,
        ext: str,
        magic: bytes,
    ) -> None:
        """Each format writes bytes to output_path and still returns the embedded resource."""
        from gemini_research_mcp.server import export_research_session

        out = tmp_path / f"export.{ext}"

        with patch(
            "gemini_research_mcp.server.get_research_session",
            return_value=quantum_session,
        ):
            result = await export_research_session(
                interaction_id=quantum_session.interaction_id,
                format=fmt,
                output_path=str(out),
            )

        # File must be written to disk with the correct magic bytes
        assert out.exists(), f"{fmt} export was not written to {out}"
        assert out.read_bytes().startswith(magic), (
            f"{fmt} file does not start with expected magic {magic!r}"
        )
        assert out.stat().st_size > 0

        # Tool must still return [TextContent, EmbeddedResource]
        assert isinstance(result, list)
        assert len(result) == 2
        text_content, embedded = result
        assert "Saved to" in text_content.text
        assert str(out.resolve()) in text_content.text
        # EmbeddedResource bytes match disk bytes (backward compat preserved)
        resource = embedded.resource
        if hasattr(resource, "blob"):
            import base64

            assert base64.b64decode(resource.blob) == out.read_bytes()
        else:
            assert resource.text.encode("utf-8") == out.read_bytes()

    @pytest.mark.asyncio
    async def test_output_path_missing_parent_returns_error(
        self,
        tmp_path,
        quantum_session: ResearchSession,
    ) -> None:
        """Missing parent directory returns an error JSON (no silent mkdir)."""
        from gemini_research_mcp.server import export_research_session

        bogus = tmp_path / "does-not-exist" / "out.docx"

        with patch(
            "gemini_research_mcp.server.get_research_session",
            return_value=quantum_session,
        ):
            result = await export_research_session(
                interaction_id=quantum_session.interaction_id,
                format="docx",
                output_path=str(bogus),
            )

        assert isinstance(result, str)
        data = json.loads(result)
        assert "error" in data
        assert "does not exist" in data["error"]
        assert "hint" in data
        assert not bogus.exists()

    @pytest.mark.asyncio
    async def test_output_path_omitted_auto_writes_to_default_dir(
        self,
        quantum_session: ResearchSession,
        tmp_path,
        monkeypatch,
    ) -> None:
        """When output_path is None, the file is still written to disk.

        This is the v0.13.0 disk-first contract: headless clients that
        cannot render EmbeddedResource blobs must still get a usable file
        path back, without needing to know about ``output_path``.
        """
        from gemini_research_mcp.server import export_research_session

        monkeypatch.setenv("GEMINI_RESEARCH_EXPORT_DIR", str(tmp_path))

        with patch(
            "gemini_research_mcp.server.get_research_session",
            return_value=quantum_session,
        ):
            result = await export_research_session(
                interaction_id=quantum_session.interaction_id,
                format="json",
            )

        assert isinstance(result, list)
        text_content = result[0]
        # "Saved to:" must be the first line so CLI hosts parse it easily.
        assert text_content.text.lstrip().startswith("✅ **Saved to:**")
        assert "Resource URI" in text_content.text

        # File actually landed on disk under the configured export dir.
        written = list(tmp_path.iterdir())
        assert len(written) == 1
        assert written[0].suffix == ".json"
        assert written[0].stat().st_size > 0

    @pytest.mark.asyncio
    async def test_output_path_omitted_keeps_embedded_resource(
        self,
        quantum_session: ResearchSession,
        tmp_path,
        monkeypatch,
    ) -> None:
        """GUI clients still receive the EmbeddedResource alongside the path."""
        from mcp.types import EmbeddedResource

        from gemini_research_mcp.server import export_research_session

        monkeypatch.setenv("GEMINI_RESEARCH_EXPORT_DIR", str(tmp_path))

        with patch(
            "gemini_research_mcp.server.get_research_session",
            return_value=quantum_session,
        ):
            result = await export_research_session(
                interaction_id=quantum_session.interaction_id,
                format="json",
            )

        assert isinstance(result, list)
        assert any(isinstance(item, EmbeddedResource) for item in result)
