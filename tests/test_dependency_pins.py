"""Dependency verification tests for the FastMCP 4 stable stack.

These tests do NOT import gemini_research_mcp.server (which requires the
FastMCP 4 core migration to be completed first). They only verify that the
dependency stack resolved from pyproject.toml/uv.lock meets the floors expected
for this migration, so a future accidental downgrade is caught immediately
instead of surfacing as a confusing downstream failure.
"""

import re
from importlib.metadata import version

import pytest


def _release(package: str) -> tuple[int, ...]:
    raw = version(package)
    assert re.fullmatch(r"\d+(\.\d+)*", raw), f"{package} {raw} is not a stable release"
    return tuple(int(p) for p in raw.split("."))


def test_fastmcp_is_stable_4_1_or_newer() -> None:
    """fastmcp must be a stable 4.x release, at least 4.1.3."""
    release = _release("fastmcp")
    assert release[0] == 4
    assert release >= (4, 1, 3)


@pytest.mark.parametrize("package", ["fastmcp-slim", "fastmcp-tasks"])
def test_fastmcp_companions_match_fastmcp(package: str) -> None:
    """fastmcp-slim and fastmcp-tasks are released in lockstep with fastmcp."""
    assert version(package) == version("fastmcp")


def test_mcp_sdk_is_v2() -> None:
    """FastMCP 4 builds on MCP Python SDK v2; verify the major version landed."""
    mcp_version = version("mcp")
    major = int(mcp_version.split(".", maxsplit=1)[0])
    assert major >= 2, f"expected MCP SDK v2+, got {mcp_version}"


def test_google_genai_is_latest_verified_stable() -> None:
    """google-genai should be at or above the latest stable verified at plan time.

    2.32.0 was the latest stable release on PyPI when this migration was
    planned. This is a floor, not a ceiling: a newer stable release passing
    this test is expected and correct.
    """
    genai_version = version("google-genai")
    parts = tuple(int(p) for p in genai_version.split(".")[:2])
    assert parts >= (2, 32), f"expected google-genai>=2.32, got {genai_version}"


def test_tasks_extension_importable() -> None:
    """fastmcp_tasks.TasksExtension is the FastMCP 4 replacement for task=True wiring."""
    from fastmcp_tasks import TasksExtension

    assert TasksExtension is not None


def test_task_config_moved_to_utilities() -> None:
    """TaskConfig moved from fastmcp.server.tasks to fastmcp.utilities.tasks in v4."""
    from fastmcp.utilities.tasks import TaskConfig

    config = TaskConfig(mode="optional")
    assert config.mode == "optional"


def test_bm25_search_transform_importable() -> None:
    """BM25SearchTransform lives under fastmcp.server.transforms.search in v4."""
    from fastmcp.server.transforms.search import BM25SearchTransform

    assert BM25SearchTransform is not None


def test_legacy_task_config_import_path_removed() -> None:
    """The FastMCP 3 import path must be gone; a future accidental use should fail loudly."""
    with pytest.raises(ModuleNotFoundError):
        from fastmcp.server.tasks.config import TaskConfig  # noqa: F401
