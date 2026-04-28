import pytest

from gemini_research_mcp.deep import build_interactions_tools


def test_build_interactions_tools_combines_file_search_and_mcp() -> None:
    tools = build_interactions_tools(
        file_search_store_names=["fileSearchStores/market"],
        mcp_servers=[
            {
                "name": "Market Researcher MCP",
                "url": "https://mcp.example.com/mcp",
                "headers": {"Authorization": "Bearer secret"},
                "allowed_tools": ["market_get_mission", "market_generate_report"],
            }
        ],
    )

    assert tools == [
        {
            "type": "file_search",
            "file_search_store_names": ["fileSearchStores/market"],
        },
        {
            "type": "mcp_server",
            "name": "Market Researcher MCP",
            "url": "https://mcp.example.com/mcp",
            "headers": {"Authorization": "Bearer secret"},
            "allowed_tools": ["market_get_mission", "market_generate_report"],
        },
    ]


def test_build_interactions_tools_rejects_non_https_remote_mcp() -> None:
    with pytest.raises(ValueError, match="must be HTTPS"):
        build_interactions_tools(mcp_servers=[{"url": "http://mcp.example.com/mcp"}])


def test_build_interactions_tools_allows_explicit_localhost_dev_override() -> None:
    tools = build_interactions_tools(
        mcp_servers=[{"url": "http://127.0.0.1:8000/mcp", "allow_insecure_localhost": True}]
    )

    assert tools == [{"type": "mcp_server", "url": "http://127.0.0.1:8000/mcp"}]


def test_build_interactions_tools_validates_allowed_tools() -> None:
    with pytest.raises(ValueError, match="allowed_tools"):
        build_interactions_tools(
            mcp_servers=[{"url": "https://mcp.example.com/mcp", "allowed_tools": ["ok", 42]}]
        )
