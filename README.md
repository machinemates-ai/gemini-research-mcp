# Gemini Research MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MCP server for AI-powered research using **Gemini**. Combines fast grounded search with comprehensive Deep Research capabilities.

## Tools

| Tool | Description | Latency | API Used |
|------|-------------|---------|----------|
| `research_quick` | Fast web search with citations | 5-30 sec | Gemini + Google Search grounding |
| `research_deep` | Multi-step autonomous research | 3-20 min | Deep Research Agent (Interactions API) |
| `research_status` | Check status of background tasks | instant | Interactions API |
| `research_followup` | Continue conversation after research | 5-30 sec | Interactions API |

### Workflow

```
research_quick  ─── quick lookup ───▶  Got what you need?  ── yes ──▶ Done
       │                                        │
       │                                       no
       │                                        ▼
       └──────────────────────────────▶  research_deep  ──▶  Wait for results
                                                │
                                                ▼
                                        research_followup  ──▶  Dive deeper
```

### Advanced Features

- **File Search**: Search your own data alongside web using `file_search_store_names`
- **Follow-up**: Continue conversations with `previous_interaction_id`
- **Format Instructions**: Control report structure (sections, tables, tone)
- **Real-time Progress**: [MCP Tasks](https://spec.modelcontextprotocol.io/specification/draft/server/tasks/) with streaming updates

## Installation

### From PyPI

```bash
pip install gemini-research-mcp
# or
uv add gemini-research-mcp
```

### From Source

```bash
git clone https://github.com/fortaine/gemini-research-mcp
cd gemini-research-mcp
uv sync
```

## Configuration

```bash
# Required
GEMINI_API_KEY=your-api-key-here

# Optional: Override default models
GEMINI_MODEL=gemini-2.5-flash
DEEP_RESEARCH_AGENT=deep-research-pro-preview-12-2025
```

Get your key from https://aistudio.google.com/apikey

## Usage

### VS Code (Recommended)

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "gemini-research": {
      "command": "uvx",
      "args": ["gemini-research-mcp"],
      "env": {
        "GEMINI_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Command Line

```bash
# After pip install
gemini-research-mcp

# Or with uvx (no install needed)
uvx gemini-research-mcp
```

## Architecture

- **Transport**: stdio (VS Code spawns as child process)
- **Tasks**: [MCP Tasks](https://spec.modelcontextprotocol.io/specification/draft/server/tasks/) (SEP-1732) for `research_deep`
- **Framework**: [FastMCP](https://github.com/jlowin/fastmcp) 2.5+

## Module Structure

```
gemini_research_mcp/
├── __init__.py     # Package exports
├── server.py       # MCP server (FastMCP tools)
├── config.py       # Configuration management
├── types.py        # Data types and exceptions
├── quick.py        # Quick research (grounded search)
├── deep.py         # Deep research (multi-step agent)
└── citations.py    # Citation extraction and URL resolution
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run mypy src/
uv run ruff check src/
```

## Why "Gemini Research"?

This server provides two distinct research capabilities:

1. **research_quick** - Uses Gemini Flash + Google Search grounding (NOT Deep Research)
2. **research_deep** - Uses Gemini Deep Research Agent

The name "gemini-research-mcp" accurately reflects that both tools are Gemini-powered research capabilities, rather than implying everything uses Deep Research.

## License

MIT
