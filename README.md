# Gemini Research MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MCP server for AI-powered research using **Gemini**. Fast grounded search + comprehensive Deep Research.

## Tools

| Tool | Description | Latency |
|------|-------------|---------|
| `research_web` | Fast web search with citations | 5-30 sec |
| `research_deep` | Multi-step autonomous research | 3-20 min |
| `research_followup` | Continue conversation after research | 5-30 sec |

### Workflow

```
research_web  ─── quick lookup ───▶  Got what you need?  ── yes ──▶ Done
       │                                        │
       │                                       no
       │                                        ▼
       └──────────────────────────────▶  research_deep  ──▶  Comprehensive report
                                                 │
                                                 ▼
                                        research_followup  ──▶  Dive deeper
```

### Features

- **Auto-Clarification**: `research_deep` asks clarifying questions for vague queries via [MCP Elicitation](https://modelcontextprotocol.io/specification/2025-11-25/client/elicitation)
- **MCP Tasks**: [Real-time progress](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks) with streaming updates
- **File Search**: Search your own data alongside web using `file_search_store_names`
- **Format Instructions**: Control report structure (sections, tables, tone)
- **Models Resource**: Discover available models via `research://models`

## Installation

```bash
pip install gemini-research-mcp
# or
uv add gemini-research-mcp
```

From source:

```bash
git clone https://github.com/fortaine/gemini-research-mcp
cd gemini-research-mcp
uv sync
```

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | **Yes** | — | [Google AI Studio API key](https://aistudio.google.com/apikey) |
| `GEMINI_MODEL` | No | `gemini-3-flash-preview` | Model for `research_web` |
| `DEEP_RESEARCH_AGENT` | No | `deep-research-pro-preview-12-2025` | Agent for `research_deep` |

```bash
cp .env.example .env
# Edit .env with your API key
```

## Usage

### VS Code MCP

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

Or run from source:

```json
{
  "servers": {
    "gemini-research": {
      "command": "uv",
      "args": ["--directory", "path/to/gemini-research-mcp", "run", "gemini-research-mcp"],
      "envFile": "${workspaceFolder}/path/to/gemini-research-mcp/.env"
    }
  }
}
```

### Command Line

```bash
uv run gemini-research-mcp
# or
uvx gemini-research-mcp
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run mypy src/
uv run ruff check src/
```

### Tests

```bash
uv run pytest                    # Unit tests
uv run pytest -m e2e             # E2E tests (requires GEMINI_API_KEY)
uv run pytest --cov=src/gemini_research_mcp  # With coverage
```

## Pricing

| Tool | Typical Cost |
|------|-------------|
| `research_web` | ~$0.01-0.05 per query |
| `research_deep` | ~$2-5 per task |

*Deep Research uses ~80-160 searches and ~250k-900k tokens per task.*

## License

MIT
