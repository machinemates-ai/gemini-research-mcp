# Gemini Research MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The **reference MCP server** for AI-powered research using **Gemini**. Combines fast grounded search with comprehensive Deep Research capabilities, plus async patterns for maximum flexibility.

## Tools

| Tool | Description | Latency | API Used |
|------|-------------|---------|----------|
| `research_web` | Fast web search with citations | 5-30 sec | Gemini 3 Flash + Google Search grounding |
| `research_deep` | Multi-step autonomous research | 3-20 min | Deep Research Agent (Interactions API) |
| `research_followup` | Continue conversation after research | 5-30 sec | Interactions API |

### Resources

| Resource | Description |
|----------|-------------|
| `research://models` | Lists available models & agents with capabilities |

### Workflow

```
research_web  ─── quick lookup ───▶  Got what you need?  ── yes ──▶ Done
       │                                        │
       │                                       no
       │                                        ▼
       └──────────────────────────────▶  research_deep  ──▶  Comprehensive report
                                       │         │
                               (auto-clarifies   │
                                vague queries)   │
                                                 ▼
                                        research_followup  ──▶  Dive deeper
```

### Advanced Features

- **Auto-Clarification (SEP-1330)**: `research_deep` automatically asks clarifying questions for vague queries via MCP Elicitation
- **MCP Tasks (SEP-1732)**: [Real-time progress](https://spec.modelcontextprotocol.io/specification/draft/server/tasks/) with streaming updates
- **File Search**: Search your own data alongside web using `file_search_store_names`
- **Follow-up**: Continue conversations with `previous_interaction_id`
- **Format Instructions**: Control report structure (sections, tables, tone)
- **Error Categorization**: Typed error categories (`AUTH_ERROR`, `RATE_LIMIT`, `SAFETY_BLOCK`, etc.)
- **Models Resource**: Discover available models and agents via `research://models`

## Installation

### From Source

```bash
git clone https://github.com/fortaine/gemini-research-mcp
cd gemini-research-mcp
uv sync
```

### From PyPI (future)

```bash
pip install gemini-research-mcp
# or
uv add gemini-research-mcp
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | **Yes** | — | Google AI Studio API key |
| `GEMINI_MODEL` | No | `gemini-3-flash-preview` | Model for `research_web` |
| `DEEP_RESEARCH_AGENT` | No | `deep-research-pro-preview-12-2025` | Agent for `research_deep` |

Get your API key from https://aistudio.google.com/apikey

### Environment File

Create `.env` from the template:

```bash
cp .env.example .env
# Edit .env with your API key
```

## Usage

### VS Code MCP (Recommended)

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "gemini-research": {
      "command": "uv",
      "args": ["--directory", "path/to/mcp-server", "run", "gemini-research-mcp"],
      "envFile": "${workspaceFolder}/path/to/mcp-server/.env"
    }
  }
}
```

Or use uvx (no install needed):

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
# After uv sync
uv run gemini-research-mcp

# Or with uvx (no install needed)
uvx gemini-research-mcp
```

## Architecture

### Transport: stdio

The server uses **stdio transport** (VS Code spawns it as a child process):

```python
mcp.run(transport="stdio")
```

**Why stdio over HTTP?**

| Aspect | stdio | HTTP |
|--------|-------|------|
| VS Code integration | ✅ Native | ⚠️ Needs URL config |
| Latency | Microseconds | Milliseconds |
| Configuration | Zero | Complex (ports, SSL, auth) |
| Security | Process isolation | Network exposure risk |
| Concurrency | Single client | Multiple clients |

For this use case (single developer, VS Code MCP), stdio is optimal. The 5-30 second Gemini API latency dominates; transport overhead is negligible.

### MCP Tasks (SEP-1732)

`research_deep` uses [MCP Tasks](https://spec.modelcontextprotocol.io/specification/draft/server/tasks/) for background execution with real-time progress:

```python
@mcp.tool(task=TaskConfig(mode="required"))
async def research_deep(..., progress: Progress):
    await progress.set_message("Researching...")
    await progress.increment(10)
```

### Module Structure

```
src/gemini_research_mcp/
├── __init__.py     # Package exports (ErrorCategory, DeepResearchResult, etc.)
├── server.py       # MCP server (3 tools + 1 resource)
├── config.py       # Configuration management
├── types.py        # Data types, exceptions, ErrorCategory enum
├── quick.py        # research_web (grounded search)
├── deep.py         # research_deep + research_followup
├── clarifier.py    # Query analysis and clarification (internal)
├── citations.py    # Citation extraction and URL resolution
└── py.typed        # PEP 561 type marker
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
# Unit tests only (no API calls)
uv run pytest

# With coverage report
uv run pytest --cov=src/gemini_research_mcp --cov-report=term-missing

# Include E2E tests (requires GEMINI_API_KEY)
uv run pytest -m e2e
```

## Why "Gemini Research"?

This server provides two distinct research capabilities:

1. **research_web** - Uses Gemini 3 Flash + Google Search grounding (fast, ~10s)
2. **research_deep** - Uses Gemini Deep Research Agent (comprehensive, ~5-15min)

### Thinking Configuration

Use `thinking_level` parameter:
- `minimal` - Minimize latency (chat/high-throughput)
- `low` - Balance speed and quality  
- `medium` - Good reasoning depth
- `high` - Maximum reasoning (default, recommended for research)

The name "gemini-research-mcp" accurately reflects that both tools are Gemini-powered research capabilities.

## Pricing Estimate

| Tool | Typical Cost |
|------|-------------|
| `research_web` | ~$0.01-0.05 per query |
| `research_deep` | ~$2-5 per task |

*Deep Research uses ~80-160 searches and ~250k-900k tokens per task.*

## Comparison

This server implements the most complete feature set among Gemini Deep Research MCP implementations:

| Feature | gemini-research-mcp | Others |
|---------|:-------------------:|:------:|
| **Tools** | 6 | 1-3 |
| Sync deep research | ✅ | ✅ |
| Async start/poll pattern | ✅ | ✅ (some) |
| Fast web search | ✅ | ❌ |
| Follow-up conversations | ✅ | ❌ |
| MCP Tasks (real-time progress) | ✅ | ❌ |
| **MCP Features** | | |
| Elicitation (SEP-1330) | ✅ | ❌ |
| Resources | ✅ | ❌ |
| **Advanced** | | |
| Format instructions | ✅ | ⚠️ (some) |
| File search (RAG) | ✅ | ❌ |
| Typed error categories | ✅ | ❌ |
| Thinking level config | ✅ | ⚠️ (some) |
| Citation extraction | ✅ | ⚠️ (basic) |

### Key Differentiators

1. **Complete workflow coverage**: Web search → Clarify → Deep Research → Follow-up
2. **MCP showcase**: Implements SEP-1330 (Elicitation), SEP-1732 (Tasks), Resources
3. **Two execution models**: Blocking (MCP Tasks) + Non-blocking (start/check)
4. **Production-ready**: Error categorization, retry-aware exceptions, typed responses
5. **First-class RAG**: `file_search_store_names` for grounding on your data
6. **Format control**: Structure reports with `format_instructions`

## License

MIT
