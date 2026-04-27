# Gemini Research MCP for VS Code

AI-powered research tools for GitHub Copilot Chat, powered by Google Gemini.

## Features

- **🔎 Quick Research** (`research_web`): Fast web search with Gemini grounding (5-30 seconds)
- **🌐 Read URL** (`fetch_webpage`): Extract clean Markdown from a specific public URL
- **🔬 Deep Research** (`research_deep`): Default/fast autonomous research agent (3-20 minutes)
- **🔬 Deep Research Max** (`research_deep_max`): Maximum-comprehensiveness research for exhaustive/high-stakes work
- **💬 Follow-up** (`research_followup`): Continue conversation with previous research
- **🔄 Resume** (`resume_research`): Recover interrupted research sessions
- **📤 Export** (`export_research_session`): Export to Markdown, JSON, or Word

## Quick Start

1. Install this extension
2. Get a [Gemini API key](https://aistudio.google.com/apikey)
3. Set your API key in VS Code Settings → Gemini Research MCP → API Key
4. Open Copilot Chat and ask: "research the latest developments in quantum computing"

## Usage in Copilot Chat

```
@workspace research the competition for our product
@workspace do Deep Research Max due diligence on this acquisition market
@workspace do a deep dive on React Server Components vs Next.js App Router
@workspace what's the latest on the OpenAI o3 model?
@workspace read this URL and summarize it: https://example.com/blog-post
```

Use regular Deep Research for interactive, latency-sensitive research. Use Deep
Research Max when the request says Max, exhaustive, comprehensive, due diligence,
market map, literature review, high-stakes, board-ready, offline/nightly, or asks
for maximum completeness over speed.

## Requirements

- VS Code 1.96.0 or later
- GitHub Copilot Chat extension
- [uv](https://docs.astral.sh/uv/) (installed automatically if using uvx)

## Configuration

| Setting | Description |
|---------|-------------|
| `gemini-research-mcp.apiKey` | Your Gemini API key |

## Links

- [GitHub Repository](https://github.com/machinemates-ai/gemini-research-mcp)
- [PyPI Package](https://pypi.org/project/gemini-research-mcp/)
- [Documentation](https://github.com/machinemates-ai/gemini-research-mcp#readme)

## License

MIT
