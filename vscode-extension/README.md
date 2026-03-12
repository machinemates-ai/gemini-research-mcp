# Gemini Research MCP for VS Code

AI-powered research tools for GitHub Copilot Chat, powered by Google Gemini.

## Features

- **🔎 Quick Research** (`research_web`): Fast web search with Gemini grounding (5-30 seconds)
- **🔬 Deep Research** (`research_deep`): Comprehensive autonomous research agent (3-20 minutes)
- **🧰 Research Utilities** (`search_tools` + `call_tool`): Discover URL reading, follow-up, resume, session, template, and export tools on demand

## Quick Start

1. Install this extension
2. Get a [Gemini API key](https://aistudio.google.com/apikey)
3. Set your API key in VS Code Settings → Gemini Research MCP → API Key
4. Open Copilot Chat and ask: "research the latest developments in quantum computing"

## Usage in Copilot Chat

Copilot can discover the utility tools automatically through FastMCP's search transform, so prompts like URL reading, export, resume, and follow-up still work without exposing every tool up front.

```
@workspace research the competition for our product
@workspace do a deep dive on React Server Components vs Next.js App Router
@workspace what's the latest on the OpenAI o3 model?
@workspace read this URL and summarize it: https://example.com/blog-post
```

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
