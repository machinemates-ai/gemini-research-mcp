# Versioning

`gemini-research-mcp` follows [Semantic Versioning](https://semver.org/) with
the usual pre-1.0 caveat: minor releases may contain breaking changes, and every
breaking change still needs an explicit notice in the release notes.

## Rules

- **Patch** (`0.x.Y`): fixes that keep the public tool and resource contract.
- **Minor** (`0.X.0`): a coherent additive capability (for example the
  engine-neutral research jobs in 0.17). A model change alone is not a reason
  for a minor bump, and the package version is never synchronized with a model
  number (Gemini 4 Argon does not imply a `0.4` or `4.x` line).
- Pre-releases are published for validation before the final version.
- A published pre-release (such as `v0.16.0b4`) is **immutable**. It is never
  retagged or rewritten, and a stable `0.16.0` is not forced.

The Python version comes from the git tag (`hatch-vcs`), so the tag is the
source of truth for the wheel and sdist.

## Version mapping for 0.17

| Artifact | Release candidate | Final |
|----------|-------------------|-------|
| Git tag | `v0.17.0rc1` | `v0.17.0` |
| Python package (PEP 440) | `0.17.0rc1` | `0.17.0` |
| MCPB bundle (`manifest.json`) | `0.17.0-rc.1` | `0.17.0` |
| VS Code extension | independent | independent |

The VS Code extension keeps its own version and is bumped only when it is
actually rebuilt.

## Release gates

Preparing the manifest and tag is a release action and needs explicit release
authorization; a pull request, merge, tag, GitHub release and PyPI publish are
each separate. A GitHub prerelease is not PyPI delivery, so verify both
independently.

Local gates before any candidate:

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest -q
FASTMCP_MCP_CAMELCASE_COMPAT=false uv run pytest -q
git diff --check
uv build
```

Mocks prove mechanics, not real local research or quality. A final `0.17.0`
additionally requires live acceptance with separately authorized charges and
installs: one native run, one real Local Deep Research + Ollama + search run, an
explicit comparison, native recovery, honest local interruption, an attributed
bundle with no additional model calls, and child-process teardown.
