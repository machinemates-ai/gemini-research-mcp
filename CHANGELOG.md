# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-01-20

### Fixed

- **Invalid model name**: Changed `gemini-3.0-flash` to `gemini-3-flash-preview` (GA model not yet available)
  - Affected: `CLARIFIER_MODEL`, `DEFAULT_SUMMARY_MODEL`, `.env.example`, README

## [0.2.1] - 2026-01-20

### Changed

- **VS Code MCP config**: Updated README with correct `uvx --from` syntax for DOCX support ([#5](https://github.com/fortaine/gemini-research-mcp/pull/5))
- **Configuration docs**: Added `GEMINI_SUMMARY_MODEL` to configuration table
- **DOCX section**: Clarified "Pure Python" (no Pandoc binary required)

### Fixed

- **`uv run` command order**: Fixed `args` order (`run` before `--directory`) ([#6](https://github.com/fortaine/gemini-research-mcp/pull/6))
- **`.env.example`**: Removed non-existent `research_vendor_docs` feature and `EXTERNAL_API_*` variables
- **Model examples**: Updated to current defaults (`gemini-3-flash-preview`)

## [0.2.0] - 2026-01-20

### Added

- **Professional DOCX export** with:
  - Cover page with title, date, and research metadata
  - Clickable Table of Contents with navigation
  - Professional typography (Calibri fonts, 1-inch margins, 1.5x line spacing)
  - Executive summary with elegant formatting
  - Full research report with proper heading hierarchy (H1-H4)
  - Sources section with full clickable URLs
  - Metadata table with session details
- **`[docx]` optional dependency**: `pip install 'gemini-research-mcp[docx]'`
- ARM64/Apple Silicon installation instructions

### Changed

- Switched from `skelmis-docx` to `marko` + `python-docx` for DOCX generation (pure Python, no Pandoc binary needed)
- Updated dependencies

## [0.1.4] - 2026-01-19

### Fixed

- CI: Only lint `src/` directory, not tests

## [0.1.3] - 2026-01-18

### Added

- Automatic versioning from git tags using `hatch-vcs`
- CI tests and linting before publish

### Changed

- Updated author email

[0.2.2]: https://github.com/fortaine/gemini-research-mcp/compare/0.2.1...0.2.2
[0.2.1]: https://github.com/fortaine/gemini-research-mcp/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/fortaine/gemini-research-mcp/compare/0.1.4...0.2.0
[0.1.4]: https://github.com/fortaine/gemini-research-mcp/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/fortaine/gemini-research-mcp/releases/tag/0.1.3
