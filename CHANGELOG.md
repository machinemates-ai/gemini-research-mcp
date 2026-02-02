# Changelog

All notable changes to this project will be documented in this file.

## [0.10.0] - 2025-02-02

### Added

- **Grounded Fact-Check Mode**: New `grounded` parameter for `research_deep` that enables Google Search verification of research claims
  - Uses `GoogleSearch()` tool (same as ADK's `google_search`) to verify key claims
  - Returns `GroundedCritiqueResult` with `fact_check_rating`, `claims_verified`, `claims_disputed`, `sources`
  - Ratings: `VERIFIED`, `PARTIALLY_VERIFIED`, `DISPUTED`, `INSUFFICIENT_DATA`
- **Format Templates**: 10 pre-built output templates for consistent research formatting
  - Categories: `business`, `analysis`, `technical`, `academic`
  - Templates: `executive_summary`, `market_analysis`, `competitive_intel`, `trend_analysis`, `technology_comparison`, `risk_assessment`, `technical_report`, `implementation_guide`, `literature_review`, `research_brief`
- **Auto-Refine with Critique**: Iterative quality improvement loop using `auto_refine` parameter
  - Generates critique with actionable suggestions
  - Automatically refines research up to `max_refine_iterations` times
- **Research Planning**: `research_deep_planned` tool with MCP elicitation for human-in-the-loop approval
  - Generates research plan before execution
  - Presents plan to user for approval via MCP elicitation
- **Session Resume**: `resume_research` tool to recover interrupted research sessions
- **Session Followup**: `research_followup` tool to continue conversation with previous sessions

### Changed

- Upgraded from `gemini-2.0-flash` to `deep-research-pro-preview-12-2025` as managed agent
- Session persistence extended to 55 days (paid tier support)
- Export now supports DOCX, JSON, and Markdown formats natively

### Fixed

- Citation resolution accuracy improved with structured extraction
- Session matching algorithm now uses semantic similarity

## [0.6.6] - 2025-01-xx

### Added

- Initial MCP server with `research_web` and `research_deep` tools
- Session storage with resume capability
- Export to Markdown and JSON

---

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
