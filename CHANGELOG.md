# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-01-21

Enable VS Code "Save As" button for all export formats.
([#12](https://github.com/fortaine/gemini-research-mcp/pull/12))

### Added

- **Save As button for Markdown and JSON exports**: Extended `EmbeddedResource` pattern to all formats
- Text formats (MD, JSON) now use `TextResourceContents` for native file download
- Format-specific emoji and labels in export metadata (📄 DOCX, 📝 Markdown, 📋 JSON)

### Changed

- Consolidated export logic into single unified code path
- Removed unused export caching for text formats (now handled via EmbeddedResource)

## [0.3.0] - 2026-01-21

Major DOCX export improvements with syntax highlighting and layout fixes.
([#11](https://github.com/fortaine/gemini-research-mcp/pull/11))

### Added

- **Pygments syntax highlighting** for code blocks in DOCX exports with GitHub-inspired color scheme
- Code blocks now have GitHub-style background shading (#F6F8FA) and subtle borders

### Changed

- Footer attribution now uses Word's proper footer section (page margin area) instead of body paragraphs
- Removed redundant page breaks between TOC and Document Information sections
- Executive Summary now flows naturally after Document Information without forced page break
- Removed cover page subtitle (query in quotes) for cleaner appearance
- Removed empty paragraphs after tables (spacing handled by next paragraph's space_before)
- Added `pygments>=2.18.0` as optional dependency for DOCX export syntax highlighting

### Fixed

- **Blank final page eliminated**: Footer text no longer causes extra page by using Word's footer section
- DOCX export now correctly renders GFM tables with direct TableRow children (fixes tables not appearing in exported documents) ([#10](https://github.com/fortaine/gemini-research-mcp/pull/10))
