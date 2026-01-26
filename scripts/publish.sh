#!/usr/bin/env bash
# Publish gemini-research-mcp to PyPI
#
# Prerequisites:
#   1. PyPI account: https://pypi.org/account/register/
#   2. API token: https://pypi.org/manage/account/token/
#   3. Store token: export PYPI_TOKEN=pypi-...
#
# Usage:
#   ./scripts/publish.sh          # Publish to PyPI
#   ./scripts/publish.sh --test   # Publish to TestPyPI first
#
set -euo pipefail

cd "$(dirname "$0")/.."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Check for test mode
TEST_MODE=false
if [[ "${1:-}" == "--test" ]]; then
    TEST_MODE=true
    info "Publishing to TestPyPI (test mode)"
fi

# Verify clean git state
if [[ -n "$(git status --porcelain)" ]]; then
    error "Working directory not clean. Commit or stash changes first."
    exit 1
fi

# Get version from pyproject.toml
VERSION=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)
info "Publishing version: $VERSION"

# Verify version tag doesn't exist
if git rev-parse "v$VERSION" >/dev/null 2>&1; then
    error "Tag v$VERSION already exists. Bump version in pyproject.toml first."
    exit 1
fi

# Run tests
info "Running tests..."
uv run pytest -v --tb=short
info "Tests passed ✓"

# Run linting
info "Running linter..."
uv run ruff check .
info "Linting passed ✓"

# Verify README renders correctly
info "Checking README rendering..."
uv run python -c "
import tomllib
from pathlib import Path

# Load pyproject.toml
with open('pyproject.toml', 'rb') as f:
    config = tomllib.load(f)

readme_file = config['project'].get('readme', 'README.md')
readme = Path(readme_file).read_text()

# Basic checks
assert len(readme) > 500, 'README too short'
assert '# ' in readme, 'README missing headers'
assert 'gemini' in readme.lower(), 'README should mention Gemini'
print(f'README: {len(readme)} chars, looks good ✓')
"

# Build package
info "Building package..."
rm -rf dist/
uv build
info "Build complete ✓"

# Show what will be uploaded
info "Package contents:"
ls -la dist/

# Verify package with twine
info "Verifying package..."
uv run --with twine twine check dist/*
info "Package verification passed ✓"

# Publish
if $TEST_MODE; then
    info "Uploading to TestPyPI..."
    uv run --with twine twine upload --repository testpypi dist/*
    echo ""
    info "Published to TestPyPI!"
    info "Test install: pip install -i https://test.pypi.org/simple/ gemini-research-mcp"
    info "View: https://test.pypi.org/project/gemini-research-mcp/"
else
    # Confirm before publishing to production
    echo ""
    warn "About to publish v$VERSION to PyPI (production)."
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Aborted."
        exit 0
    fi

    info "Uploading to PyPI..."
    uv run --with twine twine upload dist/*

    # Create git tag
    info "Creating git tag v$VERSION..."
    git tag -a "v$VERSION" -m "Release v$VERSION"
    git push origin "v$VERSION"

    echo ""
    info "✅ Published gemini-research-mcp v$VERSION to PyPI!"
    info "View: https://pypi.org/project/gemini-research-mcp/"
    info "Install: pip install gemini-research-mcp"
    info "   or:   uv add gemini-research-mcp"
fi
