"""Tests for format templates and related helper behavior.

These tests verify the reusable template features:
- list_format_templates tool

Run with: uv run pytest tests/test_v2_features.py -v --tb=short
"""

import pytest


class TestListFormatTemplates:
    """Test list_format_templates tool."""

    @pytest.mark.asyncio
    async def test_list_format_templates_returns_json(self):
        """list_format_templates should return valid JSON wrapper."""
        import json

        from gemini_research_mcp.server import list_format_templates

        result = await list_format_templates()

        # Should be parseable JSON with wrapper
        data = json.loads(result)
        assert isinstance(data, dict)
        assert "templates" in data
        assert "count" in data
        assert len(data["templates"]) == 10  # 10 templates

    @pytest.mark.asyncio
    async def test_list_format_templates_structure(self):
        """Each template should have required fields."""
        import json

        from gemini_research_mcp.server import list_format_templates

        result = await list_format_templates()
        data = json.loads(result)

        for template in data["templates"]:
            assert "name" in template
            assert "key" in template
            assert "description" in template
            assert "category" in template
            assert template["category"] in ["business", "technical", "academic", "analysis"]

    @pytest.mark.asyncio
    async def test_list_format_templates_filter_by_category(self):
        """Category filter should work."""
        import json

        from gemini_research_mcp.server import list_format_templates

        result = await list_format_templates(category="business")
        data = json.loads(result)

        assert data["count"] == 3  # 3 business templates
        for template in data["templates"]:
            assert template["category"] == "business"


class TestTemplateResolution:
    """Test template key resolution in research_deep."""

    def test_get_template_by_key(self):
        """get_template should resolve template keys."""
        from gemini_research_mcp.templates import get_template

        template = get_template("executive_briefing")
        assert template is not None
        assert "Executive Summary" in template.instructions

    def test_get_template_by_alias(self):
        """get_template should handle various formats."""
        from gemini_research_mcp.templates import get_template

        # Different formats should all resolve
        assert get_template("executive-briefing") is not None
        assert get_template("Executive Briefing") is not None
        assert get_template("EXECUTIVE_BRIEFING") is not None


class TestIntegration:
    """Integration tests combining multiple v2.0 features."""

    @pytest.mark.asyncio
    async def test_template_to_format_instructions(self):
        """Template should be usable as format_instructions."""
        from gemini_research_mcp.templates import EXECUTIVE_BRIEFING, get_template

        # Get template by key (as user would via MCP)
        template = get_template("executive_briefing")
        assert template is not None

        # Should be usable as string for format_instructions
        format_str = str(template)
        assert "Executive Summary" in format_str
        assert "Key Findings" in format_str
        assert "Recommendations" in format_str

        # Should match the predefined constant
        assert format_str == str(EXECUTIVE_BRIEFING)

    @pytest.mark.asyncio
    async def test_all_template_keys_resolvable(self):
        """All template keys from list_templates should be resolvable."""
        from gemini_research_mcp.templates import get_template, list_templates

        templates = list_templates()

        for t in templates:
            resolved = get_template(t["key"])
            assert resolved is not None, f"Template key '{t['key']}' should resolve"
            assert resolved.name == t["name"]
