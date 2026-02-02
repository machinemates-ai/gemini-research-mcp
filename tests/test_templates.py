"""Unit tests for format instruction templates.

Tests template structure, helpers, and category organization.
Run with: uv run pytest tests/test_templates.py -v
"""

import pytest

from gemini_research_mcp.templates import (
    ALL_TEMPLATES,
    API_EVALUATION,
    COMPARISON_TABLE,
    COMPETITIVE_ANALYSIS,
    CRITIQUE_PROMPT,
    DEEP_DIVE,
    EXECUTIVE_BRIEFING,
    LITERATURE_REVIEW,
    MARKET_RESEARCH,
    PROS_CONS_ANALYSIS,
    RESEARCH_BRIEF,
    RESEARCH_PLAN_PROMPT,
    TECHNICAL_OVERVIEW,
    TEMPLATES_BY_CATEGORY,
    FormatTemplate,
    TemplateCategory,
    get_template,
    list_templates,
)


class TestTemplateCategory:
    """Test TemplateCategory enum."""

    def test_category_values(self):
        """Categories should have expected values."""
        assert TemplateCategory.BUSINESS.value == "business"
        assert TemplateCategory.TECHNICAL.value == "technical"
        assert TemplateCategory.ACADEMIC.value == "academic"
        assert TemplateCategory.ANALYSIS.value == "analysis"

    def test_all_categories_exist(self):
        """All expected categories should be defined."""
        categories = list(TemplateCategory)
        assert len(categories) == 4
        assert TemplateCategory.BUSINESS in categories
        assert TemplateCategory.TECHNICAL in categories
        assert TemplateCategory.ACADEMIC in categories
        assert TemplateCategory.ANALYSIS in categories


class TestFormatTemplate:
    """Test FormatTemplate dataclass."""

    def test_construction(self):
        """FormatTemplate should be constructable with all fields."""
        template = FormatTemplate(
            name="Test Template",
            description="A test description",
            category=TemplateCategory.BUSINESS,
            instructions="Write a report...",
        )
        assert template.name == "Test Template"
        assert template.description == "A test description"
        assert template.category == TemplateCategory.BUSINESS
        assert template.instructions == "Write a report..."

    def test_str_returns_instructions(self):
        """__str__ should return the instructions for easy use."""
        template = FormatTemplate(
            name="Test",
            description="Test",
            category=TemplateCategory.TECHNICAL,
            instructions="## Header\n\nContent here",
        )
        assert str(template) == "## Header\n\nContent here"


class TestPredefinedTemplates:
    """Test all predefined templates."""

    @pytest.mark.parametrize("template,expected_category", [
        (EXECUTIVE_BRIEFING, TemplateCategory.BUSINESS),
        (COMPETITIVE_ANALYSIS, TemplateCategory.BUSINESS),
        (MARKET_RESEARCH, TemplateCategory.BUSINESS),
        (COMPARISON_TABLE, TemplateCategory.ANALYSIS),
        (PROS_CONS_ANALYSIS, TemplateCategory.ANALYSIS),
        (DEEP_DIVE, TemplateCategory.ANALYSIS),
        (TECHNICAL_OVERVIEW, TemplateCategory.TECHNICAL),
        (API_EVALUATION, TemplateCategory.TECHNICAL),
        (LITERATURE_REVIEW, TemplateCategory.ACADEMIC),
        (RESEARCH_BRIEF, TemplateCategory.ACADEMIC),
    ])
    def test_template_categories(
        self, template: FormatTemplate, expected_category: TemplateCategory
    ):
        """Each template should have the correct category."""
        assert template.category == expected_category

    @pytest.mark.parametrize("template", [
        EXECUTIVE_BRIEFING,
        COMPETITIVE_ANALYSIS,
        MARKET_RESEARCH,
        COMPARISON_TABLE,
        PROS_CONS_ANALYSIS,
        DEEP_DIVE,
        TECHNICAL_OVERVIEW,
        API_EVALUATION,
        LITERATURE_REVIEW,
        RESEARCH_BRIEF,
    ])
    def test_template_has_required_fields(self, template: FormatTemplate):
        """Each template should have all required fields populated."""
        assert template.name, "Template should have a name"
        assert template.description, "Template should have a description"
        assert template.instructions, "Template should have instructions"
        assert len(template.name) > 3, "Name should be meaningful"
        assert len(template.description) > 10, "Description should be meaningful"
        assert len(template.instructions) > 100, "Instructions should be substantial"

    @pytest.mark.parametrize("template", [
        EXECUTIVE_BRIEFING,
        COMPETITIVE_ANALYSIS,
        MARKET_RESEARCH,
        COMPARISON_TABLE,
        PROS_CONS_ANALYSIS,
        DEEP_DIVE,
        TECHNICAL_OVERVIEW,
        API_EVALUATION,
        LITERATURE_REVIEW,
        RESEARCH_BRIEF,
    ])
    def test_template_has_sources_section(self, template: FormatTemplate):
        """Each template should include a sources section for citations."""
        instructions_lower = template.instructions.lower()
        assert "sources" in instructions_lower or "references" in instructions_lower, \
            f"Template {template.name} should include sources/references section"


class TestAllTemplates:
    """Test the ALL_TEMPLATES registry."""

    def test_contains_all_templates(self):
        """ALL_TEMPLATES should contain all 10 templates."""
        assert len(ALL_TEMPLATES) == 10

    def test_expected_keys(self):
        """ALL_TEMPLATES should have expected keys."""
        expected_keys = {
            "executive_briefing",
            "competitive_analysis",
            "market_research",
            "comparison_table",
            "pros_cons",
            "deep_dive",
            "technical_overview",
            "api_evaluation",
            "literature_review",
            "research_brief",
        }
        assert set(ALL_TEMPLATES.keys()) == expected_keys

    def test_all_values_are_templates(self):
        """All values should be FormatTemplate instances."""
        for key, template in ALL_TEMPLATES.items():
            assert isinstance(template, FormatTemplate), f"{key} should be FormatTemplate"


class TestTemplatesByCategory:
    """Test the TEMPLATES_BY_CATEGORY grouping."""

    def test_all_categories_present(self):
        """All categories should have templates."""
        assert TemplateCategory.BUSINESS in TEMPLATES_BY_CATEGORY
        assert TemplateCategory.TECHNICAL in TEMPLATES_BY_CATEGORY
        assert TemplateCategory.ACADEMIC in TEMPLATES_BY_CATEGORY
        assert TemplateCategory.ANALYSIS in TEMPLATES_BY_CATEGORY

    def test_business_templates(self):
        """Business category should have 3 templates."""
        templates = TEMPLATES_BY_CATEGORY[TemplateCategory.BUSINESS]
        assert len(templates) == 3
        assert EXECUTIVE_BRIEFING in templates
        assert COMPETITIVE_ANALYSIS in templates
        assert MARKET_RESEARCH in templates

    def test_technical_templates(self):
        """Technical category should have 2 templates."""
        templates = TEMPLATES_BY_CATEGORY[TemplateCategory.TECHNICAL]
        assert len(templates) == 2
        assert TECHNICAL_OVERVIEW in templates
        assert API_EVALUATION in templates

    def test_academic_templates(self):
        """Academic category should have 2 templates."""
        templates = TEMPLATES_BY_CATEGORY[TemplateCategory.ACADEMIC]
        assert len(templates) == 2
        assert LITERATURE_REVIEW in templates
        assert RESEARCH_BRIEF in templates

    def test_analysis_templates(self):
        """Analysis category should have 3 templates."""
        templates = TEMPLATES_BY_CATEGORY[TemplateCategory.ANALYSIS]
        assert len(templates) == 3
        assert COMPARISON_TABLE in templates
        assert PROS_CONS_ANALYSIS in templates
        assert DEEP_DIVE in templates


class TestGetTemplate:
    """Test the get_template helper function."""

    def test_exact_key(self):
        """Should find template by exact key."""
        template = get_template("executive_briefing")
        assert template is not None
        assert template.name == "Executive Briefing"

    def test_case_insensitive(self):
        """Should be case-insensitive."""
        template = get_template("EXECUTIVE_BRIEFING")
        assert template is not None
        assert template.name == "Executive Briefing"

        template = get_template("Executive_Briefing")
        assert template is not None

    def test_space_to_underscore(self):
        """Should convert spaces to underscores."""
        template = get_template("executive briefing")
        assert template is not None
        assert template.name == "Executive Briefing"

    def test_dash_to_underscore(self):
        """Should convert dashes to underscores."""
        template = get_template("executive-briefing")
        assert template is not None
        assert template.name == "Executive Briefing"

    def test_mixed_case_and_separators(self):
        """Should handle mixed case and separators."""
        template = get_template("Competitive-Analysis")
        assert template is not None
        assert template.name == "Competitive Analysis"

    def test_not_found(self):
        """Should return None for unknown template."""
        template = get_template("nonexistent_template")
        assert template is None

    def test_empty_string(self):
        """Should return None for empty string."""
        template = get_template("")
        assert template is None


class TestListTemplates:
    """Test the list_templates helper function."""

    def test_returns_list(self):
        """Should return a list of dicts."""
        templates = list_templates()
        assert isinstance(templates, list)
        assert len(templates) == 10

    def test_dict_structure(self):
        """Each dict should have expected keys."""
        templates = list_templates()
        for t in templates:
            assert "name" in t
            assert "key" in t
            assert "description" in t
            assert "category" in t

    def test_category_is_string(self):
        """Category should be the string value, not enum."""
        templates = list_templates()
        categories = {t["category"] for t in templates}
        assert "business" in categories
        assert "technical" in categories
        assert "academic" in categories
        assert "analysis" in categories

    def test_keys_match_all_templates(self):
        """Keys should match ALL_TEMPLATES keys."""
        templates = list_templates()
        keys = {t["key"] for t in templates}
        assert keys == set(ALL_TEMPLATES.keys())


class TestResearchPlanPrompt:
    """Test the RESEARCH_PLAN_PROMPT constant."""

    def test_has_query_placeholder(self):
        """Should have {query} placeholder."""
        assert "{query}" in RESEARCH_PLAN_PROMPT

    def test_format_works(self):
        """Should be formattable with query."""
        formatted = RESEARCH_PLAN_PROMPT.format(query="test query")
        assert "test query" in formatted
        assert "{query}" not in formatted

    def test_includes_task_types(self):
        """Should document RESEARCH and DELIVERABLE task types."""
        assert "[RESEARCH]" in RESEARCH_PLAN_PROMPT
        assert "[DELIVERABLE]" in RESEARCH_PLAN_PROMPT


class TestCritiquePrompt:
    """Test the CRITIQUE_PROMPT constant."""

    def test_has_placeholders(self):
        """Should have {query} and {report} placeholders."""
        assert "{query}" in CRITIQUE_PROMPT
        assert "{report}" in CRITIQUE_PROMPT

    def test_format_works(self):
        """Should be formattable with query and report."""
        formatted = CRITIQUE_PROMPT.format(
            query="test query",
            report="# Report\n\nContent here"
        )
        assert "test query" in formatted
        assert "Content here" in formatted
        assert "{query}" not in formatted
        assert "{report}" not in formatted

    def test_includes_rating_options(self):
        """Should document PASS and NEEDS_REFINEMENT ratings."""
        assert "PASS" in CRITIQUE_PROMPT
        assert "NEEDS_REFINEMENT" in CRITIQUE_PROMPT

    def test_includes_output_format(self):
        """Should include expected output format."""
        assert "RATING:" in CRITIQUE_PROMPT
        assert "GAPS IDENTIFIED:" in CRITIQUE_PROMPT
        assert "FOLLOW_UP_QUESTIONS:" in CRITIQUE_PROMPT
