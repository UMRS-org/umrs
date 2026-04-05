"""Tests for B7 — Pipeline Templates."""

import pytest

from umrs.pipeline_sdk.templates import (
    ParameterType,
    PipelineTemplate,
    TemplateParameter,
    TemplateRegistry,
)


class TestTemplateParameter:
    def test_string_param(self):
        p = TemplateParameter(
            name="model_name",
            param_type=ParameterType.STRING,
            description="Model to use",
            default="e5-base",
        )
        assert p.name == "model_name"
        assert p.default == "e5-base"

    def test_integer_param_validation(self):
        p = TemplateParameter(
            name="batch_size",
            param_type=ParameterType.INTEGER,
            min_value=1,
            max_value=128,
        )
        assert p.min_value == 1
        assert p.max_value == 128

    def test_choice_param(self):
        p = TemplateParameter(
            name="mode",
            param_type=ParameterType.CHOICE,
            choices=["fast", "accurate", "balanced"],
            default="balanced",
        )
        assert "fast" in p.choices

    def test_validate_integer_range(self):
        p = TemplateParameter(
            name="batch_size",
            param_type=ParameterType.INTEGER,
            min_value=1,
            max_value=128,
        )
        assert p.validate(64) is None
        assert p.validate(200) is not None  # exceeds max
        assert p.validate(0) is not None  # below min


class TestPipelineTemplate:
    def test_validate_params_valid(self):
        tmpl = PipelineTemplate(
            template_id="test-tmpl",
            name="test-template",
            description="Test",
            parameters=[
                TemplateParameter(
                    name="model",
                    param_type=ParameterType.STRING,
                    required=True,
                ),
                TemplateParameter(
                    name="batch_size",
                    param_type=ParameterType.INTEGER,
                    default=32,
                ),
            ],
            workflow_skeleton={"model": "{{model}}", "batch_size": "{{batch_size}}"},
        )
        errors = tmpl.validate_params({"model": "clip"})
        assert len(errors) == 0

    def test_validate_params_missing_required(self):
        tmpl = PipelineTemplate(
            template_id="test-tmpl",
            name="test-template",
            description="Test",
            parameters=[
                TemplateParameter(
                    name="model",
                    param_type=ParameterType.STRING,
                    required=True,
                ),
            ],
            workflow_skeleton={"model": "{{model}}"},
        )
        errors = tmpl.validate_params({})
        assert len(errors) > 0

    def test_render_substitution(self):
        tmpl = PipelineTemplate(
            template_id="test-tmpl",
            name="test-template",
            description="Test",
            parameters=[
                TemplateParameter(name="name", param_type=ParameterType.STRING, default="default"),
                TemplateParameter(name="count", param_type=ParameterType.INTEGER, default=10),
            ],
            workflow_skeleton={"name": "{{name}}", "count": "{{count}}"},
        )
        result = tmpl.render({"name": "myPipeline", "count": 42})
        assert isinstance(result, dict)
        assert result["name"] == "myPipeline"
        assert result["count"] == 42

    def test_render_uses_defaults(self):
        tmpl = PipelineTemplate(
            template_id="test-tmpl",
            name="test-template",
            description="Test",
            parameters=[
                TemplateParameter(name="x", param_type=ParameterType.STRING, default="hello"),
            ],
            workflow_skeleton={"value": "{{x}}"},
        )
        result = tmpl.render({})
        assert result["value"] == "hello"


class TestTemplateRegistry:
    def test_register_and_list(self):
        registry = TemplateRegistry(include_builtins=False)
        tmpl = PipelineTemplate(
            template_id="my-template",
            name="my-template",
            description="Custom template",
            parameters=[],
            workflow_skeleton={},
        )
        registry.register(tmpl)
        templates = registry.list_templates()
        ids = [t.template_id for t in templates]
        assert "my-template" in ids

    def test_get(self):
        registry = TemplateRegistry(include_builtins=False)
        tmpl = PipelineTemplate(
            template_id="find-me",
            name="find-me",
            description="Findable template",
            parameters=[],
            workflow_skeleton={},
        )
        registry.register(tmpl)
        found = registry.get("find-me")
        assert found is not None
        assert found.template_id == "find-me"

    def test_get_nonexistent(self):
        registry = TemplateRegistry(include_builtins=False)
        assert registry.get("nope") is None

    def test_builtin_templates_exist(self):
        """Check that built-in templates are discoverable."""
        registry = TemplateRegistry(include_builtins=True)
        templates = registry.list_templates()
        assert isinstance(templates, list)
        assert len(templates) >= 5  # 5 built-in templates
        ids = [t.template_id for t in templates]
        assert "image_embedding" in ids
        assert "text_analysis" in ids

    def test_unregister(self):
        registry = TemplateRegistry(include_builtins=False)
        tmpl = PipelineTemplate(
            template_id="removable",
            name="removable",
            description="remove me",
            parameters=[],
            workflow_skeleton={},
        )
        registry.register(tmpl)
        assert registry.unregister("removable")
        assert not registry.unregister("removable")
