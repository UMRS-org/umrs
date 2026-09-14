"""Tests for the FormScreen modal form."""

from __future__ import annotations

import pytest
from umrs.tui.forms.screens import FormField, FormSchema, FormScreen


def _make_schema():
    return FormSchema(
        title="Submit Job",
        fields=[
            FormField(name="job_type", label="Job Type", value="index_build", required=True),
            FormField(name="project", label="Project", value="demo"),
        ],
    )


@pytest.mark.asyncio
async def test_form_constructs():
    schema = _make_schema()
    screen = FormScreen(schema)
    assert screen._schema.title == "Submit Job"


@pytest.mark.asyncio
async def test_form_validation():
    schema = FormSchema(
        title="Register Model",
        fields=[
            FormField(name="name", label="Name", required=True),
            FormField(name="path", label="Path"),
        ],
    )
    screen = FormScreen(schema)
    assert screen._schema.fields[0].required
