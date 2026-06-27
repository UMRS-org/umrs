"""Schema-validated modal form screens.

``FormScreen`` is a :class:`textual.screen.ModalScreen` composed from a
declarative field schema.  It supports text inputs, select (dropdown)
fields, and boolean toggles.  Validation uses ``jsonschema`` (already a
base dependency) so JSON spec fields get proper schema validation with
inline error messages.  User input is preserved across validation errors
so the user doesn't lose their work on a typo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static, Switch


@dataclass
class FormField:
    """Declarative form field definition."""

    name: str
    label: str
    kind: str = "text"  # "text", "select", "bool"
    value: Any = ""
    choices: list[tuple[str, Any]] | None = None
    required: bool = False
    placeholder: str = ""


@dataclass
class FormSchema:
    """Schema for a modal form."""

    title: str
    fields: list[FormField]
    submit_label: str = "Submit"
    cancel_label: str = "Cancel"
    # Optional jsonschema for validating the assembled form values
    json_schema: dict[str, Any] | None = None


class FormScreen(ModalScreen[dict[str, Any]]):
    """A modal form screen composed from a :class:`FormSchema`."""

    DEFAULT_CSS = """
    FormScreen {
        align: center middle;
    }
    FormScreen > Vertical {
        width: 70;
        height: auto;
        max-height: 80%;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }
    FormScreen Label.title {
        text-style: bold;
        margin-bottom: 1;
    }
    FormScreen .field-label {
        margin-top: 1;
        color: $text;
    }
    FormScreen .error {
        color: $error;
        margin-top: 1;
    }
    FormScreen .buttons {
        margin-top: 2;
        height: 3;
        layout: horizontal;
    }
    FormScreen .buttons Button {
        margin-right: 2;
    }
    """

    def __init__(self, schema: FormSchema) -> None:
        super().__init__()
        self._schema = schema
        self._values: dict[str, Any] = {
            f.name: f.value for f in schema.fields
        }
        self._error: str = ""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._schema.title, classes="title")
            for field in self._schema.fields:
                yield Label(field.label, classes="field-label")
                if field.kind == "select":
                    yield Select(
                        field.choices or [],
                        value=field.value,
                        id=f"field-{field.name}",
                    )
                elif field.kind == "bool":
                    yield Switch(value=bool(field.value), id=f"field-{field.name}")
                else:
                    yield Input(
                        value=str(field.value),
                        placeholder=field.placeholder,
                        id=f"field-{field.name}",
                    )
            if self._error:
                yield Label(self._error, classes="error")
            with Horizontal():
                yield Button(self._schema.submit_label, id="submit", variant="primary")
                yield Button(self._schema.cancel_label, id="cancel", variant="default")

    @on(Button.Pressed, "#submit")
    def _on_submit(self) -> None:
        # Gather current values from widgets
        for field in self._schema.fields:
            widget = self.query_one(f"#field-{field.name}")
            if isinstance(widget, Input):
                self._values[field.name] = widget.value
            elif isinstance(widget, Select):
                self._values[field.name] = widget.value
            elif isinstance(widget, Switch):
                self._values[field.name] = widget.value

        # Validate required fields
        for field in self._schema.fields:
            if field.required and not self._values.get(field.name):
                self._error = f"{field.label} is required"
                self._refresh_error()
                return

        # Validate against jsonschema if provided
        if self._schema.json_schema:
            import jsonschema
            try:
                jsonschema.validate(self._values, self._schema.json_schema)
            except jsonschema.ValidationError as e:
                self._error = f"Validation error: {e.message}"
                self._refresh_error()
                return

        self.dismiss(self._values)

    @on(Button.Pressed, "#cancel")
    def _on_cancel(self) -> None:
        self.dismiss({})

    def _refresh_error(self) -> None:
        try:
            error_label = self.query_one(".error", Label)
            error_label.update(self._error)
        except Exception:
            pass
