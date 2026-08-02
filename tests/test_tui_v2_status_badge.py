"""Tests for monitor-v2 status badge glyphs (Wave-1 UX upgrade)."""

from __future__ import annotations

from umrs.tui_v2.widgets.status_badge import status_style, status_text


class TestStatusStyleUnchanged:
    """Legacy API keeps working — v1 widgets and summaries depend on it."""

    def test_running_uses_cyan_markup(self) -> None:
        assert status_style("running") == "[cyan]running[/cyan]"

    def test_unknown_status_returns_plain(self) -> None:
        assert status_style("weird-state") == "weird-state"


class TestStatusText:
    def test_known_status_gets_glyph_and_color(self) -> None:
        text = status_text("running")
        assert text.plain == "● running"
        # Style is applied to the whole cell.
        assert "cyan" in str(text.style)

    def test_terminal_statuses_have_success_glyphs(self) -> None:
        assert status_text("succeeded").plain.startswith("✓")
        assert status_text("failed").plain.startswith("✗")
        assert status_text("cancelled").plain.startswith("⊘")

    def test_transitional_statuses(self) -> None:
        assert status_text("provisioning").plain == "◐ provisioning"
        assert status_text("queued").plain == "◌ queued"

    def test_case_insensitive(self) -> None:
        assert status_text("Running").plain == "● Running"

    def test_unknown_status_falls_back_to_dim_circle(self) -> None:
        text = status_text("hibernating")
        assert text.plain == "○ hibernating"
        assert "bright_black" in str(text.style)

    def test_empty_status_shows_unknown(self) -> None:
        assert status_text("").plain == "○ unknown"

    def test_none_like_blank_status(self) -> None:
        assert status_text(None).plain == "○ unknown"  # type: ignore[arg-type]
