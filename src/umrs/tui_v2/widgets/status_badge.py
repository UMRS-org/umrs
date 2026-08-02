"""Status badge styling helper for monitor v2."""

from __future__ import annotations

from rich.text import Text


_STATUS_STYLES = {
    "running": "[cyan]{}[/cyan]",
    "queued": "[yellow]{}[/yellow]",
    "pending": "[yellow]{}[/yellow]",
    "succeeded": "[green]{}[/green]",
    "completed": "[green]{}[/green]",
    "done": "[green]{}[/green]",
    "failed": "[red]{}[/red]",
    "cancelled": "[dim]{}[/dim]",
    "lost": "[dim]{}[/dim]",
    "unknown": "[dim]{}[/dim]",
}

# Lifecycle glyph + color per status. Icon-first so status reads correctly
# even on monochrome or color-blind terminals. Colours mirror _STATUS_STYLES.
_STATUS_GLYPHS: dict[str, tuple[str, str]] = {
    "running": ("●", "cyan"),
    "provisioning": ("◐", "cyan"),
    "starting": ("◐", "cyan"),
    "stopping": ("◐", "yellow"),
    "queued": ("◌", "yellow"),
    "pending": ("◌", "yellow"),
    "succeeded": ("✓", "green"),
    "completed": ("✓", "green"),
    "done": ("✓", "green"),
    "ready": ("✓", "green"),
    "failed": ("✗", "red"),
    "error": ("✗", "red"),
    "lost": ("✗", "bright_black"),
    "stopped": ("○", "bright_black"),
    "cancelled": ("⊘", "bright_black"),
    "canceled": ("⊘", "bright_black"),
}

_DEFAULT_GLYPH: tuple[str, str] = ("○", "bright_black")


def status_style(status: str) -> str:
    """Return a Rich markup styled version of the status string."""
    template = _STATUS_STYLES.get(status.lower() if status else "", "{}")
    return template.format(status)


def status_text(status: str) -> Text:
    """Return a glyph-prefixed, colour-coded Text cell for a lifecycle status."""
    key = (status or "").strip().lower()
    glyph, color = _STATUS_GLYPHS.get(key, _DEFAULT_GLYPH)
    return Text(f"{glyph} {status or 'unknown'}", style=color)
