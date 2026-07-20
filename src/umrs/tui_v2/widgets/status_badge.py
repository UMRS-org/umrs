"""Status badge styling helper for monitor v2."""

from __future__ import annotations


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


def status_style(status: str) -> str:
    """Return a Rich markup styled version of the status string."""
    template = _STATUS_STYLES.get(status.lower() if status else "", "{}")
    return template.format(status)
