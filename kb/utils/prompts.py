"""Prompt building helpers."""

from __future__ import annotations

import config


def render_prompt(template: str, **values: str | int) -> str:
    """Substitute <<KEY>> placeholders in a prompt template.

    Plain str.format cannot be used here: prompt templates contain the curly
    braces of the JSON output examples.
    """
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"<<{key.upper()}>>", str(value))
    return rendered


def format_categories() -> str:
    """Render the configured categories as a bullet list for a prompt."""
    return "\n".join(
        f"- {name} — {description}" for name, description in config.CATEGORIES.items()
    )


def format_category_names() -> str:
    """Render the configured category keys as a comma separated list."""
    return ", ".join(config.CATEGORIES)
