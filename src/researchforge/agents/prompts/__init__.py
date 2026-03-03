"""Prompt template loader."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template by name (without extension)."""
    path = _PROMPTS_DIR / f"{name}.txt"
    return path.read_text().strip()
