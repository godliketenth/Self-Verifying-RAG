"""
Helpers shared by the pipeline stages: loading prompt templates and
formatting passages for a prompt.

Prompts live in src/prompts/*.txt rather than inside Python strings, for two
reasons: a prompt change shows up as a clean one-file diff in git, and the
LLM cache key includes the prompt text, so editing a prompt automatically
invalidates exactly that stage's cached results and nothing else.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.types import Passage

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Read src/prompts/<name>.txt. Cached, since prompts never change mid-run."""
    f = PROMPT_DIR / f"{name}.txt"
    if not f.exists():
        raise FileNotFoundError(f"Prompt template not found: {f}")
    return f.read_text()


def format_passages(passages: list[Passage], include_pid: bool = True) -> str:
    """
    Render passages for a prompt.

    The [pid] prefix matters: it is how the model refers back to a specific
    passage when it reports which one justified its verdict, which is what
    lets the UI link a claim to its evidence.
    """
    if not passages:
        return "(no passages retrieved)"
    out = []
    for p in passages:
        head = f"[{p.pid}] {p.title}" if include_pid else p.title
        out.append(f"{head}\n{p.text}")
    return "\n\n".join(out)
