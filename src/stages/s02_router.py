"""
Stage 2 - Complexity router (Adaptive-RAG, Jeong et al. 2024).

Decides how much retrieval a question deserves before spending anything on it.
The insight from Adaptive-RAG: treating every query with the same retrieval
budget is wasteful at the easy end and insufficient at the hard end.

In our ablation this stage is OFF in arm A and ON in arms B and C.
"""

from __future__ import annotations

from src.config import Config
from src.llm import LLMClient
from src.stages.base import load_prompt
from src.types import Route

STAGE = "s02_router"


def route_query(question: str, client: LLMClient, cfg: Config) -> tuple[Route, str]:
    """
    Classify a question's retrieval needs.

    Returns (route, reason). On an unrecognised label we fall back to
    SINGLE_HOP rather than crashing: a mis-route costs some accuracy, but a
    crash costs the whole run, and the fallback is recorded in the reason
    string so it is visible in the logs.
    """
    prompt = load_prompt(STAGE).format(question=question)
    data, _ = client.complete_json(prompt, stage=STAGE,
                                   model=cfg.llm.helper_model)

    raw = str(data.get("route", "")).strip().upper()
    reason = str(data.get("reason", ""))
    try:
        return Route(raw), reason
    except ValueError:
        return Route.SINGLE_HOP, f"[fallback: router returned {raw!r}] {reason}"
