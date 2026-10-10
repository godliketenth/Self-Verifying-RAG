"""
Stage 5 - Relevance evaluator (CRAG, Yan et al. 2024).

Retrieval returns the top-k passages whether or not any of them are any good -
similarity search always returns *something*. This stage asks a model to score
each retrieved passage against the question and drop the ones that do not help.

Two reasons it earns its cost:
1. Fewer irrelevant passages in the generator's context means less material
   for the generator to hallucinate from.
2. If too few passages survive, we know retrieval failed and can re-plan,
   rather than generating a confident answer from nothing.

OFF in arm A, ON in arms B and C.
"""

from __future__ import annotations

from src.config import Config
from src.llm import LLMClient
from src.stages.base import format_passages, load_prompt
from src.types import Passage

STAGE = "s05_relevance"


def evaluate_relevance(
    question: str,
    passages: list[Passage],
    client: LLMClient,
    cfg: Config,
) -> tuple[list[Passage], dict[str, float]]:
    """
    Score passages and return (kept_passages, {pid: score}).

    Scores are attached to the returned Passage objects so the UI and the
    logs can show why a passage was kept or dropped.
    """
    if not passages:
        return [], {}

    prompt = load_prompt(STAGE).format(
        question=question,
        passages=format_passages(passages),
        threshold=cfg.pipeline.relevance_threshold,
    )
    data, _ = client.complete_json(prompt, stage=STAGE,
                                   model=cfg.llm.helper_model)

    scores: dict[str, float] = {}
    for item in data.get("passages", []):
        pid = str(item.get("pid", "")).strip()
        try:
            scores[pid] = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            scores[pid] = 0.0

    kept: list[Passage] = []
    for p in passages:
        # A passage the evaluator failed to mention is kept, not dropped.
        # Dropping on silence would let one malformed reply throw away all
        # the evidence and manufacture an "unanswerable" result.
        score = scores.get(p.pid)
        if score is None:
            kept.append(p)
            continue
        p.score = score
        if score >= cfg.pipeline.relevance_threshold:
            kept.append(p)

    return kept, scores


def needs_replan(kept: list[Passage], cfg: Config) -> bool:
    """
    True when too little relevant evidence survived, so retrieval should be
    re-planned. This is the trigger for the Stage 5 -> Stage 3 corrective loop.
    """
    return len(kept) < cfg.pipeline.min_relevant_passages
