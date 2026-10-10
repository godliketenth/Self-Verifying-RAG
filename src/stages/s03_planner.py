"""
Stage 3 - Retrieval planner.

Turns one complex question into a small set of self-contained search queries.
This is what makes multi-hop retrieval possible: "Who directed the film that
won Best Picture in 1994?" cannot be answered by one lookup, because the film
must be identified before its director can be found.

Scope limit from the proposal: at most 2-3 hops (config: max_subqueries).
OFF in arm A, ON in arms B and C.
"""

from __future__ import annotations

from src.config import Config
from src.llm import LLMClient, as_object
from src.stages.base import load_prompt
from src.types import Route

STAGE = "s03_planner"


def plan_retrieval(
    question: str,
    route: Route,
    client: LLMClient,
    cfg: Config,
    previous: list[str] | None = None,
) -> list[str]:
    """
    Produce the list of queries to retrieve with.

    A SINGLE_HOP question needs no decomposition, so we skip the LLM call
    entirely and use the question verbatim - cheaper and strictly better than
    asking a model to paraphrase it.

    `previous` is passed on a re-plan (the Stage 5 -> 3 loop). We append a
    note telling the planner its last attempt retrieved nothing relevant, so
    it produces different queries rather than repeating itself.
    """
    if route is Route.SINGLE_HOP:
        return [question]

    prompt = load_prompt(STAGE).format(
        question=question, max_subqueries=cfg.pipeline.max_subqueries
    )
    if previous:
        prompt += (
            "\n\nNOTE: these queries were already tried and returned nothing "
            "relevant:\n"
            + "\n".join(f"- {q}" for q in previous)
            + "\nWrite DIFFERENT queries. Try other phrasings, other entity "
              "names, or break the question down differently."
        )

    raw, _ = client.complete_json(prompt, stage=STAGE,
                                  model=cfg.llm.helper_model)
    data = as_object(raw, "subqueries")

    subqueries: list[str] = []
    for q in data.get("subqueries", []):
        q = str(q).strip()
        # Drop blanks and duplicates; cap at the configured hop limit.
        if q and q not in subqueries:
            subqueries.append(q)
    subqueries = subqueries[: cfg.pipeline.max_subqueries]

    # A planner that returns nothing must not silently disable retrieval.
    return subqueries or [question]
