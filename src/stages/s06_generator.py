"""
Stage 6 - Generator: produces the draft answer.

This is the ONLY stage that uses the expensive model, and it uses the SAME
model in all three arms. That is deliberate and it is what makes the ablation
valid: if arm C generated with a stronger model than arm A, any improvement
we measured could be the model rather than the pipeline.

This file also holds the claim regenerator used by the Stage 9 correction
loop, because rewriting a claim is a generation job.
"""

from __future__ import annotations

from src.config import Config
from src.llm import LLMClient
from src.stages.base import format_passages, load_prompt
from src.types import Claim, Passage

STAGE = "s06_generator"
CORRECTOR_STAGE = "s09_corrector"

# The exact sentence the prompt tells the generator to use when the passages
# are insufficient. Recognising it lets the decomposer skip decomposition
# (there are no factual claims in a refusal to answer).
REFUSAL = ("The retrieved passages do not contain enough information to "
           "answer this question.")


def generate_answer(
    question: str,
    passages: list[Passage],
    client: LLMClient,
    cfg: Config,
) -> str:
    """Write the draft answer from the retrieved passages."""
    prompt = load_prompt(STAGE).format(
        question=question, passages=format_passages(passages)
    )
    resp = client.complete(
        prompt,
        stage=STAGE,
        model=cfg.llm.generator_model,   # the fixed, expensive model
    )
    return resp.text.strip()


def regenerate_claim(
    claim: Claim,
    evidence: list[Passage],
    client: LLMClient,
    cfg: Config,
) -> tuple[str, bool]:
    """
    Rewrite one failed claim against freshly retrieved evidence.

    Returns (new_claim_text, dropped). This is the "regenerate only the failed
    claims, not the whole answer" behaviour from the proposal: the cost of
    correction scales with the number of bad claims, not with answer length.

    Uses the HELPER model, not the generator. Rewriting a single sentence to
    match given evidence is a much easier task than composing the original
    answer, and this keeps the correction loop cheap.
    """
    prompt = load_prompt(CORRECTOR_STAGE).format(
        claim=claim.text,
        verdict=claim.verdict.value if claim.verdict else "UNKNOWN",
        evidence=format_passages(evidence),
    )
    data, _ = client.complete_json(prompt, stage=CORRECTOR_STAGE,
                                   model=cfg.llm.helper_model)

    dropped = bool(data.get("dropped", False))
    new_text = str(data.get("claim", "")).strip()
    # An empty rewrite is a drop, whatever the flag claimed.
    if not new_text:
        return "", True
    return new_text, dropped
