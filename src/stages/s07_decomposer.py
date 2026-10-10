"""
Stage 7 - Claim decomposer (FActScore, Min et al. 2023; JointCQ, Xu et al. 2025).

Turns the draft answer into a list of atomic, self-contained claims, each with
a search query for finding its evidence.

WHY DECOMPOSE AT ALL
--------------------
"Is this paragraph faithful?" is a question with one blurry answer. A
three-sentence answer where two sentences are solid and one is invented is
neither faithful nor unfaithful. Decomposing first makes the question sharp:
each claim gets its own verdict, so the user learns WHICH sentence to distrust
rather than receiving one confidence score for the whole paragraph.

WHY GENERATE THE QUERY AT THE SAME TIME (JointCQ)
-------------------------------------------------
The alternative is to extract claims first, then write a query for each in a
second pass. Doing both in one call is cheaper and better grounded: the model
still has the surrounding answer in view when it decides what evidence the
claim would need.
"""

from __future__ import annotations

import re

from src.config import Config
from src.llm import LLMClient, as_object
from src.stages.base import load_prompt
from src.stages.s06_generator import REFUSAL
from src.types import Claim

STAGE = "s07_decomposer"


_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace - for comparison only."""
    return " ".join(_PUNCT_RE.sub(" ", text.lower()).split())


def _dedupe(claims: list[Claim], threshold: float) -> list[Claim]:
    """
    Remove near-duplicate claims.

    A decomposer sometimes emits the same fact twice in slightly different
    words. Verifying both wastes money and double-counts in the faithfulness
    score. We compare normalised word sets with the Jaccard index rather than
    embeddings - it needs no model, and for near-identical short sentences it
    is entirely adequate.

    Normalisation strips punctuation as well as case: without that, "France."
    and "France" count as different words and two otherwise identical claims
    survive de-duplication.
    """
    kept: list[Claim] = []
    seen: list[set[str]] = []
    for c in claims:
        words = set(_normalize(c.text).split())
        if not words:
            continue
        is_dupe = False
        for prev in seen:
            overlap = len(words & prev) / max(1, len(words | prev))
            if overlap >= threshold:
                is_dupe = True
                break
        if not is_dupe:
            kept.append(c)
            seen.append(words)
    return kept


def decompose_answer(
    answer: str,
    client: LLMClient,
    cfg: Config,
    is_shadow: bool = False,
    question: str = "",
) -> list[Claim]:
    """
    Split the draft answer into verifiable claims.

    The QUESTION is passed in only so a terse answer can be made
    self-contained. HotpotQA rewards short answers, so a draft is often a bare
    name like "Diamond White." - which asserts no proposition on its own and
    decomposes to nothing, leaving faithfulness undefined for that question.
    With the question available, it becomes "Diamond White voiced Frankie
    Greene", which can be verified.

    THIS DOES NOT WEAKEN THE ASYMMETRY RULE. That rule constrains Stage 9, the
    verifier, which must never see generator-side context. The decomposer is
    generator-side itself - it is reading the generator's output by
    definition. The prompt forbids importing facts from the question that the
    answer did not assert, because verifying those would test the question
    rather than the answer.

    `is_shadow=True` marks this as a measurement-only call (arms A and B),
    so its cost is excluded from that arm's reported total.

    A refusal is returned as zero claims: "the passages do not contain enough
    information" asserts no fact about the world, so there is nothing to
    verify. Treating it as a claim would produce a spurious UNSUPPORTED and
    silently depress the faithfulness score of whichever arm refuses most.
    """
    answer = answer.strip()
    if not answer or REFUSAL.lower() in answer.lower():
        return []

    prompt = load_prompt(STAGE).format(
        answer=answer, max_claims=cfg.pipeline.max_claims,
        question=question or "(not provided)",
    )
    raw, _ = client.complete_json(prompt, stage=STAGE,
                                  model=cfg.llm.helper_model,
                                  is_shadow=is_shadow)
    data = as_object(raw, "claims")

    claims: list[Claim] = []
    for item in data.get("claims", []):
        text = str(item.get("claim", "")).strip()
        if not text:
            continue
        query = str(item.get("query", "")).strip() or text
        src = str(item.get("source_sentence", "")).strip()
        # The model is told to copy source_sentence verbatim, but it sometimes
        # paraphrases. If it is not actually in the answer, fall back to the
        # claim text - the RAGTruth span mapping needs a real offset, and a
        # hallucinated "source sentence" would silently corrupt it.
        if src and src not in answer:
            src = ""
        claims.append(Claim(text=text, verification_query=query,
                            source_sentence=src))

    claims = _dedupe(claims, cfg.pipeline.claim_dedup_threshold)
    return claims[: cfg.pipeline.max_claims]
