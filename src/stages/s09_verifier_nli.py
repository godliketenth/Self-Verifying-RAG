"""
Stage 9, alternative backend: an NLI cross-encoder instead of an LLM judge.

WHY THIS EXISTS
---------------
The LLM verifier measured precision 0.247 against RAGTruth's human labels -
it catches most real hallucinations but three of every four flags are false
positives, which is what drove arm C to delete correct content. That leaves
an open question the main experiment cannot answer: is the problem THIS
model, or the task framing?

A DeBERTa cross-encoder trained on MNLI answers it. It is a purpose-built
entailment classifier rather than a general model being asked to behave like
one, it runs locally at no cost, and its three labels map exactly onto ours:

    entailment    -> ENTAILED
    contradiction -> CONTRADICTED
    neutral       -> UNSUPPORTED

If the cross-encoder is markedly more precise, the fault was the 7B model
pretending to be a classifier. If it is not, the fault is the task - claims
and retrieved passages are harder to adjudicate than NLI benchmark pairs.
Either answer is useful, which is why this is the first item in the future
work list.

THE ASYMMETRY RULE APPLIES HERE TOO. The model receives the claim as the
hypothesis and the evidence as the premise. There is no slot for the draft
answer or the question, structurally as well as by convention.
"""

from __future__ import annotations

from functools import lru_cache

from src.config import Config
from src.types import Evidence, Passage, Verdict

STAGE = "s09_verifier_nli"

# The label set the cross-encoder reports, lowercased, mapped to our verdicts.
_LABEL_MAP = {
    "entailment": Verdict.ENTAILED,
    "contradiction": Verdict.CONTRADICTED,
    "neutral": Verdict.UNSUPPORTED,
}


@lru_cache(maxsize=2)
def _load(model_name: str):
    """Load once per process - the model is ~400MB and loading is slow."""
    import torch
    from sentence_transformers import CrossEncoder

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return CrossEncoder(model_name, device=device)


def verify_claim_nli(
    claim_text: str,
    evidence: list[Passage],
    cfg: Config,
) -> tuple[Verdict, list[Evidence], str]:
    """
    Verify one claim with the cross-encoder.

    Mirrors verify_claim()'s signature deliberately, minus the LLM client, so
    the two backends are interchangeable and the asymmetry guarantee reads the
    same way: a claim string and evidence passages, and nothing else.

    Each passage is scored separately against the claim and the STRONGEST
    verdict wins, in the order CONTRADICTED > ENTAILED > UNSUPPORTED. Rationale:
    a single passage that contradicts the claim matters even when another is
    merely silent, and a single passage that entails it is enough to support
    it. Averaging the scores across passages would let two silent passages
    drown out one that actually settles the question.
    """
    if not evidence:
        return Verdict.UNSUPPORTED, [], "No evidence was retrieved for this claim."

    model = _load(cfg.retrieval.nli_model)

    # CrossEncoder expects (premise, hypothesis). The EVIDENCE is the premise
    # and the CLAIM is the hypothesis - reversing them asks a different and
    # wrong question ("does this claim imply the passage?").
    pairs = [(p.text, claim_text) for p in evidence]
    scores = model.predict(pairs, show_progress_bar=False)

    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

    best: tuple[Verdict, float, Passage] | None = None
    rank = {Verdict.CONTRADICTED: 3, Verdict.ENTAILED: 2, Verdict.UNSUPPORTED: 1}

    for passage, row in zip(evidence, scores):
        idx = int(row.argmax())
        verdict = _LABEL_MAP.get(id2label.get(idx, ""), Verdict.UNSUPPORTED)
        confidence = float(row[idx])
        if best is None or rank[verdict] > rank[best[0]] or (
            rank[verdict] == rank[best[0]] and confidence > best[1]
        ):
            best = (verdict, confidence, passage)

    assert best is not None
    verdict, confidence, passage = best

    ev_list: list[Evidence] = []
    if verdict is Verdict.ENTAILED or verdict is Verdict.CONTRADICTED:
        # The deciding passage is the citation.
        ev_list.append(Evidence(pid=passage.pid, title=passage.title,
                                span=passage.text[:220], full_text=passage.text))
    else:
        # Nothing supported the claim; show what was examined, matching the
        # LLM backend's behaviour so the UI renders both identically.
        for p in evidence[:2]:
            ev_list.append(Evidence(pid=p.pid, title=p.title, span="",
                                    full_text=p.text))

    return (verdict, ev_list,
            f"NLI cross-encoder: {verdict.value.lower()} (p={confidence:.2f})")
