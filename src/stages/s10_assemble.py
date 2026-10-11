"""
Stage 10 - Assemble the verified answer.

Produces the user-facing output: the answer with a verdict badge on every
claim and a pointer to the evidence that justified it.

This stage does no LLM work. It exists as its own file because the proposal
describes it as stage 10, and because "what the user finally sees" is worth
being explicit about rather than hiding at the bottom of the orchestrator.
"""

from __future__ import annotations

from src.types import Claim, Verdict

BADGE = {
    Verdict.ENTAILED: "[OK]",
    Verdict.CONTRADICTED: "[CONTRADICTED]",
    Verdict.UNSUPPORTED: "[UNSUPPORTED]",
}


def assemble_answer(draft: str, claims: list[Claim]) -> str:
    """
    Build the final answer text.

    When nothing was verified (arm A and B, where verification is shadow-only
    or absent) we return the draft unchanged - that IS those arms' answer.

    When claims were verified and corrected, the final answer is composed from
    the surviving claims. Dropped claims are omitted; corrected claims appear
    in their corrected form. If every claim was dropped we say so plainly
    rather than returning an empty string.
    """
    if not claims:
        return draft

    surviving = [c for c in claims if c.text.strip()]
    if not surviving:
        return ("Every claim in the draft answer failed verification against "
                "the retrieved evidence, so no answer can be supported.")

    return " ".join(c.text.strip() for c in surviving)


def format_verified_answer(draft: str, claims: list[Claim]) -> str:
    """A plain-text rendering with badges, for the CLI and the logs."""
    lines = [assemble_answer(draft, claims), ""]
    if not claims:
        return lines[0]
    lines.append("Claim-level verification:")
    for n, c in enumerate(claims, start=1):
        badge = BADGE.get(c.verdict, "[UNVERIFIED]") if c.verdict else "[UNVERIFIED]"
        lines.append(f"  {n}. {badge} {c.text}")
        if c.was_corrected:
            lines.append(f"       corrected from: {c.original_text}")
        if c.evidence and c.evidence[0].span:
            e = c.evidence[0]
            lines.append(f"       evidence [{e.pid}] {e.title}: \"{e.span[:140]}\"")
        elif c.evidence:
            # Empty span means "considered, but did not support the claim".
            titles = ", ".join(f"[{e.pid}] {e.title}" for e in c.evidence)
            lines.append(f"       considered (none supported this): {titles}")
        if c.reasoning:
            lines.append(f"       reason: {c.reasoning}")
    return "\n".join(lines)
