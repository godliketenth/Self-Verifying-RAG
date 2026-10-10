"""
Stage 9 - Claim verifier. The core of the project.

Produces a three-way verdict per claim (RT4CHART-style labels, Yu et al. 2026)
plus the evidence span that decided it.

================================================================================
 THE INFORMATION ASYMMETRY RULE  -  do not weaken this
================================================================================
The verifier sees ONLY the claim text and the retrieved evidence. It never
sees:
  - the draft answer
  - the original question
  - the generator's prompt, reasoning, or chain of thought
  - which passages the generator was given
  - any earlier verdict on the same claim

WHY. MARCH (Li et al. 2026) shows that a model asked to check its own output
reproduces its own mistakes: shown its reasoning, it finds that reasoning
persuasive and confirms the error. This is confirmation bias, and it is the
reason naive "self-check" prompts report high confidence on hallucinations.
Withholding the generator's output breaks the loop - the verifier has nothing
to agree with, so it must actually read the evidence.

HOW IT IS ENFORCED. Three ways, deliberately redundant:
  1. `verify_claim` takes only (claim_text, evidence). There is no parameter
     through which the draft answer COULD be passed.
  2. `assert_asymmetry` checks at runtime that no forbidden text appears in
     the prompt, and raises if it does.
  3. tests/test_asymmetry.py fails the build if either guard is removed.

If you are tempted to pass the question in "just to help the verifier judge
relevance" - that is exactly the leak MARCH warns about. The claim is
self-contained by construction (Stage 7 guarantees it), so the question adds
nothing the verifier legitimately needs.
================================================================================
"""

from __future__ import annotations

from src.config import Config
from src.llm import LLMClient, as_object
from src.stages.base import format_passages, load_prompt
from src.types import Claim, Evidence, Passage, Verdict

STAGE = "s09_verifier"


class AsymmetryViolation(RuntimeError):
    """Raised when generator-side information leaks into a verifier prompt."""


def assert_asymmetry(prompt: str, forbidden: list[str],
                     claim_text: str = "") -> None:
    """
    Runtime guard: none of `forbidden` may appear in the verifier's prompt,
    OUTSIDE the claim itself.

    The "outside the claim" part is essential and was learned the hard way.
    The claim is EXTRACTED from the draft answer, so when the draft is a
    single sentence the claim is nearly identical to it:

        draft: "Family Matters was a spin-off of Perfect Strangers."
        claim: "Family Matters was a spin-off of Perfect Strangers."

    A naive check for "does the draft appear in the prompt?" fires on that,
    even though nothing leaked - the claim is the one piece of generator-side
    text the verifier is SUPPOSED to see. So we remove the claim from the
    prompt first, then look for the forbidden text in what remains.

    This keeps the guard's real purpose intact: if anything passed the draft
    or the question into the prompt IN ADDITION to the claim, that extra copy
    survives the removal and is still caught.

    We compare on normalised whitespace and ignore strings under 25
    characters, because a short fragment can coincide innocently.
    """
    haystack = " ".join(prompt.lower().split())

    # Remove the claim (one occurrence) - it is legitimately present.
    claim_norm = " ".join((claim_text or "").lower().split())
    if claim_norm:
        haystack = haystack.replace(claim_norm, " ", 1)

    for text in forbidden:
        needle = " ".join((text or "").lower().split())
        if len(needle) < 25:        # too short to be a meaningful leak
            continue
        if needle in haystack:
            raise AsymmetryViolation(
                "Verifier prompt contains generator-side text OUTSIDE the "
                "claim, which breaks the information asymmetry this project "
                "depends on.\n"
                f"Leaked text: {text[:120]!r}"
            )


def verify_claim(
    claim_text: str,
    evidence: list[Passage],
    client: LLMClient,
    cfg: Config,
    is_shadow: bool = False,
    _forbidden: list[str] | None = None,
) -> tuple[Verdict, list[Evidence], str]:
    """
    Verify ONE claim against ONE evidence set.

    Note the signature: a plain string and a passage list. The draft answer
    cannot reach this function because there is nowhere to put it.

    `_forbidden` is used only by the runtime guard and by the test - the
    caller passes the draft answer and question here so this function can
    PROVE they are absent from the prompt. They are never added to it.

    Returns (verdict, evidence_list, reasoning).
    """
    if not evidence:
        # No evidence retrieved means we cannot claim support. UNSUPPORTED is
        # the honest verdict; calling an LLM with an empty evidence block
        # would just invite it to answer from parametric knowledge.
        return Verdict.UNSUPPORTED, [], "No evidence was retrieved for this claim."

    prompt = load_prompt(STAGE).format(
        claim=claim_text,
        evidence=format_passages(evidence),
    )

    # Enforce the rule before the prompt can ever reach the API.
    assert_asymmetry(prompt, _forbidden or [], claim_text=claim_text)

    raw, _ = client.complete_json(prompt, stage=STAGE,
                                  model=cfg.llm.helper_model,
                                  is_shadow=is_shadow)
    data = as_object(raw, "verdict")

    raw = str(data.get("verdict", "")).strip().upper()
    try:
        verdict = Verdict(raw)
    except ValueError:
        # An unparseable verdict must not be optimistically read as ENTAILED.
        # UNSUPPORTED is the conservative default: it flags the claim for the
        # user rather than silently vouching for it.
        verdict = Verdict.UNSUPPORTED

    reasoning = str(data.get("reasoning", ""))
    span = str(data.get("evidence_span", "")).strip()
    pid = data.get("evidence_pid")
    pid = str(pid).strip() if pid else ""

    # Models copy the pid back in whatever form they saw it, including the
    # square brackets the prompt renders ("[p00042]" or even "[p1]"). Strip
    # non-pid characters before matching, or evidence linking silently fails
    # and every claim loses its citation in the UI.
    pid = pid.strip().strip("[]").strip()

    ev_list: list[Evidence] = []
    match = next((p for p in evidence if p.pid == pid), None)
    if match is None and pid:
        # Tolerate a shortened or zero-stripped pid ("p1" for "p00001").
        digits = "".join(ch for ch in pid if ch.isdigit())
        if digits:
            match = next(
                (p for p in evidence
                 if "".join(ch for ch in p.pid if ch.isdigit()).lstrip("0")
                 == digits.lstrip("0")),
                None,
            )
    if match is None and span:
        # The model named no pid (or a wrong one) but quoted a span; find the
        # passage that actually contains the quote.
        match = next((p for p in evidence if span[:60] in p.text), None)
    if match is not None:
        ev_list.append(Evidence(pid=match.pid, title=match.title,
                                span=span or match.text[:200],
                                full_text=match.text))
    return verdict, ev_list, reasoning


def verify_claims(
    claims: list[Claim],
    evidence_by_claim: dict[str, list[Passage]],
    client: LLMClient,
    cfg: Config,
    is_shadow: bool = False,
    forbidden: list[str] | None = None,
) -> list[Claim]:
    """
    Verify every claim, filling in its verdict, evidence and reasoning
    in place. Returns the same list for convenience.
    """
    for claim in claims:
        verdict, ev, reasoning = verify_claim(
            claim_text=claim.text,
            evidence=evidence_by_claim.get(claim.text, []),
            client=client, cfg=cfg, is_shadow=is_shadow,
            _forbidden=forbidden,
        )
        claim.verdict = verdict
        claim.evidence = ev
        claim.reasoning = reasoning
    return claims
