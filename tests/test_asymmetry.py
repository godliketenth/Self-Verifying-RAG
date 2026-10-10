"""
THE VIVA-CRITICAL TEST.

This file is the executable proof of the design claim that distinguishes this
project: that the claim verifier cannot inherit the generator's errors,
because it never sees the generator's output.

If an examiner asks "how do you know your verifier isn't just agreeing with
itself?", the answer is this test. If it ever goes red, the project's central
contribution is broken - not merely untested.
"""

import inspect

import pytest

from src.config import load_config
from src.llm import LLMClient
from src.stages import s09_verifier
from src.stages.s09_verifier import (
    AsymmetryViolation,
    assert_asymmetry,
    verify_claim,
)
from src.types import Passage, Verdict

cfg_real = load_config()

DRAFT = ("Scott Derrickson is an American film director born in 1966 who "
         "directed the horror film Sinister, and Ed Wood was also American.")
QUESTION = "Were Scott Derrickson and Ed Wood of the same nationality?"


def test_verify_claim_signature_cannot_accept_the_draft_answer():
    """
    Structural guarantee: there is no parameter through which the draft
    answer could be passed into the verifier prompt.

    `_forbidden` exists only so the function can PROVE the draft is absent;
    it is passed to the guard, never into the prompt. The test below
    (test_forbidden_text_is_not_sent_to_the_model) verifies that.
    """
    params = set(inspect.signature(verify_claim).parameters)
    assert params == {
        "claim_text", "evidence", "client", "cfg", "is_shadow", "_forbidden"
    }, f"verify_claim signature changed: {sorted(params)}"

    # No parameter name may suggest generator-side context.
    for banned in ("draft", "answer", "question", "reasoning", "prompt",
                   "generation", "context"):
        assert not any(banned in p for p in params - {"_forbidden"}), (
            f"verify_claim gained a {banned!r}-like parameter - that is the "
            "leak MARCH (Li et al. 2026) warns about"
        )


def test_verifier_prompt_template_has_no_slot_for_the_draft():
    """
    The prompt template itself must only have {claim} and {evidence} slots.
    A {question} or {answer} placeholder would mean a leak was intended.
    """
    from src.stages.base import load_prompt
    template = load_prompt("s09_verifier")
    import re
    slots = set(re.findall(r"\{(\w+)\}", template))
    assert slots == {"claim", "evidence"}, (
        f"verifier prompt has unexpected slots: {slots}"
    )


def test_guard_raises_when_draft_text_is_present():
    """The runtime guard must actually fire on a real leak."""
    leaky = f"Statement: X.\n\nEvidence: [p1] foo.\n\nFor context: {DRAFT}"
    with pytest.raises(AsymmetryViolation):
        assert_asymmetry(leaky, [DRAFT, QUESTION])


def test_guard_allows_a_clean_prompt():
    """And must not fire on a legitimate prompt."""
    clean = ("Statement: Scott Derrickson is American.\n\n"
             "Evidence: [p1] Scott Derrickson\nScott Derrickson is an "
             "American director.")
    assert_asymmetry(clean, [DRAFT, QUESTION])   # must not raise


def test_guard_ignores_short_incidental_overlaps():
    """
    A few shared words between claim and draft are unavoidable - the claim is
    EXTRACTED from the draft. The guard must only catch substantial leaks, or
    it would be useless in practice.
    """
    assert_asymmetry("Statement: Ed Wood was American.", ["Ed Wood"])


def test_forbidden_text_is_not_sent_to_the_model(cfg):
    """
    End-to-end proof: run a real verification and capture the exact prompt
    that reached the LLM client. Neither the draft nor the question may
    appear in it.
    """
    captured: list[str] = []

    class Spy(LLMClient):
        # *args/**kw because complete() is called positionally internally.
        def complete(self, prompt, *args, **kw):      # type: ignore[override]
            captured.append(prompt)
            return super().complete(prompt, *args, **kw)

    client = Spy(cfg, mock=True)
    evidence = [Passage(pid="p00001", title="Scott Derrickson",
                        text="Scott Derrickson is an American film director.")]

    verify_claim(
        claim_text="Scott Derrickson is American.",
        evidence=evidence,
        client=client, cfg=cfg,
        _forbidden=[DRAFT, QUESTION],
    )

    assert captured, "no prompt was sent"
    sent = " ".join(" ".join(p.split()) for p in captured).lower()
    assert DRAFT.lower() not in sent, "the draft answer reached the verifier"
    assert QUESTION.lower() not in sent, "the question reached the verifier"
    # And the things it legitimately needs ARE there.
    assert "scott derrickson is american" in sent
    assert "american film director" in sent


def test_pipeline_passes_forbidden_text_to_the_guard(cfg):
    """
    The guard is only useful if the pipeline actually arms it. Verify that
    src/pipeline.py passes the draft and question as `forbidden`, so a future
    edit that drops that argument is caught here.
    """
    src = inspect.getsource(__import__("src.pipeline", fromlist=["run"]))
    assert "forbidden=[draft, question]" in src, (
        "pipeline.py no longer arms the asymmetry guard with the draft and "
        "question - the runtime check would silently pass on a real leak"
    )


def test_empty_evidence_yields_unsupported_without_an_llm_call(cfg):
    """
    With no evidence, the honest verdict is UNSUPPORTED. Calling the model
    with an empty evidence block would invite it to answer from its own
    parametric knowledge - exactly what the verifier must not do.
    """
    client = LLMClient(cfg, mock=True)
    verdict, ev, reason = verify_claim("Anything at all.", [], client, cfg)
    assert verdict is Verdict.UNSUPPORTED
    assert ev == []
    assert client.n_calls == 0, "an LLM was called with no evidence"


def test_unparseable_verdict_defaults_to_unsupported_not_entailed(cfg):
    """
    A malformed reply must never be read optimistically. Defaulting to
    ENTAILED would silently vouch for unverified claims and inflate
    faithfulness.
    """
    class Garbage(LLMClient):
        def complete_json(self, prompt, stage, **kw):   # type: ignore[override]
            from src.types import LLMResponse
            return ({"verdict": "PROBABLY_FINE"},
                    LLMResponse("", "mock", 0, 0, 0.0, 0.0, False))

    client = Garbage(cfg, mock=True)
    evidence = [Passage(pid="p1", title="t", text="some evidence text")]
    verdict, _, _ = verify_claim("A claim.", evidence, client, cfg)
    assert verdict is Verdict.UNSUPPORTED


# --- regression: found by a live run, 2026-10-11 ---------------------------

def test_single_sentence_draft_does_not_false_positive():
    """
    REGRESSION. When the draft is one sentence, the claim extracted from it is
    nearly identical to the draft. The claim is the one piece of
    generator-side text the verifier is SUPPOSED to see, so this must not
    raise - the original guard did, and it killed real questions mid-run.
    """
    draft = "Family Matters was a spin-off of Perfect Strangers."
    claim = "Family Matters was a spin-off of Perfect Strangers."
    prompt = (f"Statement:\n{claim}\n\nEvidence:\n[p1] Family Matters\n"
              "Family Matters is an American sitcom.")
    assert_asymmetry(prompt, [draft, "some question text here that is long"],
                     claim_text=claim)      # must not raise


def test_leak_still_caught_when_draft_appears_beyond_the_claim():
    """
    The other half of the regression: removing the claim must not create a
    hole. A draft passed in ADDITION to the claim is still a real leak.
    """
    draft = ("Family Matters was a spin-off of Perfect Strangers, and it ran "
             "for nine seasons on ABC before moving to CBS.")
    claim = "Family Matters was a spin-off of Perfect Strangers."
    prompt = (f"Statement:\n{claim}\n\nFor context the model wrote: {draft}\n\n"
              "Evidence:\n[p1] Family Matters\nAn American sitcom.")
    with pytest.raises(AsymmetryViolation):
        assert_asymmetry(prompt, [draft], claim_text=claim)


def test_question_leak_still_caught_with_claim_removal():
    """A leaked question must still be caught after the claim is removed."""
    claim = "Silphium is a genus of flowering plants."
    prompt = (f"Statement:\n{claim}\n\nThe user asked: {QUESTION}\n\n"
              "Evidence:\n[p1] Silphium\nA genus of plants.")
    with pytest.raises(AsymmetryViolation):
        assert_asymmetry(prompt, [QUESTION], claim_text=claim)


def test_evidence_pid_matching_tolerates_bracketed_and_short_ids(cfg):
    """
    Models echo the pid in whatever form the prompt showed it - '[p00001]',
    '[p1]', 'p1'. If matching were exact-only, evidence linking would fail
    silently and every claim would lose its citation in the UI.
    """
    from src.types import LLMResponse

    for returned in ("[p00001]", "p00001", "[p1]", "p1"):
        class Fixed(LLMClient):
            def complete_json(self, prompt, stage, **kw):
                return ({"verdict": "ENTAILED", "evidence_pid": returned,
                         "evidence_span": "Scott Derrickson is American.",
                         "reasoning": "ok"},
                        LLMResponse("", "mock", 0, 0, 0.0, 0.0, False))

        ev = [Passage(pid="p00001", title="Scott Derrickson",
                      text="Scott Derrickson is American.")]
        verdict, evidence, _ = verify_claim("Scott Derrickson is American.",
                                            ev, Fixed(cfg, mock=True), cfg)
        assert verdict is Verdict.ENTAILED
        assert evidence, f"evidence lost when model returned pid {returned!r}"
        assert evidence[0].pid == "p00001"
