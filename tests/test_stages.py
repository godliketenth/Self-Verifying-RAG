"""
Milestone 3/4 tests: each stage behaves correctly, including its failure
modes. These run on the mock backend, so they are free and deterministic.

The emphasis is on DEFENSIVE behaviour. An LLM stage that works on a good
reply but corrupts the run on a bad one is not finished, and over a
300-question eval bad replies are a certainty, not a hypothetical.
"""

import pytest

from src.config import ARMS, load_config
from src.llm import LLMClient
from src.stages.s02_router import route_query
from src.stages.s03_planner import plan_retrieval
from src.stages.s05_relevance import evaluate_relevance, needs_replan
from src.stages.s07_decomposer import _dedupe, decompose_answer
from src.stages.s10_assemble import assemble_answer, format_verified_answer
from src.types import Claim, Passage, Route, Verdict


@pytest.fixture
def client(cfg):
    return LLMClient(cfg, mock=True)


# --- stage 2: router ------------------------------------------------------

def test_router_returns_a_valid_route(client, cfg):
    route, reason = route_query("Who directed Jaws?", client, cfg)
    assert isinstance(route, Route)


def test_router_falls_back_rather_than_crashing(cfg):
    """An unrecognised label must degrade to SINGLE_HOP, not kill the run."""
    class Bad(LLMClient):
        def complete_json(self, prompt, stage, **kw):
            from src.types import LLMResponse
            return ({"route": "MAYBE", "reason": "nonsense"},
                    LLMResponse("", "mock", 0, 0, 0.0, 0.0, False))
    route, reason = route_query("q", Bad(cfg, mock=True), cfg)
    assert route is Route.SINGLE_HOP
    assert "fallback" in reason


# --- stage 3: planner -----------------------------------------------------

def test_planner_skips_the_llm_for_single_hop(client, cfg):
    """
    A single-hop question needs no decomposition, so we must not pay for a
    call that can only paraphrase it.
    """
    subs = plan_retrieval("Who directed Jaws?", Route.SINGLE_HOP, client, cfg)
    assert subs == ["Who directed Jaws?"]
    assert client.n_calls == 0


def test_planner_respects_the_hop_cap(cfg):
    """The proposal's scope limit is 2-3 hops; more must be truncated."""
    class Many(LLMClient):
        def complete_json(self, prompt, stage, **kw):
            from src.types import LLMResponse
            return ({"subqueries": [f"q{i}" for i in range(10)]},
                    LLMResponse("", "mock", 0, 0, 0.0, 0.0, False))
    subs = plan_retrieval("q", Route.MULTI_HOP, Many(cfg, mock=True), cfg)
    assert len(subs) <= cfg.pipeline.max_subqueries


def test_planner_never_returns_nothing(cfg):
    """An empty plan would silently disable retrieval entirely."""
    class Empty(LLMClient):
        def complete_json(self, prompt, stage, **kw):
            from src.types import LLMResponse
            return ({"subqueries": []},
                    LLMResponse("", "mock", 0, 0, 0.0, 0.0, False))
    subs = plan_retrieval("my question", Route.MULTI_HOP, Empty(cfg, mock=True), cfg)
    assert subs == ["my question"]


def test_replan_prompt_tells_the_planner_what_already_failed(cfg):
    """The re-plan must differ from the first attempt, or the loop is useless."""
    seen = []
    class Spy(LLMClient):
        def complete_json(self, prompt, stage, **kw):
            from src.types import LLMResponse
            seen.append(prompt)
            return ({"subqueries": ["new q"]},
                    LLMResponse("", "mock", 0, 0, 0.0, 0.0, False))
    plan_retrieval("q", Route.MULTI_HOP, Spy(cfg, mock=True), cfg,
                   previous=["old query one"])
    assert "old query one" in seen[0]
    assert "DIFFERENT" in seen[0]


# --- stage 5: relevance ---------------------------------------------------

def _passages(n=3):
    return [Passage(pid=f"p{i:05d}", title=f"T{i}", text=f"text {i}")
            for i in range(n)]


def test_relevance_keeps_high_scorers_and_drops_low(cfg):
    class Scored(LLMClient):
        def complete_json(self, prompt, stage, **kw):
            from src.types import LLMResponse
            return ({"passages": [
                        {"pid": "p00000", "score": 0.9},
                        {"pid": "p00001", "score": 0.1},
                        {"pid": "p00002", "score": 0.7}]},
                    LLMResponse("", "mock", 0, 0, 0.0, 0.0, False))
    kept, scores = evaluate_relevance("q", _passages(3), Scored(cfg, mock=True), cfg)
    assert {p.pid for p in kept} == {"p00000", "p00002"}


def test_unmentioned_passage_is_kept_not_dropped(cfg):
    """
    If the evaluator omits a passage, keeping it is the safe default.
    Dropping on silence would let one malformed reply discard all evidence
    and manufacture a false "unanswerable".
    """
    class Partial(LLMClient):
        def complete_json(self, prompt, stage, **kw):
            from src.types import LLMResponse
            return ({"passages": [{"pid": "p00000", "score": 0.9}]},
                    LLMResponse("", "mock", 0, 0, 0.0, 0.0, False))
    kept, _ = evaluate_relevance("q", _passages(3), Partial(cfg, mock=True), cfg)
    assert len(kept) == 3


def test_needs_replan_triggers_below_the_minimum(cfg):
    assert needs_replan([], cfg) is True
    assert needs_replan(_passages(1), cfg) is True       # min is 2
    assert needs_replan(_passages(5), cfg) is False


# --- stage 7: decomposer --------------------------------------------------

def test_decomposer_returns_claims_with_queries(client, cfg):
    claims = decompose_answer("Some answer with two facts.", client, cfg)
    assert claims
    assert all(c.text and c.verification_query for c in claims)


def test_refusal_produces_no_claims(client, cfg):
    """
    A refusal asserts nothing about the world. Decomposing it would create a
    spurious UNSUPPORTED claim and silently depress the faithfulness of
    whichever arm refuses most often.
    """
    from src.stages.s06_generator import REFUSAL
    assert decompose_answer(REFUSAL, client, cfg) == []
    assert decompose_answer("", client, cfg) == []
    assert client.n_calls == 0


def test_dedupe_removes_near_identical_claims():
    claims = [
        Claim("Paris is the capital of France.", "q"),
        Claim("paris is the capital of france", "q"),     # same, different case
        Claim("Berlin is the capital of Germany.", "q"),  # different
    ]
    kept = _dedupe(claims, 0.95)
    assert len(kept) == 2


def test_decomposer_rejects_a_fabricated_source_sentence(cfg):
    """
    source_sentence must be copied from the answer. A paraphrase would give a
    bogus character offset and corrupt the RAGTruth span-to-claim mapping.
    """
    class Fake(LLMClient):
        def complete_json(self, prompt, stage, **kw):
            from src.types import LLMResponse
            return ({"claims": [{"claim": "A fact.", "query": "q",
                                 "source_sentence": "NOT IN THE ANSWER AT ALL"}]},
                    LLMResponse("", "mock", 0, 0, 0.0, 0.0, False))
    claims = decompose_answer("The real answer text.", Fake(cfg, mock=True), cfg)
    assert claims[0].source_sentence == ""      # rejected, not trusted


# --- stage 10: assembly ---------------------------------------------------

def test_assemble_returns_draft_when_nothing_was_verified():
    assert assemble_answer("the draft", []) == "the draft"


def test_assemble_omits_dropped_claims():
    claims = [Claim("Kept fact.", "q", Verdict.ENTAILED),
              Claim("", "q", Verdict.UNSUPPORTED, was_corrected=True,
                    original_text="Dropped fact.")]
    out = assemble_answer("draft", claims)
    assert "Kept fact." in out
    assert "Dropped fact." not in out


def test_assemble_says_so_when_everything_failed():
    claims = [Claim("", "q", Verdict.UNSUPPORTED, was_corrected=True,
                    original_text="gone")]
    out = assemble_answer("draft", claims)
    assert "failed verification" in out.lower()


def test_badges_appear_in_the_formatted_output():
    claims = [Claim("Good.", "q", Verdict.ENTAILED),
              Claim("Bad.", "q", Verdict.CONTRADICTED)]
    out = format_verified_answer("draft", claims)
    assert "[OK]" in out and "[CONTRADICTED]" in out


# --- regression: failures found in a live local-model run, 2026-10-11 -------

def test_unescaped_quotes_in_evidence_span_are_repaired():
    """
    REGRESSION. A 7B model copies evidence verbatim into a JSON field without
    escaping it, producing structurally complete but invalid JSON:

        {"evidence_span": "...the term "Oldies But Goodies"."}

    This killed 4 of 36 questions in a live run. Prompting alone does not
    fix it reliably at this model size, so the parser repairs it.
    """
    from src.llm import extract_json
    bad = ('{"verdict": "ENTAILED", "evidence_pid": "p1", "evidence_span": '
           '"Art Laboe coined the term "Oldies But Goodies".", '
           '"reasoning": "stated in evidence"}')
    d = extract_json(bad)
    assert d["verdict"] == "ENTAILED"
    assert "Oldies But Goodies" in d["evidence_span"]


def test_bare_list_response_is_coerced_not_crashed():
    """
    REGRESSION. The model returned a bare array where the prompt asked for an
    object wrapping it, and .get() on a list raised AttributeError, killing
    the question.
    """
    from src.llm import as_object
    assert as_object([{"claim": "x"}], "claims") == {"claims": [{"claim": "x"}]}
    assert as_object({"claims": []}, "claims") == {"claims": []}
    assert as_object("nonsense", "claims") == {}


def test_decomposer_survives_a_bare_list_reply(cfg):
    """End to end: the decomposer must not crash on a bare-array reply."""
    class BareList(LLMClient):
        def complete_json(self, prompt, stage, **kw):
            from src.types import LLMResponse
            return ([{"claim": "A fact.", "query": "q",
                      "source_sentence": "A fact."}],
                    LLMResponse("", "mock", 0, 0, 0.0, 0.0, False))
    claims = decompose_answer("A fact.", BareList(cfg, mock=True), cfg)
    assert len(claims) == 1
    assert claims[0].text == "A fact."


def test_planner_survives_a_bare_list_reply(cfg):
    class BareList(LLMClient):
        def complete_json(self, prompt, stage, **kw):
            from src.types import LLMResponse
            return (["sub one", "sub two"],
                    LLMResponse("", "mock", 0, 0, 0.0, 0.0, False))
    subs = plan_retrieval("q", Route.MULTI_HOP, BareList(cfg, mock=True), cfg)
    assert subs == ["sub one", "sub two"]


# --- upgrade 3: warn-only mode (arm D) -------------------------------------

def test_arm_d_keeps_flagged_claims_arm_c_deletes_them(cfg):
    """
    Arms C and D differ ONLY in what happens to a claim that fails
    verification. C deletes it; D keeps it and lets the badge carry the
    warning. With verifier precision measured at 0.247, that difference is
    the point of the experiment, not a cosmetic option.
    """
    from src.config import ARMS
    assert ARMS["C"].drop_failed is True
    assert ARMS["D"].drop_failed is False
    # Everything else about the two arms must be identical, or the comparison
    # would measure more than one change.
    for field in ("router", "planner", "relevance", "verify",
                  "verify_shadow", "correction_rounds"):
        assert getattr(ARMS["C"], field) == getattr(ARMS["D"], field), field


def test_assemble_keeps_a_warned_claim_in_the_answer():
    """A claim kept by warn-only mode must still appear in the answer text."""
    from src.types import Claim, Verdict
    kept = Claim("Possibly wrong but kept.", "q", Verdict.UNSUPPORTED,
                 was_corrected=True, original_text="Possibly wrong but kept.")
    out = assemble_answer("draft", [kept])
    assert "Possibly wrong but kept." in out
    rendered = format_verified_answer("draft", [kept])
    assert "[UNSUPPORTED]" in rendered
