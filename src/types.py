"""
Shared data types for the whole pipeline.

Every stage takes one of these and returns another. Having them all in one
small file means you can read this single file and know what data flows
through the system - which is exactly what you want when explaining it.

We use @dataclass (standard library) rather than pydantic because these are
plain containers, not validators, and fewer dependencies is fewer things to
explain.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums: the fixed vocabularies of the system
# ---------------------------------------------------------------------------

class Route(str, Enum):
    """
    Output of Stage 2, the complexity router (Adaptive-RAG, Jeong et al. 2024).

    The idea: not every question needs the same amount of retrieval. Spending
    three retrieval hops on "What is 2+2?" is waste; spending one on a
    two-entity bridge question is not enough.
    """
    NO_RETRIEVAL = "NO_RETRIEVAL"   # answerable from parametric knowledge alone
    SINGLE_HOP = "SINGLE_HOP"       # one retrieval pass is enough
    MULTI_HOP = "MULTI_HOP"         # needs decomposition into sub-queries


class Verdict(str, Enum):
    """
    Output of Stage 9, the claim verifier. Three-way labels follow RT4CHART
    (Yu et al. 2026) rather than a binary supported/unsupported split.

    The distinction between CONTRADICTED and UNSUPPORTED matters: the first
    means the evidence actively disagrees with the claim (a factual error),
    the second means the evidence simply does not cover it (a grounding gap).
    They have different causes and different fixes.
    """
    ENTAILED = "ENTAILED"            # evidence supports the claim
    CONTRADICTED = "CONTRADICTED"     # evidence contradicts the claim
    UNSUPPORTED = "UNSUPPORTED"       # evidence neither supports nor refutes

    @property
    def is_failure(self) -> bool:
        """A claim that needs correcting. Drives the Stage 9 -> 8 loop."""
        return self is not Verdict.ENTAILED


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@dataclass
class Passage:
    """One Wikipedia paragraph in the search corpus."""
    pid: str                  # stable id, e.g. "p00042"
    title: str                # Wikipedia article title
    text: str                 # the paragraph itself
    score: float = 0.0        # fusion score; meaningful only after retrieval
    dense_rank: int | None = None   # rank from the embedding retriever
    sparse_rank: int | None = None  # rank from BM25

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    """
    A passage cited as the reason for a verdict, plus the exact sentence the
    verifier pointed at. The span is what lets the UI show the user *why*.
    """
    pid: str
    title: str
    span: str                 # the supporting/contradicting sentence
    full_text: str = ""       # the whole passage, for the UI's expander

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """
    One atomic, self-contained factual statement extracted from the draft
    answer (FActScore, Min et al. 2023), together with the query used to find
    evidence for it (JointCQ, Xu et al. 2025 - generate claim and query
    together rather than in two separate passes).

    "Atomic" means it asserts exactly one fact, and "self-contained" means it
    has no dangling pronouns - "He was born in 1946" is neither, whereas
    "Donald Trump was born in 1946" is both.
    """
    text: str
    verification_query: str
    verdict: Verdict | None = None
    evidence: list[Evidence] = field(default_factory=list)
    reasoning: str = ""                 # verifier's short justification
    source_sentence: str = ""           # sentence of the draft it came from
    was_corrected: bool = False         # rewritten by the correction loop?
    original_text: str = ""             # pre-correction text, if corrected

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value if self.verdict else None
        return d


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------

@dataclass
class StageLog:
    """
    One record per LLM call, appended as the pipeline runs. This is how we
    answer "what did verification actually cost?" - the central question of
    the ablation. Note that token counts are the PRIMARY cost metric and
    wall-clock latency is secondary, because requests route through
    OpenRouter, which adds variance we do not control.
    """
    stage: str                 # e.g. "s09_verifier"
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cache_hit: bool = False
    # True for calls made only to MEASURE an arm, not to build its answer
    # (shadow-mode verification on arms A and B). Keeping these separate is
    # what stops arm A's reported cost from being inflated by our metrics.
    is_shadow: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LLMResponse:
    """What src/llm.py hands back from a single chat completion."""
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float
    cache_hit: bool
    parse_failures: int = 0    # JSON reparse attempts spent on this call


# ---------------------------------------------------------------------------
# The pipeline's final output
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Everything that happened for one question. Serialised as one line of
    runs/<run>/log.jsonl, so the whole experiment is reconstructible from
    disk without re-calling any API.
    """
    question: str
    arm: str                              # "A", "B", or "C"
    gold_answer: str = ""

    # agentic layer (stages 2-5)
    route: Route | None = None
    subqueries: list[str] = field(default_factory=list)
    retrieved: list[Passage] = field(default_factory=list)
    kept_passages: list[Passage] = field(default_factory=list)
    replan_rounds: int = 0

    # generation (stage 6)
    draft_answer: str = ""

    # verification (stages 7-10)
    claims: list[Claim] = field(default_factory=list)
    final_answer: str = ""
    correction_rounds: int = 0

    # accounting
    stage_logs: list[StageLog] = field(default_factory=list)
    total_latency_ms: float = 0.0
    error: str = ""                       # set if the query failed outright

    # ---- derived accounting helpers --------------------------------------
    # `shadow` calls are excluded by default: they exist to measure an arm,
    # not to produce its answer, so counting them as the arm's cost would
    # misreport what that arm actually costs to run.

    def _logs(self, include_shadow: bool) -> list[StageLog]:
        return [l for l in self.stage_logs if include_shadow or not l.is_shadow]

    def total_tokens(self, include_shadow: bool = False) -> int:
        return sum(l.tokens_in + l.tokens_out for l in self._logs(include_shadow))

    def total_cost_usd(self, include_shadow: bool = False) -> float:
        return sum(l.cost_usd for l in self._logs(include_shadow))

    def n_llm_calls(self, include_shadow: bool = False) -> int:
        return len(self._logs(include_shadow))

    def latency_ms(self, include_shadow: bool = False) -> float:
        """
        Wall-clock latency for this question, excluding measurement-only work.

        total_latency_ms is the raw wall clock, which for arms A and B
        INCLUDES shadow verification - calls made purely so we can report
        their faithfulness, and which those arms would never make in
        deployment. Counting that time made arm B look slower than arm C even
        though C does strictly more work.

        Cost already excludes shadow calls, so latency must too, or the
        cost/latency table contradicts itself. We subtract the shadow calls'
        own latency and keep everything else, so local retrieval time (FAISS
        and BM25, which cost no API call but real milliseconds) still counts.
        """
        if include_shadow:
            return self.total_latency_ms
        shadow_ms = sum(l.latency_ms for l in self.stage_logs if l.is_shadow)
        return max(0.0, self.total_latency_ms - shadow_ms)

    def faithfulness(self) -> float | None:
        """
        Fraction of claims the verifier marked ENTAILED.

        Returns None (not 0.0) when there are no claims - "we did not measure"
        and "we measured zero faithfulness" are different facts, and averaging
        a missing value as 0 would silently understate every arm.
        """
        verdicted = [c for c in self.claims if c.verdict is not None]
        if not verdicted:
            return None
        n_ok = sum(1 for c in verdicted if c.verdict is Verdict.ENTAILED)
        return n_ok / len(verdicted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "arm": self.arm,
            "gold_answer": self.gold_answer,
            "route": self.route.value if self.route else None,
            "subqueries": self.subqueries,
            "retrieved": [p.to_dict() for p in self.retrieved],
            "kept_passages": [p.to_dict() for p in self.kept_passages],
            "replan_rounds": self.replan_rounds,
            "draft_answer": self.draft_answer,
            "claims": [c.to_dict() for c in self.claims],
            "final_answer": self.final_answer,
            "correction_rounds": self.correction_rounds,
            "stage_logs": [l.to_dict() for l in self.stage_logs],
            "total_latency_ms": self.total_latency_ms,
            "latency_ms": self.latency_ms(),
            "total_tokens": self.total_tokens(),
            "total_cost_usd": self.total_cost_usd(),
            "n_llm_calls": self.n_llm_calls(),
            "faithfulness": self.faithfulness(),
            "error": self.error,
        }
