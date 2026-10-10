"""
All the metrics the project reports.

Five families:
  1. ANSWER CORRECTNESS - HotpotQA's official exact match and token F1, plus
     an LLM judge for the cases string matching gets wrong.
  2. FAITHFULNESS - fraction of claims the verifier marked ENTAILED.
  3. CLAIM DETECTION - precision/recall/F1 of flagging hallucinated claims,
     measured against RAGTruth's human span labels.
  4. COST - LLM calls and tokens (primary), USD (secondary).
  5. LATENCY - median and p90 wall clock.

Why tokens are the PRIMARY cost metric and wall-clock the secondary one:
requests route through OpenRouter, which adds queueing and routing variance we
do not control. Token counts are a property of our pipeline; latency is partly
a property of someone else's infrastructure.
"""

from __future__ import annotations

import re
import statistics
import string
from collections import Counter
from typing import Any

from src.config import Config
from src.llm import LLMClient
from src.stages.base import load_prompt
from src.types import Claim, PipelineResult, Verdict


# ===========================================================================
# 1. Answer correctness - HotpotQA's official metric implementation
# ===========================================================================

def normalize_answer(s: str) -> str:
    """
    HotpotQA's official normalisation: lowercase, strip punctuation, strip
    articles, collapse whitespace. Reproduced here so our numbers are
    comparable with published HotpotQA results rather than being our own
    invented variant.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if the normalised strings are identical."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """
    Token-overlap F1, HotpotQA style.

    This is the metric that handles our generator being chatty. The gold
    answer is often one word ("yes", "1994") while our answer is a sentence,
    so EM is harsh and F1 is the fairer reading - though note F1 also
    PENALISES a correct answer for being verbose, which is why we add an LLM
    judge as a third view.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0

    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def judge_answer(
    question: str,
    predicted: str,
    gold: str,
    client: LLMClient,
    cfg: Config,
) -> bool:
    """
    LLM-as-judge correctness.

    Needed because EM and F1 both misjudge common cases: gold "yes" against
    predicted "Yes, both directors were American" scores EM=0 and a mediocre
    F1, yet is plainly correct. The judge call is tagged is_shadow=True - it
    measures the arm, it is not part of answering, so it must not be charged
    to the arm's cost.
    """
    if not predicted.strip():
        return False
    prompt = load_prompt("judge").format(
        question=question, gold=gold, predicted=predicted
    )
    try:
        data, _ = client.complete_json(prompt, stage="judge",
                                       model=cfg.llm.helper_model,
                                       is_shadow=True)
        return bool(data.get("correct", False))
    except ValueError:
        # A judge that cannot answer must not silently score the answer as
        # correct. False is the conservative reading.
        return False


# ===========================================================================
# 3. Claim detection against RAGTruth's human labels
# ===========================================================================

def map_spans_to_claims(
    claims: list[Claim],
    response: str,
    gold_spans: list[dict[str, Any]],
    overlap_threshold: float,
) -> tuple[list[str], int]:
    """
    Convert RAGTruth's character-level span labels into per-claim gold labels.

    A gold span is attributed to a claim when the span overlaps the sentence
    that claim was extracted from, by at least `overlap_threshold` of the
    span's length.

    Returns (gold_labels, n_unmapped) where gold_labels[i] is the verdict
    RAGTruth implies for claims[i] - CONTRADICTED, UNSUPPORTED, or ENTAILED
    for a claim no span touches.

    THIS MAPPING IS LOSSY, and the report says so. Our own decomposer picks
    the source sentences, so decomposition error leaks into the gold labels.
    `n_unmapped` counts gold spans that landed on no claim at all - usually
    because the decomposer dropped that sentence - and is reported alongside
    the precision/recall figures so the reader can judge how much to trust
    them.
    """
    # Locate each claim's source sentence in the response.
    claim_ranges: list[tuple[int, int] | None] = []
    for c in claims:
        src = c.source_sentence or c.text
        pos = response.find(src)
        if pos == -1:
            # Fall back to the first few words, in case of trivial whitespace
            # differences between the claim and the response text.
            head = " ".join(src.split()[:6])
            pos = response.find(head) if head else -1
            if pos == -1:
                claim_ranges.append(None)
                continue
            claim_ranges.append((pos, pos + len(head)))
        else:
            claim_ranges.append((pos, pos + len(src)))

    gold = [Verdict.ENTAILED.value] * len(claims)
    mapped_spans = 0

    for span in gold_spans:
        s_start, s_end = int(span["start"]), int(span["end"])
        s_len = max(1, s_end - s_start)
        hit = False
        for i, rng in enumerate(claim_ranges):
            if rng is None:
                continue
            c_start, c_end = rng
            overlap = max(0, min(s_end, c_end) - max(s_start, c_start))
            if overlap / s_len >= overlap_threshold:
                # CONTRADICTED is the stronger label: if a claim is touched by
                # both kinds of span, the conflict dominates.
                if (gold[i] == Verdict.ENTAILED.value
                        or span["verdict"] == Verdict.CONTRADICTED.value):
                    gold[i] = span["verdict"]
                hit = True
        if hit:
            mapped_spans += 1

    return gold, len(gold_spans) - mapped_spans


def claim_detection_scores(
    predicted: list[str], gold: list[str]
) -> dict[str, float]:
    """
    Precision, recall and F1 for the binary task "is this claim hallucinated?"

    A claim counts as FLAGGED when its verdict is not ENTAILED. We report the
    binary detection task rather than 3-way accuracy because that is the
    decision the system actually makes for the user: show a warning or not.
    Three-way agreement is reported separately as `verdict_accuracy`.
    """
    assert len(predicted) == len(gold)

    tp = sum(1 for p, g in zip(predicted, gold)
             if p != Verdict.ENTAILED.value and g != Verdict.ENTAILED.value)
    fp = sum(1 for p, g in zip(predicted, gold)
             if p != Verdict.ENTAILED.value and g == Verdict.ENTAILED.value)
    fn = sum(1 for p, g in zip(predicted, gold)
             if p == Verdict.ENTAILED.value and g != Verdict.ENTAILED.value)
    tn = sum(1 for p, g in zip(predicted, gold)
             if p == Verdict.ENTAILED.value and g == Verdict.ENTAILED.value)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    exact = sum(1 for p, g in zip(predicted, gold) if p == g)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "verdict_accuracy": exact / len(gold) if gold else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_claims": len(gold),
    }


# ===========================================================================
# Aggregation across a whole run
# ===========================================================================

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(results: list[PipelineResult],
              judged: list[bool] | None = None) -> dict[str, Any]:
    """
    Roll a list of per-question results up into the numbers that go in the
    ablation table.

    Faithfulness averages only over questions that HAD claims. Treating a
    question with no claims as faithfulness 0 would punish an arm for
    correctly refusing to answer, which is the opposite of what we want to
    measure.
    """
    ok = [r for r in results if not r.error]
    if not ok:
        return {"n": 0, "n_errors": len(results)}

    faiths = [f for f in (r.faithfulness() for r in ok) if f is not None]
    latencies = [r.total_latency_ms for r in ok]

    verdicts: Counter[str] = Counter()
    for r in ok:
        for c in r.claims:
            if c.verdict:
                verdicts[c.verdict.value] += 1

    n_claims = sum(len(r.claims) for r in ok)
    n_corrected = sum(1 for r in ok for c in r.claims if c.was_corrected)
    n_dropped = sum(1 for r in ok for c in r.claims
                    if c.was_corrected and not c.text.strip())

    out: dict[str, Any] = {
        "n": len(ok),
        "n_errors": len(results) - len(ok),

        # correctness
        "exact_match": _mean([exact_match(r.final_answer, r.gold_answer) for r in ok]),
        "token_f1": _mean([token_f1(r.final_answer, r.gold_answer) for r in ok]),

        # faithfulness
        "faithfulness": _mean(faiths),
        "n_questions_with_claims": len(faiths),

        # claim-level behaviour
        "claims_per_answer": n_claims / len(ok),
        "pct_claims_flagged": (
            100 * (verdicts["CONTRADICTED"] + verdicts["UNSUPPORTED"]) / n_claims
            if n_claims else 0.0
        ),
        "verdict_counts": dict(verdicts),
        "n_claims_corrected": n_corrected,
        "n_claims_dropped": n_dropped,

        # agentic behaviour
        "replan_rate": _mean([float(r.replan_rounds > 0) for r in ok]),
        "correction_rate": _mean([float(r.correction_rounds > 0) for r in ok]),
        "route_counts": dict(Counter(
            r.route.value for r in ok if r.route
        )),

        # cost: the arm's own cost excludes shadow measurement calls
        "llm_calls_per_query": _mean([float(r.n_llm_calls()) for r in ok]),
        "tokens_per_query": _mean([float(r.total_tokens()) for r in ok]),
        "usd_per_query": _mean([r.total_cost_usd() for r in ok]),
        "usd_total": sum(r.total_cost_usd() for r in ok),
        "usd_total_incl_shadow": sum(
            r.total_cost_usd(include_shadow=True) for r in ok
        ),

        # latency
        "latency_median_ms": statistics.median(latencies) if latencies else 0.0,
        "latency_p90_ms": (
            statistics.quantiles(latencies, n=10)[-1]
            if len(latencies) >= 10 else max(latencies, default=0.0)
        ),
    }

    if judged is not None:
        judged_ok = [j for j, r in zip(judged, results) if not r.error]
        out["judge_correct"] = _mean([float(j) for j in judged_ok])

    return out
