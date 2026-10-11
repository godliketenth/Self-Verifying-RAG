"""
Re-score EM and token-F1 on a short-answer SPAN, as HotpotQA intends.

    python -m src.rescore_em --run <run_dir>

WHY THIS IS NEEDED, AND WHY IT IS NOT CHEATING
----------------------------------------------
HotpotQA's gold answers are spans: "Perfect Strangers", "yes", "1994" - on
average 2.8 words. Our generator is prompted to write a one-to-three sentence
answer, because the claim decomposer and verifier need prose to work on. The
average generated answer is 16.5 words.

The result is that exact match scores 0.020 while the gold string is actually
PRESENT in 0.408 of answers, and an LLM judge marks 0.429 correct. EM was
measuring output formatting, not correctness.

The honest fix is to score what HotpotQA scores. This reads each already-
logged answer, extracts the shortest answering span with one cheap local
call, and recomputes EM and token-F1 against the gold span. It re-runs no
part of the pipeline: the answers, claims, verdicts and costs are untouched,
and so are faithfulness and the LLM-judge score.

Both sets of numbers are reported. The sentence-level EM is not deleted - it
is a true measurement of a real property of the system, namely that it
answers in prose. The span-level EM is what makes the comparison with
published HotpotQA results meaningful.

The extraction call is tagged is_shadow, because it measures an arm rather
than producing its answer, and so must not be charged to that arm's cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.llm import LLMClient, as_object
from src.metrics import exact_match, normalize_answer, token_f1
from src.stages.base import load_prompt

STAGE = "rescore_short_answer"


def extract_span(question: str, answer: str, client: LLMClient, cfg) -> str:
    """One cheap call: pull the shortest answering span out of a prose answer."""
    if not answer.strip():
        return ""
    prompt = load_prompt("short_answer").format(question=question, answer=answer)
    try:
        raw, _ = client.complete_json(prompt, stage=STAGE,
                                      model=cfg.llm.helper_model,
                                      is_shadow=True, max_tokens=120)
        span = str(as_object(raw, "span").get("span", "")).strip()
    except (ValueError, RuntimeError):
        return ""
    if span.upper() == "NONE":
        return ""
    # The span must come FROM the answer. A model that invents one would be
    # scoring its own guess rather than the system's output.
    if span and normalize_answer(span) not in normalize_answer(answer):
        return ""
    return span


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()

    cfg = load_config(require_api_key=True, provider=args.provider)
    runs = [d for d in sorted(cfg.runs_dir.iterdir())
            if d.is_dir() and not d.name.startswith("ragtruth")]
    run_dir = cfg.runs_dir / args.run if args.run else runs[-1]

    client = LLMClient(cfg)
    print(f"Re-scoring {run_dir.name} on short-answer spans\n")
    print(f"{'arm':5}{'EM sentence':>13}{'EM span':>10}{'F1 sentence':>13}{'F1 span':>10}")
    print("-" * 51)

    for arm_dir in sorted(run_dir.glob("arm*")):
        log = arm_dir / "log.jsonl"
        if not log.exists():
            continue
        rows = [json.loads(l) for l in open(log)]
        ok = [r for r in rows if not r.get("error")]

        em_s = f1_s = em_p = f1_p = 0.0
        spans = {}
        for r in ok:
            gold, pred = r["gold_answer"], r["final_answer"]
            em_s += exact_match(pred, gold)
            f1_s += token_f1(pred, gold)
            span = extract_span(r["question"], pred, client, cfg)
            spans[r["qid"]] = span
            # An empty extraction falls back to the full answer rather than
            # scoring zero, so a failed extraction cannot flatter the result.
            scored = span or pred
            em_p += exact_match(scored, gold)
            f1_p += token_f1(scored, gold)

        n = len(ok)
        print(f"{arm_dir.name.replace('arm',''):5}{em_s/n:13.3f}{em_p/n:10.3f}"
              f"{f1_s/n:13.3f}{f1_p/n:10.3f}")

        summary_path = arm_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["exact_match_sentence"] = em_s / n
        summary["token_f1_sentence"] = f1_s / n
        summary["exact_match_span"] = em_p / n
        summary["token_f1_span"] = f1_p / n
        summary_path.write_text(json.dumps(summary, indent=2))
        (arm_dir / "short_answers.json").write_text(json.dumps(spans, indent=2))

    print(f"\nExtraction calls: {client.n_calls} "
          f"(all tagged shadow, so arm costs are unchanged)")


if __name__ == "__main__":
    main()
