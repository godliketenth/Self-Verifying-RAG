"""
Evaluate the VERIFIER against RAGTruth's human span labels.

    python -m src.run_ragtruth
    python -m src.run_ragtruth --n 10 --mock

This is a different experiment from run_eval. run_eval asks "does the whole
pipeline answer questions well?". This asks the narrower and more important
question: "when the verifier says a claim is unsupported, is it right?"

HOW IT WORKS
------------
RAGTruth gives us real RAG outputs with human-marked hallucination spans. For
each response we:
  1. decompose it into claims with our Stage 7 decomposer,
  2. verify each claim against the response's OWN retrieved passages, using
     the Stage 9 verifier under the same asymmetry rule,
  3. convert the human spans into per-claim gold labels,
  4. score precision, recall and F1 for flagging a hallucinated claim.

TWO HONEST CAVEATS, both reported in the output:
  - The sample is ENRICHED (~50% hallucinated vs ~18% naturally), so
    precision here is optimistic relative to the wild.
  - The span-to-claim mapping is lossy, because our decomposer chooses the
    sentences the mapping keys on. `unmapped_spans` counts the gold spans that
    landed on no claim, which is the honest measure of that loss.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from src.config import load_config
from src.llm import LLMClient
from src.metrics import claim_detection_scores, map_spans_to_claims
from src.stages.s07_decomposer import decompose_answer
from src.stages.s09_verifier import verify_claims
from src.stages.s09_verifier_nli import verify_claim_nli
from src.types import Passage, Verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="0 = all 50")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--provider", default=None,
                    help="override config.yaml: ollama | gemini | openrouter")
    ap.add_argument("--verifier", default="llm", choices=["llm", "nli"],
                    help="which Stage 9 backend to score: llm judge or NLI "
                         "cross-encoder (default: llm)")
    args = ap.parse_args()

    cfg = load_config(require_api_key=not args.mock, provider=args.provider)
    path = cfg.data_dir / "ragtruth_qa.json"
    if not path.exists():
        raise SystemExit("Run `python -m src.data.build_ragtruth` first.")

    rows = json.loads(path.read_text())
    if args.n:
        rows = rows[: args.n]

    client = LLMClient(cfg, use_cache=not args.no_cache, mock=args.mock)

    all_pred: list[str] = []
    all_gold: list[str] = []
    unmapped_total = 0
    n_claims_total = 0
    per_row = []

    print(f"Verifying {len(rows)} RAGTruth QA responses "
          f"({'MOCK' if args.mock else 'LIVE'}) ...")

    for i, row in enumerate(rows, start=1):
        response = row["response"]

        # 1. decompose the response into claims
        claims = decompose_answer(response, client, cfg,
                                  question=row["question"])
        if not claims:
            continue

        # 2. verify each claim against THIS response's own passages.
        #    We use the given passages rather than our Wikipedia index,
        #    because the question here is whether the verifier can judge a
        #    claim against the evidence the original system actually had.
        evidence = [Passage(pid=p["pid"], title=p["title"], text=p["text"])
                    for p in row["passages"]]
        if args.verifier == "nli":
            # The cross-encoder takes the claim and the evidence and nothing
            # else, so the asymmetry rule holds structurally.
            for c in claims:
                v, ev, reason = verify_claim_nli(c.text, evidence, cfg)
                c.verdict, c.evidence, c.reasoning = v, ev, reason
        else:
            verify_claims(
                claims, {c.text: evidence for c in claims}, client, cfg,
                # The asymmetry rule still applies: the verifier must not see
                # the full response it came from, nor the original question.
                forbidden=[response, row["question"]],
            )

        # 3. human spans -> per-claim gold labels
        gold, unmapped = map_spans_to_claims(
            claims, response, row["gold_spans"],
            cfg.data.span_overlap_threshold,
        )
        unmapped_total += unmapped
        n_claims_total += len(claims)

        pred = [c.verdict.value if c.verdict else Verdict.UNSUPPORTED.value
                for c in claims]
        all_pred.extend(pred)
        all_gold.extend(gold)

        per_row.append({
            "rid": row["rid"],
            "has_hallucination": row["has_hallucination"],
            "n_claims": len(claims),
            "n_gold_spans": len(row["gold_spans"]),
            "unmapped_spans": unmapped,
            "claims": [
                {"text": c.text, "predicted": p, "gold": g,
                 "reasoning": c.reasoning}
                for c, p, g in zip(claims, pred, gold)
            ],
        })

        if i % 10 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] {n_claims_total} claims verified")

    # 4. score
    scores = claim_detection_scores(all_pred, all_gold)
    scores["unmapped_spans"] = unmapped_total
    scores["n_responses"] = len(per_row)
    scores["span_mapping_loss_pct"] = (
        100 * unmapped_total / max(1, sum(r["n_gold_spans"] for r in per_row))
    )

    scores["verifier_backend"] = args.verifier
    out_dir = cfg.runs_dir / (f"ragtruth_{args.verifier}_"
                              f"{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    (out_dir / "per_response.json").write_text(json.dumps(per_row, indent=2))

    print(f"\n{'=' * 60}")
    print(f"VERIFIER vs RAGTruth HUMAN LABELS  (backend: {args.verifier})")
    print(f"{'=' * 60}")
    print(f"  responses            : {scores['n_responses']}")
    print(f"  claims               : {scores['n_claims']}")
    print(f"  precision            : {scores['precision']:.3f}")
    print(f"  recall               : {scores['recall']:.3f}")
    print(f"  F1                   : {scores['f1']:.3f}")
    print(f"  3-way verdict acc.   : {scores['verdict_accuracy']:.3f}")
    print(f"  confusion            : TP={scores['tp']} FP={scores['fp']} "
          f"FN={scores['fn']} TN={scores['tn']}")
    print(f"\n  CAVEATS")
    print(f"  enriched base rate   : ~50% hallucinated (natural rate ~18%),")
    print(f"                         so precision here is optimistic")
    print(f"  span mapping loss    : {scores['unmapped_spans']} gold spans "
          f"({scores['span_mapping_loss_pct']:.0f}%) mapped to no claim")
    print(f"\n  Wrote {out_dir}")


if __name__ == "__main__":
    main()
