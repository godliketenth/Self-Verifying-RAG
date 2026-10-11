"""
Recompute a run's summary.json from its log.jsonl, without re-running anything.

    python -m src.rebuild_summaries                 # newest run
    python -m src.rebuild_summaries --run <name>

Why this exists: the per-query logs record everything (claims, verdicts,
per-call tokens, latency and shadow flags), so any metric change can be
applied to past runs for free. When the latency metric was corrected to
exclude shadow-verification time, this rebuilt every affected summary in a
second rather than costing another hour of inference.

It is also a reproducibility guarantee: the tables in the report are derived
from the logs on disk, not from numbers that were computed once and trusted
thereafter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.metrics import aggregate
from src.types import Claim, Evidence, Passage, PipelineResult, Route, StageLog, Verdict


def result_from_dict(d: dict) -> PipelineResult:
    """Reconstruct a PipelineResult from one line of log.jsonl."""
    r = PipelineResult(
        question=d["question"], arm=d["arm"], gold_answer=d.get("gold_answer", ""),
        draft_answer=d.get("draft_answer", ""),
        final_answer=d.get("final_answer", ""),
        subqueries=d.get("subqueries", []),
        replan_rounds=d.get("replan_rounds", 0),
        correction_rounds=d.get("correction_rounds", 0),
        total_latency_ms=d.get("total_latency_ms", 0.0),
        error=d.get("error", ""),
    )
    if d.get("route"):
        r.route = Route(d["route"])
    r.retrieved = [Passage(**{k: p[k] for k in
                             ("pid", "title", "text", "score",
                              "dense_rank", "sparse_rank") if k in p})
                   for p in d.get("retrieved", [])]
    r.kept_passages = [Passage(**{k: p[k] for k in
                                  ("pid", "title", "text", "score",
                                   "dense_rank", "sparse_rank") if k in p})
                       for p in d.get("kept_passages", [])]
    for c in d.get("claims", []):
        claim = Claim(
            text=c.get("text", ""),
            verification_query=c.get("verification_query", ""),
            reasoning=c.get("reasoning", ""),
            source_sentence=c.get("source_sentence", ""),
            was_corrected=c.get("was_corrected", False),
            original_text=c.get("original_text", ""),
        )
        if c.get("verdict"):
            claim.verdict = Verdict(c["verdict"])
        claim.evidence = [Evidence(**{k: e[k] for k in
                                      ("pid", "title", "span", "full_text")
                                      if k in e})
                          for e in c.get("evidence", [])]
        r.claims.append(claim)
    r.stage_logs = [StageLog(**{k: l[k] for k in
                                ("stage", "model", "tokens_in", "tokens_out",
                                 "latency_ms", "cost_usd", "cache_hit",
                                 "is_shadow", "note") if k in l})
                    for l in d.get("stage_logs", [])]
    return r


def rebuild(run_dir: Path) -> None:
    for arm_dir in sorted(run_dir.glob("arm*")):
        log = arm_dir / "log.jsonl"
        if not log.exists():
            continue
        rows = [json.loads(l) for l in open(log)]
        results = [result_from_dict(d) for d in rows]
        judged = [bool(d.get("judge_correct")) for d in rows]

        summary_path = arm_dir / "summary.json"
        old = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        new = aggregate(results, judged=judged)
        # Preserve fields that describe the RUN rather than the metrics.
        for k in ("arm", "arm_description", "wall_clock_s", "llm_client",
                  "aborted"):
            if k in old:
                new[k] = old[k]
        summary_path.write_text(json.dumps(new, indent=2))

        print(f"  {arm_dir.name}: n={new['n']} "
              f"faithfulness={new['faithfulness']:.3f} "
              f"latency_median={new['latency_median_ms']/1000:.1f}s "
              f"(was {old.get('latency_median_ms', 0)/1000:.1f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="run dir name; default = all runs")
    args = ap.parse_args()
    cfg = load_config()

    runs = ([cfg.runs_dir / args.run] if args.run else
            [d for d in sorted(cfg.runs_dir.iterdir())
             if d.is_dir() and not d.name.startswith("ragtruth")])
    for d in runs:
        if not d.exists():
            raise SystemExit(f"Run not found: {d}")
        print(f"{d.name}:")
        rebuild(d)


if __name__ == "__main__":
    main()
