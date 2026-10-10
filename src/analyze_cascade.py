"""
Evidence for the open question: verify only the final answer, or intermediate
agent steps too?

    python -m src.analyze_cascade --run <run_dir>

CHARM (Mishra 2026) argues that errors early in a multi-step agent pipeline
CASCADE: a bad retrieval at stage 4-5 produces a bad answer at stage 6, which
produces unsupported claims at stage 9. If that is true here, verifying
intermediate steps would catch errors earlier and cheaper. If it is not - if
unsupported claims appear just as often after GOOD retrieval - then
intermediate verification would spend money watching a stage that was not the
problem.

Rather than argue from intuition, this script measures it. Our per-query logs
record the retrieval state and the claim verdicts for the same question, so we
can directly ask: conditional on retrieval having gone badly, how much more
likely is an unsupported claim?

The retrieval-quality proxy is `kept_passages`: the number of passages that
survived the CRAG relevance filter. Few survivors means the retriever did not
find much that was relevant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config


def load_run(arm_dir: Path) -> list[dict]:
    f = arm_dir / "log.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def analyse(rows: list[dict], arm: str) -> dict:
    """Split questions by retrieval quality and compare claim outcomes."""
    poor, good = [], []
    for r in rows:
        if r.get("error") or not r.get("claims"):
            continue
        # "poor retrieval" = at or below the minimum the re-plan trigger uses,
        # or a re-plan actually fired (the pipeline itself judged it poor).
        n_kept = len(r.get("kept_passages", []))
        (poor if (n_kept <= 2 or r.get("replan_rounds", 0) > 0) else good).append(r)

    def stats(group: list[dict]) -> dict:
        n_claims = sum(len(r["claims"]) for r in group)
        n_bad = sum(1 for r in group for c in r["claims"]
                    if c["verdict"] and c["verdict"] != "ENTAILED")
        n_correct = sum(1 for r in group if r.get("judge_correct"))
        return {
            "n_questions": len(group),
            "n_claims": n_claims,
            "pct_claims_flagged": 100 * n_bad / n_claims if n_claims else 0.0,
            "pct_answers_correct": 100 * n_correct / len(group) if group else 0.0,
        }

    return {"arm": arm, "poor_retrieval": stats(poor), "good_retrieval": stats(good)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    args = ap.parse_args()

    cfg = load_config()
    runs = sorted(d for d in cfg.runs_dir.iterdir()
                  if d.is_dir() and not d.name.startswith("ragtruth"))
    if not runs:
        raise SystemExit("No runs found.")
    run_dir = cfg.runs_dir / args.run if args.run else runs[-1]

    print(f"Cascade analysis for {run_dir.name}\n")
    print("Question: do unsupported claims cluster after POOR retrieval?")
    print("If yes, intermediate-step verification would pay. If no, it would")
    print("spend money watching a stage that was not the problem.\n")

    out = []
    for arm_dir in sorted(run_dir.glob("arm*")):
        arm = arm_dir.name.replace("arm", "")
        rows = load_run(arm_dir)
        if not rows:
            continue
        res = analyse(rows, arm)
        out.append(res)

        p, g = res["poor_retrieval"], res["good_retrieval"]
        print(f"--- ARM {arm} ---")
        print(f"{'':22s}{'poor retrieval':>16s}{'good retrieval':>16s}")
        print(f"{'questions':22s}{p['n_questions']:>16d}{g['n_questions']:>16d}")
        print(f"{'claims':22s}{p['n_claims']:>16d}{g['n_claims']:>16d}")
        print(f"{'% claims flagged':22s}{p['pct_claims_flagged']:>15.1f}%"
              f"{g['pct_claims_flagged']:>15.1f}%")
        print(f"{'% answers correct':22s}{p['pct_answers_correct']:>15.1f}%"
              f"{g['pct_answers_correct']:>15.1f}%")
        if g["pct_claims_flagged"] > 0:
            ratio = p["pct_claims_flagged"] / g["pct_claims_flagged"]
            print(f"\n  claims are {ratio:.2f}x more likely to be flagged after "
                  f"poor retrieval")
        print()

    (run_dir / "cascade_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {run_dir / 'cascade_analysis.json'}")


if __name__ == "__main__":
    main()
