"""
Run the ablation. This is the main entry point for the experiment.

    python -m src.run_eval --arm C --n 50
    python -m src.run_eval --arm all --n 100
    python -m src.run_eval --arm all --n 5 --mock     # free plumbing check

Every run writes runs/<timestamp>_arm<X>/
    log.jsonl     one JSON object per question: the full stage trace
    summary.json  the aggregated metrics for the arm
    config.json   the settings that produced it

Because every question's full trace is on disk, the tables and plots can be
rebuilt later with `python -m src.report` without re-calling any API.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import ARMS, Config, get_arm, load_config
from src.llm import LLMClient
from src.metrics import aggregate, judge_answer
from src.pipeline import run
from src.retrieval.hybrid import HybridRetriever
from src.types import PipelineResult


def load_questions(cfg: Config, n: int, seed: int) -> list[dict]:
    """
    Load the evaluation questions.

    The order is shuffled with a fixed seed and then truncated, so that
    `--n 5` is a random-but-reproducible sample rather than always the first
    five questions, which would over-represent whatever the sort order put
    first. Critically, EVERY ARM GETS THE SAME QUESTIONS IN THE SAME ORDER.
    """
    path = cfg.data_dir / "eval_questions.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run `python -m src.data.build_hotpot` first."
        )
    questions = json.loads(path.read_text())
    random.Random(seed).shuffle(questions)
    return questions[:n] if n > 0 else questions


def run_arm(
    arm_name: str,
    questions: list[dict],
    cfg: Config,
    retriever: HybridRetriever,
    mock: bool,
    use_cache: bool,
    judge: bool,
    out_root: Path,
) -> dict:
    """Run one arm over all questions and write its logs. Returns the summary."""
    arm = get_arm(arm_name)
    client = LLMClient(cfg, use_cache=use_cache, mock=mock)

    out_dir = out_root / f"arm{arm.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    print(f"\n{'=' * 72}")
    print(f"ARM {arm.name}: {arm.description}")
    print(f"{'=' * 72}")

    results: list[PipelineResult] = []
    judged: list[bool] = []
    t_start = time.perf_counter()

    with open(log_path, "w") as logf:
        for i, q in enumerate(questions, start=1):
            try:
                result = run(
                    question=q["question"], arm=arm, client=client,
                    retriever=retriever, cfg=cfg, gold_answer=q["answer"],
                )
            except Exception as e:                   # noqa: BLE001
                # One bad question must not destroy a run that costs money.
                # We record the failure and carry on; n_errors is reported.
                result = PipelineResult(
                    question=q["question"], arm=arm.name,
                    gold_answer=q["answer"], error=f"{type(e).__name__}: {e}",
                )
                client.take_logs()                   # discard partial logs
                print(f"  [{i}/{len(questions)}] ERROR: {e}")

            # LLM-as-judge correctness (tagged shadow inside judge_answer)
            is_correct = False
            if judge and not result.error:
                is_correct = judge_answer(q["question"], result.final_answer,
                                          q["answer"], client, cfg)
                result.stage_logs.extend(client.take_logs())
            judged.append(is_correct)

            results.append(result)
            record = result.to_dict()
            record["qid"] = q["qid"]
            record["judge_correct"] = is_correct
            logf.write(json.dumps(record) + "\n")
            logf.flush()      # survive a crash with partial results intact

            if i % 10 == 0 or i == len(questions):
                spent = sum(r.total_cost_usd(include_shadow=True) for r in results)
                print(f"  [{i}/{len(questions)}] "
                      f"spent ${spent:.3f} | "
                      f"cache {client.n_cache_hits}/{client.n_calls}")

    elapsed = time.perf_counter() - t_start
    summary = aggregate(results, judged=judged)
    summary["arm"] = arm.name
    summary["arm_description"] = arm.description
    summary["wall_clock_s"] = elapsed
    summary["llm_client"] = client.summary()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  ARM {arm.name} done in {elapsed:.0f}s")
    print(f"    exact match      : {summary['exact_match']:.3f}")
    print(f"    token F1         : {summary['token_f1']:.3f}")
    if "judge_correct" in summary:
        print(f"    judge correct    : {summary['judge_correct']:.3f}")
    print(f"    faithfulness     : {summary['faithfulness']:.3f} "
          f"(over {summary['n_questions_with_claims']} questions with claims)")
    print(f"    claims/answer    : {summary['claims_per_answer']:.1f}")
    print(f"    % claims flagged : {summary['pct_claims_flagged']:.1f}%")
    print(f"    llm calls/query  : {summary['llm_calls_per_query']:.1f}")
    print(f"    tokens/query     : {summary['tokens_per_query']:.0f}")
    print(f"    USD/query        : ${summary['usd_per_query']:.5f}")
    print(f"    latency median   : {summary['latency_median_ms'] / 1000:.1f}s")
    if summary["n_errors"]:
        print(f"    ERRORS           : {summary['n_errors']}")
    if client.n_parse_failures:
        print(f"    JSON parse fails : {client.n_parse_failures}")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the Self-Verifying Agentic RAG ablation."
    )
    ap.add_argument("--arm", default="all",
                    help="A, B, C, or all (default: all)")
    ap.add_argument("--n", type=int, default=20,
                    help="number of questions; 0 means all (default: 20)")
    ap.add_argument("--mock", action="store_true",
                    help="use the mock LLM: no API key, no cost")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the disk cache and force fresh LLM calls")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip LLM-as-judge correctness scoring")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the config seed for question sampling")
    ap.add_argument("--tag", default="",
                    help="suffix for the run directory name")
    args = ap.parse_args()

    cfg = load_config(require_api_key=not args.mock)
    seed = args.seed if args.seed is not None else cfg.seed

    arms = ["A", "B", "C"] if args.arm.lower() == "all" else [args.arm.upper()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"Unknown arm {a!r}. Choose A, B, C or all.")

    questions = load_questions(cfg, args.n, seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}{'_' + args.tag if args.tag else ''}"
    if args.mock:
        name += "_mock"
    out_root = cfg.runs_dir / name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Run       : {out_root.name}")
    print(f"Questions : {len(questions)} (seed {seed})")
    print(f"Arms      : {', '.join(arms)}")
    print(f"Mode      : {'MOCK (free)' if args.mock else 'LIVE'}"
          f"{' , cache OFF' if args.no_cache else ''}")
    print(f"Generator : {cfg.llm.generator_model}")
    print(f"Helper    : {cfg.llm.helper_model}")

    # Loading the retriever (embedding model + FAISS) takes a few seconds, so
    # we do it once and share it across arms.
    retriever = HybridRetriever(cfg)

    (out_root / "config.json").write_text(json.dumps({
        "seed": seed, "n_questions": len(questions), "arms": arms,
        "mock": args.mock, "cache": not args.no_cache,
        "generator_model": cfg.llm.generator_model,
        "helper_model": cfg.llm.helper_model,
        "top_k": cfg.retrieval.top_k,
        "claim_top_k": cfg.retrieval.claim_top_k,
        "max_correction_rounds": cfg.pipeline.max_correction_rounds,
        "qids": [q["qid"] for q in questions],
    }, indent=2))

    summaries = {}
    for a in arms:
        summaries[a] = run_arm(
            a, questions, cfg, retriever,
            mock=args.mock, use_cache=not args.no_cache,
            judge=not args.no_judge, out_root=out_root,
        )

    # ---- side-by-side comparison ----------------------------------------
    if len(summaries) > 1:
        print(f"\n{'=' * 72}")
        print("ABLATION SUMMARY")
        print(f"{'=' * 72}")
        hdr = f"{'metric':24s}" + "".join(f"{'arm ' + a:>12s}" for a in arms)
        print(hdr)
        print("-" * len(hdr))
        rows = [
            ("exact match", "exact_match", "{:.3f}"),
            ("token F1", "token_f1", "{:.3f}"),
            ("judge correct", "judge_correct", "{:.3f}"),
            ("faithfulness", "faithfulness", "{:.3f}"),
            ("claims / answer", "claims_per_answer", "{:.1f}"),
            ("% claims flagged", "pct_claims_flagged", "{:.1f}"),
            ("llm calls / query", "llm_calls_per_query", "{:.1f}"),
            ("tokens / query", "tokens_per_query", "{:.0f}"),
            ("USD / query", "usd_per_query", "${:.5f}"),
            ("latency median (s)", "latency_median_ms", "{:.1f}"),
        ]
        for label, key, fmt in rows:
            vals = []
            for a in arms:
                v = summaries[a].get(key)
                if v is None:
                    vals.append(f"{'-':>12s}")
                else:
                    if key == "latency_median_ms":
                        v = v / 1000
                    vals.append(f"{fmt.format(v):>12s}")
            print(f"{label:24s}" + "".join(vals))

    total = sum(s["usd_total_incl_shadow"] for s in summaries.values())
    print(f"\nTotal spend this run: ${total:.3f}")
    print(f"Logs: {out_root}")
    print(f"\nNext: python -m src.report --run {out_root.name}")


if __name__ == "__main__":
    sys.exit(main())
