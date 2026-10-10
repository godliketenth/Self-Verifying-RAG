"""
The orchestrator: runs one question through the pipeline.

THIS IS THE MOST IMPORTANT FILE TO UNDERSTAND. Everything else is a stage;
this is the control flow that connects them, including the two corrective
loops.

The three ablation arms are not three functions. They are this one function
run with different flags on the `arm` object:

    arm A: router=off planner=off relevance=off verify=off  (shadow on)
    arm B: router=on  planner=on  relevance=on  verify=off  (shadow on)
    arm C: router=on  planner=on  relevance=on  verify=ON   corrections=1

Reading the code with that in mind, you can see exactly what each arm does
and - more importantly - that no arm gets a secretly different code path.

ON SHADOW VERIFICATION. Arms A and B run stages 7-9 too, but only to RECORD
verdicts; they never act on them. Without this, arms A and B would have no
faithfulness number and two thirds of the results table would be empty. The
calls are tagged is_shadow=True so they stay out of those arms' cost figures.
"""

from __future__ import annotations

import time

from src.config import Arm, Config
from src.llm import LLMClient
from src.retrieval.hybrid import HybridRetriever
from src.stages.base import load_prompt
from src.stages.s02_router import route_query
from src.stages.s03_planner import plan_retrieval
from src.stages.s05_relevance import evaluate_relevance, needs_replan
from src.stages.s06_generator import generate_answer, regenerate_claim
from src.stages.s07_decomposer import decompose_answer
from src.stages.s08_claim_retrieval import retrieve_for_claim, retrieve_for_claims
from src.stages.s09_verifier import verify_claim, verify_claims
from src.stages.s10_assemble import assemble_answer
from src.types import Claim, PipelineResult, Route


def run(
    question: str,
    arm: Arm,
    client: LLMClient,
    retriever: HybridRetriever,
    cfg: Config,
    gold_answer: str = "",
) -> PipelineResult:
    """Run one question end to end and return everything that happened."""
    t_start = time.perf_counter()
    result = PipelineResult(question=question, arm=arm.name,
                            gold_answer=gold_answer)

    # ======================================================================
    # STAGE 2 - complexity router
    # ======================================================================
    if arm.router:
        route, _reason = route_query(question, client, cfg)
    else:
        # Arm A is baseline RAG: it always retrieves exactly once, which is
        # what SINGLE_HOP means here. No LLM call is spent.
        route = Route.SINGLE_HOP
    result.route = route

    # ======================================================================
    # STAGES 3-5 - plan, retrieve, filter, and re-plan if the evidence is poor
    # ======================================================================
    passages: list = []
    kept: list = []

    if route is Route.NO_RETRIEVAL:
        # The router judged this answerable without lookup (Adaptive-RAG's
        # cheapest branch). We honour that decision rather than quietly
        # retrieving anyway - if the router is wrong, the cost of being wrong
        # should show up in the results, not be hidden by a safety net.
        result.subqueries = []
    else:
        # Stage 3
        if arm.planner:
            subqueries = plan_retrieval(question, route, client, cfg)
        else:
            subqueries = [question]

        max_attempts = cfg.pipeline.max_replan_rounds + 1
        for attempt in range(max_attempts):
            result.subqueries = subqueries

            # Stage 4 - hybrid retrieval (local, no API cost)
            if len(subqueries) == 1:
                passages = retriever.search(subqueries[0], k=cfg.retrieval.top_k)
            else:
                passages = retriever.search_many(subqueries, k=cfg.retrieval.top_k)

            # Stage 5 - relevance filter
            if arm.relevance:
                kept, _scores = evaluate_relevance(question, passages, client, cfg)

                # ---- CORRECTIVE LOOP 1: stage 5 -> stage 3 ----
                # Too little relevant evidence survived, so the retrieval plan
                # was wrong. Re-plan with different queries rather than
                # generating a confident answer from poor context.
                if needs_replan(kept, cfg) and attempt < max_attempts - 1:
                    result.replan_rounds += 1
                    subqueries = plan_retrieval(
                        question, route, client, cfg, previous=subqueries
                    )
                    continue
            else:
                kept = passages        # arm A: no filtering at all
            break

    result.retrieved = passages
    result.kept_passages = kept

    # ======================================================================
    # STAGE 6 - generate the draft answer (the FIXED model, in every arm)
    # ======================================================================
    if route is Route.NO_RETRIEVAL:
        prompt = load_prompt("s06_generator_norag").format(question=question)
        draft = client.complete(prompt, stage="s06_generator",
                                model=cfg.llm.generator_model).text.strip()
    else:
        draft = generate_answer(question, kept, client, cfg)
    result.draft_answer = draft

    # ======================================================================
    # STAGES 7-9 - decompose, retrieve per claim, verify
    # ======================================================================
    do_verify = arm.verify or arm.verify_shadow
    if do_verify:
        # shadow = measure only, never act
        shadow = not arm.verify

        # Stage 7
        claims = decompose_answer(draft, client, cfg, is_shadow=shadow,
                                  question=question)

        if claims:
            # Stage 8 - fresh evidence per claim (local, free)
            evidence = retrieve_for_claims(claims, retriever, cfg)

            # Stage 9 - verdicts.
            # `forbidden` is how we PROVE asymmetry: we hand the verifier the
            # draft and question only so it can assert they are absent from
            # its own prompt. See src/stages/s09_verifier.py.
            claims = verify_claims(
                claims, evidence, client, cfg,
                is_shadow=shadow,
                forbidden=[draft, question],
            )

            # ---- CORRECTIVE LOOP 2: stage 9 -> stage 8/6 ----
            # Only arm C acts on the verdicts. We re-retrieve and regenerate
            # ONLY the failed claims - not the whole answer - so the cost of
            # correction scales with the number of errors, not answer length.
            if arm.verify:
                claims = _run_correction_loop(
                    claims, arm, client, retriever, cfg,
                    draft=draft, question=question, result=result,
                )

        result.claims = claims

    # ======================================================================
    # STAGE 10 - assemble what the user sees
    # ======================================================================
    # Arms A and B return the draft verbatim: their verdicts were shadow-only,
    # so acting on them here would turn them into arm C.
    result.final_answer = (
        assemble_answer(draft, result.claims) if arm.verify else draft
    )

    result.stage_logs = client.take_logs()
    result.total_latency_ms = (time.perf_counter() - t_start) * 1000
    return result


def _run_correction_loop(
    claims: list[Claim],
    arm: Arm,
    client: LLMClient,
    retriever: HybridRetriever,
    cfg: Config,
    draft: str,
    question: str,
    result: PipelineResult,
) -> list[Claim]:
    """
    The verification corrective loop (arm C only).

    For each failed claim, up to `arm.correction_rounds` times:
      1. retrieve again with a BROADER query (the claim text itself, which
         differs from the narrow verification query that just failed),
      2. ask the model to rewrite the claim to match that evidence, or drop it,
      3. re-verify the rewrite - a correction we do not re-check is just a
         second unverified assertion.

    Dropped claims are removed from the answer entirely. That is the honest
    outcome when the corpus simply does not support a statement.
    """
    rounds = min(arm.correction_rounds, cfg.pipeline.max_correction_rounds)

    for _ in range(rounds):
        failed = [c for c in claims
                  if c.verdict is not None and c.verdict.is_failure]
        if not failed:
            break
        result.correction_rounds += 1

        for claim in failed:
            # 1. broader re-retrieval: search on the claim itself
            fresh = retriever.search(claim.text, k=cfg.retrieval.claim_top_k)
            # Union with the original narrow retrieval, so a correction never
            # has LESS evidence available than the failed verification did.
            seen = {p.pid for p in fresh}
            for p in retrieve_for_claim(claim, retriever, cfg):
                if p.pid not in seen:
                    fresh.append(p)

            # 2. rewrite or drop
            new_text, dropped = regenerate_claim(claim, fresh, client, cfg)
            if dropped or not new_text:
                claim.original_text = claim.text
                claim.was_corrected = True
                claim.text = ""          # removed from the final answer
                continue

            # 3. re-verify the rewrite under the same asymmetry rule
            verdict, ev, reasoning = verify_claim(
                claim_text=new_text,
                evidence=fresh,
                client=client, cfg=cfg,
                _forbidden=[draft, question],
            )
            claim.original_text = claim.text
            claim.text = new_text
            claim.was_corrected = True
            claim.verdict = verdict
            claim.evidence = ev
            claim.reasoning = reasoning

    # A claim whose text was emptied is dropped from the answer.
    return [c for c in claims if c.text.strip() or c.was_corrected]
