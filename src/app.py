"""
Streamlit demo - the live "catches an unsupported claim" moment.

    streamlit run src/app.py

Shows, for one question:
  - the agent's retrieval decisions (route, sub-queries, what it kept/dropped)
  - the draft answer
  - every claim with its verdict badge and the evidence span behind it
  - what the correction loop changed

The point of the UI is the thing the proposal argued for: a user should not
get one confidence score for a paragraph, they should see WHICH sentence is
unsupported and WHY.
"""

from __future__ import annotations

import streamlit as st

from src.config import ARMS, load_config
from src.llm import LLMClient
from src.pipeline import run
from src.retrieval.hybrid import HybridRetriever
from src.types import Verdict

st.set_page_config(page_title="Self-Verifying Agentic RAG",
                   page_icon="*", layout="wide")

BADGE = {
    Verdict.ENTAILED:     ("ENTAILED",     "#0ca30c"),
    Verdict.UNSUPPORTED:  ("UNSUPPORTED",  "#fab219"),
    Verdict.CONTRADICTED: ("CONTRADICTED", "#d03b3b"),
}

# Two kinds of demo question, and the contrast between them IS the demo.
#
# ANSWERABLE: the supporting Wikipedia articles are in the indexed corpus, so
# retrieval finds real evidence and claims should come back ENTAILED.
#
# EVIDENCE MISSING: real HotpotQA questions whose supporting articles were
# never indexed (we index 1,000 questions' paragraphs out of 7,405). Nothing
# in the corpus can support an answer. Retrieval still returns SOMETHING -
# similarity search always does - so this is precisely the failure mode the
# project exists to catch: a fluent answer built on loosely-related passages,
# revealed only by checking each claim against evidence.
#
# These are honest demonstrations, not staged failures: the questions are
# genuine and the system is behaving correctly by flagging them.
ANSWERABLE = [
    "Were Scott Derrickson and Ed Wood of the same nationality?",
    "What government position was held by the woman who portrayed "
    "Corliss Archer in the film Kiss and Tell?",
    "Which magazine was started first, Arthur's Magazine or First for Women?",
]


def load_demo_questions(cfg) -> list[dict]:
    """Questions whose evidence is deliberately absent from the corpus."""
    f = cfg.data_dir / "demo_questions.json"
    if not f.exists():
        return []
    import json
    return json.loads(f.read_text())


def load_verified_demos(cfg) -> list[dict]:
    """
    Questions that DID flag or correct a claim in the measured evaluation run.

    Mined from the logs rather than hand-picked, so the demo shows behaviour
    that was actually observed and is reproducible at the same seed.
    """
    f = cfg.data_dir / "demo_verified.json"
    if not f.exists():
        return []
    import json
    return json.loads(f.read_text())


# Loading the embedding model and FAISS index takes a few seconds; cache it
# so only the first question in a session pays that cost.
@st.cache_resource
def get_retriever():
    cfg = load_config()
    return HybridRetriever(cfg), cfg


def main() -> None:
    st.title("Self-Verifying Agentic RAG")
    st.caption("Claim-level hallucination detection in an agentic retrieval "
               "pipeline - Aditya Khairnar (C221)")

    try:
        retriever, cfg = get_retriever()
    except SystemExit as e:
        st.error(f"{e}\n\nRun the data build commands in START_HERE.md first.")
        return

    # ---- controls --------------------------------------------------------
    with st.sidebar:
        st.header("Settings")
        arm_name = st.radio(
            "Ablation arm", ["C", "B", "A"],
            format_func=lambda a: f"Arm {a} - {ARMS[a].description.split(':')[0]}",
            help="Arm C is the full system. A and B are the baselines.",
        )
        mock = st.checkbox("Mock mode (no API key, free)", value=False)
        st.divider()
        arm = ARMS[arm_name]
        st.markdown("**This arm's stages**")
        for label, on in [("Complexity router", arm.router),
                          ("Retrieval planner", arm.planner),
                          ("Relevance evaluator", arm.relevance),
                          ("Claim verification", arm.verify),
                          ("Correction loop", arm.correction_rounds > 0)]:
            st.markdown(f"{'ON ' if on else 'off'} - {label}")
        st.divider()
        st.caption(f"Generator: `{cfg.llm.generator_model}`")
        st.caption(f"Helper: `{cfg.llm.helper_model}`")

    # Keep the chosen question across reruns, so clicking an example button
    # then pressing Run does not reset to the default.
    if "question" not in st.session_state:
        st.session_state.question = ANSWERABLE[0]

    tab_catch, tab_ok, tab_missing = st.tabs(
        ["Verification CATCHES a claim", "Clean answer",
         "Evidence missing from corpus"]
    )
    with tab_catch:
        st.caption(
            "These questions flagged or corrected at least one claim in the "
            "measured evaluation run. They are taken from the logs, not "
            "hand-picked, and reproduce at the same seed. **Run these on arm "
            "C to see the correction loop fire.**"
        )
        for d in load_verified_demos(cfg)[:5]:
            badge = ("contradiction found" if d["has_contradiction"]
                     else f"{d['n_flagged']} claim(s) flagged")
            if st.button(d["question"], key=f"catch_{d['question'][:24]}",
                         use_container_width=True):
                st.session_state.question = d["question"]
            st.caption(f"observed: {badge} - gold answer: *{d['gold_answer']}*")
    with tab_ok:
        st.caption("Supporting articles are indexed. Claims should verify as "
                   "ENTAILED.")
        for ex in ANSWERABLE:
            if st.button(ex, key=f"ok_{ex[:24]}", use_container_width=True):
                st.session_state.question = ex
    with tab_missing:
        st.caption(
            "Real HotpotQA questions whose supporting Wikipedia articles were "
            "never indexed. Retrieval still returns passages - similarity "
            "search always returns something - so the system should either "
            "refuse, or produce claims the verifier marks UNSUPPORTED. "
            "**This is the demo moment.**"
        )
        demo_qs = load_demo_questions(cfg)
        if not demo_qs:
            st.info("Run `python -m src.data.build_demo_questions` to populate "
                    "these.")
        for d in demo_qs[:6]:
            if st.button(d["question"], key=f"miss_{d['question'][:24]}",
                         use_container_width=True):
                st.session_state.question = d["question"]
            st.caption(f"gold answer: *{d['answer']}* - missing articles: "
                       f"{', '.join(d['gold_titles'])}")

    question = st.text_input("Question", value=st.session_state.question)

    if not st.button("Run pipeline", type="primary"):
        st.info("Enter a question and press **Run pipeline**.")
        return

    if not mock and not cfg.llm.api_key:
        st.error("No OPENROUTER_API_KEY found. Add it to .env, or tick "
                 "**Mock mode** in the sidebar.")
        return

    # ---- run -------------------------------------------------------------
    client = LLMClient(cfg, mock=mock)
    with st.spinner(f"Running arm {arm_name} ..."):
        result = run(question, arm, client, retriever, cfg)

    # ---- headline numbers ------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LLM calls", result.n_llm_calls(include_shadow=True))
    c2.metric("Tokens", f"{result.total_tokens(include_shadow=True):,}")
    c3.metric("Cost", f"${result.total_cost_usd(include_shadow=True):.4f}")
    c4.metric("Latency", f"{result.total_latency_ms / 1000:.1f}s")

    # ---- the answer ------------------------------------------------------
    st.subheader("Answer")
    st.success(result.final_answer or "(no answer produced)")

    if result.claims and result.draft_answer != result.final_answer:
        dropped = [c for c in result.claims
                   if c.was_corrected and not c.text.strip()]
        st.markdown("**Draft answer, before verification**")
        st.info(result.draft_answer)
        if dropped:
            # Showing this is the point, not an embarrassment. Measured
            # verifier precision is 0.247, so some dropped claims were
            # correct. A viewer must be able to see what verification COST,
            # not only what it caught - that is the honest comparison, and it
            # is the evidence behind ANALYSIS.md's recommendation to surface
            # verdicts as warnings rather than delete content automatically.
            st.warning(
                f"Verification removed {len(dropped)} claim(s). On the "
                "RAGTruth evaluation this verifier had **precision 0.247** - "
                "roughly three in four flags are false positives - so some "
                "removed claims were correct. Compare the draft above with "
                "the verified answer to judge the trade-off."
            )

    # ---- per-claim verdicts: the core of the demo ------------------------
    if result.claims:
        n_bad = sum(1 for c in result.claims
                    if c.verdict and c.verdict.is_failure)
        st.subheader(f"Claim-level verification "
                     f"({len(result.claims)} claims, {n_bad} flagged)")
        if not arm.verify:
            st.caption("This arm runs verification in SHADOW mode: verdicts "
                       "are recorded for measurement but the answer is not "
                       "corrected. Switch to arm C to see the correction loop.")

        for i, claim in enumerate(result.claims, start=1):
            label, color = BADGE.get(claim.verdict, ("UNVERIFIED", "#52514e"))
            shown = claim.text or claim.original_text
            # Built outside the f-string: Python 3.11 disallows backslash
            # escapes inside f-string expressions.
            corrected_tag = ""
            if claim.was_corrected:
                corrected_tag = (
                    "<span style='color:#52514e;font-size:.8rem'>"
                    " (corrected)</span>"
                )
            st.markdown(
                f"<div style='border-left:4px solid {color};"
                f"padding:.5rem .8rem;margin:.4rem 0;background:#fafaf9'>"
                f"<span style='color:{color};font-weight:700;font-size:.8rem'>"
                f"{label}</span>{corrected_tag}"
                f"<br><span style='color:#0b0b0b'>{shown}</span></div>",
                unsafe_allow_html=True,
            )
            with st.expander(f"Evidence for claim {i}"):
                if claim.was_corrected:
                    st.markdown(f"**Original claim:** {claim.original_text}")
                st.markdown(f"**Verification query:** `{claim.verification_query}`")
                if claim.reasoning:
                    st.markdown(f"**Verifier's reasoning:** {claim.reasoning}")
                cited = [e for e in claim.evidence if e.span]
                considered = [e for e in claim.evidence if not e.span]
                if cited:
                    for ev in cited:
                        st.markdown(f"**[{ev.pid}] {ev.title}**")
                        st.markdown(f"> {ev.span}")
                        if ev.full_text:
                            st.caption(ev.full_text[:600])
                if considered:
                    st.markdown("**Passages considered** - none of these "
                                "supported the claim:")
                    for ev in considered:
                        st.markdown(f"- **[{ev.pid}] {ev.title}**")
                        st.caption(ev.full_text[:400])
                if not claim.evidence:
                    st.caption("No passages were retrieved for this claim.")

    # ---- the agent's retrieval decisions ---------------------------------
    st.subheader("Pipeline trace")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"**Route:** `{result.route.value if result.route else '-'}`")
        st.markdown(f"**Re-plan rounds:** {result.replan_rounds}")
        st.markdown(f"**Correction rounds:** {result.correction_rounds}")
        if result.subqueries:
            st.markdown("**Sub-queries:**")
            for q in result.subqueries:
                st.markdown(f"- {q}")
    with t2:
        st.markdown(f"**Retrieved:** {len(result.retrieved)} passages, "
                    f"**kept:** {len(result.kept_passages)}")
        kept_ids = {p.pid for p in result.kept_passages}
        for p in result.retrieved[:10]:
            mark = "KEPT   " if p.pid in kept_ids else "dropped"
            st.caption(f"{mark} [{p.pid}] {p.title}")

    with st.expander("Per-stage cost breakdown"):
        import pandas as pd
        rows = [{
            "stage": l.stage, "model": l.model.split("/")[-1],
            "tokens in": l.tokens_in, "tokens out": l.tokens_out,
            "USD": round(l.cost_usd, 6),
            "ms": round(l.latency_ms),
            "cached": l.cache_hit, "shadow": l.is_shadow,
        } for l in result.stage_logs]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


if __name__ == "__main__":
    main()
