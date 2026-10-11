"""
Streamlit demo for Self-Verifying Agentic RAG.

    streamlit run src/app.py

Three views, in the order a viva needs them:

  Verify   - one question, one configuration, every claim audited
  Compare  - the SAME question through two configurations, side by side.
             This is the project's actual contribution made visible: the
             ablation is the result, and a demo that runs one arm at a time
             cannot show it.
  Results  - the measured numbers, so the findings live in the demo rather
             than only in the report.

DESIGN NOTE. The verdict colour on each claim is a left rule, not a card.
That rule encodes the verdict - it is information, not decoration - and the
claim text stays the largest thing on screen, because the claim under audit
is what the audience is here to look at.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run src/app.py` executes this file directly, which puts src/ on
# sys.path rather than the project root - so `import src.config` fails with
# ModuleNotFoundError. (`python -m src.app` works, because -m adds the current
# directory.) Putting the project root on the path first makes the documented
# command work from anywhere. This must run BEFORE any src.* import.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import time

import streamlit as st

from src.config import ARMS, load_config
from src.llm import LLMClient
from src.pipeline import run
from src.retrieval.hybrid import HybridRetriever
from src.types import PipelineResult, Verdict

st.set_page_config(page_title="Self-Verifying Agentic RAG",
                   page_icon="◆", layout="wide")

# Verdict colours are the reserved status palette used throughout the report,
# so a verdict here and a verdict in a chart mean the same thing.
VERDICT = {
    Verdict.ENTAILED:     ("Entailed", "#0ca30c",
                           "the evidence supports this"),
    Verdict.UNSUPPORTED:  ("Unsupported", "#b07d00",
                           "the evidence is silent on this"),
    Verdict.CONTRADICTED: ("Contradicted", "#d03b3b",
                           "the evidence disagrees with this"),
}

CSS = """
<style>
  /* The claim under audit is the largest thing on screen. */
  .claim-row { border-left: 3px solid var(--rule); padding: .55rem 0 .55rem .85rem;
               margin: .1rem 0 .55rem 0; }
  .claim-text { font-size: 1.03rem; line-height: 1.5; color: #1a1a1a; }
  .claim-verdict { font-size: .78rem; font-weight: 700; letter-spacing: .01em;
                   color: var(--rule); }
  .claim-why { font-size: .78rem; color: #6b6762; margin-left: .5rem;
               font-weight: 400; letter-spacing: 0; }
  .claim-was { font-size: .82rem; color: #6b6762; margin-top: .25rem; }
  .ev { border-left: 2px solid #ddd9d2; padding-left: .8rem; margin:.4rem 0;
        font-size: .86rem; color: #3b3835; }
  .ev-src { font-size: .76rem; color: #8a857e; }
  .answer { font-size: 1.12rem; line-height: 1.6; color: #16150f; }
  .muted { color: #6b6762; font-size: .85rem; }
</style>
"""


@st.cache_resource
def boot():
    cfg = load_config()
    return HybridRetriever(cfg), cfg


def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def claim_block(claim, show_evidence: bool = True) -> None:
    """One audited claim: verdict rule, the claim, then what settled it."""
    label, colour, why = VERDICT.get(
        claim.verdict, ("Unverified", "#8a857e", "no verdict recorded"))
    text = claim.text.strip() or claim.original_text.strip()
    removed = claim.was_corrected and not claim.text.strip()

    st.markdown(
        f"<div class='claim-row' style='--rule:{colour}'>"
        f"<div class='claim-verdict'>{label}"
        f"<span class='claim-why'>{why}</span></div>"
        f"<div class='claim-text'>{text}</div>"
        + (f"<div class='claim-was'>Removed from the answer.</div>"
           if removed else "")
        + (f"<div class='claim-was'>Rewritten from: {claim.original_text}</div>"
           if claim.was_corrected and not removed and claim.original_text
           else "")
        + "</div>",
        unsafe_allow_html=True)

    if not show_evidence:
        return
    cited = [e for e in claim.evidence if e.span]
    looked = [e for e in claim.evidence if not e.span]
    if cited:
        for e in cited:
            st.markdown(
                f"<div class='ev'>{e.span}"
                f"<div class='ev-src'>{e.pid} · {e.title}</div></div>",
                unsafe_allow_html=True)
    elif looked:
        titles = ", ".join(f"{e.pid} · {e.title}" for e in looked)
        st.markdown(
            f"<div class='ev'><span class='muted'>Searched, and none of these "
            f"supported the claim:</span><div class='ev-src'>{titles}</div>"
            f"</div>", unsafe_allow_html=True)


def cost_line(r: PipelineResult) -> str:
    return (f"{r.n_llm_calls(include_shadow=True)} model calls · "
            f"{r.total_tokens(include_shadow=True):,} tokens · "
            f"{r.total_latency_ms / 1000:.1f}s")


def execute(question: str, arm_key: str, mock: bool, retriever, cfg):
    client = LLMClient(cfg, mock=mock)
    t0 = time.perf_counter()
    result = run(question, ARMS[arm_key], client, retriever, cfg)
    result.total_latency_ms = (time.perf_counter() - t0) * 1000
    return result


# ---------------------------------------------------------------------------

def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Self-Verifying Agentic RAG")
    st.markdown(
        "<div class='muted'>Every sentence of the answer is checked against "
        "retrieved evidence before you see it. Aditya Khairnar · C221</div>",
        unsafe_allow_html=True)
    st.write("")

    try:
        retriever, cfg = boot()
    except SystemExit as e:
        st.error(f"{e}\n\nBuild the data first — see START_HERE.md.")
        return

    with st.sidebar:
        st.subheader("Configuration")
        arm_key = st.radio(
            "Pipeline", ["D", "C", "B", "A"], index=0,
            format_func=lambda a: {
                "A": "A · Baseline RAG",
                "B": "B · Agentic retrieval",
                "C": "C · Verified, removes failures",
                "D": "D · Verified, flags failures",
            }[a])
        st.caption(ARMS[arm_key].description)
        mock = st.checkbox("Mock mode (no model, instant)", value=False)
        st.divider()
        arm = ARMS[arm_key]
        st.caption("Stages active in this configuration")
        for label, on in [("Complexity router", arm.router),
                          ("Retrieval planner", arm.planner),
                          ("Relevance filter", arm.relevance),
                          ("Claim verification", arm.verify),
                          ("Correction loop", arm.correction_rounds > 0)]:
            st.markdown(f"{'●' if on else '○'} {label}")
        st.divider()
        st.caption(f"Generator `{cfg.llm.generator_model}` via "
                   f"`{cfg.llm.provider}`, identical in every configuration")

    tab_verify, tab_compare, tab_results = st.tabs(
        ["Verify an answer", "Compare configurations", "Measured results"])

    # ---- VERIFY ----------------------------------------------------------
    with tab_verify:
        demos = load_json(cfg.data_dir / "demo_verified.json") or []
        if "q" not in st.session_state:
            st.session_state.q = (demos[0]["question"] if demos else
                                  "Were Scott Derrickson and Ed Wood of the "
                                  "same nationality?")

        if demos:
            st.caption("Questions that flagged a claim in the measured run — "
                       "taken from the logs, so they reproduce")
            for d in demos[:4]:
                tag = ("a contradiction" if d["has_contradiction"]
                       else f"{d['n_flagged']} flagged")
                if st.button(f"{d['question']}", key=f"d{d['question'][:20]}",
                             use_container_width=True):
                    st.session_state.q = d["question"]
                st.caption(f"found {tag} · expected answer *{d['gold_answer']}*")

        question = st.text_input("Question", value=st.session_state.q)
        if not st.button("Verify", type="primary"):
            return

        if not mock and cfg.llm.provider != "ollama" and not cfg.llm.api_key:
            st.error("No API key for this provider. Tick mock mode, or set the "
                     "key in .env.")
            return

        with st.spinner(f"Running configuration {arm_key}…"):
            r = execute(question, arm_key, mock, retriever, cfg)

        st.markdown(f"<div class='answer'>{r.final_answer or '—'}</div>",
                    unsafe_allow_html=True)
        st.caption(cost_line(r))

        if r.claims:
            n_bad = sum(1 for c in r.claims
                        if c.verdict and c.verdict.is_failure)
            st.divider()
            st.subheader(f"{len(r.claims)} claims audited, {n_bad} flagged")
            if not arm.verify:
                st.caption("This configuration records verdicts but does not "
                           "act on them. Switch to C or D to see the "
                           "correction loop.")
            for c in r.claims:
                claim_block(c)

            dropped = [c for c in r.claims
                       if c.was_corrected and not c.text.strip()]
            if dropped:
                st.warning(
                    f"Verification removed {len(dropped)} claim(s). Measured "
                    "precision of this verifier is 0.247, so roughly three in "
                    "four flags are false positives and some removed claims "
                    "were correct. Configuration D keeps them and flags them "
                    "instead.")
                with st.expander("Answer before verification"):
                    st.write(r.draft_answer)

        with st.expander("How the answer was built"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"Route **{r.route.value if r.route else '—'}**")
                st.markdown(f"Re-planned **{r.replan_rounds}×**, corrected "
                            f"**{r.correction_rounds}×**")
                if r.subqueries:
                    st.caption("Searched for")
                    for q in r.subqueries:
                        st.markdown(f"- {q}")
            with c2:
                kept = {p.pid for p in r.kept_passages}
                st.caption(f"{len(r.retrieved)} passages retrieved, "
                           f"{len(kept)} kept")
                for p in r.retrieved[:8]:
                    mark = "kept" if p.pid in kept else "dropped"
                    st.markdown(f"<span class='muted'>{mark} · {p.pid} · "
                                f"{p.title}</span>", unsafe_allow_html=True)

    # ---- COMPARE ---------------------------------------------------------
    with tab_compare:
        st.markdown(
            "The same question through two configurations. Everything is "
            "held constant except the stages each one runs, so any difference "
            "you see is caused by those stages.")
        c1, c2 = st.columns(2)
        left = c1.selectbox("Left", ["A", "B", "C", "D"], index=0,
                            format_func=lambda a: f"{a} · "
                            f"{ARMS[a].description.split(':')[0]}")
        right = c2.selectbox("Right", ["A", "B", "C", "D"], index=3,
                             format_func=lambda a: f"{a} · "
                             f"{ARMS[a].description.split(':')[0]}")
        cq = st.text_input("Question to compare",
                           value=st.session_state.get("q", ""), key="cmpq")

        if st.button("Run both", type="primary"):
            cols = st.columns(2)
            for col, key in ((cols[0], left), (cols[1], right)):
                with col:
                    st.markdown(f"**Configuration {key}**")
                    with st.spinner(f"Running {key}…"):
                        r = execute(cq, key, mock, retriever, cfg)
                    st.markdown(f"<div class='answer'>{r.final_answer or '—'}"
                                f"</div>", unsafe_allow_html=True)
                    st.caption(cost_line(r))
                    st.divider()
                    if r.claims:
                        n_bad = sum(1 for c in r.claims
                                    if c.verdict and c.verdict.is_failure)
                        st.caption(f"{len(r.claims)} claims, {n_bad} flagged")
                        for c in r.claims:
                            claim_block(c, show_evidence=False)
                    else:
                        st.caption("No claims audited in this configuration.")

    # ---- RESULTS ---------------------------------------------------------
    with tab_results:
        runs = [d for d in sorted(cfg.runs_dir.iterdir())
                if d.is_dir() and (d / "armA" / "summary.json").exists()]
        if not runs:
            st.info("No completed run yet. See START_HERE.md step 5.")
            return
        latest = runs[-1]
        st.caption(f"From `{latest.name}` — 49 questions per configuration, "
                   "HotpotQA distractor")

        rows = {}
        for arm_dir in sorted(latest.glob("arm*")):
            s = load_json(arm_dir / "summary.json")
            if s:
                rows[arm_dir.name.replace("arm", "")] = s
        if rows:
            import pandas as pd
            keys = sorted(rows)
            table = pd.DataFrame({
                "Answer correct (judge)": [rows[k].get("judge_correct") for k in keys],
                "Exact match (span)": [rows[k].get("exact_match_span") for k in keys],
                "Faithfulness": [rows[k].get("faithfulness") for k in keys],
                "Tokens per question": [rows[k].get("tokens_per_query") for k in keys],
            }, index=[f"Configuration {k}" for k in keys]).round(3)
            st.dataframe(table, use_container_width=True)

        st.markdown(
            "Verification raises faithfulness. Removing flagged claims is what "
            "costs accuracy — configuration D keeps them and recovers it, at "
            "the same token cost. Measured verifier precision is 0.247, which "
            "is why flagging beats deleting. Full discussion in `ANALYSIS.md`.")

        shots = [("plot_faithfulness.png", "Faithfulness"),
                 ("plot_correctness.png", "Answer correctness"),
                 ("plot_cost_quality.png", "What faithfulness costs"),
                 ("plot_verdicts.png", "Verdict mix")]
        pics = [(latest / f, cap) for f, cap in shots if (latest / f).exists()]
        for i in range(0, len(pics), 2):
            cc = st.columns(2)
            for col, (path, cap) in zip(cc, pics[i:i + 2]):
                col.image(str(path), caption=cap, use_container_width=True)


if __name__ == "__main__":
    main()
