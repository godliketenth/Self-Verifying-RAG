# Analysis

What the experiment measured, what it found, and what the findings do not
support. Every number here comes from `runs/20261011_050216_final` and
`runs/ragtruth_20261011_055308`, and can be regenerated with
`python -m src.report` without re-running inference.

---

## 1. What was run

| | |
|---|---|
| Questions | 50 from HotpotQA distractor (validation), seed 42 |
| Completed | 49 per arm (one question failed identically in all three) |
| Corpus | 9,735 Wikipedia paragraphs, built from 1,000 questions |
| Generator | `qwen2.5:7b-instruct`, run locally, **identical in all three arms** |
| Helper stages | same model |
| Verifier test set | 50 RAGTruth QA responses, 25 hallucinated + 25 clean |
| Cache | disabled, so every arm is measured on equal terms |

**The generator is a 7B local model.** This is a consequence of budget, not
design: OpenRouter's free tier allows 50 requests/day and Gemini's caps the
relevant model at 20/day, against roughly 1,150 calls needed here. Absolute
accuracy is therefore far below what a frontier model would produce, and the
EM/F1 numbers should not be read as a claim about RAG in general. The
ablation itself remains valid, because the only requirement is that the
generator be *fixed across arms*, and it is.

---

## 2. Headline results

| metric | A baseline | B agentic | C full |
|---|---|---|---|
| Exact match | **0.020** | 0.000 | 0.000 |
| Token F1 | **0.127** | 0.109 | 0.080 |
| LLM-judge correct | **0.429** | 0.367 | 0.367 |
| Faithfulness | 0.753 | 0.753 | **0.788** |
| Claims flagged | 28.6% | 32.2% | 27.1% |
| LLM calls / query | 1.0 | 3.7 | 5.9 |
| Tokens / query | 857 | 2,775 | **4,353** |
| Latency median | 1.3s | 7.5s | 8.2s |

Two results stand out, and they point in opposite directions.

**Claim-level verification improves faithfulness.** Arm C is the only arm
that raises it (0.753 → 0.788), and it does so over arm B, which shares its
retrieval exactly. The verification loop is the only difference between them,
so the gain is attributable to it.

**It does not improve answer correctness, and costs 5.1x the tokens.**
LLM-judge correctness falls from 0.429 to 0.367 and token F1 from 0.127 to
0.080. Section 3 explains why.

**The agentic layer alone buys nothing.** Arm B matches arm A's faithfulness
(0.753 vs 0.753) while scoring *worse* on every correctness metric, for 3.2x
the tokens. On this corpus and at this model size, routing and planning and
corrective retrieval did not pay for themselves.

---

## 3. Why correctness falls: the verifier over-flags

Measured against RAGTruth's human span labels (`runs/ragtruth_20261011_055308`):

| | |
|---|---|
| Precision | **0.247** |
| Recall | **0.688** |
| F1 | 0.364 |
| 3-way verdict accuracy | 0.706 |
| Confusion | TP 22, FP 67, FN 10, TN 173 |

The verifier **catches roughly 69% of genuine hallucinations**, which is the
useful half of the result. But **three of every four claims it flags are
actually fine**: 67 false positives against 22 true positives.

This is the mechanism behind arm C's correctness drop. The correction loop
acted on 19 claims and dropped 15 of them. If three quarters of those flags
were wrong, arm C deleted correct content, which lowers F1 and judge
correctness while raising the proportion of surviving claims the verifier is
happy with. The system is trading recall of *content* for grounding of what
remains - a conservative bias that is defensible for a high-stakes
application and harmful for a benchmark score.

### A circularity that must be stated

**Faithfulness is measured by the same verifier that does the correcting.**
Arm C's +3.5 point gain is therefore partly an artefact: the arm is optimised
against the metric's own judge. The honest reading is that arm C produces
answers *its own verifier approves of*, which is weaker than "arm C produces
more faithful answers".

The RAGTruth evaluation exists precisely to break that circle, because it
compares the verifier against **human** labels rather than itself. What it
shows is that the verifier's approval is a noisy signal of actual
groundedness. Any faithfulness figure in this report should be read with that
precision of 0.247 in mind.

---

## 4. A relevance rubric written for the wrong task

The first complete run produced a worse result: arm C below arm A on every
quality metric. Investigation found a single cause, and it was a prompt bug
rather than a model failure.

The CRAG-style relevance evaluator scored **gold passages at a mean of 0.50
and distractors at 0.03** - excellent discrimination. The rubric, however,
defined 1.0 as "contains a fact directly needed to answer" and 0.5 as "useful
background but not the answer". That is a **single-hop** rubric. In a
multi-hop question no single passage contains the answer - each supplies one
hop - so the model correctly scored gold passages 0.5, and the keep-threshold
sat at exactly 0.5.

| | passages kept | gold recall |
|---|---|---|
| Arm A, no filter | 5.0 | **0.806** |
| Arm B, original rubric | 2.1 | 0.520 |
| Arm B, fixed rubric | 3.3 | 0.643 |

**35% of gold evidence was being discarded.** The fix states that supplying
one hop is fully relevant, and lowers the threshold to 0.4 so it sits below
the gold band rather than on it. The pre-fix run is kept in
`runs/BEFORE_relevance_fix`.

Note that even after the fix, the filter still costs recall (0.643 vs 0.806
unfiltered). A CRAG-style filter driven by a 7B model removes more good
evidence than noise on this task.

**This was a prompt change made after seeing evaluation results**, which is a
form of fitting to the test set. It is reported as a bug fix - the rubric
contradicted the task definition - rather than tuning, and both runs are kept
so a reader can judge that for themselves. No further threshold adjustment
was made.

---

## 5. Should verification run on intermediate steps too?

This was the open question in the proposal. CHARM (Mishra, 2026) argues that
errors early in an agent pipeline cascade and amplify, which would favour
verifying intermediate steps. The logs allow a direct test
(`src/analyze_cascade.py`): split questions by whether retrieval went badly,
then compare how often claims are flagged.

| | poor retrieval | good retrieval |
|---|---|---|
| Arm B, % claims flagged | 21.2% | **46.2%** |
| Arm C, % claims flagged | 15.2% | **42.3%** |

**The result is the opposite of the cascade prediction.** Claims are flagged
roughly twice as often after *good* retrieval.

The mechanism is visible in the outputs: with poor retrieval the generator
hedges and produces fewer, more generic claims, several of which are
refusals carrying no factual assertion at all. With good retrieval it writes
longer and more specific answers, and specificity is what creates
opportunities to exceed the evidence.

**Conclusion for this pipeline: verify the final answer, not the intermediate
steps.** A bad retrieval does not reliably produce a bad claim, so
intermediate verification would spend tokens monitoring a stage that was not
the source of the failures. This conclusion is scoped to this setup - a
2-3 hop corpus QA task with a small generator - and CHARM's argument may
still hold for longer agent chains with tool use and side effects, where an
early error changes what later steps *do* rather than only what they say.

---

## 6. Cost

| | A | B | C |
|---|---|---|---|
| Tokens / query | 857 | 2,775 | 4,353 |
| vs baseline | 1.00x | 3.24x | **5.08x** |
| Latency median | 1.3s | 7.5s | 8.2s |

USD is zero because inference ran locally. This is why the project treats
**tokens and call counts** as the primary cost metric throughout: they remain
meaningful regardless of who is billing, and they are what a hosted
deployment would pay for.

Latency figures exclude shadow-verification time. Arms A and B run the
verifier only so their faithfulness is measurable and would never do so in
deployment; charging them that time initially made arm B appear slower than
arm C, which does strictly more work.

**Is verification worth 5x the tokens?** On this evidence, not for a
benchmark score - it lowers measured accuracy. It is worth it when the cost
of an ungrounded claim exceeds the cost of a missing one, which is the
high-stakes setting the proposal motivated. With precision at 0.247, however,
a deployment would want the verdicts surfaced as *warnings for a human* rather
than used to delete content automatically, which is how the demo presents
them.

---

## 7. Limitations

1. **Small generator.** `qwen2.5:7b-instruct` produces far weaker answers
   than a frontier model. Absolute EM of 0.020 reflects the model, not the
   architecture. The relative comparison between arms is the result; the
   absolute numbers are not.
2. **n = 49.** One failed question per arm, from a malformed JSON reply the
   repair pass could not salvage. Differences of a few points are within
   noise at this sample size, and no significance testing was performed.
3. **Faithfulness is self-judged**, as set out in section 3.
4. **RAGTruth sample is enriched** to ~50% hallucinated against a natural rate
   of ~18%, so reported precision is *optimistic* relative to deployment.
5. **Span-to-claim mapping is lossy**: 34% of gold spans mapped to no claim,
   usually because our decomposer did not produce a claim for that sentence.
   What is measured is the verifier *as used inside this pipeline*, not in
   isolation.
6. **One prompt was changed after seeing results** (section 4).
7. **Wikipedia, not a high-stakes domain.** The proposal motivates the work
   with legal, medical and governance text. The mechanism is
   domain-independent, but those domains have longer documents, denser
   cross-references, and a much higher cost for a false ENTAILED. Nothing
   here validates performance in them.

---

## 8. What I would do next

1. **Fix verifier precision.** At 0.247 it is the binding constraint on the
   whole system - it is what makes arm C delete correct content. A DeBERTa-
   MNLI cross-encoder would test whether the problem is this model or the
   task framing, and it runs locally at no cost.
2. **Re-run on a frontier generator.** One $5 top-up switches the provider
   back to Claude Sonnet 5 with a single flag, and would separate "what the
   architecture does" from "what a 7B model does".
3. **Stop auto-dropping flagged claims.** Given the precision, surfacing
   verdicts as warnings would likely preserve correctness while keeping the
   grounding signal.
4. **SelfCheckGPT as a reference-free baseline**, to quantify how much the
   retrieval in verification actually contributes over self-consistency.
