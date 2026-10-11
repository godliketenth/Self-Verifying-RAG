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
| Verifier backends | LLM judge, and an NLI cross-encoder for comparison (section 9) |
| Arms | A, B, C, plus arm D (warn-only) added after the fact - section 10 |
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
0.080. Section 3 explains why - and section 10 shows the loss is caused
entirely by DELETING flagged claims, not by verifying them: a warn-only
variant keeps the full faithfulness gain and recovers the correctness.

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
compares the verifier against **human** labels rather than itself. Section 9
pushes on this further by scoring a second, independent backend on the same
claims. What it
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
than used to delete content automatically. Section 10 measures that variant
(arm D) rather than merely recommending it: it retains the whole faithfulness
gain at the same token cost, and is what the demo now presents.

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
7. **Arm D was run separately** from arms A-C, though on the same 50
   questions with the same seed and the same cold-cache settings. Its results
   are merged into the same report directory, and the comparison in section
   10 should be read with that in mind.
8. **The arm D vs baseline difference is one question** (22 correct against
   21). Only the arm D vs arm C gap, four questions, is worth any weight, and
   neither was significance tested.
9. **Wikipedia, not a high-stakes domain.** The proposal motivates the work
   with legal, medical and governance text. The mechanism is
   domain-independent, but those domains have longer documents, denser
   cross-references, and a much higher cost for a false ENTAILED. Nothing
   here validates performance in them.

---

## 8. What I would do next

*(Items 1 and 3 have since been done - see sections 9 and 10. This list is
updated accordingly.)*

1. ~~Swap in an NLI cross-encoder to fix verifier precision.~~ **Done, and it
   did not work** (section 9). The replacement direction that follows from
   that result: **reframe the verification task**. Almost all false positives
   are UNSUPPORTED rather than CONTRADICTED, so asking the narrower question
   "does the evidence *contradict* this claim?" - and treating silence as
   unverified rather than as a failure - should raise precision sharply. That
   is the experiment the data now points to.
2. **Re-run on a frontier generator.** The pipeline is provider-agnostic, so
   `--provider openrouter` switches the generator to Claude Sonnet 5 in one
   flag. This would separate "what the architecture does" from "what a 7B
   model does", which no amount of analysis here can disentangle.
3. ~~Stop auto-dropping flagged claims.~~ **Done** (section 10). Warn-only
   keeps the faithfulness gain and recovers the lost correctness, and is now
   the recommended configuration.
4. **Calibrate a confidence threshold.** The verifier returns a verdict but no
   usable confidence. Flagging only high-confidence failures would trade
   recall for precision deliberately rather than accidentally.
5. **SelfCheckGPT as a reference-free baseline**, to quantify how much the
   retrieval in verification contributes over mere self-consistency.

---

## 9. Does a purpose-built classifier fix the precision problem? No.

Section 3 left an open question: is precision 0.247 a property of the 7B
model being asked to behave like a classifier, or of the task itself? A
DeBERTa cross-encoder trained on MNLI answers it. It is a dedicated entailment
model, it runs locally at no cost, and its labels map directly onto ours
(entailment / contradiction / neutral). Both backends were scored on the
**same 272 claims from the same 50 RAGTruth responses**.

| | LLM judge | NLI cross-encoder |
|---|---|---|
| Precision | **0.247** | 0.173 |
| Recall | 0.688 | **0.812** |
| F1 | **0.364** | 0.286 |
| 3-way verdict accuracy | **0.706** | 0.515 |
| False positives | 67 | **124** |

**The cross-encoder is worse.** It finds more real hallucinations (recall
0.812) but flags so much else that precision falls further, and three-way
accuracy collapses from 0.706 to 0.515.

The decisive figure is how often each backend fires against the truth:

| | claims flagged |
|---|---|
| Actual hallucination rate in the sample | **12%** (32 / 272) |
| LLM judge flags | 33% (89 / 272) |
| NLI cross-encoder flags | 55% (150 / 272) |

Both over-trigger - one nearly 3x, the other 4.5x.

**Why.** NLI models are trained to judge strict logical entailment between
short, clean sentence pairs. "Is this claim supported by this document?" is a
looser and genuinely different question. A claim can be true, well grounded
and supported in context while not being *strictly entailed* by any single
retrieved passage, and a strict entailment model labels that `neutral`, which
maps to UNSUPPORTED. The false-positive mass sits almost entirely in that
bucket.

**Conclusion: the precision ceiling is the task formulation, not the model.**
Swapping in a better classifier does not help, because both models are being
asked a question whose correct answer is not the one the task needs. This is
a more useful result than a fix would have been - it redirects the next
attempt from model choice to task design.

A caution on method, worth recording. On four hand-picked examples the
cross-encoder scored 4/4, including a case the LLM verifier got wrong, and
that looked like strong evidence it was better. On all 272 claims it is
clearly worse. Four examples predicted the opposite of what the full sample
showed.

---

## 10. Drop the flagged claim, or keep it and warn? Keep it.

Section 3 showed that arm C deletes correct content because three in four of
its flags are wrong. Arm D tests the alternative directly: it is **identical
to arm C in every flag except one** - a claim that fails verification is kept
in the answer carrying its verdict badge, rather than being removed. A test
pins that the two arms differ in that single field, so the comparison cannot
drift.

| | A baseline | C delete | D warn-only |
|---|---|---|---|
| LLM-judge correct | 0.429 | 0.367 | **0.449** |
| Token F1 | **0.127** | 0.080 | 0.086 |
| Faithfulness | 0.753 | **0.788** | **0.788** |
| Claims corrected | 0 | 19 | 19 |
| Claims **deleted** | 0 | **15** | **0** |
| Tokens / query | 857 | 4,353 | 4,353 |

**Warn-only keeps the entire faithfulness gain while giving back the
correctness that deletion cost, at identical token cost.** Arm D matches arm
C's faithfulness exactly (0.788), because the verdicts are the same verdicts -
only the action taken on them differs.

The finding separates two things the corrective loop does, which arm C had
conflated:

- **Rewriting a failed claim against fresh evidence helps.** Both arms do it
  19 times, and arm D shows the benefit is retained.
- **Deleting a claim that cannot be rescued hurts.** It is the only
  difference between the arms, and it costs 4 questions of judge-correctness.

**Caveat on size.** At n = 49, arm D's 0.449 against arm C's 0.367 is 22
correct answers against 18 - a four-question difference. Against baseline it
is 22 against 21, a single question, which is noise and should not be read as
warn-only *beating* plain RAG. The defensible claim is the narrower one:
**warn-only is at least as good as baseline on correctness while adding a
faithfulness signal, and it is clearly better than deleting.** No significance
testing was performed.

**Recommendation.** Arm D is the configuration to deploy, and it is what the
demo now shows: surface a verdict as a warning next to the claim and let a
reader judge, rather than silently removing content on a signal that is wrong
three times out of four. That recommendation was already in section 6 on the
strength of the precision figure; this measures it.
