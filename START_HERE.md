# Start here

Five steps from a clone to a running system. Most of the time is dataset
downloads; nothing here costs money.

## 0. Prerequisites

- Python 3.11
- [Ollama](https://ollama.com) for local inference (no API key, no quota)

```bash
ollama serve &                      # start the local model server
ollama pull qwen2.5:7b-instruct     # ~4.7GB, the generator and helper model
```

**Why a local model?** Not preference - necessity. OpenRouter's free tier
allows 50 requests/day and Gemini caps the relevant model at 20/day, against
roughly 1,150 calls needed for one 50-question ablation. Local inference has
no quota. Model choice matters too: Qwen3 and Gemma 4 are *reasoning* models
that spend thousands of characters thinking before answering (~83s per call,
sometimes producing no answer at all). `qwen2.5-instruct` predates that shift
and just answers.

## 1. Install

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

(No `uv`? `python3.11 -m venv .venv && source .venv/bin/activate &&
pip install -r requirements.txt`.)

## 2. API key — only if you switch providers

The default provider is `ollama`, which needs **no key at all**. Skip this
step unless you want a hosted model.

```bash
cp .env.example .env     # then paste a key for whichever provider you use
```

The project is provider-agnostic. `config.yaml` holds a `providers` map and
one flag switches the whole stack:

```bash
python -m src.run_eval --arm all --n 50                      # local (default)
python -m src.run_eval --arm all --n 50 --provider gemini    # needs GEMINI_API_KEY
python -m src.run_eval --arm all --n 50 --provider openrouter # needs credits
```

## 3. Build the data (~3 minutes, no model needed)

```bash
python -m src.data.build_hotpot          # 100 eval questions + 9,735 passages
python -m src.data.build_ragtruth        # 50 human-labelled responses
python -m src.data.build_index           # FAISS + BM25 (~30s on an M1)
python -m src.data.build_demo_questions  # demo questions with no indexed evidence
```

## 4. Check it works (free, no model calls)

```bash
pytest -q                                     # 63 tests, all should pass
python -m src.run_eval --arm all --n 3 --mock # canned replies, zero cost
```

`--mock` runs the whole pipeline against fixed replies. The numbers are
meaningless; if it completes, the plumbing is sound.

## 5. Run the experiment

```bash
python -m src.run_eval --arm all --n 50 --no-cache   # arms A, B, C (~35 min)
python -m src.run_eval --arm D   --n 50 --no-cache   # warn-only variant
python -m src.run_ragtruth                           # verifier vs human labels
python -m src.run_ragtruth --verifier nli            # the NLI backend, for comparison
python -m src.report                                 # tables + plots into runs/
python -m src.analyze_cascade                        # intermediate-verification evidence
```

Use `--no-cache` for anything where **latency** matters: arms B and C share
stages, so a warm cache makes arm C look faster than it is. Quality metrics
are unaffected by caching.

Start smaller to watch it first: `--arm C --n 5`.

## 6. The demo

```bash
streamlit run src/app.py
```

Pick **arm D**, open the "Verification CATCHES a claim" tab, and run one of
the listed questions. Those are mined from the measured run, so they
reproduce. Expand a flagged claim to see the evidence that was examined.

---

## Where to read next

| File | What it gives you |
|---|---|
| `ANALYSIS.md` | **The results and what they mean. Read this first.** |
| `VIVA.md` | Likely examiner questions, with answers |
| `EXPLANATION.md` | Every file explained, and the concepts behind them |
| `README.md` | Overview and architecture diagram |
| `CLAUDE.md` | Conventions, including the asymmetry rule |

## If something breaks

| Symptom | Fix |
|---|---|
| `Connection refused` on localhost:11434 | `ollama serve &` |
| `model not found` | `ollama pull qwen2.5:7b-instruct` |
| `dense.faiss not found` | Run step 3 |
| `eval_questions.json not found` | Run step 3 |
| Tests skipped with "run build_hotpot first" | Run step 3 |
| `GEMINI_API_KEY is not set` | You passed `--provider gemini`; either set the key or drop the flag |
| Run aborts: "N questions failed in a row" | Infrastructure, not a result - check the model server or your quota |
