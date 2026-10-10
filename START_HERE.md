# Start here

Five commands, in order. Total setup time: about five minutes, most of it
downloading the datasets.

## 0. Prerequisites

- Python 3.11
- An OpenRouter API key with a few dollars of credit: https://openrouter.ai/keys

## 1. Install

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

(No `uv`? Use `python3.11 -m venv .venv && source .venv/bin/activate &&
pip install -r requirements.txt`.)

## 2. Add your API key

```bash
cp .env.example .env
```

Open `.env` and paste your key over the placeholder. `.env` is gitignored and
must never be committed.

## 3. Build the data (no API key needed, ~3 minutes)

```bash
python -m src.data.build_hotpot      # 100 eval questions + 9,735 passages
python -m src.data.build_ragtruth    # 50 human-labelled responses
python -m src.data.build_index       # FAISS + BM25 (~30s on an M1)
```

## 4. Check everything works (free)

```bash
pytest -q                                        # should be all green
python -m src.run_eval --arm all --n 3 --mock    # free: no API calls at all
```

`--mock` runs the entire pipeline against canned replies. The numbers are
meaningless, but if this completes, the plumbing is sound.

## 5. Run it for real

```bash
python -m src.run_eval --arm all --n 100   # the full ablation, about $3
python -m src.run_ragtruth                 # verifier vs human labels, ~$0.40
python -m src.report                       # tables + plots into runs/
```

Start smaller if you want to watch it first: `--arm C --n 5`.

## 6. The demo

```bash
streamlit run src/app.py
```

Pick arm C, ask one of the example questions, and open a claim's evidence
expander. That is the demo moment: a specific sentence flagged, with the
passage that justified the flag.

---

## Where to read next

| File | What it gives you |
|---|---|
| `EXPLANATION.md` | Every file explained, and the concepts behind them |
| `VIVA.md` | Likely examiner questions, with answers |
| `ANALYSIS.md` | The results, and what they mean |
| `README.md` | Architecture diagram and the short overview |
| `CLAUDE.md` | Conventions, including the asymmetry rule |

## If something breaks

| Symptom | Fix |
|---|---|
| `OPENROUTER_API_KEY is not set` | Step 2, or add `--mock` |
| `dense.faiss not found` | Run step 3 |
| `eval_questions.json not found` | Run step 3 |
| Tests skipped with "run build_hotpot first" | Run step 3 |
| A run costs more than expected | The cache is per-prompt; check you did not pass `--no-cache` |
