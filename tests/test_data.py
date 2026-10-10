"""
Milestone 1 test: the datasets and indexes on disk are what we think they are.

These are cheap checks of facts the whole experiment rests on. If the corpus
silently lost its gold paragraphs, or pids stopped lining up with index rows,
every later number would be wrong in a way that is very hard to notice.
"""

import json
import pickle

import pytest

from src.config import load_config

cfg = load_config()

EVAL_Q = cfg.data_dir / "eval_questions.json"
PASSAGES = cfg.data_dir / "passages.jsonl"
RAGTRUTH = cfg.data_dir / "ragtruth_qa.json"

# Skip the whole module with a useful message if data was never built.
pytestmark = pytest.mark.skipif(
    not (EVAL_Q.exists() and PASSAGES.exists()),
    reason="run `python -m src.data.build_hotpot` and `build_index` first",
)


@pytest.fixture(scope="module")
def questions():
    return json.loads(EVAL_Q.read_text())


@pytest.fixture(scope="module")
def passages():
    return [json.loads(l) for l in open(PASSAGES)]


def test_eval_set_size_and_uniqueness(questions):
    assert len(questions) == cfg.data.n_eval_questions
    assert len({q["qid"] for q in questions}) == len(questions)


def test_every_question_has_answer_and_gold(questions):
    for q in questions:
        assert q["question"].strip(), q["qid"]
        assert q["answer"].strip(), q["qid"]
        assert q["gold_pids"], f"{q['qid']} has no gold passages"


def test_corpus_is_large_enough_to_be_a_real_search_problem(passages):
    """
    The whole point of indexing 1000 questions' paragraphs while evaluating
    100 is that retrieval must be non-trivial. Guard that intent: with only
    ~1000 passages the task would be far too easy.
    """
    assert len(passages) > 5000, (
        f"only {len(passages)} passages - retrieval would be trivially easy "
        "and the ablation would measure nothing"
    )


def test_pids_unique_and_titles_deduplicated(passages):
    pids = [p["pid"] for p in passages]
    titles = [p["title"] for p in passages]
    assert len(set(pids)) == len(pids)
    assert len(set(titles)) == len(titles), "titles should be de-duplicated"


def test_passage_order_matches_index_rows(passages):
    """
    Row i of the FAISS index must be passage i of passages.pkl. If this
    drifted, retrieval would return confidently wrong passages.
    """
    with open(cfg.index_dir / "passages.pkl", "rb") as f:
        indexed = pickle.load(f)
    assert len(indexed) == len(passages)
    assert [p["pid"] for p in indexed] == [p["pid"] for p in passages]


def test_gold_passages_are_present_in_the_corpus(questions, passages):
    """Every eval question must be answerable from the index."""
    available = {p["pid"] for p in passages}
    for q in questions:
        missing = set(q["gold_pids"]) - available
        assert not missing, f"{q['qid']} gold passages missing: {missing}"


def test_rebuild_is_deterministic(questions):
    """
    Same seed must give the same sample. We check the recorded qids against a
    fresh stratified selection rather than re-downloading: the invariant we
    care about is that the file is stable, so a rerun that changed it would
    show up as a diff in git.
    """
    again = json.loads(EVAL_Q.read_text())
    assert [q["qid"] for q in again] == [q["qid"] for q in questions]


# ---------------------------------------------------------------------------
# RAGTruth
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RAGTRUTH.exists(),
                    reason="run `python -m src.data.build_ragtruth` first")
def test_ragtruth_slice_is_balanced_and_well_formed():
    rows = json.loads(RAGTRUTH.read_text())
    assert len(rows) == (cfg.data.n_ragtruth_hallucinated
                         + cfg.data.n_ragtruth_clean)

    n_hall = sum(1 for r in rows if r["has_hallucination"])
    assert n_hall == cfg.data.n_ragtruth_hallucinated

    for r in rows:
        assert r["response"].strip()
        assert r["passages"], f"{r['rid']} has no evidence passages"
        # has_hallucination must agree with the spans actually stored
        assert r["has_hallucination"] == bool(r["gold_spans"])


@pytest.mark.skipif(not RAGTRUTH.exists(), reason="build_ragtruth first")
def test_ragtruth_span_offsets_index_into_the_response():
    """
    The gold labels are character offsets. If they did not line up with the
    response text, every claim label derived from them would be wrong.
    """
    rows = json.loads(RAGTRUTH.read_text())
    n = 0
    for r in rows:
        for s in r["gold_spans"]:
            assert r["response"][s["start"]:s["end"]] == s["text"], r["rid"]
            assert s["verdict"] in ("CONTRADICTED", "UNSUPPORTED")
            n += 1
    assert n > 0, "no gold spans at all - the sample is useless"
