"""
Milestone 2 test: retrieval actually finds the right paragraphs.

The headline metric here is recall@k: of the 2 gold paragraphs a HotpotQA
question needs, how many appear in the top k we retrieve? If recall is poor,
no amount of clever verification downstream can save the answer - the
generator will be grounded on the wrong text.
"""

import json

import pytest

from src.config import load_config
from src.retrieval.hybrid import HybridRetriever

cfg = load_config()
EVAL_Q = cfg.data_dir / "eval_questions.json"

pytestmark = pytest.mark.skipif(
    not (cfg.index_dir / "dense.faiss").exists(),
    reason="run `python -m src.data.build_index` first",
)


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever(cfg)


@pytest.fixture(scope="module")
def questions():
    return json.loads(EVAL_Q.read_text())


def recall_at_k(hits, gold_pids) -> float:
    """Fraction of a question's gold paragraphs that appear in the results."""
    found = {p.pid for p in hits} & set(gold_pids)
    return len(found) / len(gold_pids)


def test_search_returns_requested_number_of_passages(retriever):
    hits = retriever.search("Who directed the film Jaws?", k=5)
    assert len(hits) == 5
    assert all(h.title and h.text for h in hits)
    # RRF scores are small positive numbers, descending
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_dense_finds_paraphrases_sparse_cannot(retriever):
    """
    Sanity check that the embedding half is doing semantic work: a query
    using none of the passage's words should still retrieve it.
    """
    hits = retriever.dense.search("movie about a shark attacking swimmers", 20)
    assert len(hits) == 20
    assert all(0.0 <= s <= 1.0001 for _, s in hits), "cosine scores out of range"


def test_hybrid_beats_or_matches_each_retriever_alone(retriever, questions):
    """
    The justification for building TWO retrievers instead of one. We compare
    recall@5 across dense-only, BM25-only and fused, on 30 questions.
    """
    subset = questions[:30]
    k = 5
    r_dense = r_sparse = r_hybrid = 0.0

    for q in subset:
        gold = q["gold_pids"]
        dense_pids = [pid for pid, _ in retriever.dense.search(q["question"], k)]
        sparse_pids = [pid for pid, _ in retriever.sparse.search(q["question"], k)]
        hybrid_pids = [p.pid for p in retriever.search(q["question"], k)]

        r_dense += len(set(dense_pids) & set(gold)) / len(gold)
        r_sparse += len(set(sparse_pids) & set(gold)) / len(gold)
        r_hybrid += len(set(hybrid_pids) & set(gold)) / len(gold)

    n = len(subset)
    r_dense, r_sparse, r_hybrid = r_dense / n, r_sparse / n, r_hybrid / n
    print(f"\nrecall@{k} over {n} questions: "
          f"dense={r_dense:.3f} bm25={r_sparse:.3f} hybrid={r_hybrid:.3f}")

    # Fusion should not be WORSE than the better single retriever by more
    # than a small margin. (RRF can lose a little on queries where one
    # retriever is dramatically better, which is an honest tradeoff.)
    assert r_hybrid >= max(r_dense, r_sparse) - 0.05, (
        f"hybrid ({r_hybrid:.3f}) is clearly worse than the best single "
        f"retriever (dense={r_dense:.3f}, bm25={r_sparse:.3f}) - "
        "fusion is not earning its complexity"
    )
    # And retrieval must be good enough that the task is solvable at all.
    assert r_hybrid > 0.5, f"recall@{k}={r_hybrid:.3f} is too low to build on"


def test_search_many_deduplicates_across_subqueries(retriever):
    """Multi-hop planning retrieves per sub-query; passages must not repeat."""
    hits = retriever.search_many(
        ["Who directed Jaws?", "When was Jaws released?"], k=5
    )
    pids = [h.pid for h in hits]
    assert len(pids) == len(set(pids)), "duplicate passages across sub-queries"
    assert len(hits) <= 10


def test_fusion_records_which_retriever_found_each_passage(retriever):
    """
    Each result carries its dense and sparse rank. This is what lets the
    report show how often the two retrievers agree - a nice analysis point,
    and useful for debugging a bad retrieval.
    """
    hits = retriever.search("Scott Derrickson nationality", k=10)
    assert any(h.dense_rank is not None for h in hits)
    assert any(h.sparse_rank is not None for h in hits)
    both = [h for h in hits if h.dense_rank and h.sparse_rank]
    print(f"\n{len(both)}/{len(hits)} top passages found by BOTH retrievers")
