"""
Hybrid retrieval - Stage 4 proper. Fuses the dense and sparse rankings.

THE FUSION METHOD: Reciprocal Rank Fusion (RRF)
-----------------------------------------------
BM25 scores are unbounded (they can be 3.0 or 40.0 depending on the query);
cosine scores sit in roughly [0, 1]. Adding them directly would let BM25
drown out the embeddings, and normalising them per-query is fragile because
the scale shifts with every query.

RRF sidesteps the problem by throwing the scores away and using only the
RANKS:

    rrf_score(passage) = sum over retrievers of  1 / (k + rank)

with rank starting at 1 and k=60 (the value from the original RRF paper,
Cormack et al. 2009). A passage ranked 1st by both retrievers scores
2/(60+1) = 0.0328; one ranked 1st by only one scores 0.0164. So agreement
between the two retrievers is rewarded, and no retriever's score scale can
dominate. The constant k damps the difference between top ranks, preventing a
single retriever's #1 from automatically winning.
"""

from __future__ import annotations

import pickle

from src.config import Config
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.types import Passage


class HybridRetriever:
    """
    The single retrieval interface the rest of the pipeline uses.

    Both the main retrieval (Stage 4) and the per-claim evidence retrieval
    (Stage 8) call this same object - they differ only in `k`. Reusing one
    retriever means claim verification searches the same corpus the generator
    drew from, so an UNSUPPORTED verdict means "not in this corpus", not "I
    looked somewhere else".
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.dense = DenseRetriever(cfg)
        self.sparse = SparseRetriever(cfg)
        with open(cfg.index_dir / "passages.pkl", "rb") as f:
            self.by_pid: dict[str, dict] = {
                p["pid"]: p for p in pickle.load(f)
            }

    def search(self, query: str, k: int | None = None) -> list[Passage]:
        """
        Retrieve the top-k passages for one query using RRF fusion.

        We ask each retriever for `candidates_per_retriever` results (default
        20) and fuse those deeper lists before cutting to k. Fusing only the
        top-k of each would throw away exactly the passages where the two
        retrievers disagree, which is where fusion earns its keep.
        """
        k = k or self.cfg.retrieval.top_k
        depth = self.cfg.retrieval.candidates_per_retriever
        rrf_k = self.cfg.retrieval.rrf_k

        dense_hits = self.dense.search(query, depth)
        sparse_hits = self.sparse.search(query, depth)

        # rank is 1-based, as RRF assumes
        dense_rank = {pid: r for r, (pid, _) in enumerate(dense_hits, start=1)}
        sparse_rank = {pid: r for r, (pid, _) in enumerate(sparse_hits, start=1)}

        fused: dict[str, float] = {}
        for ranks in (dense_rank, sparse_rank):
            for pid, rank in ranks.items():
                fused[pid] = fused.get(pid, 0.0) + 1.0 / (rrf_k + rank)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]

        results: list[Passage] = []
        for pid, score in ordered:
            rec = self.by_pid[pid]
            results.append(Passage(
                pid=pid, title=rec["title"], text=rec["text"],
                score=score,
                dense_rank=dense_rank.get(pid),
                sparse_rank=sparse_rank.get(pid),
            ))
        return results

    def search_many(self, queries: list[str], k: int | None = None) -> list[Passage]:
        """
        Retrieve for several sub-queries and merge into one list.

        Used when the Stage 3 planner has split a multi-hop question. We keep
        each passage's BEST score across sub-queries and de-duplicate, because
        a passage relevant to two sub-queries should appear once, not twice.
        """
        k = k or self.cfg.retrieval.top_k
        best: dict[str, Passage] = {}
        for q in queries:
            for p in self.search(q, k):
                prev = best.get(p.pid)
                if prev is None or p.score > prev.score:
                    best[p.pid] = p
        merged = sorted(best.values(), key=lambda p: p.score, reverse=True)
        # Allow more passages through than a single query would yield, since a
        # multi-hop question genuinely needs evidence from several articles -
        # but cap it so the generator's context stays bounded.
        return merged[: k * 2]
