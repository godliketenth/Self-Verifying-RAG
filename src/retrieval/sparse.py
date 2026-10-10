"""
Sparse (lexical) retrieval with BM25 - Stage 4, the other half.

BM25 scores a passage by how many of the query's words it contains, weighted
so that rare words count for more than common ones, and so that repeating a
word has diminishing returns. It has no notion of meaning: "film director"
will not match "movie-maker". But it is excellent at the thing embeddings are
worst at - rare exact strings like proper nouns, years and numbers - which is
precisely why pairing them works.
"""

from __future__ import annotations

import pickle

from src.config import Config
from src.data.build_index import tokenize   # same tokeniser as at index time


class SparseRetriever:
    """Loads the fitted BM25 model once and answers queries against it."""

    def __init__(self, cfg: Config) -> None:
        idx_dir = cfg.index_dir
        if not (idx_dir / "sparse_bm25.pkl").exists():
            raise SystemExit(
                f"{idx_dir}/sparse_bm25.pkl not found. "
                "Run `python -m src.data.build_index` first."
            )
        with open(idx_dir / "sparse_bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(idx_dir / "passages.pkl", "rb") as f:
            self.passages: list[dict] = pickle.load(f)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return the k highest-BM25 passages as (pid, bm25_score)."""
        scores = self.bm25.get_scores(tokenize(query))
        # argsort descending, take k. The corpus is ~10k so a full sort is
        # cheap and keeps the code obvious.
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.passages[i]["pid"], float(scores[i])) for i in top]
