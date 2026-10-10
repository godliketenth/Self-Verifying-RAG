"""
Dense (embedding-based) retrieval - Stage 4, half of the hybrid retriever.

How it works: every passage was turned into a 384-dimensional vector by
bge-small at index time. A query is turned into a vector the same way, and we
return the passages whose vectors point in the most similar direction.

Because every vector was L2-normalised when indexed, the inner product that
FAISS computes IS the cosine similarity, so scores are interpretable: 1.0 is
identical direction, 0.0 is unrelated.
"""

from __future__ import annotations

import json
import pickle

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.config import Config


class DenseRetriever:
    """Loads the FAISS index once and answers queries against it."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        idx_dir = cfg.index_dir
        if not (idx_dir / "dense.faiss").exists():
            raise SystemExit(
                f"{idx_dir}/dense.faiss not found. "
                "Run `python -m src.data.build_index` first."
            )
        self.index = faiss.read_index(str(idx_dir / "dense.faiss"))
        with open(idx_dir / "passages.pkl", "rb") as f:
            self.passages: list[dict] = pickle.load(f)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(cfg.retrieval.embedding_model,
                                         device=device)

        meta = json.loads((idx_dir / "meta.json").read_text())
        # Querying an index built by a different model would return silent
        # nonsense rather than an error, so check it explicitly.
        if meta["embedding_model"] != cfg.retrieval.embedding_model:
            raise RuntimeError(
                f"Index was built with {meta['embedding_model']} but config "
                f"says {cfg.retrieval.embedding_model}. Rebuild the index."
            )

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query.

        bge models were trained with an instruction prefix on QUERIES only
        (never on passages). Omitting it measurably hurts retrieval, which is
        why the prefix lives in config.yaml rather than being forgotten here.
        """
        text = self.cfg.retrieval.query_prefix + query
        vec = self.model.encode([text], convert_to_numpy=True,
                                normalize_embeddings=True)
        return vec.astype(np.float32)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return the k most similar passages as (pid, cosine_score)."""
        scores, idxs = self.index.search(self.embed_query(query), k)
        out: list[tuple[str, float]] = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:            # FAISS pads with -1 when k > corpus size
                continue
            out.append((self.passages[i]["pid"], float(score)))
        return out
