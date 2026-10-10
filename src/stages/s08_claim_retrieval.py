"""
Stage 8 - Per-claim evidence retrieval.

Each claim gets its OWN retrieval pass, using the verification query the
decomposer wrote for it.

WHY NOT JUST REUSE THE PASSAGES THE GENERATOR SAW?
--------------------------------------------------
Because that would only detect one kind of error. If the generator was given
five passages and invented a detail none of them contained, re-checking
against those same five passages would catch it. But if retrieval failed and
the right evidence was never fetched, reusing the generator's context would
mark a TRUE claim as UNSUPPORTED - and, worse, a claim that happens to be
loosely echoed in an irrelevant passage as ENTAILED.

Retrieving fresh evidence per claim, with a query written specifically for
that claim, is a genuinely independent check. It also means an UNSUPPORTED
verdict carries a clear meaning: "this is not in the corpus", not "I did not
look".
"""

from __future__ import annotations

from src.config import Config
from src.retrieval.hybrid import HybridRetriever
from src.types import Claim, Passage


def retrieve_for_claim(
    claim: Claim,
    retriever: HybridRetriever,
    cfg: Config,
) -> list[Passage]:
    """Fetch the top-k passages for one claim's verification query."""
    return retriever.search(claim.verification_query, k=cfg.retrieval.claim_top_k)


def retrieve_for_claims(
    claims: list[Claim],
    retriever: HybridRetriever,
    cfg: Config,
) -> dict[str, list[Passage]]:
    """
    Retrieve evidence for every claim. Keyed by claim text.

    Retrieval is local (FAISS + BM25 over ~10k passages, no network), so doing
    this once per claim costs milliseconds, not money. The expensive part of
    verification is the LLM call in Stage 9, not this.
    """
    return {c.text: retrieve_for_claim(c, retriever, cfg) for c in claims}
