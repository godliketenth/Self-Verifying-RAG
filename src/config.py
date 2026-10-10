"""
Loads config.yaml into typed dataclasses, and defines the three ablation arms.

The important idea in this file is the `Arm` dataclass. The three arms of the
experiment are NOT three different implementations - they are the same
pipeline with different boolean flags. That is what makes the comparison fair:
if arm C were a separate code path, any difference we measured could be an
artefact of that code rather than of verification itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root = the directory containing config.yaml (one level above src/).
ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# The ablation arms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Arm:
    """
    One experimental condition. Every flag here switches a pipeline stage on
    or off; nothing else differs between arms.

    verify_shadow is the subtle one. To compare faithfulness across arms we
    need claim verdicts for arms A and B too - but those arms must not ACT on
    the verdicts, or they would secretly become arm C. So shadow mode runs the
    verifier purely to record verdicts, never to trigger correction, and tags
    those LLM calls with is_shadow=True so they are excluded from the arm's
    reported cost.
    """
    name: str
    description: str
    router: bool              # stage 2: complexity routing
    planner: bool             # stage 3: query decomposition into sub-queries
    relevance: bool           # stage 5: CRAG-style relevance filter + re-plan
    verify: bool              # stages 7-9 acting on the answer (correction on)
    verify_shadow: bool       # stages 7-9 for measurement only (no correction)
    correction_rounds: int    # stage 9 -> 8 loop; 0 disables it


ARMS: dict[str, Arm] = {
    "A": Arm(
        name="A",
        description="Baseline RAG: retrieve once, generate once.",
        router=False, planner=False, relevance=False,
        verify=False, verify_shadow=True, correction_rounds=0,
    ),
    "B": Arm(
        name="B",
        description="Agentic only: routing, planning and corrective retrieval, "
                    "but the answer itself is never audited.",
        router=True, planner=True, relevance=True,
        verify=False, verify_shadow=True, correction_rounds=0,
    ),
    "C": Arm(
        name="C",
        description="Full system: agentic retrieval plus the claim-level "
                    "verification loop.",
        router=True, planner=True, relevance=True,
        verify=True, verify_shadow=False, correction_rounds=1,
    ),
}


# ---------------------------------------------------------------------------
# config.yaml -> typed sections
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    base_url: str
    generator_model: str
    helper_model: str
    temperature: float
    max_tokens: int
    timeout_s: int
    max_retries: int
    pricing: dict[str, dict[str, float]]
    api_key: str = ""

    def price_of(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """USD cost of one call. Unknown models cost 0 and are not guessed at."""
        p = self.pricing.get(model)
        if not p:
            return 0.0
        return (tokens_in * p["input"] + tokens_out * p["output"]) / 1_000_000


@dataclass
class RetrievalConfig:
    embedding_model: str
    embedding_dim: int
    query_prefix: str
    top_k: int
    claim_top_k: int
    candidates_per_retriever: int
    rrf_k: int


@dataclass
class DataConfig:
    hotpot_dataset: str
    hotpot_config: str
    hotpot_split: str
    n_eval_questions: int
    n_corpus_questions: int
    ragtruth_dataset: str
    n_ragtruth_hallucinated: int
    n_ragtruth_clean: int
    span_overlap_threshold: float


@dataclass
class PipelineConfig:
    max_subqueries: int
    max_claims: int
    claim_dedup_threshold: float
    relevance_threshold: float
    min_relevant_passages: int
    max_replan_rounds: int
    max_correction_rounds: int


@dataclass
class Config:
    seed: int
    llm: LLMConfig
    retrieval: RetrievalConfig
    data: DataConfig
    pipeline: PipelineConfig
    paths: dict[str, str]

    # --- absolute paths, resolved against the project root ----------------
    def path(self, key: str) -> Path:
        # `ROOT / "/abs/path"` correctly yields "/abs/path", so absolute
        # values in config (used by tests) pass through unchanged.
        return ROOT / self.paths[key]

    @property
    def index_dir(self) -> Path: return self.path("index_dir")
    @property
    def data_dir(self) -> Path: return self.path("data_dir")
    @property
    def runs_dir(self) -> Path: return self.path("runs_dir")
    @property
    def cache_dir(self) -> Path: return self.path("cache_dir")


def load_config(path: str | Path | None = None, require_api_key: bool = False) -> Config:
    """
    Read config.yaml (and .env for the API key) into a Config object.

    require_api_key is False by default so that data building, index building
    and the mock-LLM tests all work with no key present at all.
    """
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if require_api_key and not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set.\n"
            "Fix: cp .env.example .env, then paste your key from "
            "https://openrouter.ai/keys into it."
        )

    cfg = Config(
        seed=raw["seed"],
        llm=LLMConfig(**raw["llm"], api_key=api_key),
        retrieval=RetrievalConfig(**raw["retrieval"]),
        data=DataConfig(**raw["data"]),
        pipeline=PipelineConfig(**raw["pipeline"]),
        paths=raw["paths"],
    )
    # Create the directories we are going to write into.
    for key in ("data_dir", "index_dir", "runs_dir", "cache_dir"):
        cfg.path(key).mkdir(parents=True, exist_ok=True)
    return cfg


def get_arm(name: str) -> Arm:
    """Look up an arm by letter, with a readable error for a typo."""
    key = name.strip().upper()
    if key not in ARMS:
        raise ValueError(f"Unknown arm {name!r}. Choose one of: {sorted(ARMS)}")
    return ARMS[key]
