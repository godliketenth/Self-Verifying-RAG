"""
Milestone 0 test: the LLM cache and the accounting are trustworthy.

Why this test matters: every later milestone depends on being able to re-run
an experiment cheaply. If caching silently failed, a 100-question re-run would
cost real money each time, and the project's cost figures would be wrong.
"""

from src.llm import LLMClient, extract_json
from src.types import StageLog


def test_mock_mode_needs_no_api_key(cfg):
    """The whole pipeline must be runnable with zero credentials."""
    cfg.llm.api_key = ""
    client = LLMClient(cfg, mock=True)
    resp = client.complete("anything", stage="s06_generator")
    assert resp.text
    assert resp.cost_usd == 0.0


def test_cache_hit_on_identical_call(cfg):
    """
    A repeated identical call must be served from disk: same text, flagged as
    a hit, and costing nothing.
    """
    client = LLMClient(cfg, mock=True)
    client.mock = False                       # exercise the cache path...
    client._client = None                     # ...without a real network client

    key = client._cache_key("m", "", "p", 0.0, 10)
    client._cache_write(key, {"text": "cached reply", "model": "m",
                              "tokens_in": 11, "tokens_out": 7})

    resp = client.complete("p", stage="s02_router", model="m", max_tokens=10)
    assert resp.cache_hit is True
    assert resp.text == "cached reply"
    assert resp.cost_usd == 0.0               # no money spent on a hit
    assert resp.tokens_in == 11               # but tokens still reported
    assert client.n_cache_hits == 1


def test_cache_key_changes_with_prompt_and_model(cfg):
    """Editing a prompt must invalidate only that stage's cache."""
    c = LLMClient(cfg, mock=True)
    base = c._cache_key("m1", "sys", "prompt", 0.0, 100)
    assert base != c._cache_key("m1", "sys", "prompt CHANGED", 0.0, 100)
    assert base != c._cache_key("m2", "sys", "prompt", 0.0, 100)
    assert base != c._cache_key("m1", "sys", "prompt", 0.7, 100)
    assert base == c._cache_key("m1", "sys", "prompt", 0.0, 100)   # stable


def test_cost_accounting_matches_pricing():
    """
    1M input tokens must cost exactly the listed input price, per provider.

    Loaded per-provider rather than from the default config, so this keeps
    testing real pricing arithmetic whichever provider is active.
    """
    from src.config import load_config
    orc = load_config(provider="openrouter").llm
    assert abs(orc.price_of("anthropic/claude-sonnet-5", 1_000_000, 0) - 2.0) < 1e-9
    assert abs(orc.price_of("anthropic/claude-haiku-4.5", 0, 1_000_000) - 5.0) < 1e-9
    # An unknown model must cost 0, never a guessed number.
    assert orc.price_of("some/unknown-model", 1000, 1000) == 0.0

    # A free tier is genuinely zero - not "unknown". Token counts, not
    # dollars, are what make the arms comparable there.
    gem = load_config(provider="gemini").llm
    assert gem.price_of(gem.generator_model, 1_000_000, 1_000_000) == 0.0


def test_provider_switching_changes_endpoint_and_models():
    """--provider must swap the whole recipe, not just the model name."""
    from src.config import load_config
    g = load_config(provider="gemini").llm
    o = load_config(provider="openrouter").llm
    assert g.base_url != o.base_url
    assert g.generator_model != o.generator_model
    assert g.api_key_env == "GEMINI_API_KEY"
    assert o.api_key_env == "OPENROUTER_API_KEY"
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match="Unknown provider"):
        load_config(provider="not-a-provider")


def test_shadow_calls_excluded_from_arm_cost(cfg):
    """
    The heart of the faithfulness design: measurement calls must not be
    counted as the arm's own cost.
    """
    from src.types import PipelineResult
    r = PipelineResult(question="q", arm="A")
    r.stage_logs = [
        StageLog(stage="s06_generator", cost_usd=0.003, tokens_in=100, tokens_out=50),
        StageLog(stage="s09_verifier", cost_usd=0.002, tokens_in=80, tokens_out=20,
                 is_shadow=True),
    ]
    assert abs(r.total_cost_usd() - 0.003) < 1e-9              # shadow excluded
    assert abs(r.total_cost_usd(include_shadow=True) - 0.005) < 1e-9
    assert r.n_llm_calls() == 1
    assert r.n_llm_calls(include_shadow=True) == 2


def test_complete_json_parses_mock_replies(cfg):
    """Every mock stage reply must be valid JSON where the stage expects it."""
    client = LLMClient(cfg, mock=True)
    for stage in ("s02_router", "s03_planner", "s07_decomposer", "s09_verifier"):
        data, _ = client.complete_json("p [p00001]", stage=stage)
        assert isinstance(data, dict)


def test_take_logs_resets_between_queries(cfg):
    """Logs are per-query; a leak would corrupt per-query cost figures."""
    client = LLMClient(cfg, mock=True)
    client.complete("a", stage="s06_generator")
    client.complete("b", stage="s06_generator")
    assert len(client.take_logs()) == 2
    assert client.take_logs() == []


def test_latency_excludes_shadow_calls_like_cost_does():
    """
    REGRESSION. Arm B's wall clock included shadow verification - calls made
    only to measure its faithfulness, never made in deployment. That made arm
    B appear SLOWER than arm C in a live run, although C does strictly more
    work, and it contradicted the cost column, which already excluded shadow.
    """
    from src.types import PipelineResult, StageLog
    r = PipelineResult(question="q", arm="B")
    r.total_latency_ms = 10_000.0
    r.stage_logs = [
        StageLog(stage="s06_generator", latency_ms=4_000.0),
        StageLog(stage="s09_verifier", latency_ms=3_000.0, is_shadow=True),
    ]
    assert r.latency_ms() == 7_000.0                     # shadow removed
    assert r.latency_ms(include_shadow=True) == 10_000.0
    # Retrieval time (wall clock not attributable to any LLM call) is kept:
    # 10000 - 3000 shadow = 7000, which is more than the 4000 of LLM time.
    assert r.latency_ms() > 4_000.0
