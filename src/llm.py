"""
The single place where this project talks to an LLM.

Three jobs, and nothing else:

1. CALL   - one function sends a prompt to OpenRouter and returns the text.
2. CACHE  - identical calls are served from disk, so re-running a 100-question
            experiment after fixing a plotting bug costs nothing. This is the
            difference between an affordable project and an expensive one.
3. COUNT  - every call records its tokens, cost and latency, because "what did
            verification cost?" is one of the questions the project answers.

Note on the client: OpenRouter exposes an OpenAI-COMPATIBLE endpoint, so we
use the `openai` SDK pointed at openrouter.ai. The models behind it are
Anthropic's (Sonnet 5 and Haiku 4.5), but the wire format is OpenAI's.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import Config
from src.types import LLMResponse, StageLog


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> Any:
    """
    Pull a JSON value out of an LLM response.

    Models wrap JSON in markdown fences, add a sentence of preamble, or both,
    even when told not to. Rather than fight that with ever-stricter prompts,
    we parse tolerantly: try the whole string, then any fenced block, then the
    outermost {...} or [...] we can find.

    Raises ValueError if nothing parses - callers retry, and a persistent
    failure is recorded as a metric rather than hidden.
    """
    candidates: list[str] = [text.strip()]

    m = _FENCE_RE.search(text)
    if m:
        candidates.append(m.group(1).strip())

    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start:end + 1])

    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"No parseable JSON in response: {text[:300]!r}")


# ---------------------------------------------------------------------------
# Mock backend (zero cost, deterministic)
# ---------------------------------------------------------------------------

class MockBackend:
    """
    Canned replies keyed by stage name, so the entire pipeline can be run and
    tested without an API key or a single cent of spend. Used by
    `--mock` and by the unit tests.

    These are deliberately crude. Their job is to prove the PLUMBING works
    (data flows between stages, loops fire, logs accumulate), not to produce
    good answers.
    """

    def reply(self, stage: str, prompt: str) -> str:
        if "router" in stage:
            return '{"route": "MULTI_HOP", "reason": "mock: two entities"}'
        if "planner" in stage:
            return '{"subqueries": ["mock sub-query one", "mock sub-query two"]}'
        if "relevance" in stage:
            # Keep every passage; ids are discovered from the prompt text.
            pids = re.findall(r"\[(p\d+)\]", prompt)
            items = ", ".join(
                f'{{"pid": "{p}", "score": 0.9, "keep": true}}' for p in pids
            ) or '{"pid": "p00000", "score": 0.9, "keep": true}'
            return f'{{"passages": [{items}]}}'
        if "generator" in stage:
            return "Mock draft answer. The first fact is true. The second fact is invented."
        if "decomposer" in stage:
            return (
                '{"claims": ['
                '{"claim": "The first fact is true.", "query": "first fact", '
                '"source_sentence": "The first fact is true."},'
                '{"claim": "The second fact is invented.", "query": "second fact", '
                '"source_sentence": "The second fact is invented."}'
                "]}"
            )
        if "verifier" in stage:
            # Fail the "invented" claim so the correction loop gets exercised.
            if "invented" in prompt:
                return ('{"verdict": "UNSUPPORTED", "evidence_pid": null, '
                        '"evidence_span": "", "reasoning": "mock: no evidence"}')
            return ('{"verdict": "ENTAILED", "evidence_pid": "p00000", '
                    '"evidence_span": "mock span", "reasoning": "mock: supported"}')
        if "judge" in stage:
            return '{"correct": true, "reasoning": "mock judge"}'
        if "corrector" in stage:
            return '{"claim": "Corrected mock claim.", "dropped": false}'
        return '{"ok": true}'


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    A simple sliding-window throttle: at most `rpm` requests in any 60s window.

    This exists because OpenRouter caps new accounts at 20 requests per minute
    per model, and this pipeline makes 6-9 calls per question back to back.
    Without a client-side throttle the run dies on HTTP 429 within seconds -
    which is exactly what happened on the first live attempt.

    Throttling here rather than relying on retries is deliberate: retrying
    after a 429 wastes a round trip and the provider may count the rejected
    request anyway. Pacing the requests avoids the error in the first place.
    """

    def __init__(self, rpm: int) -> None:
        self.rpm = max(1, rpm)
        self._times: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a request slot is free. Returns seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                # drop timestamps older than the window
                self._times = [t for t in self._times if now - t < 60.0]
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return waited
                sleep_for = 60.0 - (now - self._times[0]) + 0.05
            time.sleep(max(0.05, sleep_for))
            waited += max(0.05, sleep_for)


class InsufficientCredits(RuntimeError):
    """Raised on HTTP 402 so the run can stop immediately rather than burn
    through every remaining question producing identical failures."""


def _retry_after_seconds(err: Exception, default: float) -> float:
    """
    Honour a Retry-After header when the provider sends one.

    Guessing a backoff when the server has told us exactly how long to wait
    is how you get stuck in a retry loop that never succeeds.
    """
    for attr in ("response", "http_response"):
        resp = getattr(err, attr, None)
        headers = getattr(resp, "headers", None)
        if not headers:
            continue
        for key in ("retry-after", "Retry-After", "x-ratelimit-reset"):
            val = headers.get(key) if hasattr(headers, "get") else None
            if not val:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            # Some providers send an epoch-millis reset rather than a delay.
            if v > 1e11:
                v = max(0.0, v / 1000.0 - time.time())
            return min(max(v, 1.0), 90.0)
    return default


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Wraps OpenRouter with caching and accounting.

    Usage:
        client = LLMClient(cfg)
        resp = client.complete("Say hi", stage="demo")
        data, resp = client.complete_json(prompt, stage="s02_router")

    Every call appends a StageLog to `self.logs`. The pipeline reads that list
    at the end of a query to produce the per-query cost record.
    """

    def __init__(
        self,
        cfg: Config,
        use_cache: bool = True,
        mock: bool = False,
    ) -> None:
        self.cfg = cfg
        self.use_cache = use_cache
        self.mock = mock
        self.cache_dir: Path = cfg.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs: list[StageLog] = []
        self._mock_backend = MockBackend()
        self._limiter = RateLimiter(cfg.llm.requests_per_minute)
        self.throttle_wait_s = 0.0      # total time spent pacing requests

        # Counters for the run summary.
        self.n_calls = 0
        self.n_cache_hits = 0
        self.n_parse_failures = 0

        self._client: OpenAI | None = None
        if not mock:
            if not cfg.llm.api_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not set, and mock=False.\n"
                    "Fix: put your key in .env, or pass --mock to run free."
                )
            self._client = OpenAI(
                base_url=cfg.llm.base_url,
                api_key=cfg.llm.api_key,
                timeout=cfg.llm.timeout_s,
                max_retries=0,   # we retry ourselves, with logging
            )

    # -- cache ------------------------------------------------------------

    def _cache_key(self, model: str, system: str, prompt: str,
                   temperature: float, max_tokens: int) -> str:
        """
        sha256 over everything that could change the answer. Because the key
        includes the model and the prompt, editing a prompt automatically
        invalidates just that stage's cache - which is what you want.
        """
        blob = json.dumps(
            {"model": model, "system": system, "prompt": prompt,
             "temperature": temperature, "max_tokens": max_tokens},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_read(self, key: str) -> dict[str, Any] | None:
        f = self.cache_dir / f"{key}.json"
        if not f.exists():
            return None
        try:
            return json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            return None   # a corrupt cache entry is a miss, never a crash

    def _cache_write(self, key: str, payload: dict[str, Any]) -> None:
        try:
            (self.cache_dir / f"{key}.json").write_text(json.dumps(payload))
        except OSError:
            pass          # losing a cache write must never fail a run

    # -- the main call ----------------------------------------------------

    def complete(
        self,
        prompt: str,
        stage: str,
        model: str | None = None,
        system: str = "",
        max_tokens: int | None = None,
        is_shadow: bool = False,
    ) -> LLMResponse:
        """
        Send one prompt, return the text plus accounting.

        `stage` is required: it labels the StageLog, and it is how mock mode
        knows which canned reply to give. `is_shadow=True` marks a call made
        only to measure an arm, so it can be excluded from that arm's cost.
        """
        model = model or self.cfg.llm.helper_model
        max_tokens = max_tokens or self.cfg.llm.max_tokens
        temperature = self.cfg.llm.temperature

        # --- mock: no network, no cache, no cost ---
        if self.mock:
            text = self._mock_backend.reply(stage, prompt)
            resp = LLMResponse(text=text, model="mock", tokens_in=0, tokens_out=0,
                               latency_ms=0.0, cost_usd=0.0, cache_hit=False)
            self.n_calls += 1
            self._log(stage, resp, is_shadow, note="mock")
            return resp

        key = self._cache_key(model, system, prompt, temperature, max_tokens)

        # --- cache hit ---
        if self.use_cache:
            hit = self._cache_read(key)
            if hit is not None:
                self.n_calls += 1
                self.n_cache_hits += 1
                # A cache hit costs nothing. We still report the token counts
                # of the original call, because the experiment's token figures
                # should describe the pipeline, not our cache luck - but
                # cost_usd is zero, because no money changed hands.
                resp = LLMResponse(
                    text=hit["text"], model=hit["model"],
                    tokens_in=hit["tokens_in"], tokens_out=hit["tokens_out"],
                    latency_ms=0.0, cost_usd=0.0, cache_hit=True,
                )
                self._log(stage, resp, is_shadow, note="cache")
                return resp

        # --- live call, with retries ---
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err: Exception | None = None
        for attempt in range(self.cfg.llm.max_retries):
            # Pace requests BEFORE sending, to avoid the 429 rather than
            # retry through it.
            self.throttle_wait_s += self._limiter.acquire()
            t0 = time.perf_counter()
            try:
                assert self._client is not None
                raw = self._client.chat.completions.create(
                    model=model,
                    messages=messages,            # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                text = (raw.choices[0].message.content or "").strip()
                tin = raw.usage.prompt_tokens if raw.usage else 0
                tout = raw.usage.completion_tokens if raw.usage else 0
                cost = self.cfg.llm.price_of(model, tin, tout)

                if self.use_cache:
                    self._cache_write(key, {"text": text, "model": model,
                                            "tokens_in": tin, "tokens_out": tout})

                self.n_calls += 1
                resp = LLMResponse(text=text, model=model, tokens_in=tin,
                                   tokens_out=tout, latency_ms=latency_ms,
                                   cost_usd=cost, cache_hit=False)
                self._log(stage, resp, is_shadow)
                return resp

            except Exception as e:                # noqa: BLE001 - see comment
                # Deliberately broad: OpenRouter surfaces rate limits, upstream
                # provider errors and transport failures as different classes,
                # and for a 3-day project "retry with backoff, then report
                # honestly" is the right behaviour for most of them.
                last_err = e
                status = getattr(e, "status_code", None)
                text = str(e)

                # 402 = out of credits. Retrying cannot help, and continuing
                # would burn the whole question set producing identical
                # failures, so fail loudly and immediately.
                if status == 402 or "402" in text[:80]:
                    raise InsufficientCredits(
                        "OpenRouter rejected the request: no credits "
                        "available.\n"
                        "Add credits at https://openrouter.ai/settings/credits "
                        "(about $5 covers a full 100-question run)."
                    ) from e

                if attempt < self.cfg.llm.max_retries - 1:
                    # Honour Retry-After on a 429; otherwise exponential backoff.
                    if status == 429 or "429" in text[:80]:
                        delay = _retry_after_seconds(e, default=20.0)
                    else:
                        delay = 2 ** attempt      # 1s, 2s, 4s, 8s
                    time.sleep(delay)

        raise RuntimeError(
            f"LLM call failed at stage {stage!r} after "
            f"{self.cfg.llm.max_retries} attempts: {last_err}"
        ) from last_err

    def complete_json(
        self,
        prompt: str,
        stage: str,
        model: str | None = None,
        system: str = "",
        max_tokens: int | None = None,
        is_shadow: bool = False,
        attempts: int = 2,
    ) -> tuple[Any, LLMResponse]:
        """
        Like complete(), but parses the reply as JSON.

        On a parse failure we re-ask once with an explicit correction appended,
        bypassing the cache so we do not just re-read the same bad reply. If it
        still fails we raise - and the count is tracked in
        self.n_parse_failures, which the run summary reports as a metric. A
        silently swallowed parse failure would corrupt results invisibly.
        """
        resp = self.complete(prompt, stage, model, system, max_tokens, is_shadow)
        try:
            return extract_json(resp.text), resp
        except ValueError:
            self.n_parse_failures += 1

        for _ in range(attempts - 1):
            retry_prompt = (
                prompt
                + "\n\nIMPORTANT: your previous reply was not valid JSON. "
                  "Reply with ONLY a single valid JSON value. No markdown "
                  "fences, no explanation, no text before or after."
            )
            saved, self.use_cache = self.use_cache, False   # force a fresh ask
            try:
                resp = self.complete(retry_prompt, stage, model, system,
                                     max_tokens, is_shadow)
            finally:
                self.use_cache = saved
            try:
                parsed = extract_json(resp.text)
                resp.parse_failures = 1
                return parsed, resp
            except ValueError:
                self.n_parse_failures += 1

        raise ValueError(
            f"Stage {stage!r} did not return valid JSON after {attempts} "
            f"attempts. Last reply: {resp.text[:300]!r}"
        )

    # -- bookkeeping ------------------------------------------------------

    def _log(self, stage: str, resp: LLMResponse, is_shadow: bool,
             note: str = "") -> None:
        self.logs.append(StageLog(
            stage=stage, model=resp.model,
            tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
            latency_ms=resp.latency_ms, cost_usd=resp.cost_usd,
            cache_hit=resp.cache_hit, is_shadow=is_shadow, note=note,
        ))

    def take_logs(self) -> list[StageLog]:
        """Hand over the logs for one query and reset, ready for the next."""
        logs, self.logs = self.logs, []
        return logs

    def summary(self) -> dict[str, Any]:
        """Totals for the whole run, printed at the end of run_eval."""
        return {
            "n_calls": self.n_calls,
            "n_cache_hits": self.n_cache_hits,
            "cache_hit_rate": (self.n_cache_hits / self.n_calls) if self.n_calls else 0.0,
            "n_parse_failures": self.n_parse_failures,
        }
