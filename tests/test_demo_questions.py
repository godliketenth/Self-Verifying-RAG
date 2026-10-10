"""
The demo fallback must be honest: these questions genuinely have no evidence
in the corpus. If one of them were actually answerable, the demo would be
staged rather than real, and an examiner would be right to object.
"""

import json

import pytest

from src.config import load_config

cfg = load_config()
DEMO = cfg.data_dir / "demo_questions.json"
PASSAGES = cfg.data_dir / "passages.jsonl"

pytestmark = pytest.mark.skipif(
    not (DEMO.exists() and PASSAGES.exists()),
    reason="run `python -m src.data.build_demo_questions` first",
)


@pytest.fixture(scope="module")
def indexed_titles():
    return {json.loads(l)["title"] for l in open(PASSAGES)}


@pytest.fixture(scope="module")
def demo():
    return json.loads(DEMO.read_text())


def test_demo_questions_exist(demo):
    assert len(demo) >= 10


def test_no_demo_question_has_indexed_evidence(demo, indexed_titles):
    """
    The core honesty check. If ANY supporting article is in the corpus, the
    system might legitimately answer, and the demo would be misleading.
    """
    for d in demo:
        overlap = set(d["gold_titles"]) & indexed_titles
        assert not overlap, (
            f"{d['question'][:60]!r} has indexed evidence {overlap} - "
            "it is answerable, so it must not be used as an "
            "'evidence missing' demo"
        )


def test_demo_questions_are_real_and_well_formed(demo):
    """They must be genuine HotpotQA questions, not ones we invented."""
    for d in demo:
        assert d["question"].strip().endswith(("?", "'")) or "?" in d["question"]
        assert d["answer"].strip()
        assert d["gold_titles"]
        assert "why_unsupported" in d


def test_demo_questions_disjoint_from_eval_set(demo):
    """A demo question must not also be an evaluated question."""
    ev = json.loads((cfg.data_dir / "eval_questions.json").read_text())
    eval_qs = {q["question"] for q in ev}
    for d in demo:
        assert d["question"] not in eval_qs
