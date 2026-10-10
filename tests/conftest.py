"""Shared pytest fixtures."""
import shutil
import tempfile
from pathlib import Path

import pytest

from src.config import load_config


@pytest.fixture
def cfg(tmp_path: Path):
    """
    A Config whose cache points at a throwaway directory, so tests never
    read or pollute the real .cache/llm.
    """
    c = load_config()
    c.paths = dict(c.paths)
    cache = tmp_path / "llmcache"
    cache.mkdir(parents=True, exist_ok=True)
    # paths are resolved relative to ROOT, so store an absolute path and
    # override the property lookup by pointing paths at the temp dir.
    c.paths["cache_dir"] = str(cache)
    return c
