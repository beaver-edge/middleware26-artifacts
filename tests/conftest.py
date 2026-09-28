from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from base.base_processor import BaseProcessor  # type: ignore[import]
from base.llm_strategy import FakeLLMStrategy  # type: ignore[import]


class DummyProcessor(BaseProcessor):
    def __init__(self, llm_strategy=None, **kwargs):
        super().__init__(
            llm_strategy or FakeLLMStrategy(),
            trace_id=kwargs.pop("trace_id", "abcd1234"),
            task_name=kwargs.pop("task_name", "dummy_processor"),
            **kwargs,
        )

    def get_user_input(self):
        return None

    def run(self):
        return None


@pytest.fixture
def dummy_processor(monkeypatch):
    return DummyProcessor()


@pytest.fixture(autouse=True)
def isolated_generated_artifacts(monkeypatch, tmp_path):
    """Keep per-attempt evidence created by tests out of the repository tree."""
    monkeypatch.setenv("BEAVER_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
