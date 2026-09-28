"""Opt-in live integration checks for configured LLM providers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from factories.llm_factory import LLMFactory
from processors.data_processor import DataProcessor


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)

pytestmark = pytest.mark.skipif(
    os.getenv("BEAVER_EDGE_RUN_LLM_INTEGRATION", "").lower()
    not in {"1", "true", "yes", "on"},
    reason="set BEAVER_EDGE_RUN_LLM_INTEGRATION=true to make paid live API calls",
)


@pytest.mark.parametrize("provider", ["openrouter", "ollama"])
def test_live_invocation_and_json_parsing(provider):
    strategy = LLMFactory.create_llm(provider)
    response_text = strategy.invoke(
        [
            "You are testing a JSON response contract.",
            "Return only valid JSON, with no Markdown fence or explanation.",
            'Return exactly this object: {"smoke_test":"ok"}',
        ],
        metadata_={"purpose": "integration-test", "provider": provider},
    )

    parser = DataProcessor.__new__(DataProcessor)
    assert parser.extract_processing_suggestions(response_text) == {
        "smoke_test": "ok"
    }
