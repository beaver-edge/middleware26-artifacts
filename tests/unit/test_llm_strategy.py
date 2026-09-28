from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from base.llm_strategy import FakeLLMStrategy, OllamaStrategy, OpenRouterStrategy
from factories.llm_factory import LLMFactory

def test_openrouter_exclusively_uses_unified_settings(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://router.example/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL_NAME", "vendor/model")
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])
    constructor = Mock(return_value=client)
    monkeypatch.setattr("base.llm_strategy.OpenAI", constructor)
    strategy = OpenRouterStrategy(parameters=True)
    assert strategy.invoke(["context", "rules", "task"], {}) == "answer"
    assert constructor.call_args.kwargs["base_url"] == "https://router.example/v1"
    assert constructor.call_args.kwargs["api_key"] == "test-key"
    args = client.chat.completions.create.call_args.kwargs
    assert args["model"] == "vendor/model"
    assert args["messages"][0]["content"] == "context\nrules"
    assert args["extra_body"] == {"reasoning": {"enabled": False}}
    assert "metadata" not in args

def test_empty_response_fails(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    strategy = OpenRouterStrategy("vendor/model")
    strategy.client = Mock()
    strategy.client.chat.completions.create.return_value = SimpleNamespace(choices=[])
    with pytest.raises(ValueError, match="no completion choices"):
        strategy.invoke("hello", {})

def test_reasoning_field_does_not_replace_final_content(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    strategy = OpenRouterStrategy("vendor/reasoning-model")
    strategy.client = Mock()
    strategy.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"result": "ok"}',
                    reasoning="private reasoning",
                )
            )
        ]
    )
    assert strategy.invoke("hello", {}) == '{"result": "ok"}'

def test_reasoning_only_response_fails_clearly(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    strategy = OpenRouterStrategy("vendor/reasoning-model")
    strategy.client = Mock()
    strategy.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, reasoning="private reasoning")
            )
        ]
    )
    with pytest.raises(ValueError, match="reasoning was present.*not a final answer"):
        strategy.invoke("hello", {})

def test_missing_credentials_fail(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterStrategy("vendor/model")

def test_fake_and_factory():
    assert isinstance(LLMFactory.create_llm("fake", model_name="test"), FakeLLMStrategy)
    assert "copy_dataset" in FakeLLMStrategy().invoke("suggestion table", {})
    with pytest.raises(ValueError):
        LLMFactory.create_llm("unknown")

def test_ollama_uses_its_own_configuration(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.example/v1")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
    monkeypatch.setenv("OLLAMA_MODEL_NAME", "reasoning-model:latest")
    constructor = Mock()
    monkeypatch.setattr("base.llm_strategy.OpenAI", constructor)

    strategy = LLMFactory.create_llm("ollama")

    assert isinstance(strategy, OllamaStrategy)
    assert strategy.model_name == "reasoning-model:latest"
    assert constructor.call_args.kwargs["base_url"] == "https://ollama.example/v1"
    assert constructor.call_args.kwargs["api_key"] == "ollama-key"
