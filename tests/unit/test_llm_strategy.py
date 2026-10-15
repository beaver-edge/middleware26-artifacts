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


def _call_log(tmp_path):
    import json
    path = tmp_path / "llm_calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_live_call_is_logged_with_request_response_and_usage(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key-value")
    strategy = OpenRouterStrategy("vendor/model", parameters=True)
    strategy.client = Mock()
    strategy.client.chat.completions.create.return_value = SimpleNamespace(
        id="gen-1",
        model="vendor/model-served",
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="<think>x</think>answer"),
        )],
        usage=SimpleNamespace(
            prompt_tokens=11, completion_tokens=7, total_tokens=18, cost=0.0004
        ),
    )
    assert strategy.invoke(["ctx", "rules", "task"], {"generation_name": "g"}) == "answer"
    (entry,) = _call_log(tmp_path)
    assert entry["call_index"] == 1
    assert entry["provider"] == "openrouter"
    assert entry["metadata"] == {"generation_name": "g"}
    assert entry["request"]["messages"][0] == {"role": "system", "content": "ctx\nrules"}
    assert entry["request"]["extra_body"] == {"reasoning": {"enabled": False}}
    assert entry["response"]["content"] == "<think>x</think>answer"  # raw, before stripping
    assert entry["response"]["extracted_text"] == "answer"
    assert entry["response"]["finish_reason"] == "stop"
    assert entry["usage"]["total_tokens"] == 18
    assert entry["error"] is None
    assert "secret-key-value" not in (tmp_path / "llm_calls.jsonl").read_text()
    assert strategy.usage_totals == {
        "llm_calls": 1, "prompt_tokens": 11, "completion_tokens": 7,
        "total_tokens": 18, "cost": 0.0004,
    }


def test_failed_call_is_logged_and_still_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    strategy = OpenRouterStrategy("vendor/model")
    strategy.client = Mock()
    strategy.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="length",
            message=SimpleNamespace(content=None, reasoning="private reasoning"),
        )],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=100, total_tokens=105),
    )
    with pytest.raises(ValueError, match="not a final answer"):
        strategy.invoke("hello", {})
    (entry,) = _call_log(tmp_path)
    assert entry["error"]["type"] == "ValueError"
    assert entry["response"]["reasoning"] == {"reasoning": "private reasoning"}
    assert entry["usage"]["completion_tokens"] == 100
    assert strategy.last_call_index == 1
    assert strategy.usage_totals["cost"] is None


def test_api_exception_is_logged(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    strategy = OpenRouterStrategy("vendor/model")
    strategy.client = Mock()
    strategy.client.chat.completions.create.side_effect = TimeoutError("timed out")
    with pytest.raises(TimeoutError):
        strategy.invoke("hello", {})
    (entry,) = _call_log(tmp_path)
    assert entry["error"] == {"type": "TimeoutError", "message": "timed out"}
    assert entry["response"]["content"] is None
    assert strategy.usage_totals["llm_calls"] == 1


def test_unwritable_call_log_never_breaks_a_run(monkeypatch, tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file")
    monkeypatch.setenv("BEAVER_LLM_CALL_LOG", str(blocker / "llm_calls.jsonl"))
    strategy = FakeLLMStrategy()
    assert "copy_dataset" in strategy.invoke("suggestion table", {})
    assert strategy.last_call_index == 1


def test_fake_strategy_numbers_and_logs_calls(tmp_path):
    strategy = FakeLLMStrategy()
    strategy.invoke("suggestion table", {"generation_name": "a"})
    strategy.invoke("python", {"generation_name": "b"})
    entries = _call_log(tmp_path)
    assert [e["call_index"] for e in entries] == [1, 2]
    assert [e["metadata"]["generation_name"] for e in entries] == ["a", "b"]
    assert strategy.usage_totals["llm_calls"] == 2
