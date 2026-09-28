"""Construct a live OpenRouter client or an offline test double."""
try:
    from ..base.llm_strategy import FakeLLMStrategy, OllamaStrategy, OpenRouterStrategy
except ImportError:
    from base.llm_strategy import FakeLLMStrategy, OllamaStrategy, OpenRouterStrategy


class LLMFactory:
    @staticmethod
    def create_llm(model_provider, **kwargs):
        strategies = {
            "fake": FakeLLMStrategy,
            "ollama": OllamaStrategy,
            "openrouter": OpenRouterStrategy,
        }
        if model_provider not in strategies:
            raise ValueError(f"Unsupported LLM type: {model_provider}")
        model_name = kwargs.get("model_name")
        if model_provider == "fake":
            model_name = model_name or "fake-model"
        return strategies[model_provider](model_name, kwargs.get("parameters", False))
