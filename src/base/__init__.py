"""Shared workflow and language model interfaces."""
from .base_processor import BaseProcessor
from .llm_strategy import LLMStrategy, FakeLLMStrategy, OllamaStrategy, OpenRouterStrategy

__all__ = [
    "BaseProcessor",
    "LLMStrategy",
    "FakeLLMStrategy",
    "OllamaStrategy",
    "OpenRouterStrategy",
]
