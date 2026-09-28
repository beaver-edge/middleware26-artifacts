"""OpenRouter and deterministic offline language model strategies."""
import inspect
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import ClassVar
from openai import OpenAI

class LLMStrategy(ABC):
    """!
    Abstract base class defining the interface for language model strategies.

    This class enforces a consistent interface across different LLM implementations,
    making them interchangeable in the application.
    """
    
    model_provider: ClassVar[str]

    def __init__(self, model_name: str, parameters: dict|bool=False):
        self.parameters = parameters
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not inspect.isabstract(cls):
            if not getattr(cls, "model_provider", None):
                raise TypeError(
                    f"{cls.__name__} must define class attribute 'model_provider'"
                )
                
    @abstractmethod
    def invoke(
        self,
        prompts: list[str] | str,
        metadata_: dict,
        model_provider: str | None = None,
    ) -> str:
        """Invoke the language model."""
        pass
        
    @abstractmethod
    def get_endpoint_url(self) -> str:
        """!
        Retrieves the API endpoint URL for the language model.
        """
        pass

 
    
    def get_generation_parameters(self) -> dict:
        if isinstance(self.parameters, dict):
            return self.parameters

        if isinstance(self.parameters, bool):
            if self.parameters:
                return {
                    "temperature": 0.1,
                    "top_p": 0.3,
                }
            return {}

        raise ValueError("Invalid parameters")


    def get_model_provider(self) -> str:
        return self.model_provider
    
    def _validate_metadata(self, metadata_: dict) -> None:
        if not isinstance(metadata_, dict):
            raise TypeError("metadata_ must be a dictionary.")
    
    def _build_messages(self, prompts: list[str] | str) -> list[dict[str, str]]:
        if isinstance(prompts, list):
            if len(prompts) != 3:
                raise ValueError(
                    "Expected prompts to contain exactly three items: "
                    "[system_context, system_constraints, user_prompt]."
                )

            return [
                {
                    "role": "system",
                    "content": str(prompts[0]) + "\n" + str(prompts[1]),
                },
                {
                    "role": "user",
                    "content": str(prompts[2]),
                },
            ]

        return [{"role": "user", "content": str(prompts)}]

    def _strip_thinking(self, response_content: str) -> str:
        if response_content.lstrip().startswith("<think>") or "<think>" in response_content:
            return re.sub(
                r"<think>.*?</think>",
                "",
                response_content,
                flags=re.DOTALL,
            ).strip()

        return response_content

    def _extract_chat_completion_text(self, response) -> str:
        """Return final answer text without treating reasoning as the answer."""
        if not response.choices:
            raise ValueError(f"{self.model_provider} returned no completion choices")

        message = response.choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return self._strip_thinking(content)

        reasoning_fields = (
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "thinking",
        )
        populated_reasoning = [
            name for name in reasoning_fields if getattr(message, name, None)
        ]
        detail = ""
        if populated_reasoning:
            detail = (
                "; reasoning was present in "
                + ", ".join(populated_reasoning)
                + " but is not a final answer"
            )
        raise ValueError(
            f"{self.model_provider} returned no final text in message.content{detail}"
        )




class OpenRouterStrategy(LLMStrategy):
    """All live requests use the three OPENROUTER settings exclusively."""

    model_provider = "openrouter"
    # OpenRouter-specific request field; the OpenAI SDK has no `reasoning`
    # argument, so it is sent through extra_body. Disabling reasoning avoids
    # long reasoning-only responses that contain no final answer.
    REQUEST_EXTRA_BODY = {"reasoning": {"enabled": False}}

    def __init__(self, model_name=None, parameters=False):
        super().__init__(model_name or os.getenv("OPENROUTER_MODEL_NAME", ""), parameters)
        self.api_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key or not self.model_name:
            raise ValueError("Set OPENROUTER_API_KEY and OPENROUTER_MODEL_NAME in .env")
        self.client = OpenAI(base_url=self.api_base, api_key=api_key, timeout=180, max_retries=2)

    def invoke(self, prompts, metadata_, model_provider=None):
        self._validate_metadata(metadata_)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self._build_messages(prompts),
            extra_body=self.REQUEST_EXTRA_BODY,
            **self.get_generation_parameters(),
        )
        return self._extract_chat_completion_text(response)

    def get_endpoint_url(self):
        return self.api_base


class OllamaStrategy(LLMStrategy):
    """Ollama Cloud through its OpenAI-compatible Chat Completions API."""

    model_provider = "ollama"

    def __init__(self, model_name=None, parameters=False):
        super().__init__(model_name or os.getenv("OLLAMA_MODEL_NAME", ""), parameters)
        self.api_base = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1")
        api_key = os.getenv("OLLAMA_API_KEY")
        if not api_key or not self.model_name:
            raise ValueError("Set OLLAMA_API_KEY and OLLAMA_MODEL_NAME in .env")
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=api_key,
            timeout=180,
            max_retries=2,
        )

    def invoke(self, prompts, metadata_, model_provider=None):
        self._validate_metadata(metadata_)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self._build_messages(prompts),
            **self.get_generation_parameters(),
        )
        return self._extract_chat_completion_text(response)

    def get_endpoint_url(self):
        return self.api_base


class FakeLLMStrategy(LLMStrategy):
    """Deterministic offline strategy for smoke tests and unit tests."""
    
    
    model_provider="fake"
    
    
    def __init__(self, model_name="fake-model", parameters: dict | bool = False):
        super().__init__(model_name, parameters)
        self.calls = []
        
        
    def _get_model_config(self) -> dict:
        return {
            "model": self.model_name,
        }
        
        
    def invoke(
        self,
        prompts: list[str] | str,
        metadata_: dict,
    ) -> str:
        self._validate_metadata(metadata_)

        self.calls.append((prompts, metadata_))
        prompt_text = "\n".join(str(prompt) for prompt in prompts) if isinstance(prompts, list) else str(prompts)
        lowered = prompt_text.lower()
        if "single valid json dictionary" in lowered or "suggestion table" in lowered:
            return '{"copy_dataset": "Copy the sample dataset without modification."}'
        if "python" in lowered or "tflite" in lowered or "tpu" in lowered:
            body = [
                "from pathlib import Path",
                "model_path = 'model.tflite'",
                "input_path = 'input.dat'",
                "output_path = 'output.dat'",
                "label_path = 'labels.txt'",
                "confidence_threshold = 0.5",
                "Path('work/data').mkdir(parents=True, exist_ok=True)",
                "print('fake workflow executed')",
            ]
            body.extend(f"# filler line {index}" for index in range(60))
            return "```python\n" + "\n".join(body) + "\n```\n"
        if "arduino" in lowered or "sketch" in lowered:
            return "```cpp\nvoid setup() { Serial.begin(9600); }\nvoid loop() {}\n```"
        return "```python\nprint('fake workflow executed')\n```\n"


    def get_endpoint_url(self) -> str:
        return "fake://local"
