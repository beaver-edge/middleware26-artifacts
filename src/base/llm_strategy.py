"""OpenRouter and deterministic offline language model strategies."""
import inspect
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from openai import OpenAI

# One JSON object per LLM request; logs/ is bind-mounted by run-artifact.sh.
CALL_LOG_ENV = "BEAVER_LLM_CALL_LOG"
DEFAULT_CALL_LOG = "logs/llm_calls.jsonl"
REASONING_FIELDS = ("reasoning", "reasoning_content", "reasoning_details", "thinking")

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
        # Request bookkeeping for logs/llm_calls.jsonl and the outcome summary.
        self.call_count = 0
        self.last_call_index = None
        self.usage_totals = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": None,
        }

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

    @classmethod
    def _plain(cls, value, depth: int = 0):
        """Best-effort JSON-ready copy of an SDK object (pydantic model, dict, namespace)."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if depth > 6:
            return str(value)
        dump = getattr(value, "model_dump", None)
        if callable(dump):
            try:
                return cls._plain(dump(), depth + 1)
            except Exception:
                return str(value)
        if isinstance(value, dict):
            return {str(k): cls._plain(v, depth + 1) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._plain(v, depth + 1) for v in value]
        if hasattr(value, "__dict__"):
            return {
                k: cls._plain(v, depth + 1)
                for k, v in vars(value).items()
                if not k.startswith("_")
            }
        return str(value)

    def _begin_call(self):
        """Number a request before it is sent, so failed requests are numbered too."""
        self.call_count += 1
        self.last_call_index = self.call_count
        return self.call_count, time.monotonic()

    def _record_call(self, index, started, metadata_, request, response=None, text=None, error=None):
        """Append one request/response record to the call log.

        Observability only: a failure to write the log is reported as a warning
        and never changes the outcome of the workflow.
        """
        try:
            latency = time.monotonic() - started
            usage = self._plain(getattr(response, "usage", None))
            choices = getattr(response, "choices", None) or []
            choice = choices[0] if choices else None
            message = getattr(choice, "message", None)
            reasoning = {
                name: self._plain(getattr(message, name, None))
                for name in REASONING_FIELDS
                if getattr(message, name, None)
            }
            entry = {
                "call_index": index,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "provider": self.model_provider,
                "model": self.model_name,
                "endpoint": self.get_endpoint_url(),
                "metadata": self._plain(metadata_),
                "request": self._plain(request),
                "response": {
                    "id": self._plain(getattr(response, "id", None)),
                    "served_model": self._plain(getattr(response, "model", None)),
                    "finish_reason": self._plain(getattr(choice, "finish_reason", None)),
                    "content": self._plain(getattr(message, "content", None)),
                    "reasoning": reasoning or None,
                    "extracted_text": text,
                },
                "usage": usage,
                "latency_seconds": round(latency, 3),
                "error": None if error is None else {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
            self._add_usage(usage)
            path = Path(os.getenv(CALL_LOG_ENV, DEFAULT_CALL_LOG))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str) + "\n"
                )
        except Exception as exc:  # never let logging break a run
            self.logger.warning("Could not record LLM call %s: %s", index, exc)

    def _add_usage(self, usage):
        self.usage_totals["llm_calls"] += 1
        if not isinstance(usage, dict):
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                self.usage_totals[key] += value
        cost = usage.get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            self.usage_totals["cost"] = (self.usage_totals["cost"] or 0.0) + cost

    def _chat_completion(self, prompts, metadata_, **request_options):
        """Run one Chat Completions request and append it to the call log."""
        request = {
            "model": self.model_name,
            "messages": self._build_messages(prompts),
            **request_options,
            **self.get_generation_parameters(),
        }
        index, started = self._begin_call()
        response = None
        try:
            response = self.client.chat.completions.create(**request)
            text = self._extract_chat_completion_text(response)
        except Exception as exc:
            self._record_call(index, started, metadata_, request, response=response, error=exc)
            raise
        self._record_call(index, started, metadata_, request, response=response, text=text)
        return text

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
        return self._chat_completion(
            prompts, metadata_, extra_body=self.REQUEST_EXTRA_BODY
        )

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
        return self._chat_completion(prompts, metadata_)

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
        index, started = self._begin_call()
        text = self._respond(prompts)
        request = {"model": self.model_name, "prompts": prompts}
        self._record_call(index, started, metadata_, request, text=text)
        return text

    def _respond(self, prompts: list[str] | str) -> str:
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
