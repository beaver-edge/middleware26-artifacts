"""Exercise every configured live LLM and the project JSON output parser."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factories.llm_factory import LLMFactory
from processors.data_processor import DataProcessor


EXPECTED_OUTPUT = {"smoke_test": "ok"}
PROVIDERS = ("openrouter", "ollama")
REASONING_FIELDS = (
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "thinking",
)


def validate_api_base(provider: str, api_base: str) -> None:
    if api_base.rstrip("/").endswith("/chat/completions"):
        raise ValueError(
            f"{provider.upper()}_BASE_URL must be the API base, not the full "
            "/chat/completions endpoint. The OpenAI client appends that path."
        )


def run_provider(provider: str) -> None:
    strategy = LLMFactory.create_llm(provider)
    validate_api_base(provider, strategy.get_endpoint_url())
    print(f"[{provider}] Endpoint: {strategy.get_endpoint_url()}")
    print(f"[{provider}] Model: {strategy.model_name}")

    captured = {}
    create_completion = strategy.client.chat.completions.create

    def capture_response(*args, **kwargs):
        response = create_completion(*args, **kwargs)
        captured["response"] = response
        return response

    strategy.client.chat.completions.create = capture_response
    response_text = strategy.invoke(
        [
            "You are testing a JSON response contract.",
            "Return only valid JSON, with no Markdown fence or explanation.",
            'Return exactly this object: {"smoke_test":"ok"}',
        ],
        metadata_={"purpose": "llm-smoke-test", "provider": provider},
    )

    message = captured["response"].choices[0].message
    populated_reasoning = [
        name for name in REASONING_FIELDS if getattr(message, name, None)
    ]
    print(
        f"[{provider}] Response fields: content=present; "
        f"reasoning={populated_reasoning or 'none'}"
    )

    parser = DataProcessor.__new__(DataProcessor)
    parsed = parser.extract_processing_suggestions(response_text)
    if parsed != EXPECTED_OUTPUT:
        raise AssertionError(
            f"[{provider}] parsed output did not match "
            f"{EXPECTED_OUTPUT!r}: {parsed!r}"
        )

    print(f"[{provider}] Parsed response: {parsed}")
    print(f"[{provider}] PASS")


def main(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        action="append",
        choices=PROVIDERS,
        help="Provider to test; repeat the option to test more than one.",
    )
    args = parser.parse_args(argv)

    providers = args.provider or list(PROVIDERS)
    failures = []
    for provider in providers:
        try:
            run_provider(provider)
        except Exception as exc:
            failures.append((provider, exc))
            print(f"[{provider}] FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)

    if failures:
        print(f"FAIL: {len(failures)} provider(s) failed.", file=sys.stderr)
        return 1
    print(f"PASS: {len(providers)} provider(s) passed invocation and parsing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
