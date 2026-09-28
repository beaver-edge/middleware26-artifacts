"""Clone-friendly command runner for TinyML Autopilot workflows."""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv

try:
    from .factories.llm_factory import LLMFactory
    from .processors.ard_sketch_generator import ArduinoSketchGenerator
    from .processors.data_processor import DataProcessor
    from .processors.model_converter import ModelConverter
    from .processors.py_sketch_generator import PythonSketchGenerator
    from .processors.tpu_sketch_generator import TPUSketchGenerator
except ImportError:
    from factories.llm_factory import LLMFactory
    from processors.ard_sketch_generator import ArduinoSketchGenerator
    from processors.data_processor import DataProcessor
    from processors.model_converter import ModelConverter
    from processors.py_sketch_generator import PythonSketchGenerator
    from processors.tpu_sketch_generator import TPUSketchGenerator


TASKS = {
    "data": DataProcessor,
    "convert": ModelConverter,
    "ardsketch": ArduinoSketchGenerator,
    "pysketch": PythonSketchGenerator,
    "tpusketch": TPUSketchGenerator,
}


def clean_model_name(input_string: str) -> str:
    """Remove suffixes that make run/session names noisy."""
    return re.sub(r"(:latest)$", "", input_string)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one TinyML Autopilot workflow from a fresh clone."
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default=os.getenv("DEFAULT_EDGEML_TASK", "data"),
        help="Workflow to run. Defaults to DEFAULT_EDGEML_TASK or data.",
    )
    parser.add_argument(
        "--model-provider",
        choices=["fake", "ollama", "openrouter"],
        default="openrouter",
        help="LLM provider. Use fake for offline setup verification.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name for the chosen provider.",
    )
    parser.add_argument(
        "--parameters",
        action="store_true",
        help="Use conservative default generation parameters.",
    )
    parser.add_argument(
        "--num-runs",
        "--num-run",
        dest="num_runs",
        type=int,
        default=int(os.getenv("DEFAULT_NUM_RUN", "1")),
        help="Number of repeated runs to execute. Defaults to DEFAULT_NUM_RUN or 1.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Tag the run as a benchmark. This does not create batch runs.",
    )
    parser.add_argument("--outcome-file", default=None, help=argparse.SUPPRESS)
    return parser


def make_session_id(args: argparse.Namespace) -> str:
    stamp = str(uuid.uuid4()).split("-")[0][:4]
    model_name = clean_model_name(args.model_name).replace("/", "_")
    return f"{model_name}_{stamp}_{args.task}_batch"


def normalize_outcome(outcome: dict) -> dict:
    """Separate framework completion from the generated artifact outcome."""
    raw_status = str(outcome.get("status", "failed")).lower()
    last_error = str(outcome.get("last_error", ""))
    if raw_status == "success":
        return {
            "execution_status": "completed",
            "artifact_status": "generated",
            "termination": "artifact_validated",
        }
    retry_exhausted = re.search(
        r"\bmax(?:imum)?(?:\s+\d+)?\s+(?:retries|attempts)\b",
        last_error,
        flags=re.IGNORECASE,
    )
    if raw_status == "failed" and retry_exhausted and "[FATAL]" not in last_error:
        return {
            "execution_status": "completed",
            "artifact_status": "not_generated",
            "termination": "retry_budget_exhausted",
            "last_error": last_error,
        }
    return {
        "execution_status": "failed",
        "artifact_status": "unknown",
        "termination": "unexpected_error",
        "last_error": last_error or "Workflow did not report an outcome",
    }


def run_task(args: argparse.Namespace, num_run: int, session_id: str) -> dict:
    
    BENCHMARK_THRESHOLD = 20
    
    benchmark_flag = args.benchmark or args.num_runs > BENCHMARK_THRESHOLD
        
    llm_strategy = LLMFactory.create_llm(
        args.model_provider,
        model_name=args.model_name,
        parameters=args.parameters,
    )
    processor_class = TASKS[args.task]
    trace_id = str(uuid.uuid4()).split("-")[0]
    processor = processor_class(
        llm_strategy,
        trace_id=trace_id,
        num_run=num_run,
        benchmark=benchmark_flag,
        session_id=session_id,
    )
    print(
        f"Running task={args.task} provider={args.model_provider} "
        f"model={args.model_name} run={num_run}/{args.num_runs} "
        f"session_id={session_id} trace_id={trace_id}"
    )
    processor.run()
    outcome = processor.record.output.get("output", {})
    normalized = normalize_outcome(outcome)
    if hasattr(processor, "artifact_summary"):
        normalized.update(processor.artifact_summary())
    else:
        normalized.update(
            {
                "artifact_run_dir": "",
                "valid_attempts": 0,
                "invalid_attempts": 0,
                "archived_outputs": 0,
            }
        )
    if hasattr(processor, "finalize_artifact_manifest"):
        processor.finalize_artifact_manifest(normalized)
    outcome_file = getattr(args, "outcome_file", None)
    if outcome_file:
        destination = Path(outcome_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(normalized, indent=2) + "\n")
    print("WORKFLOW_OUTCOME " + json.dumps(normalized, sort_keys=True))
    print(
        "GENERATED_ARTIFACTS: "
        f"valid={normalized['valid_attempts']} "
        f"invalid={normalized['invalid_attempts']} "
        f"path={normalized['artifact_run_dir']}"
    )
    if normalized["execution_status"] != "completed":
        raise RuntimeError(normalized["last_error"])
    return normalized


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_runs < 1:
        parser.error("--num-runs must be at least 1")
    if not args.model_name:
        configured_models = {
            "fake": "fake-model",
            "ollama": os.getenv("OLLAMA_MODEL_NAME"),
            "openrouter": os.getenv("OPENROUTER_MODEL_NAME"),
        }
        args.model_name = configured_models[args.model_provider]
    if not args.model_name:
        parser.error(
            f"Set {args.model_provider.upper()}_MODEL_NAME in .env or pass --model-name"
        )
    session_id = make_session_id(args)
    for num_run in range(1, args.num_runs + 1):
        run_task(args, num_run=num_run, session_id=session_id)


if __name__ == "__main__":
    main()
