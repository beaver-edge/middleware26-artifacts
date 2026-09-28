from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import src.main as main
from base.llm_strategy import FakeLLMStrategy


def test_clean_model_name():
    assert main.clean_model_name("phi4:latest") == "phi4"
    assert main.clean_model_name("qwen:32b") == "qwen:32b"


def test_build_parser_defaults(monkeypatch):
    monkeypatch.setenv("DEFAULT_EDGEML_TASK", "ardsketch")
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "fake")
    parser = main.build_parser()
    args = parser.parse_args([])
    assert args.task == "ardsketch"
    assert args.model_provider == "openrouter"


def test_main_uses_provider_specific_model(monkeypatch):
    calls = []
    monkeypatch.setenv("OLLAMA_MODEL_NAME", "ollama-model:latest")
    monkeypatch.setattr(
        main,
        "run_task",
        lambda args, num_run, session_id: calls.append(args),
    )

    main.main(["--task", "data", "--model-provider", "ollama"])

    assert calls[0].model_name == "ollama-model:latest"


def test_run_task_dispatches(monkeypatch):
    calls = {}

    class DummyTask:
        def __init__(self, llm_strategy, **kwargs):
            self.record = SimpleNamespace(output={"output": {"status": "success"}})
            calls["llm_strategy"] = llm_strategy
            calls["kwargs"] = kwargs

        def run(self):
            calls["ran"] = True

    monkeypatch.setitem(main.TASKS, "data", DummyTask)
    args = SimpleNamespace(
        task="data",
        model_provider="fake",
        model_name="fake-model",
        parameters=False,
        num_runs=2,
        benchmark=False,
    )
    main.run_task(args, num_run=2, session_id="auto_session")
    assert calls["llm_strategy"].__class__.__name__ == "FakeLLMStrategy"
    assert calls["kwargs"]["trace_id"]
    assert calls["kwargs"]["num_run"] == 2
    assert calls["kwargs"]["session_id"] == "auto_session"
    assert calls["ran"] is True


def test_retry_exhaustion_is_completed_framework_outcome(monkeypatch, tmp_path):
    class ExhaustedTask:
        def __init__(self, *args, **kwargs):
            self.record = SimpleNamespace(
                output={"output": {
                    "status": "failed",
                    "last_error": "Max 5 attempts exceeded. Last compiler error.",
                }}
            )

        def run(self):
            pass

    monkeypatch.setitem(main.TASKS, "ardsketch", ExhaustedTask)
    destination = tmp_path / "outcome.json"
    args = SimpleNamespace(
        task="ardsketch",
        model_provider="fake",
        model_name="fake-model",
        parameters=False,
        num_runs=1,
        benchmark=False,
        outcome_file=str(destination),
    )
    outcome = main.run_task(args, num_run=1, session_id="session")
    assert outcome["execution_status"] == "completed"
    assert outcome["artifact_status"] == "not_generated"
    assert outcome["termination"] == "retry_budget_exhausted"
    assert json.loads(destination.read_text()) == outcome


def test_make_session_id():
    args = SimpleNamespace(model_name="phi4:latest", task="data")
    session_id = main.make_session_id(args)
    assert session_id.endswith("_data_batch")
    assert session_id.startswith("phi4_")


def test_main_accepts_fake_data_task(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main,
        "run_task",
        lambda args, num_run, session_id: calls.append((args, num_run, session_id)),
    )
    main.main(["--task", "data", "--model-provider", "fake", "--model-name", "fake-model", "--num-runs", "2"])
    assert [call[1] for call in calls] == [1, 2]
    assert calls[0][0].task == "data"
    assert calls[0][0].model_provider == "fake"
    assert calls[0][2] == calls[1][2]

def test_failed_task_exits_nonzero(monkeypatch):
    class FailedTask:
        def __init__(self, *args, **kwargs):
            self.record = SimpleNamespace(output={"output": {"status": "failed", "last_error": "execution failed"}})
        def run(self):
            pass
    monkeypatch.setitem(main.TASKS, "data", FailedTask)
    with pytest.raises(RuntimeError, match="execution failed"):
        main.main(["--task", "data", "--model-provider", "fake", "--model-name", "fake-model"])
