from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from base.base_processor import LocalRunRecord


def test_default_trace_is_noop(dummy_processor):
    assert isinstance(dummy_processor.record, LocalRunRecord)
    dummy_processor.record.span(name="x")
    dummy_processor.record.update(output={"status": "ok"})
    assert dummy_processor.record.spans == [{"name": "x"}]
    assert dummy_processor.record.output == {"output": {"status": "ok"}}


def test_session_id_and_task_name(dummy_processor):
    assert dummy_processor.get_task_name() == "dummy_processor"
    assert dummy_processor.get_task_name(short=True) == "dp"
    assert dummy_processor.get_session_id() == "abcd1234: "


def test_session_id_requires_num_run(DummyProcessor=None):
    from conftest import DummyProcessor as Processor

    with pytest.raises(ValueError, match="num_run"):
        Processor(session_id="batch")


def test_setup_logging_does_not_duplicate_handlers(dummy_processor):
    before = len(dummy_processor.logger.handlers)
    dummy_processor.setup_logging()
    assert len(dummy_processor.logger.handlers) == before


def test_invoke_llm_records_metadata(dummy_processor):
    response = dummy_processor.invoke_llm("hello", "generation")
    assert "```python" in response
    prompts, metadata = dummy_processor.llm_strategy.calls[-1]
    assert prompts == "hello"
    assert metadata["generation_name"] == "generation"


def test_dataset_summary(tmp_path, dummy_processor):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("label,r,g,b\napple,1,2,3\nbanana,4,5,6\n")
    dummy_processor.dataset_path = str(csv_path)
    summary = dummy_processor.dataset_summary()
    assert "Dataset shape: (2, 4)" in summary
    assert "Column names" in summary


def test_prepare_execution_environment_python(tmp_path, dummy_processor):
    work_dir, tmp_file, command = dummy_processor.prepare_execution_environment(
        str(tmp_path), fixed_timestamp="20240101000000"
    )
    assert Path(work_dir).exists()
    assert tmp_file.endswith(".py")
    assert command == [sys.executable, tmp_file]


def test_prepare_execution_environment_arduino(tmp_path, dummy_processor):
    (tmp_path / "model.h").write_text("// model")
    work_dir, tmp_file, command = dummy_processor.prepare_execution_environment(
        str(tmp_path), is_arduino=True, fixed_timestamp="20240101000000"
    )
    assert Path(work_dir, "model.h").exists()
    assert tmp_file.endswith(".ino")
    assert command[:3] == ["arduino-cli", "compile", "--fqbn"]


def test_prepare_execution_environment_arduino_external_model_header(
    tmp_path, dummy_processor
):
    workspace = tmp_path / "work"
    model_header = tmp_path / "inputs" / "model.h"
    workspace.mkdir()
    model_header.parent.mkdir()
    model_header.write_text("// external model")

    work_dir, _tmp_file, _command = dummy_processor.prepare_execution_environment(
        str(workspace),
        is_arduino=True,
        fixed_timestamp="20240101000000",
        arduino_model_header=str(model_header),
    )

    assert Path(work_dir, "model.h").read_text() == "// external model"


def test_provider_model_id_is_safe_for_arduino_paths(tmp_path):
    from conftest import DummyProcessor
    from base.llm_strategy import FakeLLMStrategy
    processor = DummyProcessor(FakeLLMStrategy("vendor/model:free"))
    (tmp_path / "model.h").write_text("// model")
    work, sketch, _ = processor.prepare_execution_environment(
        str(tmp_path), is_arduino=True, fixed_timestamp="test"
    )
    assert Path(work).parent == tmp_path
    assert Path(sketch).stem == Path(work).name


def test_execute_code_success(tmp_path, dummy_processor, capsys):
    error = dummy_processor.execute_code(
        "print('ok')", str(tmp_path), artifact_label="smoke-attempt-01"
    )
    assert error is None
    valid = list(dummy_processor.artifact_run_dir.glob("valid/valid_attempt-*.py"))
    assert len(valid) == 1
    assert "smoke-attempt-01" in valid[0].name
    assert "GENERATION_VALIDATED: status=VALID" in capsys.readouterr().out


def test_execute_code_failure(tmp_path, dummy_processor, capsys):
    error = dummy_processor.execute_code(
        "raise RuntimeError('bad')",
        str(tmp_path),
        artifact_label="broken-attempt-01",
    )
    assert "RuntimeError" in error or "bad" in error
    invalid = list(
        dummy_processor.artifact_run_dir.glob("invalid/invalid_attempt-*.py")
    )
    assert len(invalid) == 1
    assert invalid[0].read_text() == "raise RuntimeError('bad')"
    manifest = json.loads(
        dummy_processor.artifact_run_dir.joinpath("manifest.json").read_text()
    )
    assert manifest["attempts"][0]["status"] == "invalid"
    assert "GENERATION_VALIDATION_FAILED: status=INVALID" in capsys.readouterr().out


def test_success_without_fresh_expected_output_is_invalid(tmp_path, dummy_processor):
    expected = tmp_path / "expected.bin"
    error = dummy_processor.execute_code(
        "print('ok')",
        str(tmp_path),
        artifact_label="missing-output-attempt-01",
        expected_outputs=[expected],
    )
    assert "did not create fresh" in error
    assert not list(dummy_processor.artifact_run_dir.glob("valid/*"))
    assert list(dummy_processor.artifact_run_dir.glob("invalid/invalid_attempt-*.py"))


def test_unreadable_expected_output_is_invalid(tmp_path, dummy_processor):
    expected = tmp_path / "output.bin"

    def reject_output(path):
        raise RuntimeError(f"unreadable output: {path}")

    code = (
        "from pathlib import Path\n"
        f"Path({str(expected)!r}).write_bytes(b'not-a-real-output')\n"
    )
    error = dummy_processor.execute_code(
        code,
        str(tmp_path),
        artifact_label="unreadable-output-attempt-01",
        expected_outputs=[expected],
        output_validator=reject_output,
    )
    assert "unreadable output" in error
    assert list(dummy_processor.artifact_run_dir.glob("invalid/invalid_attempt-*.py"))
    rejected_outputs = list(
        dummy_processor.artifact_run_dir.glob(
            "outputs/invalid/invalid_rejected_output_*.bin"
        )
    )
    assert len(rejected_outputs) == 1
    assert rejected_outputs[0].read_bytes() == b"not-a-real-output"


def test_remote_execution_command_flow(monkeypatch, tmp_path, dummy_processor):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="0", stderr="")

    monkeypatch.setenv("REMOTE_HOST", "coral")
    monkeypatch.setenv("REMOTE_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("REMOTE_EXEC_PATH", "/tmp/remote")
    monkeypatch.setattr("base.base_processor.subprocess.run", fake_run)
    monkeypatch.setattr(dummy_processor, "_stream_ssh_execution", lambda command, script_id: None)
    assert dummy_processor._execute_code_via_ssh("print('ok')", str(tmp_path)) is None
    assert any(command[0] == "scp" for command in calls)
    assert any(command[0] == "ssh" for command in calls)


def test_execute_code_remote_disabled(monkeypatch, tmp_path, dummy_processor):
    monkeypatch.setenv("REMOTE_EXECUTION_ENABLED", "false")
    error = dummy_processor.execute_code("print('ok')", str(tmp_path), remote_execution=True)
    assert "Remote execution is disabled" in error


def test_remote_success_is_classified_after_output_copy(
    monkeypatch, tmp_path, dummy_processor
):
    monkeypatch.setenv("REMOTE_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(dummy_processor, "_execute_code_via_ssh", lambda *args: None)
    removed = []
    monkeypatch.setattr(
        dummy_processor,
        "_remove_remote_output_before_validation",
        lambda path: removed.append(path),
    )
    copied = []
    monkeypatch.setattr(
        dummy_processor,
        "_copy_remote_output_to_archive",
        lambda path, validator=None, **kwargs: copied.append(
            (path, validator, kwargs)
        ),
    )
    error = dummy_processor.execute_code(
        "print('remote')",
        str(tmp_path),
        remote_execution=True,
        artifact_label="tpu-sketch-attempt-01",
        remote_output_path="/remote/output.mp4",
    )
    assert error is None
    assert removed == ["/remote/output.mp4"]
    assert copied == [
        ("/remote/output.mp4", None, {"status": "valid", "required": True})
    ]
    assert list(dummy_processor.artifact_run_dir.glob("valid/valid_attempt-*.py"))


def test_remote_failure_archives_code_and_partial_output_as_invalid(
    monkeypatch, tmp_path, dummy_processor
):
    monkeypatch.setenv("REMOTE_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(
        dummy_processor,
        "_remove_remote_output_before_validation",
        lambda path: None,
    )
    monkeypatch.setattr(
        dummy_processor,
        "_execute_code_via_ssh",
        lambda *args: "remote validation timed out",
    )
    copied = []
    monkeypatch.setattr(
        dummy_processor,
        "_copy_remote_output_to_archive",
        lambda path, validator=None, **kwargs: copied.append(
            (path, validator, kwargs)
        ),
    )

    error = dummy_processor.execute_code(
        "print('remote')",
        str(tmp_path),
        remote_execution=True,
        artifact_label="tpu-sketch-attempt-01",
        remote_output_path="/remote/output.mp4",
    )

    assert error == "remote validation timed out"
    assert copied == [
        ("/remote/output.mp4", None, {"status": "invalid", "required": False})
    ]
    assert list(dummy_processor.artifact_run_dir.glob("invalid/invalid_attempt-*.py"))


def test_handle_successful_execution_logs_output(tmp_path, dummy_processor):
    result = subprocess.CompletedProcess(["python"], 0, stdout="ok", stderr="")
    dummy_processor._handle_successful_execution(result)


def test_cleanup_execution_files(tmp_path, dummy_processor):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    tmp_file = work_dir / "x.py"
    tmp_file.write_text("print(1)")
    dummy_processor._cleanup_execution_files(str(work_dir), str(tmp_file))
    assert not work_dir.exists()


@pytest.mark.parametrize(
    "response,expected",
    [
        ("```python\nprint('x')\n```", "print('x')"),
        ("```\nprint('x')\n```", "print('x')"),
    ],
)
def test_extract_code_python(dummy_processor, response, expected):
    assert dummy_processor.extract_code(response) == expected


def test_extract_code_python_rejects_plain_text(dummy_processor):
    with pytest.raises(ValueError):
        dummy_processor.extract_code("plain text")


def test_extract_code_cpp(dummy_processor):
    response = "```cpp\nvoid setup() {}\n```\n```ino\nvoid loop() {}\n```"
    assert "void setup" in dummy_processor.extract_code_cpp(response)
    assert "void loop" in dummy_processor.extract_code_cpp(response)


def test_replace_generated_placeholders(dummy_processor):
    code = 'model_path = "x"\nconfidence_threshold = 0.1'
    updated = dummy_processor.replace_generated_placeholders(
        code, {"model_path": "model.tflite", "confidence_threshold": 0.5}
    )
    assert 'model_path = "model.tflite"' in updated
    assert "confidence_threshold = 0.5" in updated


def test_contains_error_indicator(dummy_processor):
    result = subprocess.CompletedProcess(["x"], 0, stdout="Error: failed", stderr="")
    assert dummy_processor.contains_error_indicator(result)
    clean = subprocess.CompletedProcess(["x"], 0, stdout="ok", stderr="")
    assert not dummy_processor.contains_error_indicator(clean)
