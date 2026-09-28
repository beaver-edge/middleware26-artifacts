from __future__ import annotations

import json

import pytest

from base.llm_strategy import FakeLLMStrategy
from processors.ard_sketch_generator import ArduinoSketchGenerator
from processors.data_processor import DataProcessor
from processors.model_converter import ModelConverter
from processors.py_sketch_generator import PythonSketchGenerator
from processors.tpu_sketch_generator import TPUSketchGenerator


@pytest.fixture
def fake_llm():
    return FakeLLMStrategy()


def make_processor(cls, fake_llm, monkeypatch):
    return cls(fake_llm, trace_id="test1234")


def test_data_processor_user_input_creates_playground(fake_llm, monkeypatch):
    processor = make_processor(DataProcessor, fake_llm, monkeypatch)
    processor.get_user_input()
    assert processor.dataset_path.endswith("fruit_data.csv")
    assert processor.purpose


def test_data_processor_prompt_template(fake_llm, monkeypatch):
    processor = make_processor(DataProcessor, fake_llm, monkeypatch)
    prompts = processor.get_prompt_template("processing_suggestions")
    assert len(prompts) == 3
    with pytest.raises(ValueError):
        processor.get_prompt_template("bad")


def test_data_processor_extract_processing_suggestions(fake_llm, monkeypatch):
    processor = make_processor(DataProcessor, fake_llm, monkeypatch)
    assert processor.extract_processing_suggestions('{"a": "b"}') == {"a": "b"}
    assert processor.extract_processing_suggestions('{"a": "b"}\n{"c": "d"}') == {"a": "b", "c": "d"}
    with pytest.raises(json.JSONDecodeError):
        processor.extract_processing_suggestions("not json")


def test_data_processor_update_dataset_path(fake_llm, monkeypatch):
    processor = make_processor(DataProcessor, fake_llm, monkeypatch)
    processor.update_dataset_path("```json\nout.csv\n```")
    assert processor.dataset_path == "out.csv"


def test_data_processor_update_dataset_path_uses_json_block_after_code(fake_llm, monkeypatch):
    processor = make_processor(DataProcessor, fake_llm, monkeypatch)
    response = """```python
dataset_path = "input.csv"
```
```json
{"dataset_path": "processed.csv"}
```"""
    processor.update_dataset_path(response)
    assert processor.dataset_path == "processed.csv"


def test_data_processor_operation_generation_executes_model_code_unchanged(fake_llm, monkeypatch):
    processor = make_processor(DataProcessor, fake_llm, monkeypatch)
    processor.current_operation = {"operation": "scale", "explanation": "scale rgb"}
    processor.dataset_path = "work/data/fruit_data.csv"
    processor.dataset_summary_str = "summary"
    response = """```python
import pandas as pd
dataset_path = "path/to/dataset.csv"
df = pd.read_csv(dataset_path)
```
```json
{"dataset_path": "work/data/fruit_data.csv"}
```"""
    captured = {}

    monkeypatch.setattr(processor, "invoke_llm", lambda *args, **kwargs: response)

    def fake_execute(code, workspace, **kwargs):
        captured["code"] = code
        return None

    monkeypatch.setattr(processor, "execute_code", fake_execute)
    assert processor.generate_operation_code(max_retries=1) is None
    assert 'dataset_path = "path/to/dataset.csv"' in captured["code"]


def test_data_processor_suggestion_generation(fake_llm, monkeypatch):
    processor = make_processor(DataProcessor, fake_llm, monkeypatch)
    processor.dataset_path = "inputs/data/fruit_data.csv"
    processor.dataset_intro = "fruit data"
    processor.purpose = "classify fruits"
    ok, table, error = processor.generate_processing_suggestions(max_retries=1)
    assert ok is True
    assert table == {"copy_dataset": "Copy the sample dataset without modification."}
    assert error == ""


def test_data_processor_operation_generation_success(fake_llm, monkeypatch):
    processor = make_processor(DataProcessor, fake_llm, monkeypatch)
    processor.current_operation = {"operation": "copy_dataset", "explanation": "copy"}
    processor.dataset_summary_str = "summary"
    monkeypatch.setattr(processor, "execute_code", lambda code, workspace, **kwargs: None)
    assert processor.generate_operation_code(max_retries=1) is None


def test_model_converter_user_input_and_templates(fake_llm, monkeypatch):
    processor = make_processor(ModelConverter, fake_llm, monkeypatch)
    processor.get_user_input()
    assert processor.original_model_path == "inputs/convert/model.keras"
    assert processor.dataset_path == "inputs/convert/calibration.npy"
    assert processor.converted_model_path == "work/convert/model_quant.tflite"
    assert len(processor.get_prompt_template("conversion_code")) == 3
    with pytest.raises(ValueError):
        processor.get_prompt_template("bad")


def test_model_converter_generation_success(fake_llm, monkeypatch):
    processor = make_processor(ModelConverter, fake_llm, monkeypatch)
    processor.get_user_input()
    long_code = "```python\n" + "\n".join([f"print({i})" for i in range(21)]) + "\n```"
    monkeypatch.setattr(processor, "invoke_llm", lambda *args, **kwargs: long_code)
    monkeypatch.setattr(processor, "execute_code", lambda code, workspace, **kwargs: None)
    status, code, error = processor.generate_conversion_code(max_retries=1)
    assert status is True
    assert "print(20)" in code
    assert error == ""


def test_model_converter_short_code_rejected(fake_llm, monkeypatch):
    processor = make_processor(ModelConverter, fake_llm, monkeypatch)
    processor.get_user_input()
    monkeypatch.setattr(processor, "invoke_llm", lambda *args, **kwargs: "```python\nprint(1)\n```")
    status, _code, error = processor.generate_conversion_code(max_retries=1)
    assert status is False
    assert "Failed to generate valid code" in error


def test_ard_sketch_generator_prompt_flow(fake_llm, monkeypatch):
    processor = make_processor(ArduinoSketchGenerator, fake_llm, monkeypatch)
    processor.get_user_input()
    processor.dataset_summary_str = "summary"
    fill_prompt = processor.compose_specification_filling_prompt()
    assert len(fill_prompt) == 3
    processor.filled_application_spec = '{"programming_guidelines": "programming_guidelines_placeholder_remain_this_untouched"}'
    processor.build_application_spec_with_guidelines()
    assert "programming_guidelines" in processor.application_spec_with_guidelines
    sketch_prompt = processor.compose_sketch_generation_prompt()
    assert len(sketch_prompt) == 3


def test_ard_sketch_generator_extract_code(fake_llm, monkeypatch):
    processor = make_processor(ArduinoSketchGenerator, fake_llm, monkeypatch)
    assert processor.extract_code("```cpp\nvoid setup() {}\n```") == "void setup() {}"


def test_ard_sketch_generator_uses_unified_input_and_workspace(
    fake_llm, monkeypatch
):
    processor = make_processor(ArduinoSketchGenerator, fake_llm, monkeypatch)
    processor.application_spec_with_guidelines = "{}"
    processor.dataset_summary_str = "summary"
    code = "```cpp\n" + "\n".join(["void noop() {}" for _ in range(61)]) + "\n```"
    captured = {}
    monkeypatch.setattr(processor, "invoke_llm", lambda *args, **kwargs: code)

    def fake_execute(generated_code, workspace, **kwargs):
        captured["workspace"] = workspace
        captured["model_header"] = kwargs["arduino_model_header"]
        return None

    monkeypatch.setattr(processor, "execute_code", fake_execute)
    status, _generated, error = processor.generate_arduino_sketch(max_retries=1)

    assert status is True
    assert error == ""
    assert captured["workspace"].endswith("work/ardsketch")
    assert captured["model_header"].endswith("inputs/ardsketch/model.h")


def test_sketch_generation_success(fake_llm, monkeypatch):
    processor = make_processor(ArduinoSketchGenerator, fake_llm, monkeypatch)
    processor.application_spec_with_guidelines = "{}"
    processor.dataset_summary_str = "summary"
    code = "```cpp\n" + "\n".join(["void noop() {}" for _ in range(61)]) + "\n```"
    monkeypatch.setattr(processor, "invoke_llm", lambda *args, **kwargs: code)
    monkeypatch.setattr(processor, "execute_code", lambda code, workspace, **kwargs: None)
    status, generated, error = processor.generate_arduino_sketch(max_retries=1)
    assert status is True
    assert "void noop" in generated
    assert error == ""


@pytest.mark.parametrize("cls", [PythonSketchGenerator, TPUSketchGenerator])
def test_python_sketch_prompt_composition(cls, fake_llm, monkeypatch):
    processor = make_processor(cls, fake_llm, monkeypatch)
    processor.get_user_input()
    prompt = processor.compose_python_sketch_prompt()
    assert len(prompt) == 3
    error_prompt = processor.compose_error_handling_prompt("bad code", "bad error")
    assert len(error_prompt) == 3


def test_cpu_sketch_uses_unified_inputs_and_workspace(fake_llm, monkeypatch):
    processor = make_processor(PythonSketchGenerator, fake_llm, monkeypatch)
    processor.get_user_input()

    assert processor.model_path == "inputs/pysketch/detect.tflite"
    assert processor.label_path == "inputs/pysketch/labelmap.txt"
    assert processor.input_path == "inputs/pysketch/sheeps.mp4"
    assert processor.output_path == "work/pysketch/sheeps_detections.mp4"
    assert processor._execution_workspace().as_posix().endswith(
        "work/pysketch/generated"
    )


def test_tpu_sketch_uses_unified_local_workspace(fake_llm, monkeypatch):
    processor = make_processor(TPUSketchGenerator, fake_llm, monkeypatch)
    processor.get_user_input()

    assert processor._execution_workspace().as_posix().endswith(
        "work/tpusketch/generated"
    )
    assert processor.model_path.startswith("/home/mendel/")
    assert processor.input_path.startswith("/home/mendel/")


@pytest.mark.parametrize("cls", [PythonSketchGenerator, TPUSketchGenerator])
def test_python_sketch_generation_success(cls, fake_llm, monkeypatch):
    processor = make_processor(cls, fake_llm, monkeypatch)
    processor.get_user_input()
    monkeypatch.setattr(processor, "execute_code", lambda code, workspace, **kwargs: None)
    status, generated, error = processor.generate_python_sketch(max_retries=1)
    assert status is True
    assert "fake workflow executed" in generated
    assert error == ""


def test_python_sketch_repairs_invalid_attempt_and_retains_both(
    fake_llm, monkeypatch
):
    processor = make_processor(PythonSketchGenerator, fake_llm, monkeypatch)
    processor.get_user_input()
    responses = iter(
        [
            "```python\nraise RuntimeError('first attempt')\n```",
            "```python\nprint('repaired')\n```",
        ]
    )
    monkeypatch.setattr(
        processor, "invoke_llm", lambda *args, **kwargs: next(responses)
    )
    executions = []

    def fake_execute(code, workspace, **kwargs):
        executions.append(code)
        is_valid = len(executions) == 2
        processor.archive_generated_attempt(
            code,
            valid=is_valid,
            error=None if is_valid else "first attempt failed",
            label=kwargs["artifact_label"],
        )
        return None if is_valid else "first attempt failed"

    monkeypatch.setattr(processor, "execute_code", fake_execute)

    status, generated, error = processor.generate_python_sketch(max_retries=2)

    assert status is True
    assert "repaired" in generated
    assert error == ""
    assert len(list(processor.artifact_run_dir.glob("invalid/invalid_attempt-*.py"))) == 1
    assert len(list(processor.artifact_run_dir.glob("valid/valid_attempt-*.py"))) == 1
