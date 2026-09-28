from __future__ import annotations

from dataclasses import dataclass

import pytest

from base.llm_strategy import FakeLLMStrategy
from base.prompt_template import PromptTemplate
from processors.ard_sketch_generator import ArduinoSketchGenerator
from processors.data_processor import DataProcessor
from processors.model_converter import ModelConverter
from processors.py_sketch_generator import PythonSketchGenerator
from processors.tpu_sketch_generator import TPUSketchGenerator


def make_processor(cls, monkeypatch):
    return cls(FakeLLMStrategy(), trace_id="contract1234")


def assert_prompt_triplet(prompts):
    assert isinstance(prompts, list)
    assert len(prompts) == 3
    assert isinstance(prompts[0], str)
    assert isinstance(prompts[1], str)
    assert hasattr(prompts[2], "format")


def test_local_prompt_template_matches_project_usage_contract():
    prompt = PromptTemplate.from_template("Hello {name}; keep {later}.")

    assert prompt.template == "Hello {name}; keep {later}."
    assert prompt.format(name="Ada", later="Grace") == "Hello Ada; keep Grace."
    assert prompt.input_variables == ["name", "later"]


def test_local_prompt_template_supports_explicit_partial_formatting():
    prompt = PromptTemplate.from_template("Fill {known}; defer {unknown}.")

    partially_formatted = prompt.format_partial(known="now")

    assert isinstance(partially_formatted, PromptTemplate)
    assert partially_formatted.template == "Fill now; defer {unknown}."
    assert partially_formatted.format(unknown="later") == "Fill now; defer later."


@dataclass(frozen=True)
class FormattingCase:
    processor_cls: type
    template_type: str
    values: dict
    expected_fragments: tuple[str, ...]


PROMPT_FORMATTING_CASES = [
    FormattingCase(
        DataProcessor,
        "processing_suggestions",
        {
            "purpose": "classify fruits",
            "dataset_summary": "Dataset shape: (3, 4)",
            "dataset_intro": "RGB fruit dataset",
        },
        (
            "Generate 8 practical data processing suggestions",
            "classify fruits",
            "Dataset shape: (3, 4)",
            "RGB fruit dataset",
            '{"operation-name": "description for operation"',
        ),
    ),
    FormattingCase(
        DataProcessor,
        "processing_suggestions_error_handling",
        {
            "purpose": "classify fruits",
            "dataset_summary": "Dataset shape: (3, 4)",
            "dataset_intro": "RGB fruit dataset",
            "executed_code": '{"bad": "json"}',
            "error_info": "failed to parse",
        },
        (
            "Generate a new data processing suggestion table",
            "failed to parse",
            '{"bad": "json"}',
            "RGB fruit dataset",
        ),
    ),
    FormattingCase(
        DataProcessor,
        "operation_code",
        {
            "operation_description": "Normalize RGB columns.",
            "current_operation_name": "normalize_rgb",
            "dataset_path": "data/in.csv",
            "dataset_summary": "columns: r,g,b",
            "completed_processing_steps": "none",
        },
        (
            "Implement the specified data engineering operation",
            "Normalize RGB columns.",
            "normalize_rgb",
            "data/in.csv",
            "```json",
        ),
    ),
    FormattingCase(
        DataProcessor,
        "operation_error_handling",
        {
            "current_operation": "normalize_rgb",
            "executed_code": "raise RuntimeError('bad')",
            "error_info": "RuntimeError: bad",
            "dataset_path": "data/in.csv",
            "dataset_summary": "columns: r,g,b",
            "completed_processing_steps": "none",
        },
        (
            "Regenerate Python code",
            "normalize_rgb",
            "raise RuntimeError('bad')",
            "RuntimeError: bad",
        ),
    ),
    FormattingCase(
        ModelConverter,
        "conversion_code",
        {
            "dataset_path": "data/calibration.npy",
            "original_model_path": "models/model.keras",
            "converted_model_path": "models/model.tflite",
            "quantization_requirement_text": " with 8-bit integer quantization",
            "input_datatype": "float32",
            "output_datatype": "uint8",
        },
        (
            "models/model.keras",
            "models/model.tflite",
            "with 8-bit integer quantization",
            "float32",
            "uint8",
        ),
    ),
    FormattingCase(
        ModelConverter,
        "conversion_error_handling",
        {
            "executed_code": "converter.convert()",
            "error_info": "missing representative dataset",
            "quantization_requirement_text": " with 8-bit integer quantization",
            "original_model_path": "models/model.keras",
            "converted_model_path": "models/model.tflite",
            "input_datatype": "float32",
            "output_datatype": "uint8",
        },
        (
            "converter.convert()",
            "missing representative dataset",
            "models/model.keras",
            "models/model.tflite",
        ),
    ),
    FormattingCase(
        ArduinoSketchGenerator,
        "application_specification",
        {
            "application_spec_template": '{"application": "fruit"}',
            "dataset_summary": "Dataset shape: (3, 4)",
            "board_fullname": "Arduino Nano 33 BLE Sense",
        },
        (
            "Arduino Nano 33 BLE Sense",
            "Dataset shape: (3, 4)",
            '{"application": "fruit"}',
        ),
    ),
    FormattingCase(
        ArduinoSketchGenerator,
        "arduino_sketch",
        {
            "application_spec_template": '{"programming_guidelines": []}',
            "dataset_summary": "Dataset shape: (3, 4)",
        },
        (
            "Generate",
            "Dataset shape: (3, 4)",
            '{"programming_guidelines": []}',
        ),
    ),
    FormattingCase(
        ArduinoSketchGenerator,
        "sketch_error_handling",
        {
            "executed_code": "void setup() {}",
            "error_info": "compile failed",
            "board_fullname": "Arduino Nano 33 BLE Sense",
            "dataset_summary": "Dataset shape: (3, 4)",
            "application_spec_template": '{"application": "fruit"}',
        },
        (
            "void setup() {}",
            "compile failed",
            "Arduino Nano 33 BLE Sense",
            '{"application": "fruit"}',
        ),
    ),
    FormattingCase(
        PythonSketchGenerator,
        "python_sketch",
        {
            "application_name": "Object Detection",
            "application_description": "Detect objects in images.",
            "model_path": "models/detect.tflite",
            "label_path": "models/labels.txt",
            "input_description": "Read one image.",
            "input_path": "data/image.jpg",
            "output_description": "Write detections.",
            "output_path": "results/out.txt",
            "core_logic_reference_formatted": "interpreter.invoke()",
            "confidence_threshold": 0.5,
            "target_device": "Raspberry Pi 4B",
        },
        (
            "Object Detection",
            "models/detect.tflite",
            "data/image.jpg",
            "results/out.txt",
            "interpreter.invoke()",
        ),
    ),
    FormattingCase(
        TPUSketchGenerator,
        "python_sketch",
        {
            "application_name": "TPU Object Detection",
            "application_description": "Detect objects with Edge TPU.",
            "model_path": "/home/mendel/model.tflite",
            "label_path": "/home/mendel/labels.txt",
            "input_description": "Read one video.",
            "input_path": "/home/mendel/in.mp4",
            "output_description": "Write annotated video.",
            "output_path": "/home/mendel/out.mp4",
            "core_logic_reference_formatted": "make_interpreter(model_path)",
            "confidence_threshold": 0.5,
            "target_device": "Google Coral Dev Board",
        },
        (
            "TPU Object Detection",
            "/home/mendel/model.tflite",
            "/home/mendel/in.mp4",
            "/home/mendel/out.mp4",
            "Google Coral Dev Board",
        ),
    ),
]


@pytest.mark.parametrize("case", PROMPT_FORMATTING_CASES, ids=lambda case: f"{case.processor_cls.__name__}:{case.template_type}")
def test_prompt_templates_format_to_same_user_prompt_contract(case, monkeypatch):
    processor = make_processor(case.processor_cls, monkeypatch)

    prompts = processor.get_prompt_template(case.template_type)
    assert_prompt_triplet(prompts)

    formatted_user_prompt = prompts[2].format(**case.values)

    assert isinstance(formatted_user_prompt, str)
    for expected_fragment in case.expected_fragments:
        assert expected_fragment in formatted_user_prompt

    assert "{dataset_summary}" not in formatted_user_prompt
    assert "{executed_code}" not in formatted_user_prompt
    assert "{error_info}" not in formatted_user_prompt


def test_data_processor_suggestion_prompt_sent_to_llm_is_fully_formatted(monkeypatch):
    processor = make_processor(DataProcessor, monkeypatch)
    processor.dataset_path = "inputs/data/fruit_data.csv"
    processor.dataset_intro = "Fruit RGB measurements."
    processor.purpose = "classify fruits"

    ok, table, error = processor.generate_processing_suggestions(max_retries=1)

    assert ok is True
    assert table == {"copy_dataset": "Copy the sample dataset without modification."}
    assert error == ""

    prompts, metadata = processor.llm_strategy.calls[-1]
    assert len(prompts) == 3
    assert all(isinstance(prompt, str) for prompt in prompts)
    assert "classify fruits" in prompts[2]
    assert "Fruit RGB measurements." in prompts[2]
    assert "Dataset shape:" in prompts[2]
    assert "{purpose}" not in prompts[2]
    assert "{dataset_summary}" not in prompts[2]
    assert metadata["generation_name"].endswith("_processing_suggestions_gen_attempt#1")


def test_data_processor_code_generation_prompt_sent_to_llm_is_fully_formatted(monkeypatch):
    processor = make_processor(DataProcessor, monkeypatch)
    processor.current_operation = {
        "operation": "copy_dataset",
        "explanation": "Copy the dataset without changing values.",
    }
    processor.dataset_path = "inputs/data/fruit_data.csv"
    processor.dataset_summary_str = "Dataset shape: (10, 4)"
    processor.completed_processing_steps = ["loaded_original_dataset"]
    monkeypatch.setattr(processor, "execute_code", lambda code, workspace, **kwargs: None)

    assert processor.generate_operation_code(max_retries=1) is None

    prompts, metadata = processor.llm_strategy.calls[-1]
    assert len(prompts) == 3
    assert all(isinstance(prompt, str) for prompt in prompts)
    assert "copy_dataset" in prompts[2]
    assert "Copy the dataset without changing values." in prompts[2]
    assert "inputs/data/fruit_data.csv" in prompts[2]
    assert "Dataset shape: (10, 4)" in prompts[2]
    assert "loaded_original_dataset" in prompts[2]
    assert "{current_operation_name}" not in prompts[2]
    assert "{dataset_path}" not in prompts[2]
    assert metadata["generation_name"].endswith("_operation_code_gen#2")


def test_model_converter_prompt_sent_to_llm_is_fully_formatted(monkeypatch):
    processor = make_processor(ModelConverter, monkeypatch)
    processor.get_user_input()
    long_code = "```python\n" + "\n".join(f"print({index})" for index in range(21)) + "\n```"
    captured = {}

    def fake_invoke(prompts, generation_name):
        captured["prompts"] = prompts
        captured["generation_name"] = generation_name
        return long_code

    monkeypatch.setattr(processor, "invoke_llm", fake_invoke)
    monkeypatch.setattr(processor, "execute_code", lambda code, workspace, **kwargs: None)

    status, code, error = processor.generate_conversion_code(max_retries=1)

    assert status is True
    assert "print(20)" in code
    assert error == ""
    assert all(isinstance(prompt, str) for prompt in captured["prompts"])
    assert processor.original_model_path in captured["prompts"][2]
    assert processor.converted_model_path in captured["prompts"][2]
    assert processor.input_datatype in captured["prompts"][2]
    assert processor.output_datatype in captured["prompts"][2]
    assert "{original_model_path}" not in captured["prompts"][2]
    assert captured["generation_name"].endswith("_conversion_code_gen")


def test_sketch_specification_prompt_preserves_placeholder_literals_until_llm_fills_them(monkeypatch):
    processor = make_processor(ArduinoSketchGenerator, monkeypatch)
    processor.get_user_input()
    processor.dataset_summary_str = "Dataset shape: (3, 4)"

    prompts = processor.compose_specification_filling_prompt()

    assert all(isinstance(prompt, str) for prompt in prompts)
    assert "Object Classifier by Color" in prompts[2]
    assert "Arduino Nano 33 BLE Sense" in prompts[2]
    assert "Dataset shape: (3, 4)" in prompts[2]
    assert "{application_spec_template}" not in prompts[2]
    assert "{dataset_summary}" not in prompts[2]
    assert "{decide_when_generating_code_based_on_given_board_and_application_description}" in prompts[2]
    assert "{decide_when_generating_code_based_on_given_data_sample_and_application_description}" in prompts[2]
    assert "{programming_guidelines_placeholder_remain_this_untouched}" in prompts[2]


def test_sketch_app_spec_first_format_is_intentionally_partial(monkeypatch):
    processor = make_processor(ArduinoSketchGenerator, monkeypatch)
    processor.get_user_input()
    processor.dataset_summary_str = "Dataset shape: (3, 4)"

    processor.compose_specification_filling_prompt()

    partially_formatted_spec = processor.application_spec_template
    assert isinstance(partially_formatted_spec, str)
    assert "Object Classifier by Color" in partially_formatted_spec
    assert "Arduino Nano 33 BLE Sense" in partially_formatted_spec
    assert "Apple" in partially_formatted_spec
    assert "{application_name_hinting_its_purpose}" not in partially_formatted_spec
    assert "{application_description}" not in partially_formatted_spec
    assert "{board_fullname}" not in partially_formatted_spec
    assert "{classification_classes}" not in partially_formatted_spec

    assert "{decide_when_generating_code_based_on_given_board_and_application_description}" in partially_formatted_spec
    assert "{decide_when_generating_code_based_on_given_data_sample_and_application_description}" in partially_formatted_spec
    assert "{programming_guidelines_placeholder_remain_this_untouched}" in partially_formatted_spec


def test_sketch_partial_app_spec_can_be_embedded_in_second_prompt_without_consuming_deferred_fields(monkeypatch):
    processor = make_processor(ArduinoSketchGenerator, monkeypatch)
    processor.get_user_input()
    processor.dataset_summary_str = "Dataset shape: (3, 4)"

    spec_filling_prompts = processor.compose_specification_filling_prompt()
    user_prompt_sent_between_formatting_steps = spec_filling_prompts[2]

    assert "### TARGET TO BE FILLED ###" in user_prompt_sent_between_formatting_steps
    assert processor.application_spec_template in user_prompt_sent_between_formatting_steps
    assert "{application_spec_template}" not in user_prompt_sent_between_formatting_steps
    assert "{dataset_summary}" not in user_prompt_sent_between_formatting_steps
    assert "{board_fullname}" not in user_prompt_sent_between_formatting_steps
    assert "{decide_when_generating_code_based_on_given_board_and_application_description}" in user_prompt_sent_between_formatting_steps
    assert "{decide_when_generating_code_based_on_given_data_sample_and_application_description}" in user_prompt_sent_between_formatting_steps
    assert "{programming_guidelines_placeholder_remain_this_untouched}" in user_prompt_sent_between_formatting_steps


def test_sketch_second_stage_replaces_guidelines_after_llm_fills_deferred_app_spec_fields(monkeypatch):
    processor = make_processor(ArduinoSketchGenerator, monkeypatch)
    processor.filled_application_spec = """
{
  "application_specifications": {
    "hardware": {
      "board": "Arduino Nano 33 BLE Sense",
      "sensors": {"APDS9960": "RGB color sensor"}
    },
    "software": {
      "model": {
        "input_tensor": {"dimensions": [1, 3], "data_type": "np.float32"},
        "output_tensor": {"dimensions": [1, 3], "data_type": "np.uint8"},
        "tensor_arena_size": 4096
      }
    },
    "programming_guidelines": "{programming_guidelines_placeholder_remain_this_untouched}"
  }
}
"""

    processor.build_application_spec_with_guidelines()

    completed_spec = processor.application_spec_with_guidelines
    assert "APDS9960" in completed_spec
    assert "RGB color sensor" in completed_spec
    assert '"tensor_arena_size": 4096' in completed_spec
    assert "{programming_guidelines_placeholder_remain_this_untouched}" not in completed_spec
    assert "Programming Guidelines for TFLite Micro Inference on Microcontrollers" in completed_spec
    assert "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h" in completed_spec
    assert "tflite::GetModel(model)" in completed_spec
    assert "tensor->data.uint8" in completed_spec
    assert "APDS.readColor(int& r, int& g, int& b)" in completed_spec
    assert "Do not call private `enableColor`" in completed_spec


def test_sketch_final_code_prompt_uses_second_stage_completed_spec(monkeypatch):
    processor = make_processor(ArduinoSketchGenerator, monkeypatch)
    processor.dataset_summary_str = "Dataset shape: (3, 4)"
    processor.filled_application_spec = """
{
  "application_specifications": {
    "hardware": {"sensors": {"APDS9960": "RGB color sensor"}},
    "programming_guidelines": "{programming_guidelines_placeholder_remain_this_untouched}"
  }
}
"""
    processor.build_application_spec_with_guidelines()

    code_prompts = processor.compose_sketch_generation_prompt()

    assert all(isinstance(prompt, str) for prompt in code_prompts)
    assert "APDS9960" in code_prompts[2]
    assert "RGB color sensor" in code_prompts[2]
    assert "Programming Guidelines for TFLite Micro Inference on Microcontrollers" in code_prompts[2]
    assert "{programming_guidelines_placeholder_remain_this_untouched}" not in code_prompts[2]
    assert "{application_spec_template}" not in code_prompts[2]


def test_sketch_code_generation_prompt_sent_to_llm_is_fully_formatted(monkeypatch):
    processor = make_processor(ArduinoSketchGenerator, monkeypatch)
    processor.application_spec_with_guidelines = '{"programming_guidelines": ["Use Serial"]}'
    processor.dataset_summary_str = "Dataset shape: (3, 4)"
    code = "```cpp\n" + "\n".join("void noop() {}" for _ in range(61)) + "\n```"
    captured = {}

    def fake_invoke(prompts, generation_name):
        captured["prompts"] = prompts
        captured["generation_name"] = generation_name
        return code

    monkeypatch.setattr(processor, "invoke_llm", fake_invoke)
    monkeypatch.setattr(processor, "execute_code", lambda code, workspace, **kwargs: None)

    status, generated, error = processor.generate_arduino_sketch(max_retries=1)

    assert status is True
    assert "void noop" in generated
    assert error == ""
    assert all(isinstance(prompt, str) for prompt in captured["prompts"])
    assert '{"programming_guidelines": ["Use Serial"]}' in captured["prompts"][2]
    assert "Dataset shape: (3, 4)" in captured["prompts"][2]
    assert "{application_spec_template}" not in captured["prompts"][2]
    assert captured["generation_name"].endswith("_sketch_code_gen_attempt#1")


@pytest.mark.parametrize("processor_cls", [PythonSketchGenerator, TPUSketchGenerator])
def test_python_sketch_prompt_sent_to_llm_is_fully_formatted(processor_cls, monkeypatch):
    processor = make_processor(processor_cls, monkeypatch)
    processor.get_user_input()
    monkeypatch.setattr(
        processor,
        "execute_code",
        lambda code, workspace, **kwargs: None,
    )

    status, generated, error = processor.generate_python_sketch(max_retries=1)

    assert status is True
    assert "fake workflow executed" in generated
    assert error == ""

    prompts, metadata = processor.llm_strategy.calls[-1]
    assert all(isinstance(prompt, str) for prompt in prompts)
    assert processor.application_name in prompts[2]
    assert processor.target_device in prompts[2]
    assert processor.model_path in prompts[2]
    assert processor.input_description in prompts[2]
    assert processor.input_path in prompts[2]
    assert processor.output_description in prompts[2]
    assert processor.output_path in prompts[2]
    assert str(processor.confidence_threshold) in prompts[2]
    assert "{application_name}" not in prompts[2]
    assert "{model_path}" not in prompts[2]
    assert metadata["generation_name"].endswith(f"_{processor.get_task_name(short=True)}_gen_attempt#1")


def test_llm_strategy_builds_chat_messages_from_prompt_triplet_without_template_semantics():
    strategy = FakeLLMStrategy()
    messages = strategy._build_messages(["system context", "format rules", "user task"])

    assert messages == [
        {"role": "system", "content": "system context\nformat rules"},
        {"role": "user", "content": "user task"},
    ]


def test_llm_strategy_stringifies_prompt_parts_at_invocation_boundary():
    class StringOnlyAtBoundary:
        def __str__(self):
            return "late formatted user task"

    strategy = FakeLLMStrategy()
    messages = strategy._build_messages(["system context", "format rules", StringOnlyAtBoundary()])

    assert messages[0] == {"role": "system", "content": "system context\nformat rules"}
    assert messages[1] == {"role": "user", "content": "late formatted user task"}


@pytest.mark.parametrize("prompts", [[], ["system"], ["system", "format"], ["system", "format", "user", "extra"]])
def test_llm_strategy_rejects_prompt_lists_that_are_not_triplets(prompts):
    strategy = FakeLLMStrategy()
    with pytest.raises(ValueError, match="exactly three items"):
        strategy._build_messages(prompts)
