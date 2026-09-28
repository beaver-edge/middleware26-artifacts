"""Python sketch generation for Coral Edge TPU targets."""

from __future__ import annotations

import traceback
from pathlib import Path

try:
    from ..base.base_processor import BaseProcessor
    from ..base.prompt_template import PromptTemplate
    from ..prompt_templates import tpu_sketch_prompts
except ImportError:
    import prompt_templates.tpu_sketch_prompts as tpu_sketch_prompts
    from base.base_processor import BaseProcessor
    from base.prompt_template import PromptTemplate


class TPUSketchGenerator(BaseProcessor):
    """
    Generates Python scripts for running TFLite models with Edge TPU support.

    The current default profile targets a Google Coral Dev Board and keeps the
    existing remote paths. Those values are now grouped as class data so a
    future device/application profile can replace them without rewriting the
    generation and validation flow.
    """

    DEFAULT_APPLICATION_SPEC = {
        "target_device": "Google Coral Dev Board",
        "application_name": "Object Detection via a video file with TPU",
        "application_description": (
            "This application uses a TFLite model optimized for Edge TPU "
            "acceleration on Google Coral Dev Board to detect objects in the "
            "mediafile. It loads a TFLite model with EdgeTPU delegate for "
            "hardware acceleration, reads an video file, preprocesses it "
            "(resizes the video frame based on the input_details and output "
            "details of the model, and converts to RGB format), runs inference "
            " using the TPU, maps the highest probability output index to a "
            "class name using a label file, and draw the boxes with texts of "
            "labels and calculated mAP(mean average precision) on the video "
            "frame, and stores the annotated video file to the given output_path."
        ),
        "input_description": "Read a single video file from the given input_path",
        "output_description": (
            "Output the video file with rectangles drew on the detected objects, "
            "along with texts of labels and calculated mAP(mean average precision)"
        ),
        "input_path": "/home/mendel/tinyml_autopilot/data//sheeps.mp4",
        "model_path": "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite",
        "label_path": "/home/mendel/tinyml_autopilot/models/labelmap.txt",
        "output_path": "/home/mendel/tinyml_autopilot/results/sheeps_detections.mp4",
        "confidence_threshold": 0.5,
    }
    REQUIRED_APPLICATION_FIELDS = (
        "application_description",
        "model_path",
        "input_description",
        "output_description",
    )
    EXECUTION_WORKSPACE_PARTS = ("work", "tpusketch", "generated")
    MIN_GENERATED_LINES = 50

    def get_task_name(self, short=False):
        if short:
            return "tpusg"
        return "tpu_sketch_generator"

    def __init__(
        self,
        llm_strategy,
        trace_id,
        num_run=None,
        benchmark=False,
        session_id=None,
    ):
        super().__init__(
            llm_strategy,
            trace_id=trace_id,
            task_name=self.get_task_name(),
            num_run=num_run,
            benchmark=benchmark,
            session_id=session_id,
        )
        self._clear_application_spec()
        self.validated_sketch = ""

    def _clear_application_spec(self) -> None:
        self.application_name = ""
        self.application_description = ""
        self.model_path = ""
        self.label_path = None
        self.input_description = ""
        self.input_path = None
        self.output_description = ""
        self.output_path = ""
        self.confidence_threshold = 0.5
        self.target_device = ""

    def _apply_application_spec(self, spec: dict) -> None:
        for key, value in spec.items():
            setattr(self, key, value)
        self._validate_application_spec()

    def _validate_application_spec(self) -> None:
        missing = [
            field
            for field in self.REQUIRED_APPLICATION_FIELDS
            if not getattr(self, field, None)
        ]
        if missing:
            raise ValueError(
                "Application description, model path, input description, "
                "and output description must be set!"
            )

    def get_user_input(self):
        """Loads the default Coral application profile."""
        self._apply_application_spec(self.DEFAULT_APPLICATION_SPEC)
        self.logger.info(
            f"{self.get_session_id()}User input acquired for task: "
            f"'{self.application_name}'"
        )

    def get_prompt_template(self, template_type: str) -> list[PromptTemplate | str]:
        templates = {
            "python_sketch": tpu_sketch_prompts.PYTHON_SKETCH_PROMPT,
            "python_sketch_error_handling": tpu_sketch_prompts.PYTHON_SKETCH_error_handling_PROMPT,
        }
        if template_type not in templates:
            raise ValueError(
                f"Unknown template type for TPUSketchGenerator: {template_type}"
            )

        return [
            tpu_sketch_prompts.CONTEXT_PROMPT,
            tpu_sketch_prompts.CODE_RESPONSE_FORMAT,
            PromptTemplate.from_template(templates[template_type]),
        ]

    def _prompt_values(self) -> dict:
        return {
            "application_name": self.application_name,
            "application_description": self.application_description,
            "model_path": self.model_path,
            "label_path": self.label_path,
            "input_description": self.input_description,
            "input_path": self.input_path,
            "output_description": self.output_description,
            "output_path": self.output_path,
            "core_logic_reference_formatted": tpu_sketch_prompts.CORE_LOGIC_REFERENCE,
            "confidence_threshold": self.confidence_threshold,
            "target_device": self.target_device,
        }

    def compose_python_sketch_prompt(self) -> list[str | PromptTemplate]:
        prompts = self.get_prompt_template("python_sketch")
        prompts[2] = prompts[2].format(**self._prompt_values())
        return prompts

    def compose_error_handling_prompt(
        self, code: str, error: str
    ) -> list[str | PromptTemplate]:
        prompts = self.get_prompt_template("python_sketch_error_handling")
        prompts[2] = prompts[2].format(
            **self._prompt_values(),
            faulty_code=code,
            error_message=error,
        )
        return prompts

    def _generation_name(self, attempt_number: int) -> str:
        prefix = f"{self.trace_id[:4]}_{self.get_task_name(short=True)}_gen"
        return f"{prefix}_attempt#{attempt_number}"

    def _prompt_for_attempt(
        self, attempt_number: int, generated_code: str, error_message: str
    ):
        if attempt_number == 1:
            return self.compose_python_sketch_prompt()
        return self.compose_error_handling_prompt(generated_code, error_message)

    def _placeholder_values(self) -> dict:
        return {
            "model_path": self.model_path,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "label_path": self.label_path,
            "model_name": self.model_name,
            "confidence_threshold": float(self.confidence_threshold),
        }

    def _postprocess_generated_code(self, generated_code: str) -> str:
        return self.replace_generated_placeholders(generated_code, self._placeholder_values())

    def _execution_workspace(self) -> Path:
        return Path(__file__).resolve().parents[2].joinpath(
            *self.EXECUTION_WORKSPACE_PARTS
        )

    def _execute_generated_code(
        self, generated_code: str, attempt_number: int
    ) -> str | None:
        execution_workspace = self._execution_workspace()
        execution_workspace.mkdir(parents=True, exist_ok=True)
        return self.execute_code(
            generated_code,
            str(execution_workspace),
            is_arduino=False,
            remote_execution=True,
            artifact_label=f"tpu-sketch-attempt-{attempt_number:02d}",
            remote_output_path=self.output_path,
            output_validator=self.validate_readable_video,
        )

    def _validate_generated_script_length(
        self, generated_code: str, attempt_number: int
    ) -> None:
        if len(generated_code.splitlines()) >= self.MIN_GENERATED_LINES:
            return

        error_message = (
            "Generated script is not complete. Please ensure to generate a "
            "complete Python script."
        )
        if generated_code:
            self.archive_generated_attempt(
                generated_code,
                valid=False,
                error=error_message,
                executor="remote",
                label=f"tpu-sketch-attempt-{attempt_number:02d}-precheck",
            )
        self.logger.error(self.get_session_id() + error_message)
        raise OSError(error_message)

    def generate_python_sketch(self, max_retries: int = 5) -> tuple[bool, str, str]:
        generated_code = ""
        error_message = ""

        for attempt_number in range(1, max_retries + 1):
            attempt_archived = False
            candidate_code = None
            try:
                self.logger.info(
                    self.get_session_id()
                    + f"Attempt {attempt_number} of {max_retries}..."
                )

                generation_name = self._generation_name(attempt_number)
                prompt_to_run = self._prompt_for_attempt(
                    attempt_number, generated_code, error_message
                )

                self.logger.info(
                    self.get_session_id() + f"invoking LLM: {generation_name}"
                )
                response_raw = self.invoke_llm(
                    prompts=prompt_to_run,
                    generation_name=generation_name,
                )
                self.logger.info(
                    self.get_session_id()
                    + f"LLM response received for {generation_name}. Extracting"
                )

                candidate_code = self.extract_code(response_raw, language="python")
                generated_code = candidate_code
                self._validate_generated_script_length(
                    generated_code, attempt_number
                )

                if not generated_code.strip():
                    error_message = "No Python code block found in LLM response."
                    self.logger.error(self.get_session_id() + error_message)
                    if attempt_number == max_retries:
                        return False, generated_code, error_message
                    continue

                self.logger.info(
                    self.get_session_id()
                    + f"Extracted code from LLM response for {generation_name}."
                )
                generated_code = self._postprocess_generated_code(generated_code)
                error_message = self._execute_generated_code(
                    generated_code, attempt_number
                )
                attempt_archived = True

                if error_message is None:
                    self.logger.info(
                        self.get_session_id()
                        + "Successfully generated valid Python sketch on "
                        + f"attempt {attempt_number}."
                    )
                    return True, generated_code, ""

                if attempt_number == max_retries:
                    return (
                        False,
                        generated_code,
                        f"Max retries reached with error. Last error: {error_message}.",
                    )

                self.logger.error(
                    self.get_session_id()
                    + f"Attempt {attempt_number} failed validation as reported above."
                )

            except Exception as e:
                if (
                    candidate_code
                    and not attempt_archived
                    and len(candidate_code.splitlines()) >= self.MIN_GENERATED_LINES
                ):
                    self.archive_generated_attempt(
                        candidate_code,
                        valid=False,
                        error=e,
                        executor="remote",
                        label=f"tpu-sketch-attempt-{attempt_number:02d}-precheck",
                    )
                error_text = str(e)
                if "CUDA error" in error_text:
                    return False, generated_code, f"[FATAL]: CUDA error detected: {e}"
                if "APIConnectionError" in error_text:
                    return False, generated_code, f"[FATAL]: LLM API error: {e}"

                error_message = f"Unexpected error during generation: {e}"
                if attempt_number == max_retries:
                    return (
                        False,
                        generated_code,
                        f"Max retries reached with exception: {error_message}.",
                    )

                self.logger.error(
                    self.get_session_id()
                    + f"Attempt {attempt_number} failed validation with exception"
                    + error_message
                    + f"\nTraceback: {traceback.format_exc()}"
                )

        return False, generated_code, "Unknown error occurred"

    def run(self, max_retries: int = 5) -> None:
        self.logger.info(
            self.get_session_id() + f"Starting {self.__class__.__name__} run..."
        )
        try:
            self.get_user_input()
            is_valid, generated_code, error_message = self.generate_python_sketch(
                max_retries
            )

            if is_valid:
                self.validated_sketch = generated_code
                self.record.update(
                    output={"status": "success", "sketch": self.validated_sketch}
                )
                self.logger.info(
                    self.get_session_id()
                    + f"{self.__class__.__name__} run finished successfully."
                )
            else:
                self.logger.error(
                    self.get_session_id()
                    + f"Failed to generate valid Python sketch.  {error_message}"
                )
                self.log_error(RuntimeError(f"Failed: {error_message}"))
                self.record.update(
                    output={
                        "status": "failed",
                        "last_code": generated_code,
                        "last_error": error_message,
                    }
                )
                self.logger.error(
                    self.get_session_id()
                    + f"{self.__class__.__name__} run finished with failure."
                )

        except KeyboardInterrupt as e:
            e.args = ("Keyboard interrupt received",) + e.args
            self.logger.error(self.get_session_id() + str(e))
            self.record.update(
                output={
                    "status": "failed",
                    "last_error": "Keyboard interrupt received. " + str(e),
                }
            )
            self.log_error(e)
            self.record.update(output={"status": "failed", "last_error": str(e)})
            raise

        except Exception as e:
            self.logger.error(
                self.get_session_id()
                + f"An unexpected error occurred: {traceback.format_exc()}"
            )
            self.record.update(
                output={
                    "status": "failed",
                    "last_error": "An unexpected error occurred. " + str(e),
                }
            )
            self.log_error(e)
            self.record.update(output={"status": "failed", "last_error": str(e)})
