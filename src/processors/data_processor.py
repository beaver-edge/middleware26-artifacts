import os
import sys
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

# import pandas as pd
from dotenv import load_dotenv

import prompt_templates.data_processing_prompts as data_processing_prompts
from base.base_processor import BaseProcessor
from base.prompt_template import PromptTemplate


class DataProcessor(BaseProcessor):
    """
    DataProcessor is responsible for processing datasets by generating and executing
    data processing code as per suggested operations. It interacts with an LLM to
    generate code for each data processing operation and applies it to the dataset.
    """

    load_dotenv()

    def __init__(
        self, llm_strategy, trace_id, num_run=None, benchmark=False, session_id=None
    ):
        """
        Initializes the DataProcessor with the given parameters.

        Args:
            llm_strategy: The language model strategy object providing invoke() method.
            trace_id: Unique identifier for tracing the whole run.
            num_run: Optional run number, to locate a run in a batch run.
            benchmark: Flag to indicate benchmarking(batch run).
            session_id: Optional batch identifier, to locate a batch test among different times of batch tests.
        """
        super().__init__(
            llm_strategy,
            trace_id=trace_id,
            num_run=num_run,
            benchmark=benchmark,
            session_id=session_id,
            task_name=self.get_task_name(),
        )
        self.current_operation = {}
        self.completed_processing_steps = []
        self.purpose = ""
        self.dataset_intro = ""
        self.processing_suggestions = {}
        self.current_operation_index = 0
        self.total_operation_count = 0
        self.files_in_playground_before_execution = []

        self.playground_dir = "./work/data/"

    # @lru_cache(maxsize=2)
    # def get_task_name(self, short=False):
    #     """
    #     Returns the task name.

    #     Args:
    #         short: If True, return abbreviated task name.

    #     Returns:
    #         Task name as a string.
    #     """
    #     if short:
    #         return "dp"
    #     else:
    #         return "data_processor"

    def get_user_input(self):
        """
        Acquires and sets the user input parameters required for data processing.
        Prepares the playground directory by copying necessary data files.
        """
        source_dir = "./inputs/data/"
        os.makedirs(self.playground_dir, exist_ok=True)

        # Remove all .csv files under "playground/"
        for file in os.listdir(self.playground_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(self.playground_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        # Copy the immutable input dataset into the disposable task workspace.

        shutil.copy(os.path.join(source_dir, "fruit_data.csv"), self.playground_dir)

        self.files_in_playground_before_execution = {
            str(path): (path.stat().st_mtime_ns, path.stat().st_size)
            for path in Path(self.playground_dir).iterdir()
            if path.is_file()
        }

        self.purpose = "classify fruits among apple, orange, and banana"

        self.dataset_path = "./work/data/fruit_data.csv"
        self.dataset_intro = """This dataset contains RGB data for a collection of fruits, categorized into three classes: "apple", "orange", and "banana". The dataset contains four columns, respectively represents the fruit name, and the Red Data, Green Data, and Blue data of the fruit."""

        self.logger.info(
            self.get_session_id() + "User input acquired.",
        )

    # @lru_cache(maxsize=32)
    def get_prompt_template(self, template_type) -> List[str| PromptTemplate]:

        # task_prompt is PromptTemplate, system_prompt_format is string
        if template_type == "processing_suggestions":
            task_prompt = PromptTemplate.from_template(
                data_processing_prompts.PROCESSING_SUGGESTIONS_PROMPT
            )
            system_prompt_format = data_processing_prompts.SUGGESTION_RESPONSE_FORMAT
        elif template_type == "processing_suggestions_error_handling":
            task_prompt = PromptTemplate.from_template(
                data_processing_prompts.PROCESSING_SUGGESTIONS_error_handling_PROMPT
            )
            system_prompt_format = data_processing_prompts.SUGGESTION_RESPONSE_FORMAT
        elif template_type == "operation_code":
            task_prompt = PromptTemplate.from_template(
                data_processing_prompts.OPERATION_CODE_PROMPT
            )
            system_prompt_format = data_processing_prompts.CODE_RESPONSE_FORMAT
        elif template_type == "operation_error_handling":
            task_prompt = PromptTemplate.from_template(
                data_processing_prompts.OPERATION_error_handling_PROMPT
            )
            system_prompt_format = data_processing_prompts.CODE_RESPONSE_FORMAT
        else:
            raise ValueError(f"Unknown template type: {template_type}")

        prompts_list = [
            data_processing_prompts.CONTEXT_PROMPT,
            system_prompt_format,
            task_prompt,
        ]
        return prompts_list
    
    def raise_data_processor_error(self, error_message:str):
     
        self.logger.error(self.get_session_id()+ error_message)
        self.log_error( Exception(error_message) )
        self.record.update(output={"status": "failed", "last_error": error_message})
        sys.exit(1)


    def generate_processing_suggestions(self, max_retries):
        """
        Generates a suggestion table for data processing operations using the LLM.
        Retries with error handling if JSON parsing fails.

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            A dictionary containing suggested data processing operations.
        """
        self.dataset_summary()
        response_raw = None
        error = None

        for attempt in range(1, max_retries + 1):
            try:
                if attempt == 1:
                    prompt_to_run = self.get_prompt_template("processing_suggestions")

                    prompt_to_run[2] = prompt_to_run[2].format(
                        purpose=self.purpose,
                        dataset_summary=self.dataset_summary_str,
                        dataset_intro=self.dataset_intro,
                    )
                    self.logger.info(
                        self.get_session_id()
                        + "Generating suggestion table using LLM attempt #1...",
                    )
                else:
                    self.logger.info(
                        self.get_session_id()
                        + f"Retrying suggestion table generation (attempt #{attempt})..."
                    )
                    prompt_to_run = self.get_prompt_template(
                        "processing_suggestions_error_handling"
                    )
                    prompt_to_run[2] = prompt_to_run[2].format(
                        purpose=self.purpose,
                        dataset_summary=self.dataset_summary_str,
                        dataset_intro=self.dataset_intro,
                        executed_code=response_raw,
                        error_info="Failed to parse response as valid JSON: " + error if error else "",
                    )

                response_raw = self.invoke_llm(
                    prompt_to_run,
                    generation_name=self.trace_id[:2]
                    + "_"
                    + self.get_task_name(short=True)
                    + f"_processing_suggestions_gen_attempt#{attempt}",
                )

                content = (
                    response_raw.strip()
                    .replace("```json", "")
                    .replace("```", "")
                    .replace("[", "")
                    .replace("]", "")
                    .replace("\\n'", "")
                    .strip()
                )

                self.processing_suggestions = self.extract_processing_suggestions(content)

                self.total_operation_count = len(self.processing_suggestions)

                self.logger.info(
                    self.get_session_id()
                    + "Suggestions received and parsed successfully:",
                )
                self.logger.info(
                    f"Suggestions: {json.dumps(self.processing_suggestions, indent=2)}",
                )
                return True, self.processing_suggestions, ""

            except json.JSONDecodeError as e:
                self.logger.error(
                    self.get_session_id()
                    + f"Attempt {attempt}: Failed to parse LLM response as a dictionary: {content}"
                )
                error = str(e)
                if attempt == max_retries:
                    error_message = (
                        "Failed to generate valid suggestion table after "
                        f"{max_retries} attempts. Last error: {error}"
                    )
                    self.raise_data_processor_error(error_message)
            except Exception as e:
                error = str(e)
                if "CUDA error" in error:
                    error_message=f"[FATAL]: CUDA error detected: {e}"
                    self.raise_data_processor_error(error_message)
                elif "APIConnectionError" in error:
                    error_message= f"[FATAL]: LLM API error: {e}"
                    self.raise_data_processor_error(error_message)
                else:
                    self.logger.error(
                        self.get_session_id()
                        + f"Exception caught in prompting attempt {attempt}, loop continues.\nError: {str(e)}\nTraceback: {traceback.format_exc()}"
                    )
        self.raise_data_processor_error(f"Failed to generate suggestions: {error}")

    def generate_operation_code(self, max_retries=5):
        """
        Generates and executes the code for each data processing operation using the LLM.
        Retries upon failure up to max_retries times.

        Args:
            max_retries: Maximum number of retries for code generation.

        Raises:
            Exception: If code generation fails after maximum retries.
        """
  
        code = None
        error = None
        current_operation = self.current_operation

        for attempt in range(1, max_retries + 1):

            if attempt == 1:

                prompts = self.get_prompt_template("operation_code")
                prompts[2] = prompts[2].format(
                    operation_description=current_operation["explanation"],
                    current_operation_name=current_operation["operation"],
                    dataset_path=self.dataset_path,
                    completed_processing_steps=str(
                        self.completed_processing_steps
                    ),
                    dataset_summary=self.dataset_summary_str,
                )
            else:
                self.logger.info(
                    self.get_session_id()
                    + f"Trying to solve the error, re-attempting... (attempt NO.{attempt})",
                )
                prompts = self.get_prompt_template("operation_error_handling")

                prompts[2] = prompts[2].format(
                    current_operation=f"{current_operation['operation']}: {current_operation['explanation']}",
                    executed_code=code,
                    error_info=error,
                    dataset_path=self.dataset_path,
                    completed_processing_steps=self.completed_processing_steps,
                    dataset_summary=self.dataset_summary_str,
                )

            response_raw = self.invoke_llm(
                prompts,
                generation_name=f"{self.trace_id[:2]}_{self.get_task_name(short=True)}_"
                f"{'operation_code_gen' if attempt == 1 else 'error_handling'}"
                f"#{len(self.completed_processing_steps) + 1}"
                f"{'_attempt#' + str(attempt) if attempt > 1 else ''}",
            )
            try:
                code = self.extract_code(response_raw)

                self.logger.info(
                    self.get_session_id()
                    + f"Operation #{self.current_operation_index}/{self.total_operation_count} "
                    + current_operation["operation"]
                    + ": "
                    + f"{'First attempt' if attempt == 1 else 'error_handling attempt NO.' + str(attempt)} code received. Executing...",
                )
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                error = self.execute_code(
                    code,
                    self.playground_dir + "tmp_" + timestamp + ".py",
                    artifact_label=(
                        f"operation-{self.current_operation_index:02d}-"
                        f"{current_operation['operation']}-attempt-{attempt:02d}"
                    ),
                )
                if not error:
                    self.update_dataset_path(response_raw)
                    self.logger.info(
                        self.get_session_id() + "Code execution successful."
                    )
                    return None
                else:
                    self.logger.error(
                        self.get_session_id()
                        + f"Error in code execution attempt NO.{attempt} for operation  #{self.current_operation_index}/{self.total_operation_count} "
                        + str(current_operation)
                        + ": "
                        + error
                    )
                if attempt == max_retries:
                    error_message = f"Failed to generate valid code after the max {max_retries} attempts. Last error from code execution: {error}"  
                    self.raise_data_processor_error(error_message)
                    
            except Exception as e:
                error = str(e)
                if "CUDA error" in error:
                    error_message=f"[FATAL]: CUDA error detected: {e}"
                    self.raise_data_processor_error(error_message)
                elif "APIConnectionError" in error:
                    error_message= f"[FATAL]: LLM API error: {e}"
                    self.raise_data_processor_error(error_message)
               
                elif attempt == max_retries:
                    error_message=f"Failed to generate valid code after the max {max_retries} attempts. Last error from code execution: {error}"
                    self.raise_data_processor_error(error_message)
                else:
                    self.logger.error(
                        self.get_session_id()
                        + f"Exception caught in prompting attempt {attempt}: {str(e)}, loop continues.\nError: {str(e)}\nTraceback: {traceback.format_exc()}"
                    )

    def update_dataset_path(self, content):
        """
        Extracts dataset paths from LLM response in various formats.
        Recursively searches through nested dictionaries to find all .csv paths.

        Args:
            content: The raw response from the LLM, can be string or object

        Returns:
            None

        Raises:
            ValueError: If no valid dataset paths are found
        """

        json_content = None
        if "```json" in content:
            json_content = content.split("```json", 1)[-1].split("```", 1)[0].strip()
        elif "```" in content and "```python" not in content:
            json_content = content.split("```", 1)[-1].split("```", 1)[0].strip()
        else:
            json_content = content.strip()

        try:
            parsed = json.loads(json_content)
        except json.JSONDecodeError:
            self.dataset_path = json_content.strip().strip('"')
            return

        if isinstance(parsed, str):
            self.dataset_path = parsed
            return

        if isinstance(parsed, dict):
            for value in reversed(list(parsed.values())):
                if isinstance(value, str) and value.endswith(".csv"):
                    self.dataset_path = value
                    return
            for value in parsed.values():
                if isinstance(value, str):
                    self.dataset_path = value
                    return

        raise ValueError("No dataset path found in LLM response JSON block")

    def extract_processing_suggestions(self, content):
        """
        Extract suggestion table from LLM output content.
        Handle both single dictionary and multiple dictionaries formats.

        Args:
            content (str): LLM output content

        Returns:
            dict: Combined suggestion table

        Raises:
            json.JSONDecodeError: If content cannot be parsed as JSON
        """
        # Remove any whitespace and newlines from start/end
        content = content.strip()

        # First try to parse as a single dictionary
        try:
            processing_suggestions = json.loads(content)
            return processing_suggestions
        except json.JSONDecodeError:
            # If that fails, try parsing as multiple dictionaries
            processing_suggestions = {}
            # Split by newlines and process each line
            for line in content.split("\n"):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        # Parse each line as a dictionary
                        line_dict = json.loads(line)
                        # Merge into main dictionary
                        processing_suggestions.update(line_dict)
                    except json.JSONDecodeError:
                        continue

            if processing_suggestions:  # If we successfully parsed any dictionaries
                return processing_suggestions

            # If we get here, we couldn't parse the content - raise JSONDecodeError
            # instead of ValueError to match error handling in generate_processing_suggestions
            raise json.JSONDecodeError(
                "Could not parse suggestion table from content", content, 0
            )

    def run(self):
        """
        Executes the data processing workflow by acquiring user input, generating a suggestion table, and performing each suggested data processing operation.
        """
        print(
            "Model name: " + self.llm_strategy.model_name,
            "; Trace name: " + self.trace_name,
            "; Batch ID: " + self.session_id if self.session_id else "",
        )

        self.logger.info(
                self.get_session_id() + "Data processing started.",
            )
        try:
 
            self.get_user_input()
            self.generate_processing_suggestions(max_retries=int(5))

            self.current_operation_index = 1
            for operation_name, operation_description in self.processing_suggestions.items():

                self.current_operation = {
                    "operation": operation_name,
                    "explanation": operation_description,
                }
                self.logger.info(
                    self.get_session_id()
                    + f"STARTED: Operation #{self.current_operation_index}/{self.total_operation_count} "
                    + operation_name
                    + ".",
                )
                self.generate_operation_code(max_retries=int(5))

                self.completed_processing_steps.append(
                    {str(operation_name): str(operation_description)}
                )
                self.logger.info(
                    self.get_session_id()
                    + f"COMPLETED: Operation #{self.current_operation_index}/{self.total_operation_count} "
                    + operation_name
                    + ".",
                )
                self.logger.info(self.get_session_id() + 40 * "#")
                self.current_operation_index += 1

            self.record.update(
                    output={"status": "success"}
                )
            output_paths = {Path(self.dataset_path)}
            for path in Path(self.playground_dir).iterdir():
                if not path.is_file() or path.suffix.lower() not in {".csv", ".png"}:
                    continue
                before = self.files_in_playground_before_execution.get(str(path))
                after = (path.stat().st_mtime_ns, path.stat().st_size)
                if before != after:
                    output_paths.add(path)
            for output_path in sorted(output_paths):
                self.archive_output_file(output_path, f"data_{output_path.stem}")
            self.logger.info(
                self.get_session_id()
                + f"Data processing completed, here are the executed operations:  {json.dumps(self.processing_suggestions, indent=2).strip()[1:-1]}",
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
            # Handle unexpected errors
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

 
