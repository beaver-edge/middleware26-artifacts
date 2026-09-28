"""!
BaseProcessor Module

This module defines the BaseProcessor abstract base class, which serves as the foundational class for all processors in the project. It provides shared functionalities such as environment setup, logging, LLM invocation with tracing, code execution, and error handling.
"""
#%%
import logging
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import final

import colorlog
import pandas as pd
from dotenv import load_dotenv

try:
    # Try relative import first (for package usage)
    from .llm_strategy import LLMStrategy
except ImportError:
    # Fall back to absolute import (for direct module execution)
    from llm_strategy import LLMStrategy

# Third-party imports

class LocalRunRecord:
    """Local in-memory workflow outcome; never sends telemetry."""

    def __init__(self):
        self.spans = []
        self.output = {}

    def span(self, **kwargs):
        self.spans.append(kwargs)
        return self

    def update(self, **kwargs):
        self.output.update(kwargs)
        return self


class BaseProcessor(ABC):
    """!
    BaseProcessor is an abstract base class that provides common functionalities
    for all processor classes in the project.
    """

    def __init__(
        self,
        llm_strategy: LLMStrategy,
        trace_id,
        task_name,
        num_run=None,
        benchmark=False,
        session_id=None,

    ):
        """!
        Initializes the BaseProcessor with the given parameters.

        Args:
            llm_strategy: The language model strategy object providing invoke() method.
            trace_id: Unique identifier for tracing the whole run.
            num_run: Optional run number, to locate a run in a batch run.
            benchmark: Flag to indicate benchmarking(batch run).
            session_id: Optional batch identifier, to locate a batch test among different times of batch tests.
        """
        self.llm_strategy = llm_strategy
        self.load_environment()
        self.setup_logging()

        self.current_option = {}
        self.input_datatype = ""
        self.output_datatype = ""
        self.dataset_path = ""
        self.chat_history_key_value = ""
        self.dataset_summary_str = ""

        # For tracing
        self.trace_id = trace_id
        self.tags = [
            self.get_task_name(),
            "benchmark" if benchmark else "experiment",
            self.model_name,
            self.llm_strategy.model_provider,
        ]

        if session_id:
            if num_run is None:
                raise ValueError("num_run is required if session_id is provided")
            self.trace_metadata = {"num_run": num_run,"local_host":str(socket.gethostname())}
            self.session_id = session_id
        else:
            self.trace_metadata = {}
            self.session_id = None
        self.trace_name = self.trace_id[:4] + "_" + task_name

        self.record = LocalRunRecord()
        self._generated_attempt_count = 0
        self._artifact_attempts = []
        self._artifact_outputs = []
        self._artifact_outcome = None
        self.artifact_run_dir = self._build_artifact_run_dir(num_run)
        self._write_artifact_manifest()

    def _build_artifact_run_dir(self, num_run):
        """Return the canonical, task-scoped directory for this workflow run."""
        task_names = {
            "data_processor": "data",
            "model_converter": "convert",
            "arduino_sketch_generator": "ardsketch",
            "python_sketch_generator": "pysketch",
            "tpu_sketch_generator": "tpusketch",
        }
        task = task_names.get(self.get_task_name(), self.get_task_name())
        root = Path(os.getenv("BEAVER_ARTIFACT_ROOT", "artifacts"))
        session = self.session_id or "standalone"
        safe_session = re.sub(r"[^A-Za-z0-9_.-]", "_", str(session))
        run_number = num_run if num_run is not None else 1
        run = f"run-{run_number:03d}-{self.trace_id}"
        return root / task / safe_session / run

    def _artifact_path_for_display(self, path):
        path = Path(path)
        try:
            return str(path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(path.resolve())

    def _write_artifact_manifest(self):
        self.artifact_run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "task": self.get_task_name(),
            "model": self.model_name,
            "provider": self.llm_strategy.model_provider,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "attempts": self._artifact_attempts,
            "outputs": self._artifact_outputs,
            "outcome": self._artifact_outcome,
        }
        (self.artifact_run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def archive_generated_attempt(
        self,
        code,
        *,
        valid,
        error=None,
        is_arduino=False,
        executor="local",
        label=None,
        command=None,
        returncode=None,
        extension=None,
    ):
        """Retain one generated candidate under an explicit valid/invalid path."""
        self._generated_attempt_count += 1
        attempt = self._generated_attempt_count
        status = "valid" if valid else "invalid"
        suffix = extension or (".ino" if is_arduino else ".py")
        safe_label = re.sub(
            r"[^A-Za-z0-9_.-]", "_", label or self.get_task_name(short=True)
        )
        filename = (
            f"{status}_attempt-{attempt:03d}_{safe_label}_{self.model_file_name}{suffix}"
        )
        status_dir = self.artifact_run_dir / status
        status_dir.mkdir(parents=True, exist_ok=True)
        code_path = status_dir / filename
        code_path.write_text(str(code), encoding="utf-8")

        relative_code_path = self._artifact_path_for_display(code_path)
        record = {
            "attempt": attempt,
            "status": status,
            "label": label or self.get_task_name(short=True),
            "executor": executor,
            "language": "arduino" if is_arduino else "python",
            "code_path": relative_code_path,
            "returncode": returncode,
            "command": command,
            "error": str(error) if error else None,
        }
        metadata_path = code_path.with_suffix(code_path.suffix + ".json")
        metadata_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        record["metadata_path"] = self._artifact_path_for_display(metadata_path)
        self._artifact_attempts.append(record)
        self._write_artifact_manifest()

        marker = "GENERATION_VALIDATED" if valid else "GENERATION_VALIDATION_FAILED"
        message = f"{marker}: status={status.upper()} path={relative_code_path}"
        print(message)
        if valid:
            self.logger.info(self.get_session_id() + message)
        else:
            self.logger.error(self.get_session_id() + message)
        return code_path

    def archive_output_file(self, source, label, status="valid"):
        """Copy a workflow output into this run's evaluator-facing output folder."""
        source = Path(source)
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Expected workflow output is missing or empty: {source}")
        if status not in {"valid", "invalid"}:
            raise ValueError(f"Unknown archived-output status: {status}")
        output_dir = self.artifact_run_dir / "outputs" / status
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
        destination = output_dir / f"{status}_{safe_label}_{source.name}"
        shutil.copy2(source, destination)
        record = {
            "label": label,
            "status": status,
            "source": str(source),
            "archived_path": self._artifact_path_for_display(destination),
        }
        self._artifact_outputs.append(record)
        self._write_artifact_manifest()
        print(
            "WORKFLOW_OUTPUT_ARCHIVED: "
            f"status={status.upper()} path={record['archived_path']}"
        )
        self.logger.info(
            self.get_session_id() + f"Archived workflow output: {record['archived_path']}"
        )
        return destination

    def artifact_summary(self):
        """Return paths and counts suitable for the CLI outcome JSON."""
        return {
            "artifact_run_dir": self._artifact_path_for_display(self.artifact_run_dir),
            "valid_attempts": sum(
                attempt["status"] == "valid" for attempt in self._artifact_attempts
            ),
            "invalid_attempts": sum(
                attempt["status"] == "invalid" for attempt in self._artifact_attempts
            ),
            "archived_outputs": len(self._artifact_outputs),
        }

    def finalize_artifact_manifest(self, outcome):
        """Persist the normalized workflow result beside the attempt evidence."""
        self._artifact_outcome = dict(outcome)
        self._write_artifact_manifest()

    def get_session_id(self):
        """!
        Get the batch id for batch run.
        """
        if self.session_id:
            return f"{str(self.session_id)}|{str(self.trace_id)[:4]}_{self.get_task_name(short=True)}: "
        else:
            return f"{str(self.trace_id)}: "

    def get_task_name(self, short=False):
        """
        Returns the task name based on the class name.

        Args:
            short: If True, returns abbreviated task name (first letters of each word).
                If False, returns full task name in snake_case.

        Returns:
            Task name as a string.
        """
        # Convert class name from CamelCase to snake_case
        class_name = self.__class__.__name__
        task_name = "".join(
            ["_" + c.lower() if c.isupper() else c for c in class_name]
        ).lstrip("_")

        if not short:
            return task_name

        # Get first letter of each part for short version
        return "".join(part[0].lower() for part in task_name.split("_"))

    def load_environment(self):
        """!
        Loads environment variables from the .env file and sets up necessary environment settings.
        """
        load_dotenv()
        self.model_name = self.llm_strategy.model_name
        self.model_file_name = re.sub(r"[^A-Za-z0-9_.-]", "_", self.model_name)

    @final
    def setup_logging(self):
        """!
        Sets up logging with both console and file handlers using colorlog for colored console output.
        """
        self.logger = colorlog.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if self.logger.handlers:
            self.logger.propagate = False
            return

        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)s %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        file_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        # make sure the log file is created
        if not os.path.exists(f"logs/{self.__class__.__name__}.log"):
            os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler(f"logs/{self.__class__.__name__}.log")
        console_handler = colorlog.StreamHandler()

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        file_handler.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    def invoke_llm(self, prompts, generation_name):
        return self.llm_strategy.invoke(prompts, {"generation_name": generation_name})

    def log_error(self, error):
        self.logger.error(self.get_session_id() + str(error))

    @abstractmethod
    def get_user_input(self):
        """!
        Abstract method to acquire user input. Must be implemented by subclasses.
        """
        pass

    @final
    def dataset_summary(self) -> str:
        """!
        Generates a summary of the dataset including shape, statistics, and column names.
        To give the inspiration to the LLM.

        Returns:
            A string summarizing the dataset.

        Raises:
            Exception: If reading the dataset fails.
        """
        try:
            dataframe = pd.read_csv(self.dataset_path)
            description = dataframe.describe().to_dict()
            head = dataframe.head().to_dict()

            # Create a simple string representation without dictionary formatting
            summary_str = (
                f"Dataset shape: {dataframe.shape}\n"
                f"Dataset descriptive statistics: {description}\n"
                f"Dataset first 5 rows: {head}\n"
                f"Column names: {list(dataframe.columns)}\n"
                f"Note: column names are case sensitive, remember that."
            )
            summary_str = summary_str.replace("{", "{{").replace("}", "}}")
            self.dataset_summary_str = summary_str

            self.logger.info(
                self.get_session_id()
                + f"Dataset summary generated for {self.dataset_path}"
            )
            return self.dataset_summary_str

        except Exception as e:
            self.logger.error(
                self.get_session_id()
                + f"Error generating dataset summary for {self.dataset_path}: {e}"
            )
            raise e

    @abstractmethod
    def run(self):
        """!
        Abstract method to execute the processor. Must be implemented by subclasses.
        """
        pass

    def prepare_execution_environment(
        self,
        execution_workspace,
        is_arduino=False,
        fixed_timestamp=None,
        arduino_model_header=None,
    ):
        """
        Prepare paths and directories for code execution/compilation.
        Creates an ephemeral working path and its execution command. Classified
        source is retained separately by ``archive_generated_attempt``.

        Args:
            execution_workspace: Base directory for execution
            is_arduino: Whether to prepare for Arduino (.ino) or Python execution
            fixed_timestamp: ONLY for testing, if provided, use this timestamp instead of the current time
            arduino_model_header: Source model header copied beside an Arduino sketch.

        Returns:
            tuple: (work_dir, tmp_file, execute_command)
        """
        timestamp = (
            datetime.now().strftime("%Y%m%d%H%M%S")
            if fixed_timestamp is None
            else fixed_timestamp
        )

        # Get file paths based on language type and timestamp
        work_dir, tmp_file = self._create_working_paths(
            execution_workspace, timestamp, is_arduino
        )
        # Set up the execution command and any special requirements
        execute_command = self._create_execution_command(
            work_dir,
            tmp_file,
            execution_workspace,
            is_arduino,
            arduino_model_header=arduino_model_header,
        )

        return work_dir, tmp_file, execute_command

 
    def _create_working_paths(self, execution_workspace, fixed_timestamp, is_arduino):
        """
        Create working directory and temporary file paths.

        Args:
            execution_workspace: Base directory
            fixed_timestamp: Current timestamp string
            is_arduino: Whether this is for Arduino code

        Returns:
            tuple: (work_dir, tmp_file). E.g., for task_name="test-task", model_name="gpt-3.5-turbo", timestamp="20210101120000", the paths for python code will be:
            work_dir = "tmp_20210101120000_test-task_gpt-3.5-turbo"
            tmp_file = "tmp_20210101120000_test-task_gpt-3.5-turbo.py". for arduino code, the paths will be:
            work_dir = "compiling_20210101120000_gpt-3.5-turbo"
            tmp_file = "compiling_20210101120000_gpt-3.5-turbo.ino"

        """

        file_ext = ".ino" if is_arduino else ".py"

        work_dir_name = (
            f"compiling_{fixed_timestamp}_{self.model_file_name}"
            if is_arduino
            else f"tmp_{fixed_timestamp}_{self.get_task_name(short=True)}_{self.model_file_name}"
        )

        # Create working directory
        work_dir = os.path.join(execution_workspace, work_dir_name)
        os.makedirs(work_dir, exist_ok=True)

        # Create temporary file path
        # self.logger.info(f"TEST PRINT work_dir: {work_dir}")
        tmp_file = os.path.join(work_dir, f"{work_dir_name}{file_ext}")
        # self.logger.info(
        #     f"TEST PRINT\n tmp_file is work_dir \n{work_dir}\n + tmp_file name \n{str(work_dir_name)}{file_ext}\nwhich is \n{tmp_file}"
        # )
        return work_dir, tmp_file

    def _create_execution_command(
        self,
        work_dir,
        tmp_file,
        execution_workspace,
        is_arduino,
        arduino_model_header=None,
    ):
        """
        Prepare the execution command and handle any special setup requirements.

        Args:
            work_dir: Working directory path
            tmp_file: Temporary file path
            execution_workspace: Base directory
            is_arduino: Whether this is for Arduino code

        Returns:
            list: Command to execute as list of strings
        """
        if is_arduino:
            # Copy required model.h file for Arduino
            shutil.copy(
                arduino_model_header
                or os.path.join(execution_workspace, "model.h"),
                os.path.join(work_dir, "model.h"),
            )
            execute_command = [
                "arduino-cli",
                "compile",
                "--fqbn",
                "arduino:mbed:nano33ble",
                tmp_file,
            ]
        else:
            execute_command = [sys.executable, tmp_file]

        return execute_command

    def execute_code(
        self,
        code,
        execution_workspace,
        is_arduino=False,
        local_retry=False,
        remote_execution=False,
        artifact_label=None,
        expected_outputs=None,
        remote_output_path=None,
        output_validator=None,
        arduino_model_header=None,
    ):
        """!
        Execute the code snippet and return the error message if any.

        Args:
            code (str): The code snippet to execute.
            execution_workspace (str): The path to the workspace directory.
            is_arduino (bool): Indicates if the code is an Arduino sketch (.ino) file.
            local_retry (bool): Whether this is a retry after installing dependencies.
            remote_execution (bool): Whether to execute remotely via SSH file transfer.
            arduino_model_header (str | None): Arduino model header source path.

        Returns:
            str: The error message if any.
            None: If the code executed successfully.

        """
        # Handle remote execution via SSH file transfer
        if remote_execution:
            if os.getenv("REMOTE_EXECUTION_ENABLED", "false").lower() not in {
                "1",
                "true",
                "yes",
                "on",
            }:
                error = "Remote execution is disabled. Set REMOTE_EXECUTION_ENABLED=true to enable it."
                self.archive_generated_attempt(
                    code,
                    valid=False,
                    error=error,
                    executor="remote",
                    label=artifact_label,
                )
                return error
            error = None
            if remote_output_path:
                try:
                    self._remove_remote_output_before_validation(remote_output_path)
                except Exception as exc:
                    error = str(exc)
            if error is None:
                error = self._execute_code_via_ssh(code, execution_workspace)
            if remote_output_path:
                if error is None:
                    try:
                        self._copy_remote_output_to_archive(
                            remote_output_path,
                            output_validator,
                            status="valid",
                            required=True,
                        )
                    except Exception as exc:
                        error = str(exc)
                else:
                    # A timed-out or crashing remote program may still have
                    # produced a partial file. Preserve it as invalid evidence.
                    try:
                        self._copy_remote_output_to_archive(
                            remote_output_path,
                            status="invalid",
                            required=False,
                        )
                    except Exception as archive_exc:
                        self.logger.warning(
                            self.get_session_id()
                            + "Could not archive partial remote output: "
                            + str(archive_exc)
                        )
            self.archive_generated_attempt(
                code,
                valid=error is None,
                error=error,
                executor="remote",
                label=artifact_label,
                returncode=0 if error is None else 1,
            )
            return error
        
        # Continue with local execution
        work_dir, tmp_file, execute_command = (
            self.prepare_execution_environment(
                execution_workspace,
                is_arduino,
                arduino_model_header=arduino_model_header,
            )
        )

        os.makedirs(os.path.dirname(tmp_file), exist_ok=True)

        try:
            with open(tmp_file, "w") as file:
                file.write(code)
        except (IOError, OSError) as e:
            self.logger.error(
                self.get_session_id() + f"Failed to write code to file {tmp_file}: {e}"
            )
            self.archive_generated_attempt(
                code,
                valid=False,
                error=e,
                is_arduino=is_arduino,
                label=artifact_label,
            )
            self._cleanup_execution_files(work_dir, tmp_file)
            return str(e)

        self.logger.info(
            self.get_session_id() + "Executing/compiling the code snippet..."
        )

        try:
            validation_started_ns = time.time_ns()
            result = subprocess.run(execute_command, capture_output=True, text=True)
            has_error = self.contains_error_indicator(result)

            output_error = None
            if result.returncode == 0 and not has_error:
                missing_outputs = []
                for expected_output in expected_outputs or []:
                    output_path = Path(expected_output)
                    if (
                        not output_path.is_file()
                        or output_path.stat().st_size == 0
                        or output_path.stat().st_mtime_ns < validation_started_ns
                    ):
                        missing_outputs.append(str(output_path))
                if missing_outputs:
                    output_error = (
                        "Generated code exited successfully but did not create fresh, "
                        "non-empty expected output(s): " + ", ".join(missing_outputs)
                    )
                elif output_validator:
                    try:
                        for expected_output in expected_outputs or []:
                            output_validator(Path(expected_output))
                    except Exception as exc:
                        output_error = str(exc)

            if result.returncode == 0 and not has_error and output_error is None:
                self._handle_successful_execution(result)
                self.archive_generated_attempt(
                    code,
                    valid=True,
                    is_arduino=is_arduino,
                    label=artifact_label,
                    command=execute_command,
                    returncode=result.returncode,
                )
                self._cleanup_execution_files(work_dir, tmp_file)
                return None
            else:
                # If there's an error, handle it
                error_message = output_error or result.stderr or result.stdout
                self.logger.error(
                    self.get_session_id()
                    + f"Execution failed:\n{error_message}"
                )
                for expected_output in expected_outputs or []:
                    output_path = Path(expected_output)
                    if (
                        output_path.is_file()
                        and output_path.stat().st_size > 0
                        and output_path.stat().st_mtime_ns >= validation_started_ns
                    ):
                        try:
                            self.archive_output_file(
                                output_path,
                                f"rejected_{output_path.stem}",
                                status="invalid",
                            )
                        except Exception as archive_exc:
                            self.logger.warning(
                                self.get_session_id()
                                + "Could not archive rejected local output: "
                                + str(archive_exc)
                            )
                self.archive_generated_attempt(
                    code,
                    valid=False,
                    error=error_message,
                    is_arduino=is_arduino,
                    label=artifact_label,
                    command=execute_command,
                    returncode=result.returncode,
                )
                self._cleanup_execution_files(work_dir, tmp_file)

                return error_message
        except Exception as e:
            self.logger.error(
                self.get_session_id() + f"Exception during code execution: {e}"
            )
            traceback.print_exc()
            self.archive_generated_attempt(
                code,
                valid=False,
                error=e,
                is_arduino=is_arduino,
                label=artifact_label,
                command=execute_command,
            )
            self._cleanup_execution_files(work_dir, tmp_file)
            return str(e)

    def _remove_remote_output_before_validation(self, remote_output_path):
        """Prevent a stale board-side result from validating a new candidate."""
        remote_host = os.getenv("REMOTE_HOST")
        if not remote_host:
            raise RuntimeError("REMOTE_HOST is required to prepare TPU validation")
        removed = subprocess.run(
            [
                "ssh",
                "-o",
                "ControlMaster=no",
                "-o",
                "ControlPath=none",
                remote_host,
                f"rm -f -- {shlex.quote(str(remote_output_path))}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if removed.returncode != 0:
            raise RuntimeError(
                "Failed to clear the previous remote output before validation: "
                + (removed.stderr or removed.stdout)
            )

    def _copy_remote_output_to_archive(
        self,
        remote_output_path,
        output_validator=None,
        *,
        status="valid",
        required=True,
    ):
        """Copy a successful remote result locally before marking code valid."""
        remote_host = os.getenv("REMOTE_HOST")
        if not remote_host:
            raise RuntimeError("REMOTE_HOST is required to archive the TPU output")
        ssh_options = ["-o", "ControlMaster=no", "-o", "ControlPath=none"]
        with tempfile.TemporaryDirectory() as temporary_directory:
            local_copy = Path(temporary_directory) / Path(remote_output_path).name
            copied = subprocess.run(
                [
                    "scp",
                    *ssh_options,
                    f"{remote_host}:{remote_output_path}",
                    str(local_copy),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if copied.returncode != 0:
                if not required:
                    return None
                raise RuntimeError(
                    "Failed to copy the validated remote output to the local archive: "
                    + (copied.stderr or copied.stdout)
                )
            if output_validator and status == "valid":
                try:
                    output_validator(local_copy)
                except Exception:
                    self.archive_output_file(
                        local_copy, "remote_annotated_video", status="invalid"
                    )
                    raise
            return self.archive_output_file(
                local_copy, "remote_annotated_video", status=status
            )

    @staticmethod
    def validate_readable_video(path):
        """Raise when an expected video has no decodable first frame."""
        import cv2

        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            raise RuntimeError(f"Generated video has no readable frame: {path}")

    def _execute_code_via_ssh(self, code, execution_workspace):
        """
        Execute code remotely via direct SSH file transfer and execution.
        
        This method:
        1. Creates a temporary Python file locally
        2. Transfers it to the remote machine via SCP
        3. Executes it remotely via SSH with real-time output streaming
        4. Captures stdout, stderr, and traceback information
        5. Cleans up the temporary file on the remote machine
        
        Args:
            code (str): The Python code to execute remotely
            execution_workspace (str): Path to the workspace directory (unused for direct transfer)
            
        Returns:
            str: Error message if execution failed, None if successful
        """
        # Configuration for remote machine
        remote_host = os.getenv('REMOTE_HOST')
        remote_path = os.getenv('REMOTE_EXEC_PATH')
        remote_env_activate = os.getenv('REMOTE_PYTHON_ENV')
        remote_python = os.getenv('REMOTE_PYTHON_EXECUTABLE')      
        ssh_options = ['-o', 'ControlMaster=no', '-o', 'ControlPath=none']

        # Generate unique script name with timestamp
        script_id = f"script_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        script_filename = f"{script_id}.py"
        remote_script_path = f"{remote_path}/{script_filename}"
        
        try:
            check_result = subprocess.run(['ssh', *ssh_options, remote_host, "cat /sys/class/apex/apex_0/device_owner"], capture_output=True, text=True, timeout=10)
            if check_result.returncode == 0:
                owner_pid = check_result.stdout.strip()
                self.logger.info(self.get_session_id() + f"TPU Device owner PID: {owner_pid}")
                # If the owner PID is not empty and not "0", kill the process
                if owner_pid and owner_pid != "0":
                    self.logger.warning(self.get_session_id() + f"TPU Device owned by PID {owner_pid}, killing process")
                    kill_cmd = ['ssh', *ssh_options, remote_host, f'kill {owner_pid} || true']
                    subprocess.run(kill_cmd, capture_output=True, text=True, timeout=10)
        except Exception:
            pass 
        
        try:
            # Create temporary local file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                local_script_path = temp_file.name
            
            # self.logger.info(self.get_session_id() + f"Created local temp script: {local_script_path}")
            
            # Step 0: Commands making
            
            # To create remote directory if it doesn't exist 
            mkdir_command = [
                'ssh', *ssh_options, remote_host,
                f'mkdir -p {remote_path}'
            ]
            
            # To transfer the script to the remote machine
            scp_command = [
                'scp', *ssh_options, local_script_path,
                f'{remote_host}:{remote_script_path}'
            ]
            
            # To execute the script remotely
            ssh_command = [
                'ssh', *ssh_options, remote_host,
                f'cd {remote_path} && {remote_python} {script_filename}'
            ]
            
            # To cleanup the remote script after execution
            cleanup_command = [
                'ssh', *ssh_options, remote_host, f'rm -f {remote_script_path}'
            ]
            
            # Kill leftover processes if any
            kill_command = [
                'ssh', *ssh_options, remote_host,
                f'pkill -f {script_id} || true'  # Ignore if no process found
            ]
            
            # Step 1: Create remote directory if it doesn't exist
            
            self.logger.info(self.get_session_id() + f"Creating remote directory: {remote_path}")
            mkdir_result = subprocess.run(mkdir_command, capture_output=True, text=True, timeout=30)
            
            if mkdir_result.returncode != 0:
                error_msg = f"Failed to create remote directory: {mkdir_result.stderr}"
                self.logger.error(self.get_session_id() + error_msg)
                return error_msg
            
            # Step 2: Transfer script to remote machine
            
            self.logger.info(self.get_session_id() + f"Transferring script to remote: {remote_script_path}")
            scp_result = subprocess.run(scp_command, capture_output=True, text=True, timeout=60)
            
            if scp_result.returncode != 0:
                error_msg = f"Failed to transfer script: {scp_result.stderr}"
                self.logger.error(self.get_session_id() + error_msg)
                return error_msg
            
            # Step 3: Execute script remotely with real-time output streaming
            
            self.logger.info(self.get_session_id() + f"Executing script remotely: {script_filename}")
            
            # Execute with real-time output capture
            execution_error = self._stream_ssh_execution(ssh_command, script_id)
            
            # Step 4: Cleanup remote file
            
            self.logger.info(self.get_session_id() + f"Cleaning up remote script: {remote_script_path}")
            # kill leftover processes if any
            kill_process_result = subprocess.run(kill_command, capture_output=True, text=True, timeout=30)
            if kill_process_result.returncode != 0:
                self.logger.warning(self.get_session_id() + f"Failed to kill leftover processes: {str(kill_process_result)}")
                
            cleanup_result = subprocess.run(cleanup_command, capture_output=True, text=True, timeout=30)
            
            if cleanup_result.returncode != 0:
                self.logger.warning(self.get_session_id() + f"Failed to cleanup remote file: {cleanup_result.stderr}")
            
            return execution_error
            
        except subprocess.TimeoutExpired as e:
            error_msg = f"Remote execution timeout: {e}"
            self.logger.error(self.get_session_id() + error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Remote execution failed: {e}"
            self.logger.error(self.get_session_id() + error_msg)
            traceback.print_exc()
            return error_msg
        finally:
            # Cleanup local temporary file
            try:
                if 'local_script_path' in locals():
                    os.unlink(local_script_path)
                    self.logger.debug(self.get_session_id() + f"Cleaned up local temp file: {local_script_path}")
            except Exception as cleanup_error:
                self.logger.warning(self.get_session_id() + f"Failed to cleanup local temp file: {cleanup_error}")
    
    def _stream_ssh_execution(self, ssh_command, script_id):
        """
        Execute SSH command with real-time output streaming to local CLI.
        
        Args:
            ssh_command (list): SSH command to execute
            script_id (str): Unique script identifier for logging
            
        Returns:
            str: Error message if execution failed, None if successful
        """
        try:
            # Start the SSH process with streaming output
            process = subprocess.Popen(
                ssh_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,  # Unbuffered for immediate output
                universal_newlines=True
            )
            
            self.logger.info(self.get_session_id() + f"🚀 Remote execution started for {script_id}")
            
            # Real-time output streaming with timeout
            stdout_lines = []
            stderr_lines = []
            
            # Use threading for reliable output capture
            import threading
            
            def read_stdout():
                try:
                    if process.stdout:
                        for line in iter(process.stdout.readline, ''):
                            if line:
                                line = line.rstrip('\n')
                                stdout_lines.append(line)
                                # print(f"📡 [REMOTE] {line}")
                    if stdout_lines:
                        if any("Error" in line for line in stdout_lines):
                            stderr_lines.extend(stdout_lines)
                        else:
                            self.logger.info(self.get_session_id() + f"[STDOUT] {stdout_lines}")
                except Exception:
                    pass

            def read_stderr():
                try:
                    if process.stderr:
                        for line in iter(process.stderr.readline, ''):
                            if line:
                                line = line.rstrip('\n')
                                stderr_lines.append(line)
                                # print(f"📡 [REMOTE ERROR] {line}")
                                # self.logger.error(self.get_session_id() + f"[STDERR] {line}")
                    if stderr_lines and 'Traceback (most recent call last)' not in stderr_lines:
                        self.logger.error(self.get_session_id() + f"[STDERR] {stderr_lines}")
                except Exception:
                    pass
            
            # Start reader threads
            stdout_thread = threading.Thread(target=read_stdout)
            stderr_thread = threading.Thread(target=read_stderr)
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process completion with timeout
            timeout_seconds = 60  # 1 minute timeout
            start_time = time.time()
            
            while True:
                # Check if process has finished
                if process.poll() is not None:
                    break
                
                # Check for overall timeout
                current_time = time.time()
                if current_time - start_time > timeout_seconds:
                    self.logger.error(self.get_session_id() + f"Remote execution timeout after {timeout_seconds} seconds")
                    process.terminate()
                    time.sleep(2)  # Give it time to terminate gracefully
                    if process.poll() is None:
                        process.kill()  # Force kill if still running
                    return f"Remote execution timeout after {timeout_seconds} seconds"
                
                # Sleep briefly to prevent excessive CPU usage
                time.sleep(0.1)
            
            # Wait for threads to finish reading all output
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            # Get the final exit code
            exit_code = process.returncode
            full_stdout = '\n'.join(stdout_lines)
            full_stderr = '\n'.join(stderr_lines)
            
            # Always return the combined output, regardless of success/failure
            combined_output = ""
            if full_stdout:
                combined_output += full_stdout
            if full_stderr:
                if combined_output:
                    combined_output += "\n"
                combined_output += full_stderr
            
            if exit_code == 0:
                self.logger.info(self.get_session_id() + f"✅ Remote execution completed successfully for {script_id}")
                # print(f"✅ Remote execution completed successfully!")
                # Return None for success (no error), but actual output is already logged/printed
                return None
            else:
                # For errors, return the combined output so the calling code can process it
                error_msg = f"❌ Remote execution failed with exit code {exit_code}"
                if full_stderr:
                    # Check for Python traceback in stderr
                    if 'Traceback' in full_stderr:
                        self.logger.error(self.get_session_id() + "Python traceback detected in remote execution:")
                        self.logger.error(self.get_session_id() + "🔍 [REMOTE TRACEBACK] \n"+full_stderr)
                        # print("🔍 [REMOTE TRACEBACK]")
                        # for line in full_stderr.split('\n'):
                        #     if line.strip():
                        #         print(f"🔍   {line}")
                
                self.logger.error(self.get_session_id() + error_msg)
                # print(f"❌ Remote execution failed!")
                # Return the actual error output, not just a generic message
                return combined_output if combined_output else error_msg
                
        except Exception as e:
            error_msg = f"Failed to stream remote execution: {e}"
            self.logger.error(self.get_session_id() + error_msg)
            print(f"💥 Remote execution streaming failed: {e}")
            return error_msg
    

    def _handle_successful_execution(self, result):
        """
        Handle successful code execution by logging output and saving the code.

        Args:
            result: The subprocess.run result
            result: The completed validation or compilation process.
        """
        if result.stdout and result.stdout.strip():
            self.logger.info(
                self.get_session_id() + f"Program output: {result.stdout.strip()}"
            )

    def _cleanup_execution_files(self, work_dir, tmp_file):
        """
        Clean up temporary execution files and directories.

        Args:
            work_dir: Directory to remove
            tmp_file: File to remove
        """
        try:
            if os.path.exists(work_dir):
                self.logger.debug(f"Attempting to remove directory: {work_dir}")
                shutil.rmtree(work_dir, ignore_errors=True)
                self.logger.info(f"Successfully removed directory: {work_dir}")
            if os.path.exists(tmp_file):
                self.logger.debug(f"Attempting to remove file: {tmp_file}")
                os.remove(tmp_file)
                self.logger.info(f"Successfully removed file: {tmp_file}")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

    def extract_code(
        self, response: str | object, language: str = "python"
    ) -> str | None:
        """!
        Extract code from the LLM response, handling different output formats.
        If no Python code blocks, try to extract from any code blocks
        If no code blocks, raise exception as not supported output format

        Args:
            response: The raw response from the LLM.
            language: Target language to extract ("python" or "cpp")

        Returns:
            Extracted code as a string.

        Raises:
            ValueError: If no supported code blocks are found.
        """
        content = str(response)

        if language == "cpp":
            return self.extract_code_cpp(content)
        elif language == "python":
            # First try explicit Python blocks
            pattern = r"```python\s*(.*?)\s*```"
            if match := re.search(pattern, content, re.DOTALL):
                code = match.group(1).strip()
                if code.startswith("{") and code.endswith("}"):
                    # remove the starting parts of { "somthing: " and the tail "}
                    code = code[code.index(":") + 1 : -1].strip()
       
                return str(code)
                    

            # Then try generic code blocks
            pattern = r"```\s*(.*?)\s*```"
            if match := re.search(pattern, content, re.DOTALL):
                code= match.group(1).strip()
                if code.startswith("{") and code.endswith("}"):
                    # remove the starting parts of { "somthing: " and the tail "}
                    code = code[code.index(":") + 1 : -1].strip()
       
                return str(code)

            raise ValueError("The output is not a valid code block. Follow the instructions in the prompt return a complete, correct Python script enclosed in a single ```python\n<generated_code>\n``` block.")

    # attributes is a dictionary like {"key1": "value1", "key2": "value2"}, i need to search in code line by line if there is any line containing "path_to" or "path to", is so, try to match if there is any keys in the dictionary attributes that are being the starting string in that line, then replace that line to be like key1 = "value1". Loop this for every line in code parameter. The lines cloud be seperated by newlines or "\n".
    # replacement for the same key can be check only once
    
    def replace_generated_placeholders(self, code, attributes):
        """
        Check and replace placeholders in the code with actual values from attributes.

        Args:
            code (str): The code snippet to check.
            attributes (dict): Dictionary of attributes to replace in the code.

        Returns:
            str: Code with placeholders replaced by actual values.
        """
        lines = code.splitlines()
        replaced_keys = set()  # Track keys that have been replaced
        
        for i, line in enumerate(lines):
            
            for key, value in attributes.items():
                # Skip if this key has already been replaced
                if key in replaced_keys:
                    continue
                    
                if f"{key} =" in line:
                    lines[i] = f'{key} = "{value}"' if isinstance(value, str) else f"{key} = {value}"
                    print(
                        self.get_session_id()
                        + f"Variable '{key}' is set to '{value}'"
                    )
                    replaced_keys.add(key)  # Mark key as replaced
                    break  # Move to next line after replacement
                    
                elif (key.split("_")[0] == line.split("_")[0]) or (key.split("_")[0].upper() == line.split("_")[0]):
                    assignment_target = line.split("=")[0].strip()
                    lines[i] = f'{assignment_target} = \"{value}\"' if isinstance(value, str) else f'{assignment_target} = {value}'
                    replaced_keys.add(key)  # Mark key as replaced
                    break  # Move to next line after replacement
                
        return "\n".join(lines)
        

        
    def extract_code_cpp(self, content):
        matches = list(re.finditer(r"```(?:cpp|ino)\n(.*?)```", content, re.DOTALL))

        # Combine all found code blocks
        code_blocks = [match.group(1).strip() for match in matches]
        if not code_blocks:
            raise ValueError("Unsupported output format: no cpp/ino code blocks found")
        match_count = len(matches)
        # print(f"Found {match_count} cpp/ino code blocks.")
        # Join all code blocks with newlines
        combined_code = "\n\n".join(code_blocks)

        self.logger.info(
            self.get_session_id()
            + f"Found {match_count} cpp/ino code blocks, concatenated if possible."
        )
        return combined_code

    def contains_error_indicator(self, result, indicators=None):
        """
        Check if output contains error indicators.

        Args:
            result: subprocess.run result
            indicators: List of error indicator strings to look for

        Returns:
            bool: True if any error indicators found
        """
        combined_output = (result.stdout or "") + (result.stderr or "")
        if indicators is None:
            indicators = ["error:", "exception:", "traceback:", "failed:"]

        return any(indicator in combined_output.lower() for indicator in indicators)
