"""A successful process alone must not validate stale artifacts."""
import json
import os
import subprocess
import sys
from pathlib import Path

VERIFY = Path(__file__).resolve().parents[2] / "scripts" / "container" / "verify_output.py"

def test_missing_and_stale_conversion_are_rejected(tmp_path):
    marker = tmp_path / "started"
    marker.touch()
    run = tmp_path / "artifacts/convert/session/run-001-trace"
    valid = run / "valid/valid_attempt-001_conversion_fake-model.py"
    valid.parent.mkdir(parents=True)
    valid.write_text("print('conversion')")
    metadata = valid.with_suffix(".py.json")
    metadata.write_text("{}")
    manifest = {
        "attempts": [{
            "status": "valid",
            "code_path": str(valid),
            "metadata_path": str(metadata),
        }],
        "outputs": [],
    }
    run.joinpath("manifest.json").write_text(json.dumps(manifest))
    outcome = tmp_path / "outcome.json"
    outcome.write_text(json.dumps({
        "execution_status": "completed",
        "artifact_status": "generated",
        "termination": "artifact_validated",
        "artifact_run_dir": str(run),
        "valid_attempts": 1,
        "invalid_attempts": 0,
    }))
    command = [sys.executable, str(VERIFY), "convert", str(marker), str(outcome)]
    missing = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert missing.returncode != 0
    model = tmp_path / "work/convert/model_quant.tflite"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"stale output")
    os.utime(model, (1, 1))
    stale = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert stale.returncode != 0
    assert "Missing fresh converted model" in stale.stderr


def test_generated_outcome_rejects_ambiguous_attempt_name(tmp_path):
    marker = tmp_path / "started"
    marker.touch()
    run = tmp_path / "artifacts/data/session/run-001-trace"
    ambiguous = run / "valid/code.py"
    ambiguous.parent.mkdir(parents=True)
    ambiguous.write_text("print('ok')")
    metadata = ambiguous.with_suffix(".py.json")
    metadata.write_text("{}")
    run.joinpath("manifest.json").write_text(json.dumps({
        "attempts": [{
            "status": "valid",
            "code_path": str(ambiguous),
            "metadata_path": str(metadata),
        }],
        "outputs": [],
    }))
    outcome = tmp_path / "outcome.json"
    outcome.write_text(json.dumps({
        "execution_status": "completed",
        "artifact_status": "generated",
        "termination": "artifact_validated",
        "artifact_run_dir": str(run),
        "valid_attempts": 1,
        "invalid_attempts": 0,
    }))
    checked = subprocess.run(
        [sys.executable, str(VERIFY), "data", str(marker), str(outcome)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert checked.returncode != 0
    assert "ambiguous filename" in checked.stderr


def test_retry_exhaustion_is_a_verified_completed_outcome(tmp_path):
    marker = tmp_path / "started"
    marker.touch()
    run = tmp_path / "artifacts/ardsketch/session/run-001-trace"
    invalid = run / "invalid/invalid_attempt-001_arduino_fake-model.ino"
    invalid.parent.mkdir(parents=True)
    invalid.write_text("void setup() {}")
    metadata = invalid.with_suffix(".ino.json")
    metadata.write_text("{}")
    run.joinpath("manifest.json").write_text(json.dumps({
        "attempts": [{
            "status": "invalid",
            "code_path": str(invalid),
            "metadata_path": str(metadata),
        }],
        "outputs": [],
    }))
    outcome = tmp_path / "outcome.json"
    outcome.write_text(json.dumps({
        "execution_status": "completed",
        "artifact_status": "not_generated",
        "termination": "retry_budget_exhausted",
        "last_error": "Max 5 attempts exceeded. Last compiler error.",
        "artifact_run_dir": str(run),
        "valid_attempts": 0,
        "invalid_attempts": 1,
    }))
    command = [
        sys.executable, str(VERIFY), "ardsketch", str(marker), str(outcome)
    ]
    completed = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True
    )
    assert completed.returncode == 0
    assert "Verified handled outcome" in completed.stdout


def _write_dp_run(tmp_path, statuses):
    """A DP run whose attempts have the given statuses, in archive order."""
    marker = tmp_path / "started"
    marker.touch()
    run = tmp_path / "artifacts/data/session/run-001-trace"
    attempts = []
    for number, status in enumerate(statuses, start=1):
        code = run / status / f"{status}_attempt-{number:03d}_operation_fake-model.py"
        code.parent.mkdir(parents=True, exist_ok=True)
        code.write_text("print('operation')")
        metadata = code.with_suffix(".py.json")
        metadata.write_text("{}")
        attempts.append({
            "status": status,
            "code_path": str(code),
            "metadata_path": str(metadata),
        })
    run.joinpath("manifest.json").write_text(json.dumps({
        "attempts": attempts,
        "outputs": [],
    }))
    outcome = tmp_path / "outcome.json"
    outcome.write_text(json.dumps({
        "execution_status": "completed",
        "artifact_status": "not_generated",
        "termination": "retry_budget_exhausted",
        "last_error": "Failed to generate valid code after the max 5 attempts.",
        "artifact_run_dir": str(run),
        "valid_attempts": statuses.count("valid"),
        "invalid_attempts": statuses.count("invalid"),
    }))
    return [sys.executable, str(VERIFY), "data", str(marker), str(outcome)]


def test_dp_budget_exhausted_after_earlier_valid_operations(tmp_path):
    # Operations 1-3 validated, operation 4 used its whole repair budget.
    command = _write_dp_run(tmp_path, ["valid"] * 3 + ["invalid"] * 5)
    completed = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert "Verified handled outcome" in completed.stdout


def test_budget_exhausted_outcome_cannot_end_with_valid_attempt(tmp_path):
    command = _write_dp_run(tmp_path, ["invalid", "valid"])
    checked = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True
    )
    assert checked.returncode != 0
    assert "does not end with an invalid attempt" in checked.stderr
