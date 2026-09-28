"""Check framework completion and the canonical per-attempt artifact archive."""
import json
import sys
from pathlib import Path

task, marker, outcome_file = sys.argv[1:]
started = Path(marker).stat().st_mtime_ns
outcome = json.loads(Path(outcome_file).read_text())

if outcome.get("execution_status") != "completed":
    raise RuntimeError("Framework execution did not complete")
artifact_run = Path(outcome.get("artifact_run_dir", ""))
if not artifact_run.is_dir():
    raise RuntimeError(f"Missing generated-artifact run directory: {artifact_run}")
manifest_path = artifact_run / "manifest.json"
if not manifest_path.is_file():
    raise RuntimeError(f"Missing generated-artifact manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
attempt_records = manifest.get("attempts", [])
valid_attempts = [
    Path(attempt["code_path"])
    for attempt in attempt_records
    if attempt.get("status") == "valid"
]
invalid_attempts = [
    Path(attempt["code_path"])
    for attempt in attempt_records
    if attempt.get("status") == "invalid"
]

def fresh(path):
    p = Path(path)
    return p.is_file() and p.stat().st_size > 0 and p.stat().st_mtime_ns >= started

if any(not fresh(path) for path in valid_attempts + invalid_attempts):
    raise RuntimeError("Manifest references a missing, empty, or stale code attempt")
metadata_paths = [Path(attempt.get("metadata_path", "")) for attempt in attempt_records]
if any(not fresh(path) for path in metadata_paths):
    raise RuntimeError("A code attempt is missing fresh validation metadata")

if outcome.get("valid_attempts") != len(valid_attempts):
    raise RuntimeError("Outcome and manifest disagree on valid attempt count")
if outcome.get("invalid_attempts") != len(invalid_attempts):
    raise RuntimeError("Outcome and manifest disagree on invalid attempt count")
if any(not path.name.startswith("valid_attempt-") for path in valid_attempts):
    raise RuntimeError("A valid attempt has an ambiguous filename")
if any(not path.name.startswith("invalid_attempt-") for path in invalid_attempts):
    raise RuntimeError("An invalid attempt has an ambiguous filename")
if outcome.get("artifact_status") == "not_generated":
    if outcome.get("termination") != "retry_budget_exhausted":
        raise RuntimeError("Unrecognized no-artifact outcome")
    if not outcome.get("last_error"):
        raise RuntimeError("Retry exhaustion has no recorded final error")
    if valid_attempts:
        raise RuntimeError("No-artifact outcome unexpectedly contains valid code")
    print(
        "Verified handled outcome: retry budget exhausted; "
        f"retained invalid attempts={len(invalid_attempts)} path={artifact_run}"
    )
    raise SystemExit(0)
if outcome.get("artifact_status") != "generated":
    raise RuntimeError("Completed run has an unknown artifact outcome")
if not valid_attempts:
    raise RuntimeError("Generated outcome has no explicitly valid code attempt")

def require(paths, description):
    found = [str(p) for p in paths if fresh(p)]
    if not found:
        raise RuntimeError(f"Missing fresh {description}")
    print(f"Verified {description}: " + ", ".join(found))

if task == "data":
    require(valid_attempts, "validated DP code")
    require(
        artifact_run.joinpath("outputs/valid").glob("valid_data_*.csv"),
        "archived DP dataset",
    )
elif task == "convert":
    require(valid_attempts, "validated conversion code")
    model = Path("work/convert/model_quant.tflite")
    require([model], "converted model")
    require(
        artifact_run.joinpath("outputs/valid").glob(
            "valid_converted_model_*.tflite"
        ),
        "archived converted model",
    )
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    print("Verified TFLite interpreter loading")
elif task == "ardsketch":
    require(valid_attempts, "compiled Arduino sketch")
elif task == "pysketch":
    require(valid_attempts, "executed CPU sketch")
    videos = list(
        artifact_run.joinpath("outputs/valid").glob(
            "valid_annotated_video_*.mp4"
        )
    )
    require(videos, "archived CPU output video")
    import cv2
    capture = cv2.VideoCapture(str(videos[0]))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError("Output video has no readable frame")
elif task == "tpusketch":
    require(valid_attempts, "validated TPU script")
    videos = list(
        artifact_run.joinpath("outputs/valid").glob(
            "valid_remote_annotated_video_*.mp4"
        )
    )
    require(videos, "locally archived TPU output video")
    import cv2
    capture = cv2.VideoCapture(str(videos[0]))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError("Archived TPU output video has no readable frame")
    print("Verified locally archived TPU output video has a readable frame")

print(
    "Verified generated-attempt classification: "
    f"valid={len(valid_attempts)} invalid={len(invalid_attempts)} "
    f"path={artifact_run}"
)
