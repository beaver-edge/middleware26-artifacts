"""work/ copies of archived outputs become hard links; nothing else changes."""
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "container" / "dedupe-work.sh"


def _run(task_dir):
    done = subprocess.run(
        ["bash", str(SCRIPT), str(task_dir)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.split()


def _layout(tmp_path):
    run = tmp_path / "artifacts/pysketch/session/run-001-trace"
    valid = run / "outputs/valid/valid_annotated_video_out.mp4"
    invalid = run / "outputs/invalid/invalid_rejected_partial.mp4"
    valid.parent.mkdir(parents=True)
    invalid.parent.mkdir(parents=True)
    valid.write_bytes(b"V" * 4096)
    invalid.write_bytes(b"P" * 2048)
    work = tmp_path / "work/pysketch"
    work.mkdir(parents=True)
    return valid, invalid, work


def test_identical_work_file_becomes_a_hard_link(tmp_path):
    valid, _, work = _layout(tmp_path)
    copy = work / "out.mp4"
    copy.write_bytes(b"V" * 4096)
    assert _run(tmp_path) == ["1", "4096"]
    assert os.path.samefile(copy, valid)
    assert copy.read_bytes() == b"V" * 4096
    assert "work/pysketch/out.mp4 <- " in (tmp_path / "work-links.txt").read_text()
    assert _run(tmp_path) == ["0", "0"]  # already one file: nothing to do


def test_only_byte_identical_valid_outputs_are_linked(tmp_path):
    valid, invalid, work = _layout(tmp_path)
    same_size_other_bytes = work / "other.mp4"
    same_size_other_bytes.write_bytes(b"X" * 4096)
    rejected_copy = work / "partial.mp4"
    rejected_copy.write_bytes(b"P" * 2048)   # matches an *invalid* archive only
    unrelated = work / "model_quant.tflite"
    unrelated.write_bytes(b"model")
    assert _run(tmp_path) == ["0", "0"]
    for untouched, content in (
        (same_size_other_bytes, b"X" * 4096),
        (rejected_copy, b"P" * 2048),
        (unrelated, b"model"),
    ):
        assert untouched.read_bytes() == content
        assert untouched.stat().st_nlink == 1
    assert not (tmp_path / "work-links.txt").exists()
    assert valid.stat().st_nlink == 1 and invalid.stat().st_nlink == 1


def test_nothing_to_do_without_work_or_artifacts(tmp_path):
    assert _run(tmp_path) == ["0", "0"]
    (tmp_path / "work").mkdir()
    assert _run(tmp_path) == ["0", "0"]
