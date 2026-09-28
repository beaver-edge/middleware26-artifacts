# Remote Coral Dev Board setup for BEAVER-EDGE TPUSG
> If you want to use own Coral Dev Board to work directly with existing BEAVER-EDGE workflow w/ or w/o docker container, this is a guide to set it up. 


This document describes the board-side requirements for the repository's TPU Python Sketch Generation (`tpusketch`) workflow. It is specific to the original Google Coral Dev Board (Mendel Linux), not the Dev Board Micro or a host using a USB/M.2 accelerator.

The framework does not provision the board. It generates a Python script in the artifact container, copies it to the board with SCP, executes it over SSH, and deletes the remote script afterward. The model, labels, input video, Python runtime, and Edge TPU runtime must therefore exist before TPUSG starts.

## 1. Required contract

The following are requirements imposed by the current source code, rather than
general Coral recommendations.

| Area | Requirement |
| --- | --- |
| Board | A reachable Google Coral Dev Board with a functioning Edge TPU. |
| Access | Non-interactive OpenSSH and SCP access from the artifact container. Public-key authentication and a populated `known_hosts` file are expected. |
| Remote user | May create files in the configured execution directory, run the configured Python executable, access `/dev/apex_0`, and write the output directory. The supplied profile uses user `mendel`. |
| Python | The exact executable named by `REMOTE_PYTHON_EXECUTABLE` must import `cv2`, `numpy`, `tflite_runtime.interpreter`, and `pycoral`. |
| Edge TPU | `libedgetpu.so.1`/`libedgetpu.so.1.0` must load, and the remote user must belong to the `apex` group or otherwise have device access. |
| Input model | An Edge-TPU-compiled TFLite object-detection model at `/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite`. |
| Other inputs | A matching label map at `/home/mendel/tinyml_autopilot/models/labelmap.txt` and a readable video at `/home/mendel/tinyml_autopilot/data/sheeps.mp4`. |
| Writable paths | `/home/mendel/tinyml_autopilot/tmp` and `/home/mendel/tinyml_autopilot/results`; the expected output is `results/sheeps_detections.mp4`. |
| Capacity | Enough free space for the uploaded script and output video. The framework does not enforce a numeric minimum. |

`REMOTE_PYTHON_ENV` is currently read into a variable but never executed by `src/base/base_processor.py`. Do not rely on an activation command there. Put the absolute interpreter path in `REMOTE_PYTHON_EXECUTABLE`.

The runner checks `/sys/class/apex/apex_0/device_owner`. If another process owns the TPU, it may send `kill` to that PID before the run, and it uses `pkill` to remove a leftover generated process afterward. Use a dedicated evaluation board; do not share it with an unrelated TPU workload.

## 2. Configuration verified on 2026-09-28

A read-only audit of the currently configured remote board found:

| Component | Observed value |
| --- | --- |
| OS | Mendel GNU/Linux 5.2 (Eagle), Linux `aarch64` |
| Python | 3.7.3 at `/home/mendel/tinyml_autopilot/tinyml-env/bin/python` |
| Virtual environment | `include-system-site-packages = true` |
| Edge TPU runtime | `libedgetpu1-std` 16.0 |
| PyCoral | 2.0.0 |
| TFLite runtime | 2.5.0.post1 |
| OpenCV | 3.2.0 |
| NumPy | 1.21.6 |
| TPU permissions | `/dev/apex_0` owned by `root:apex`; user `mendel` is in group `apex` |


This is a known-good configuration, not a claim that every version is a hard minimum. A prior live TPUSG run on it loaded the delegate, processed 160 frames, recorded 702 detections, and produced a readable output video.

The validated board inputs have these SHA-256 hashes:

```text
cf52602e62d11844a80fbcc60127ac7dfadc9bf76e08b3ab88e958a09497d9fc  sheeps.mp4
e4b118e5e4531945de2e659742c7c590f7536f8d0ed26d135abcfe83b4779d13  edgetpu_detect.tflite
26f6fad31a01133f6cdcb22701580928a327438584e2469d9678a18c27f511c2  labelmap.txt
```

The files bundled in this checkout are functionally corresponding inputs, but their hashes and sizes differ from the files used by that live run. Do not overwrite a validated board merely to make it match the checkout. For a new board, the bundled files are the available starting inputs and must pass the preflight below before an evaluation run.

## 3. Prepare a new board

Skip destructive setup on an already working board. Flashing Mendel erases board data; back up anything needed first. Google's official Dev Board guide covers
power, boot-switch, flash-card, MDT, network, and first-SSH setup:

- <https://coral.ai/docs/dev-board/get-started/>
- <https://coral.ai/docs/dev-board/reflash/>

For a fresh original Dev Board:

1. Flash Mendel Linux and boot from eMMC using the official instructions.
2. Use MDT over USB once to establish key-based shell access, then connect the board to Ethernet or Wi-Fi.
3. Confirm ordinary OpenSSH works from the machine that will run Docker:

   ```bash
   board="mendel@BOARD_ADDRESS"
   ssh "$board" 'uname -m; cat /etc/mendel_version'
   ```

4. On the board, install the Coral and Python packages. Coral's Linux guidance
   recommends Debian packages for PyCoral:

   ```bash
   sudo apt-get update
   sudo apt-get install -y \
     libedgetpu1-std python3-pycoral python3-tflite-runtime \
     python3-opencv python3-numpy python3-venv
   ```

   Do not install `libedgetpu1-std` and `libedgetpu1-max` together. The standard-frequency package is the configuration validated here. Avoid an
   unattended `dist-upgrade` immediately before evaluation; preserve a working image because this is a legacy Mendel/PyCoral stack.

5. Create the directory layout and an optional virtual environment:

   ```bash
   mkdir -p /home/mendel/tinyml_autopilot/{data,models,results,tmp}
   python3 -m venv --system-site-packages \
     /home/mendel/tinyml_autopilot/tinyml-env
   ```

   `--system-site-packages` is important when PyCoral and OpenCV come from Mendel's Debian packages. The currently validated environment additionally
   has NumPy 1.21.6. If exact reproduction is required and that wheel remains available for Python 3.7/aarch64, install it inside the virtual environment;
   otherwise test the packaged NumPy using the preflight rather than upgrading blindly.

6. From the repository root on the Docker host, copy inputs to a new board:

   ```bash
   board=coral-eval
   scp inputs/tpusketch/sheeps.mp4 \
     "$board":/home/mendel/tinyml_autopilot/data/sheeps.mp4
   scp inputs/tpusketch/edgetpu_detect.tflite \
     "$board":/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite
   scp inputs/tpusketch/labelmap.txt \
     "$board":/home/mendel/tinyml_autopilot/models/labelmap.txt
   ```

   The model must already be compiled for the Edge TPU. TPUSG does not compile    models on the board.

## 4. Run the board preflight

Set the first line to the same alias or `user@host` that will be used in `.env`. This check is read-only except for creating and removing a zero-byte
probe in the two required writable directories.

```bash
board=coral-eval
remote_python=/home/mendel/tinyml_autopilot/tinyml-env/bin/python

ssh "$board" "REMOTE_PYTHON='$remote_python' bash -s" <<'REMOTE'
set -eu

test "$(uname -m)" = aarch64
test -c /dev/apex_0
id -nG | tr ' ' '\n' | grep -qx apex

for path in \
  /home/mendel/tinyml_autopilot/data/sheeps.mp4 \
  /home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite \
  /home/mendel/tinyml_autopilot/models/labelmap.txt; do
  test -s "$path"
done

for directory in \
  /home/mendel/tinyml_autopilot/tmp \
  /home/mendel/tinyml_autopilot/results; do
  test -d "$directory"
  probe="$directory/.beaver-edge-write-test"
  : > "$probe"
  rm -f "$probe"
done

"$REMOTE_PYTHON" - <<'PY'
import cv2
import numpy
import pycoral
from tflite_runtime.interpreter import Interpreter, load_delegate

model = "/home/mendel/tinyml_autopilot/models/edgetpu_detect.tflite"
delegate = load_delegate("libedgetpu.so.1.0")
interpreter = Interpreter(model_path=model, experimental_delegates=[delegate])
interpreter.allocate_tensors()

capture = cv2.VideoCapture(
    "/home/mendel/tinyml_autopilot/data/sheeps.mp4"
)
ok, frame = capture.read()
capture.release()
assert ok and frame is not None, "input video has no readable frame"

print("PASS: imports, Edge TPU delegate, model allocation, and video read")
print("input shape:", interpreter.get_input_details()[0]["shape"].tolist())
print("numpy:", numpy.__version__, "opencv:", cv2.__version__)
PY

df -h /home
REMOTE
```

If delegate loading reports that the device is busy, find and stop the intended owner before evaluation. Do not kill an unidentified process on a shared board.

## 5. Configure host-side SSH and `.env`

Prepare a dedicated directory containing only the evaluation SSH runtime files:

```text
evaluation-ssh/
├── config
├── known_hosts
└── id_ed25519
```

The directory is mounted read-only at `/run/evaluation-ssh`. The launcher copies these files into `/root/.ssh` inside the ephemeral container. Therefore paths in
`config` must be container paths, for example:

```sshconfig
Host coral-eval
    HostName BOARD_ADDRESS
    User mendel
    IdentityFile /root/.ssh/id_ed25519
    IdentitiesOnly yes
```

Add the board host key deliberately; do not disable host-key checking:

```bash
board_address=BOARD_ADDRESS
ssh-keyscan -H "$board_address" > evaluation-ssh/known_hosts
chmod 600 evaluation-ssh/id_ed25519 evaluation-ssh/config
```

If the alias uses `ProxyJump` like the author did, include the jump host's key in `known_hosts` and ensure every `IdentityFile` referenced by the copied config uses a path that exists under `/root/.ssh` in the container.

Set these values in the private repository `.env`:

```dotenv
REMOTE_EXECUTION_ENABLED=true
REMOTE_HOST=coral-eval
REMOTE_EXEC_PATH=/home/mendel/tinyml_autopilot/tmp
REMOTE_PYTHON_ENV=
REMOTE_PYTHON_EXECUTABLE=/home/mendel/tinyml_autopilot/tinyml-env/bin/python
```

Do not add board credentials or SSH keys to `.env`, the image, or a source archive. The implementation does not use `REMOTE_USER` or `SSH_KEY_PATH`.

Verify the exact container-side connection before invoking an LLM:

```bash
docker run --rm -it \
  -v "$PWD/.env:/artifact/.env:ro" \
  -v /absolute/path/to/evaluation-ssh:/run/evaluation-ssh:ro \
  <image> bash   # <image>: the published image, or beaver-edge-artifact:middleware26 from ./build-image.sh

# Run the remaining commands inside that container shell.
set -a; . /artifact/.env; set +a   # provides REMOTE_HOST and REMOTE_PYTHON_EXECUTABLE
mkdir -p /root/.ssh
cp /run/evaluation-ssh/config /root/.ssh/
cp /run/evaluation-ssh/known_hosts /root/.ssh/
cp /run/evaluation-ssh/id_* /root/.ssh/
chmod 700 /root/.ssh
chmod 600 /root/.ssh/config /root/.ssh/id_*
ssh -o BatchMode=yes -o ControlMaster=no -o ControlPath=none \
  "$REMOTE_HOST" \
  "$REMOTE_PYTHON_EXECUTABLE -c 'import cv2, numpy, pycoral, tflite_runtime.interpreter; print(\"PASS\")'"
```

The commands above exercise the mounted SSH material and remote interpreter. 
Exit the temporary container shell after `PASS`.

## 6. Run and verify TPUSG

From the repository root on the host:

```bash
./run-artifact.sh --tpu /absolute/path/to/evaluation-ssh tpusketch
```

Omit `tpusketch` to run TPU-SG after the four local tasks. The SSH directory is
mounted read-only, and `scripts/container/setup-ssh-and-run-task.sh` copies the
needed files into the ephemeral container layer only. Evidence is written to
`artifact-runs/<timestamp>/tpusketch/`.

A successful generated-artifact outcome requires both:

- a retained local
  `artifacts/tpusketch/<session>/<run>/valid/valid_attempt-*.py`; and
- a locally archived
  `artifacts/tpusketch/<session>/<run>/outputs/valid/valid_remote_annotated_video_*.mp4`
  copied from the board and containing at least one OpenCV-readable frame.

Every rejected remote candidate is retained separately under the same run's `invalid/` directory with an `invalid_attempt-*.py` filename and error metadata. The remote temporary script is still deleted after execution; the classified local copy is the authoritative evidence.

The framework can also complete normally after exhausting its bounded repair budget, in which case `artifact_status=not_generated`. `run-artifact.sh` reports that outcome as PASS. SSH failures, missing board dependencies, and verifier failures are system failures and are reported as FAIL.

## 7. Troubleshooting map

| Symptom | Check |
| --- | --- |
| `Permission denied (publickey)` | The mounted private key, `IdentityFile` path, board `authorized_keys`, and `User` in SSH config. |
| Host-key failure | Add the real board/jump-host key to the dedicated `known_hosts`; do not use `StrictHostKeyChecking=no`. |
| SCP creates no script | `REMOTE_EXEC_PATH` exists or the SSH user may create it; check free space and permissions. |
| `No such file or directory` for Python | `REMOTE_PYTHON_EXECUTABLE` must be an absolute executable path; `REMOTE_PYTHON_ENV` will not activate it. |
| `libedgetpu.so` cannot load | Verify `libedgetpu1-std`, `python3-pycoral`, and `python3-tflite-runtime` are compatible and installed from the Mendel/Coral package source. |
| `/dev/apex_0` permission denied | Confirm the device exists, the user is in `apex`, then start a new login session after group changes. |
| Edge TPU already in use | Read `/sys/class/apex/apex_0/device_owner`; stop only the intended process or use a dedicated board. |
| Model has an unsupported custom op | Use an Edge-TPU-compiled `.tflite` model and load it with the delegate. |
| Video opens but output is empty | Check input codec support, model/label compatibility, output-directory permissions, and available storage. |
| SSH works on host but not in container | Check container-relative identity paths, jump-host material, `known_hosts`, and that the SSH directory was mounted. |
