#!/usr/bin/env bash
# Runs INSIDE the container (TPU-SG only), after setup-ssh-and-run-task.sh has
# installed the evaluation SSH files. Read-only preflight of the Coral board:
# it proves that the exact SSH/SCP path TPU-SG uses works, and that the board
# has the runtime and inputs TPU-SG expects. The only write is a probe file in
# REMOTE_EXEC_PATH, which is removed again. Exit 0 = board ready.
set -uo pipefail
cd /artifact
eval "$(python - <<'EOF'
import shlex
from dotenv import dotenv_values
env = dotenv_values(".env")
for name in ("REMOTE_HOST", "REMOTE_EXEC_PATH", "REMOTE_PYTHON_EXECUTABLE"):
    print(f"{name}={shlex.quote(env.get(name) or '')}")
EOF
)"
for name in REMOTE_HOST REMOTE_EXEC_PATH REMOTE_PYTHON_EXECUTABLE; do
    [[ -n ${!name} ]] || { echo "FAIL: $name is empty in .env"; exit 1; }
done
# Same options as src/base/base_processor.py uses for every SSH/SCP call.
ssh_options=(-o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o ConnectTimeout=20)
board_root=/home/mendel/tinyml_autopilot   # paths from the TPU profile in src/processors/tpu_sketch_generator.py
failed=0
check() {  # check DESCRIPTION COMMAND...
    local description=$1 output
    shift
    if output=$("$@" 2>&1); then
        echo "ok    $description${output:+: $(tail -n 1 <<<"$output")}"
    else
        echo "FAIL  $description"
        sed 's/^/        /' <<<"$output"
        failed=1
    fi
}
remote() { timeout 60 ssh "${ssh_options[@]}" "$REMOTE_HOST" "$@"; }

check "SSH login to $REMOTE_HOST" remote 'echo "$(id -un)@$(hostname)"'
[[ $failed -eq 0 ]] || { echo "Board unreachable; skipping the remaining checks."; exit 1; }
check "Python runtime and Edge TPU visible" remote "$REMOTE_PYTHON_EXECUTABLE -c 'import cv2, numpy, tflite_runtime.interpreter; from pycoral.utils import edgetpu; n = len(edgetpu.list_edge_tpus()); assert n > 0, \"no Edge TPU found\"; print(\"edge TPUs:\", n)'"
check "Board inputs present" remote "for f in $board_root/models/edgetpu_detect.tflite $board_root/models/labelmap.txt $board_root/data/sheeps.mp4; do test -r \$f || { echo missing \$f; exit 1; }; done; echo model, labels, video"
check "Output directory writable" remote "mkdir -p $board_root/results && test -w $board_root/results && echo $board_root/results"
probe=$(mktemp)
echo "beaver-edge board probe" > "$probe"
probe_remote="$REMOTE_EXEC_PATH/.beaver-edge-probe-$$"
check "SCP upload into REMOTE_EXEC_PATH" bash -c "timeout 60 ssh ${ssh_options[*]} '$REMOTE_HOST' 'mkdir -p $REMOTE_EXEC_PATH' && timeout 60 scp ${ssh_options[*]} '$probe' '$REMOTE_HOST:$probe_remote' && timeout 60 ssh ${ssh_options[*]} '$REMOTE_HOST' 'cat $probe_remote && rm -f $probe_remote'"
rm -f "$probe"
exit "$failed"
