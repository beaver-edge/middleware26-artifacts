#!/usr/bin/env bash
# BEAVER-EDGE artifact evaluation — the single entry point for reviewers.
#
#   ./run-artifact.sh                      pull image, self-test, API check, DP/MC/ArdSG/CPU-SG,
#                                          plus TPU-SG on the authors' Coral board when eval-ssh/ exists
#   ./run-artifact.sh data convert         run only the listed tasks (after the checks)
#   ./run-artifact.sh --no-tpu             skip TPU-SG even when board access is available
#   ./run-artifact.sh --tpu DIR            use another SSH directory for TPU-SG (e.g. your own board)
#   ./run-artifact.sh --local-image        use the image built by ./build-image.sh
#
# Credentials: when .env is missing and credentials.tar.gz.enc is present, the
# script asks for the password from the submission system (or reads
# BEAVER_EDGE_CREDENTIALS_PASSWORD) and unpacks .env and eval-ssh/.
#
# Every step prints PASS or FAIL. FAIL appears only for a system failure:
# unexpected behavior the framework cannot handle (a bug, a crash, a missing
# dependency, an unreachable service). A task whose LLM exhausts its bounded
# repair budget without producing valid code is an EXPECTED outcome and PASSes.
# Exit status: 0 if nothing FAILed, 1 otherwise, 2 for usage errors.
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)

# TODO(before submission): replace with the published Docker Hub image, ideally pinned by digest.
PUBLISHED_IMAGE="noahwu/beaver-edge:middleware26"
LOCAL_IMAGE="beaver-edge-artifact:middleware26"   # tag produced by ./build-image.sh
SEALED_CREDENTIALS=credentials.tar.gz.enc          # made by scripts/maintainer/eval-access.sh seal

usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; }

image=${BEAVER_EDGE_IMAGE:-$PUBLISHED_IMAGE}
use_local=0
provider=openrouter
ssh_dir=
no_tpu=0
out=
tasks=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-image) use_local=1; image=$LOCAL_IMAGE ;;
        --image) image=${2:?--image needs a value}; shift ;;
        --provider) provider=${2:?--provider needs openrouter or ollama}; shift ;;
        --tpu) ssh_dir=${2:?--tpu needs the evaluation SSH directory}; shift ;;
        --no-tpu) no_tpu=1 ;;
        --out) out=${2:?--out needs a directory}; shift ;;
        -h|--help) usage; exit 0 ;;
        data|convert|ardsketch|pysketch|tpusketch) tasks+=("$1") ;;
        all) ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
case "$provider" in openrouter|ollama) ;; *) echo "Unknown provider: $provider" >&2; exit 2 ;; esac
if [[ -n $ssh_dir ]]; then
    [[ -d $ssh_dir ]] || { echo "SSH directory not found: $ssh_dir" >&2; exit 2; }
    ssh_dir=$(cd "$ssh_dir" && pwd)
fi

stamp=$(date -u +%Y%m%dT%H%M%S)
out=${out:-artifact-runs/$stamp}
mkdir -p "$out"
out=$(cd "$out" && pwd)
cp artifacts/README.md "$out/HOW-TO-READ-ARTIFACTS.md" 2>/dev/null || true

# ---------------------------------------------------------------- reporting
summary=()
failed=0
line() { printf '%-20s %-8s %s\n' "$1" "$2" "$3"; }
record() {  # record NAME RESULT MEANING
    summary+=("$(line "$1" "$2" "$3")")
    [[ $2 == FAIL ]] && failed=1
    printf '\n>>> %s\n' "$(line "$1" "$2" "$3")"
}
task_label() {
    case "$1" in
        data) echo "DP (data)" ;;
        convert) echo "MC (convert)" ;;
        ardsketch) echo "ArdSG (ardsketch)" ;;
        pysketch) echo "CPU-SG (pysketch)" ;;
        tpusketch) echo "TPU-SG (tpusketch)" ;;
    esac
}
finish() {
    {
        echo
        echo "======================= BEAVER-EDGE evaluation summary ======================="
        line "STEP / TASK" "RESULT" "MEANING"
        for row in "${summary[@]}"; do echo "$row"; done
        echo "------------------------------------------------------------------------------"
        if [[ $failed -eq 0 ]]; then
            echo "OVERALL: PASS - no system failure; the framework behaved as designed."
        else
            echo "OVERALL: FAIL - at least one system failure (unexpected behavior). See the log named above."
        fi
        echo "PASS = expected behavior (artifact generated, OR repair budget exhausted as designed)."
        echo "FAIL = system failure: bug, crash, missing dependency, or unreachable service."
        echo "Evidence: $out"
    } | tee "$out/SUMMARY.txt"
    exit "$failed"
}
trap 'record "interrupted" FAIL "stopped by user"; finish' INT TERM

echo "BEAVER-EDGE artifact evaluation - image=$image provider=$provider"
echo "Evidence directory: $out"

# ---------------------------------------------------------------- 0. setup
if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
    record "0 setup" FAIL "Docker is not installed or its daemon is not running"
    finish
fi
if [[ $use_local -eq 1 ]]; then
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        record "0 setup" FAIL "local image $image not found - run ./build-image.sh first"
        finish
    fi
elif [[ $image == *"<"* ]]; then
    record "0 setup" FAIL "the prebuilt image is not published yet - run ./build-image.sh, then ./run-artifact.sh --local-image"
    finish
elif ! docker pull "$image" 2>&1 | tee "$out/0-setup.log"; then
    record "0 setup" FAIL "could not pull $image (see 0-setup.log)"
    finish
fi
record "0 setup" PASS "image ready: $image"

# ---------------------------------------------------------------- 0b. credentials
unlock_credentials() {  # decrypt with the image's OpenSSL, so the host needs nothing but Docker
    local password=${BEAVER_EDGE_CREDENTIALS_PASSWORD:-} tarball
    if [[ -z $password ]]; then
        [[ -t 0 ]] || { echo "no password: set BEAVER_EDGE_CREDENTIALS_PASSWORD" > "$out/0-credentials.log"; return 1; }
        read -rsp "Password for $SEALED_CREDENTIALS (from the submission system): " password
        echo
    fi
    tarball=$(mktemp)
    if BEAVER_EDGE_CREDENTIALS_PASSWORD=$password docker run --rm -i -e BEAVER_EDGE_CREDENTIALS_PASSWORD "$image" \
            openssl enc -d -aes-256-cbc -pbkdf2 -iter 600000 -pass env:BEAVER_EDGE_CREDENTIALS_PASSWORD \
            < "$SEALED_CREDENTIALS" > "$tarball" 2> "$out/0-credentials.log" \
        && tar -xzf "$tarball" .env eval-ssh 2>> "$out/0-credentials.log"; then
        rm -f "$tarball"
        chmod 700 eval-ssh && chmod 600 .env eval-ssh/id_*
        return 0
    fi
    rm -f "$tarball"
    return 1
}
if [[ -f .env ]]; then
    record "0 credentials" PASS "using the existing .env"
elif [[ -f $SEALED_CREDENTIALS ]]; then
    if unlock_credentials; then
        record "0 credentials" PASS "unpacked .env and eval-ssh/ from $SEALED_CREDENTIALS"
    else
        record "0 credentials" FAIL "could not decrypt $SEALED_CREDENTIALS - wrong or missing password (see 0-credentials.log)"
        finish
    fi
else
    record "0 credentials" FAIL "missing .env - run: cp example.env .env, then add your LLM API key"
    finish
fi
if [[ -z $ssh_dir && $no_tpu -eq 0 && -d eval-ssh ]]; then
    ssh_dir=$ROOT/eval-ssh
fi
if [[ ${#tasks[@]} -eq 0 ]]; then
    tasks=(data convert ardsketch pysketch)
    [[ -n $ssh_dir && $no_tpu -eq 0 ]] && tasks+=(tpusketch)
fi
for task in "${tasks[@]}"; do
    if [[ $task == tpusketch && -z $ssh_dir ]]; then
        record "0 credentials" FAIL "TPU-SG needs board access: eval-ssh/ or --tpu DIR"
        finish
    fi
done
echo "Tasks: ${tasks[*]}"

# ---------------------------------------------------------------- 1. offline self-test
docker run --rm "$image" python -m pytest tests -q > "$out/1-self-test.log" 2>&1
if [[ $? -eq 0 ]]; then
    record "1 self-test" PASS "offline unit tests: $(tail -n 1 "$out/1-self-test.log" | tr -d '=')"
else
    record "1 self-test" FAIL "offline unit tests failed (see 1-self-test.log)"
    finish
fi

# ---------------------------------------------------------------- 2. LLM API check
docker run --rm -v "$ROOT/.env:/artifact/.env:ro" "$image" \
    python scripts/container/check_llm_api.py --provider "$provider" > "$out/2-api-check.log" 2>&1
if [[ $? -eq 0 ]]; then
    record "2 api-check" PASS "$provider API reachable and returns parseable JSON"
else
    record "2 api-check" FAIL "$provider API check failed - check .env key/model (see 2-api-check.log)"
    for task in "${tasks[@]}"; do record "$(task_label "$task")" "NOT RUN" "skipped because the API check failed"; done
    finish
fi

# ---------------------------------------------------------------- 2b. board preflight (TPU-SG only)
if [[ " ${tasks[*]} " == *" tpusketch "* ]]; then
    docker run --rm -v "$ROOT/.env:/artifact/.env:ro" -v "$ssh_dir:/run/evaluation-ssh:ro" "$image" \
        bash scripts/container/setup-ssh-and-run-task.sh --check-board > "$out/2b-board-check.log" 2>&1
    if [[ $? -eq 0 ]]; then
        record "2b board-check" PASS "Coral board reachable over SSH, Edge TPU and inputs present"
    else
        record "2b board-check" FAIL "Coral board not usable (see 2b-board-check.log)"
        record "$(task_label tpusketch)" "NOT RUN" "skipped because the board check failed"
        remaining=()
        for task in "${tasks[@]}"; do [[ $task == tpusketch ]] || remaining+=("$task"); done
        tasks=(${remaining[@]+"${remaining[@]}"})   # bash 3.2 + set -u: empty arrays need the + guard
    fi
fi

# ---------------------------------------------------------------- 3. the workflow tasks
for task in ${tasks[@]+"${tasks[@]}"}; do
    dir="$out/$task"
    mkdir -p "$dir/logs" "$dir/artifacts"
    run=(docker run --rm --name "beaver-edge-$task-$stamp-$$"
        -v "$ROOT/.env:/artifact/.env:ro"
        -v "$dir/logs:/artifact/logs"
        -v "$dir/artifacts:/artifact/artifacts")
    command=(bash scripts/container/run-task.sh --provider "$provider" "$task")
    if [[ $task == tpusketch ]]; then
        run+=(-v "$ssh_dir:/run/evaluation-ssh:ro" -e REMOTE_EXECUTION_ENABLED=true)
        command=(bash scripts/container/setup-ssh-and-run-task.sh --provider "$provider" "$task")
    fi
    echo
    echo "=== $(task_label "$task"): running (live log: $dir/workflow.log) ==="
    started=$SECONDS
    "${run[@]}" "$image" "${command[@]}" 2>&1 | tee "$dir/workflow.log"
    elapsed="$(( (SECONDS - started) / 60 ))m$(( (SECONDS - started) % 60 ))s"

    status_line=$(grep -E "^(COMPLETE|ERROR) $task " "$dir/logs/status.log" 2>/dev/null | tail -n 1)
    if [[ $status_line == COMPLETE*artifact_status=generated* ]]; then
        record "$(task_label "$task")" PASS "artifact generated and verified [$elapsed]"
    elif [[ $status_line == COMPLETE*artifact_status=not_generated* ]]; then
        record "$(task_label "$task")" PASS "no artifact: repair budget exhausted (expected outcome, not a failure) [$elapsed]"
    else
        record "$(task_label "$task")" FAIL "system failure - see $task/workflow.log [$elapsed]"
    fi
done

finish
