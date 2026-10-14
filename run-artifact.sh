#!/usr/bin/env bash
# BEAVER-EDGE artifact evaluation — the single entry point for reviewers.
#
#   ./run-artifact.sh                      pull image, self-test, API check, DP/MC/ArdSG/Py-CPU,
#                                          plus Py-TPU on the authors' Coral board when eval-ssh/ exists
#   ./run-artifact.sh data convert         run only the listed tasks (after the checks)
#   ./run-artifact.sh --no-tpu             skip Py-TPU even when board access is available
#   ./run-artifact.sh --tpu DIR            use another SSH directory for Py-TPU (e.g. your own board)
#   ./run-artifact.sh --local-image        use the image built by ./build-image.sh
#
# Credentials: when .env is missing and credentials.tar.gz.enc is present, the
# script asks for the password from the submission system (or reads
# BEAVER_EDGE_CREDENTIALS_PASSWORD) and unpacks .env and eval-ssh/.
#
# Each check prints PASS or FAIL. Each task gets two independent verdicts:
#   RESULT     PASS  = the generated artifact passed validation and its outputs were verified
#              FAIL  = no valid artifact within the bounded repair budget (all attempts are kept)
#   FRAMEWORK  OK    = the framework ran the task to a recorded, verified outcome
#              ERROR = bug, crash, missing dependency, unreachable service, or inconsistent evidence
# RESULT judges what the LLM generated; FRAMEWORK judges BEAVER-EDGE itself.
# Exit status: 0 = every task PASS and FRAMEWORK OK; 3 = FRAMEWORK OK but at least
# one task FAIL; 1 = a check failed or FRAMEWORK ERROR; 2 = usage error.

# Re-run under bash when another shell started this script, e.g. `sh run-artifact.sh`
# (sh is dash on Debian/Ubuntu and cannot parse the arrays below). Keep this block
# POSIX sh and above every bash-only construct; it also keeps usage()'s line range valid.
if [ -z "${BASH_VERSION:-}" ]; then
    if command -v bash >/dev/null 2>&1; then
        exec bash "$0" "$@"
    fi
    echo "run-artifact.sh needs bash, which was not found on PATH." >&2
    exit 2
fi

set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)

# Git Bash (MSYS) would rewrite container paths such as /artifact/.env in the -v arguments.
export MSYS_NO_PATHCONV=1
case "$(uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*) echo "Note: Git Bash/MSYS detected. If the container mounts fail, run this script from WSL 2." >&2 ;;
esac

PUBLISHED_IMAGE="noahwu/beaver-edge:middleware26@sha256:a317b940a493e208ee3d30893344317d6a5fff85603f299e5b4ca17d67caf5c6"
LOCAL_IMAGE="beaver-edge-artifact:middleware26"   # tag produced by ./build-image.sh
SEALED_CREDENTIALS=credentials.tar.gz.enc          # made by scripts/maintainer/eval-access.sh seal

usage() { sed -n '2,22p' "$ROOT/${0##*/}" | sed 's/^# \{0,1\}//'; }   # $0 may be relative to the pre-cd directory

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
check_failed=0      # a setup check (0-2b) failed, or the run was interrupted
framework_error=0   # FRAMEWORK ERROR on at least one task
tasks_pass=0
tasks_fail=0
tasks_error=0
tasks_not_run=0
line() { printf '%-20s %-8s %-10s %s\n' "$1" "$2" "$3" "$4"; }
record() {  # record NAME RESULT FRAMEWORK DETAILS   (FRAMEWORK is "-" for checks)
    summary+=("$(line "$1" "$2" "$3" "$4")")
    printf '\n>>> %s\n' "$(line "$1" "$2" "$3" "$4")"
    if [[ $3 == - ]]; then
        [[ $2 == FAIL ]] && check_failed=1
        [[ $2 == "NOT RUN" ]] && tasks_not_run=$((tasks_not_run + 1))
        return 0
    fi
    case "$3" in
        ERROR) framework_error=1; tasks_error=$((tasks_error + 1)) ;;
        OK) if [[ $2 == PASS ]]; then tasks_pass=$((tasks_pass + 1)); else tasks_fail=$((tasks_fail + 1)); fi ;;
    esac
}
json_int() {  # json_int KEY FILE: an integer field of an outcome.json written with indent=2
    sed -n "s/^ *\"$1\": *\([0-9][0-9]*\).*/\1/p" "$2" 2>/dev/null | head -n 1
}
task_label() {
    case "$1" in
        data) echo "DP (data)" ;;
        convert) echo "MC (convert)" ;;
        ardsketch) echo "ArdSG (ardsketch)" ;;
        pysketch) echo "Py-CPU (pysketch)" ;;
        tpusketch) echo "Py-TPU (tpusketch)" ;;
    esac
}
finish() {
    local status=0
    if [[ $check_failed -eq 1 || $framework_error -eq 1 ]]; then
        status=1
    elif [[ $tasks_fail -gt 0 ]]; then
        status=3
    fi
    {
        echo
        echo "=========================== BEAVER-EDGE evaluation summary ==========================="
        line "STEP / TASK" "RESULT" "FRAMEWORK" "DETAILS"
        for row in "${summary[@]}"; do echo "$row"; done
        echo "--------------------------------------------------------------------------------------"
        if [[ $check_failed -eq 1 ]]; then
            echo "CHECKS:    FAIL - a setup check failed or the run was interrupted; see the log named above."
        else
            echo "CHECKS:    PASS"
        fi
        if [[ $((tasks_pass + tasks_fail + tasks_error)) -eq 0 ]]; then
            echo "TASKS:     none ran to completion"
        else
            printf 'TASKS:     %d PASS, %d FAIL' "$tasks_pass" "$tasks_fail"
            [[ $tasks_error -gt 0 ]] && printf ', %d without a result (FRAMEWORK ERROR)' "$tasks_error"
            [[ $tasks_not_run -gt 0 ]] && printf ', %d NOT RUN' "$tasks_not_run"
            printf '   (task results depend on the LLM and vary between runs)\n'
        fi
        if [[ $framework_error -eq 1 ]]; then
            echo "FRAMEWORK: ERROR - at least one task hit a framework error; see its workflow.log."
        elif [[ $((tasks_pass + tasks_fail)) -gt 0 ]]; then
            echo "FRAMEWORK: OK - every task that ran reached a recorded, verified outcome."
        else
            echo "FRAMEWORK: - (no task ran)"
        fi
        echo "RESULT    = the task: PASS = valid artifact generated and verified; FAIL = none within the repair budget."
        echo "FRAMEWORK = BEAVER-EDGE itself: OK = ran to a recorded outcome; ERROR = bug, crash, or unreachable service."
        echo "Exit status $status (0 = all tasks PASS; 3 = FRAMEWORK OK but a task FAILed; 1 = check failed or FRAMEWORK ERROR)."
        echo "Evidence: $out"
    } | tee "$out/SUMMARY.txt"
    exit "$status"
}
trap 'record "interrupted" FAIL - "stopped by user"; finish' INT TERM

echo "BEAVER-EDGE artifact evaluation - image=$image provider=$provider"
echo "Evidence directory: $out"

# ---------------------------------------------------------------- 0. setup
if ! command -v docker >/dev/null 2>&1; then
    record "0 setup" FAIL - "Docker is not installed - see https://docs.docker.com/get-docker/"
    finish
elif ! docker info >/dev/null 2> "$out/0-setup.log"; then
    if grep -qi "permission denied" "$out/0-setup.log"; then
        record "0 setup" FAIL - "this user cannot use the Docker daemon - on Linux, add it to the docker group and log in again (see 0-setup.log)"
    else
        record "0 setup" FAIL - "the Docker daemon is not reachable - start Docker Desktop or the docker service (see 0-setup.log)"
    fi
    finish
fi
if [[ $use_local -eq 1 ]]; then
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        record "0 setup" FAIL - "local image $image not found - run ./build-image.sh first"
        finish
    fi
elif [[ $image == *"<"* ]]; then
    record "0 setup" FAIL - "the prebuilt image is not published yet - run ./build-image.sh, then ./run-artifact.sh --local-image"
    finish
elif ! docker pull "$image" 2>&1 | tee "$out/0-setup.log"; then
    record "0 setup" FAIL - "could not pull $image (see 0-setup.log)"
    finish
fi
record "0 setup" PASS - "image ready: $image"

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
    record "0 credentials" PASS - "using the existing .env"
elif [[ -f $SEALED_CREDENTIALS ]]; then
    if unlock_credentials; then
        record "0 credentials" PASS - "unpacked .env and eval-ssh/ from $SEALED_CREDENTIALS"
    else
        record "0 credentials" FAIL - "could not decrypt $SEALED_CREDENTIALS - wrong or missing password (see 0-credentials.log)"
        finish
    fi
else
    record "0 credentials" FAIL - "missing .env - run: cp example.env .env, then add your LLM API key"
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
        record "0 credentials" FAIL - "Py-TPU needs board access: eval-ssh/ or --tpu DIR"
        finish
    fi
done
echo "Tasks: ${tasks[*]}"

# ---------------------------------------------------------------- 1. offline self-test
docker run --rm "$image" python -m pytest tests -q > "$out/1-self-test.log" 2>&1
if [[ $? -eq 0 ]]; then
    record "1 self-test" PASS - "offline unit tests: $(tail -n 1 "$out/1-self-test.log" | tr -d '=')"
else
    record "1 self-test" FAIL - "offline unit tests failed (see 1-self-test.log)"
    finish
fi

# ---------------------------------------------------------------- 2. LLM API check
docker run --rm -v "$ROOT/.env:/artifact/.env:ro" "$image" \
    python scripts/container/check_llm_api.py --provider "$provider" > "$out/2-api-check.log" 2>&1
if [[ $? -eq 0 ]]; then
    record "2 api-check" PASS - "$provider API reachable and returns parseable JSON"
else
    record "2 api-check" FAIL - "$provider API check failed - check .env key/model (see 2-api-check.log)"
    for task in "${tasks[@]}"; do record "$(task_label "$task")" "NOT RUN" - "skipped because the API check failed"; done
    finish
fi

# ---------------------------------------------------------------- 2b. board preflight (Py-TPU only)
if [[ " ${tasks[*]} " == *" tpusketch "* ]]; then
    docker run --rm -v "$ROOT/.env:/artifact/.env:ro" -v "$ssh_dir:/run/evaluation-ssh:ro" "$image" \
        bash scripts/container/setup-ssh-and-run-task.sh --check-board > "$out/2b-board-check.log" 2>&1
    if [[ $? -eq 0 ]]; then
        record "2b board-check" PASS - "Coral board reachable over SSH, Edge TPU and inputs present"
    else
        record "2b board-check" FAIL - "Coral board not usable (see 2b-board-check.log)"
        record "$(task_label tpusketch)" "NOT RUN" - "skipped because the board check failed"
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
    # status.log is append-only; read only the lines this run adds, so a reused
    # --out directory can never report a previous run's outcome.
    status_seen=$(cat "$dir/logs/status.log" 2>/dev/null | wc -l | tr -d ' ')
    "${run[@]}" "$image" "${command[@]}" 2>&1 | tee "$dir/workflow.log"
    elapsed="$(( (SECONDS - started) / 60 ))m$(( (SECONDS - started) % 60 ))s"

    # COMPLETE/ERROR is the framework verdict; artifact_status is the task verdict.
    status_line=$(tail -n "+$((status_seen + 1))" "$dir/logs/status.log" 2>/dev/null \
        | grep -E "^(COMPLETE|ERROR) $task " | tail -n 1)
    outcome_file=${status_line#*outcome_file=}
    outcome_file="$dir/${outcome_file%% *}"
    attempts="attempts kept: $(json_int valid_attempts "$outcome_file") valid, $(json_int invalid_attempts "$outcome_file") invalid"
    if [[ $status_line == COMPLETE*artifact_status=generated* ]]; then
        record "$(task_label "$task")" PASS OK "artifact generated and verified; $attempts [$elapsed]"
    elif [[ $status_line == COMPLETE*artifact_status=not_generated* ]]; then
        record "$(task_label "$task")" FAIL OK "no valid artifact: repair budget exhausted; $attempts [$elapsed]"
    else
        record "$(task_label "$task")" - ERROR "framework error, task result unknown - see $task/workflow.log [$elapsed]"
    fi
done

finish
