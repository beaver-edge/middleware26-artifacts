#!/usr/bin/env bash
# Runs INSIDE the container. Reviewers normally use ./run-artifact.sh instead.
#
# Usage: bash scripts/container/run-task.sh [--provider openrouter|ollama] [TASK...|all]
#
# Each task appends one line to logs/status.log:
#   COMPLETE <task> exit=0 artifact_status=<generated|not_generated> ...
#   ERROR    <task> exit=<n> artifact_status=<...|unknown> ...
# COMPLETE covers both a validated artifact and an exhausted repair budget;
# only ERROR (an unexpected framework failure) is an evaluation failure.
set -uo pipefail
cd /artifact
provider=openrouter
if [[ ${1:-} == --provider ]]; then
    provider=${2:?Specify openrouter or ollama after --provider}
    shift 2
fi
case "$provider" in openrouter|ollama) ;; *) echo "Unknown provider: $provider" >&2; exit 2;; esac
tasks=("$@")
if [[ ${#tasks[@]} -eq 0 || ${tasks[0]} == all ]]; then
    tasks=(data convert ardsketch pysketch)
fi
mkdir -p logs
status=0
for task in "${tasks[@]}"; do
    case "$task" in data|convert|ardsketch|pysketch|tpusketch) ;; *) echo "Unknown task: $task" >&2; exit 2;; esac
    stamp=$(date -u +%Y%m%dT%H%M%S)-$$
    log="logs/${stamp}-${task}"
    outcome_file="logs/${stamp}-${task}.outcome.json"
    echo "START $task provider=$provider $(date -u +%FT%TZ)" | tee "$log.log"
    marker="$log.started"
    touch "$marker"
    /usr/bin/time -v -o "$log.resources.txt" python -m src.main --task "$task" --model-provider "$provider" --outcome-file "$outcome_file" 2>&1 | tee -a "$log.log"
    result=${PIPESTATUS[0]}
    if [[ $result -eq 0 ]]; then
        python scripts/container/verify_output.py "$task" "$marker" "$outcome_file" 2>&1 | tee -a "$log.log"
        result=${PIPESTATUS[0]}
    fi
    artifact_status=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("artifact_status") or "unknown")' "$outcome_file" 2>/dev/null || echo unknown)
    if [[ $result -eq 0 ]]; then outcome=COMPLETE; else outcome=ERROR; status=1; fi
    echo "$outcome $task exit=$result artifact_status=$artifact_status outcome_file=$outcome_file $(date -u +%FT%TZ)" | tee -a "$log.log" logs/status.log
done
exit "$status"
