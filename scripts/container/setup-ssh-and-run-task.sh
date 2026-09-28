#!/usr/bin/env bash
# Runs INSIDE the container (TPUSG only). Copy only evaluator-provided SSH runtime files into the ephemeral container layer.
#   setup-ssh-and-run-task.sh [--provider P] TASK...   then run the task(s)
#   setup-ssh-and-run-task.sh --check-board            then run only the board preflight
set -euo pipefail
source_directory=/run/evaluation-ssh
mkdir -p /root/.ssh
for file in \
    "$source_directory/config" \
    "$source_directory/known_hosts" \
    "$source_directory/known_hosts2" \
    "$source_directory"/id_*; do
    if [[ -f $file && -r $file ]]; then
        cp "$file" /root/.ssh/
    fi
done
chmod 700 /root/.ssh
chmod 600 /root/.ssh/config /root/.ssh/id_* 2>/dev/null || true
if [[ ${1:-} == --check-board ]]; then
    exec bash scripts/container/check_board_access.sh
fi
exec bash scripts/container/run-task.sh "$@"
