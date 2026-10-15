#!/usr/bin/env bash
# Runs INSIDE a throwaway container (run-artifact.sh feeds it through stdin, as root,
# because the container wrote those files as root) or directly in the tests.
#
# Usage: bash dedupe-work.sh [TASK_DIR]        TASK_DIR defaults to /evidence
#
# work/ and artifacts/**/outputs/valid/ can hold the same bytes, e.g. the ~30 MB
# CPU output video. Every work/ file that is byte-identical to an archived valid
# output becomes a hard link to it: same path, same content, one copy on disk.
# The link is made under a temporary name and renamed over the original, so a
# failure at any point leaves the original file in place. Nothing is deleted
# otherwise. Prints "<files linked> <bytes saved>"; lists them in work-links.txt.
set -uo pipefail
cd "${1:-/evidence}" || exit 1
if [[ ! -d work || ! -d artifacts ]]; then
    echo "0 0"
    exit 0
fi
linked=0
saved=0
report=
while IFS= read -r archived; do
    size=$(wc -c < "$archived" | tr -d ' ')
    [[ $size -gt 0 ]] || continue
    while IFS= read -r copy; do
        [[ $copy -ef $archived ]] && continue
        cmp -s -- "$archived" "$copy" || continue
        temporary="$copy.link-$$"
        if ln -- "$archived" "$temporary" 2>/dev/null; then
            if mv -f -- "$temporary" "$copy"; then
                linked=$((linked + 1))
                saved=$((saved + size))
                report+="$copy <- $archived"$'\n'
            else
                rm -f -- "$temporary"
            fi
        fi
    done < <(find work -type f -size "${size}c")
done < <(find artifacts -type f -path '*/outputs/valid/*')
[[ -n $report ]] && printf '%s' "$report" > work-links.txt
echo "$linked $saved"
