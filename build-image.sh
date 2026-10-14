#!/usr/bin/env bash
# OPTIONAL: build the evaluation image from scratch instead of pulling the
# prebuilt one. This downloads Conda, TensorFlow, and the Arduino toolchain,
# and can take a long time. Afterwards run:  ./run-artifact.sh --local-image
#
#   ./build-image.sh                          build for this machine's architecture
#   ./build-image.sh --platform linux/amd64   extra arguments are passed to docker build
#
# Supported: linux/amd64 and linux/arm64. Build natively; the Arduino CLI has
# crashed under cross-architecture emulation.

# Re-run under bash when another shell started this script (e.g. `sh build-image.sh`,
# where sh is dash on Debian/Ubuntu). Keep this block POSIX sh.
if [ -z "${BASH_VERSION:-}" ]; then
    if command -v bash >/dev/null 2>&1; then
        exec bash "$0" "$@"
    fi
    echo "build-image.sh needs bash, which was not found on PATH." >&2
    exit 2
fi

set -euo pipefail
cd "$(dirname "$0")"
tag="beaver-edge-artifact:middleware26"   # the tag ./run-artifact.sh --local-image expects
docker build -t "$tag" "$@" .
echo
echo "Built $tag. Next: ./run-artifact.sh --local-image"
