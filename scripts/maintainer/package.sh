#!/usr/bin/env bash
# MAINTAINER ONLY: build the submission source archive from an explicit list;
# never include credentials/history. Reviewers do not need this script.
set -euo pipefail
export COPYFILE_DISABLE=1
cd "$(dirname "$0")/../.."
mkdir -p dist
archive=dist/beaver-edge-middleware26-source.tar.gz
tar --exclude='__pycache__' --exclude='.DS_Store' --exclude='scripts/maintainer' \
    -czf "$archive" \
    LICENSE run-artifact.sh build-image.sh Dockerfile .dockerignore README.md example.env \
    requirements.txt requirements-optional.txt scripts tests src vendor docs \
    artifacts/README.md inputs \
    $(test -f credentials.tar.gz.enc && echo credentials.tar.gz.enc)
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$archive" > "$archive.sha256"
else
    shasum -a 256 "$archive" > "$archive.sha256"
fi
echo "Created $archive and checksum (no private environment or version-control metadata)"
