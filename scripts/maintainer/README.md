# Maintainer notes

For the authors only. Evaluators do not need anything in this directory.

## Packaging the source archive

```bash
bash scripts/maintainer/package.sh    # dist/beaver-edge-middleware26-source.tar.gz + .sha256
```

Never include `.env` or SSH credentials in public archives or images. `.env` is
mounted read-only into each container at runtime and is never copied into the
image, the evidence directory, or the source archive.

## Evaluator access to the Coral board

`eval-access.sh` creates a restricted, revocable SSH key for evaluators, installs
it on the jump host and the board, checks it, seals `.env` and `eval-ssh/` into
an encrypted archive, and revokes the key after the evaluation. Run it without
arguments for usage. Send the archive password through HotCRP, never through
the repository.

## Before submission

- [TODO] Push a multi-architecture (`linux/amd64` + `linux/arm64`) image to
  Docker Hub, building each architecture natively. Replace the
  `<DOCKERHUB_USER>/beaver-edge:middleware26` placeholder in `run-artifact.sh`
  and `README.md`, ideally pinned by digest.
- [TODO] State **Artifacts Functional** explicitly in the submission abstract.
- [TODO] Decide whether to also request **Artifacts Available**. That badge
  needs a durable public location (a public Git repository or a Zenodo DOI) for
  this source and, optionally, the Docker image. If requested, link it from
  `README.md`.
- [TODO] Fill in the remaining `[TODO]` markers in `README.md` and
  `VALIDATION.md`.
