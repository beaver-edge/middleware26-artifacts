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

- [DONE] Multi-architecture image `noahwu/beaver-edge:middleware26`
  (`linux/amd64` + `linux/arm64`), pinned by digest in `run-artifact.sh`. The
  amd64 image was built under Rosetta on Apple Silicon (QEMU crashes Arduino
  CLI); `.github/workflows/build-amd64-image.yml` builds it natively instead.
  After any source change, rebuild both from `git archive HEAD`, push, and
  update the digest.
- [TODO] State **Artifacts Functional** explicitly in the submission abstract.
- [TODO] Decide whether to also request **Artifacts Available**. That badge
  needs a durable public location (a public Git repository or a Zenodo DOI) for
  this source and, optionally, the Docker image. If requested, link it from
  `README.md`.
- [TODO] Fill in the remaining `[TODO]` markers in `README.md`.
