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

