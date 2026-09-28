# BEAVER-EDGE — Middleware 2026 artifact

Artifact for the paper **Autopilots Need Parachutes: Lessons Learned from LLM-Automated Embedded ML Pipelines**.

BEAVER-EDGE uses an LLM to generate, validate, and automatically repair code for five embedded-ML pipeline tasks. This package runs all five workflows in Docker with a single script and reports PASS/FAIL for each one.

**Requested badges: Artifacts Functional, Artifacts Available.** The package is publicly available and shows that the workflows install and run as the paper describes. It does not reproduce the paper's aggregate measurements, tables, or figures, which come from many repeated experiments.

## Requirements
- **Software:** Docker Engine (Linux) or Docker Desktop (macOS/Windows), on `amd64` or `arm64`. All dependencies are in the image.
- **Hardware:** 4 CPUs, 8 GB RAM, and 20 GB free disk are suggested (estimates, not measured minimums). 
  - **Coral board (Py-TPU only):** we provide temporary SSH access to a prepared Google Coral Dev Board for artifacts evaluation.
- **Network:** internet access for the image pull and the LLM API calls.
- **LLM access:** we provide temporary API keys (see [Credentials](#credentials)).   The runs use `qwen/qwen3.8-27b` through OpenRouter. A full run costs about   **US$2–3** in API credits.  We also provide an Ollama Cloud configuration as a fallback or for trying other models (`--provider ollama`).
- **Time:** allow about **3 hours** for a full run. The exact duration depends   on the host's speed and on the LLM provider's load.

## Credentials

The repository contains `credentials.tar.gz.enc`, an encrypted archive with:

- `.env`: the temporary LLM API key and the board settings, and
- `eval-ssh/`: a dedicated SSH key that can only reach the Coral board.

**We send the password through HotCRP.** `run-artifact.sh` asks for it on the first run (or reads `BEAVER_EDGE_CREDENTIALS_PASSWORD`) and unpacks both. Your own `~/.ssh` is thus not used. The key and the API key are revoked after the evaluation.

To use your own API key instead, run `cp example.env .env` and fill in the `OPENROUTER_*` values before the first run. An existing `.env` is used as is.

## Quick start

```bash
./run-artifact.sh        # enter the password from HotCRP submission when asked
```

This pulls the prebuilt image, unpacks the credentials, and runs all checks and all five tasks, including Py-TPU on our Coral board.

> **The image is published on Docker Hub**, but you can build it locally (slow; it downloads Conda, TensorFlow, and the Arduino toolchain):
>
> ```bash
> ./build-image.sh
> ./run-artifact.sh --local-image
> ```

## Expected output

The script ends with a summary like this (illustrative; outcomes and timings vary between runs):

```text
======================= BEAVER-EDGE evaluation summary =======================
STEP / TASK          RESULT   MEANING
0 setup              PASS     image ready: noahwu/beaver-edge:middleware26
0 credentials        PASS     unpacked .env and eval-ssh/ from credentials.tar.gz.enc
1 self-test          PASS     offline unit tests: 93 passed, 2 skipped in 41.2s
2 api-check          PASS     openrouter API reachable and returns parseable JSON
DP (data)            PASS     artifact generated and verified [48m12s]
MC (convert)         PASS     artifact generated and verified [21m40s]
ArdSG (ardsketch)    PASS     no artifact: repair budget exhausted (expected outcome, not a failure) [57m03s]
Py-CPU (pysketch)    PASS     artifact generated and verified [36m55s]
Py-TPU (tpusketch)   PASS     artifact generated and verified [16m20s]
------------------------------------------------------------------------------
OVERALL: PASS - no system failure; the framework behaved as designed.
```

The summary is also saved to `artifact-runs/<timestamp>/SUMMARY.txt`. The exit status is 0 if nothing failed and 1 otherwise.

### How to interpret PASS and FAIL

BEAVER-EDGE is designed to handle both successful and unsuccessful code generation, because the paper draws lessons from both. An LLM may not produce working code within the bounded number of repair attempts. Stopping cleanly and keeping every attempt is the intended behavior in that case.

| What happened | Reported as |
| --- | --- |
| Generated code passed validation, and its outputs were verified | **PASS** — artifact generated and verified |
| Every attempt failed until the repair budget ran out; all attempts were kept | **PASS** — no artifact: repair budget exhausted (expected outcome) |
| Crash, uncaught exception, missing dependency, unreachable API or board, missing or stale outputs | **FAIL** — system failure |

Only FAIL indicates a problem with the artifact or its environment. LLM output is non-deterministic, so a rerun can move a task between the two PASS rows. If a FAIL comes from the environment (for example a network error), rerun only that task, e.g. `./run-artifact.sh ardsketch`.

## What the script runs

| Step | What it checks |
| --- | --- |
| 0 setup | Docker is running and the image is available |
| 0 credentials | `.env` exists or was unpacked; board access is available for Py-TPU |
| 1 self-test | Offline unit tests in the container (no API cost) |
| 2 api-check | One small request to the LLM provider; if it fails, the tasks are skipped |
| 2b board-check | Py-TPU only: SSH to the Coral board, Edge TPU visible, board inputs present; if it fails, only Py-TPU is skipped |
| Tasks | Each task runs in a fresh container, and then its outputs are verified |

The five tasks (the paper and the command line use different names):

| Paper name | CLI name | The LLM generates | A generated artifact is verified by |
| --- | --- | --- | --- |
| DP — Data Processing | `data` | Python transformations of a CSV | Transformed CSV and plots exist |
| MC — Model Conversion with INT8 quantization | `convert` | Keras → quantized TFLite conversion code | A TFLite interpreter loads `model_quant.tflite` |
| ArdSG — Arduino Sketch Generation | `ardsketch` | Arduino sketch for the Nano 33 BLE Sense | It compiles with the Arduino CLI (not flashed) |
| Py-CPU — CPU Sketch Generation | `pysketch` | Python object detection with TFLite on the CPU | The annotated output video has a readable frame |
| Py-TPU — TPU Sketch Generation | `tpusketch` | Python object detection for the Edge TPU | The video produced on the board is copied back and has a readable frame |


Options (can be combined):

```bash
./run-artifact.sh data convert                  # run only these tasks (checks still run first)
./run-artifact.sh --no-tpu                      # skip Py-TPU
./run-artifact.sh --provider ollama             # use Ollama Cloud instead of OpenRouter
./run-artifact.sh --out artifact-runs/my-run    # choose the evidence directory
```

## Evidence

Each run writes its evidence to `artifact-runs/<timestamp>/`, updated live:

```text
artifact-runs/<timestamp>/
├── SUMMARY.txt                         the table above
├── 0-setup.log  1-self-test.log  2-api-check.log
└── <task>/
    ├── workflow.log                    full console output
    ├── logs/                           outcome (*.outcome.json), status, resource usage
    └── artifacts/<task>/.../<run>/     valid/ and invalid/ code, outputs/, manifest.json
```

Every generated candidate is kept, whether it passed validation or not. [`artifacts/README.md`](artifacts/README.md) explains this layout; a copy is included in each run as `HOW-TO-READ-ARTIFACTS.md`. `*.resources.txt` records elapsed time, CPU, and peak memory of the workflow in the container. It does not include resources used on the board or by the API provider.

## Py-TPU details

The container generates the script, copies it to the board over SSH, runs it there, and copies the output video back. It runs automatically when `eval-ssh/` exists; `./run-artifact.sh tpusketch` runs it alone. The SSH directory is mounted read-only into the container.

To use your own Coral Dev Board instead, prepare it as described in [`docs/CORAL_REMOTE_SETUP.md`](docs/CORAL_REMOTE_SETUP.md) and pass `--tpu /absolute/path/to/ssh-dir`. The runner may stop another process that holds the TPU, so use a dedicated board.

## Repository contents

| Path | Purpose |
| --- | --- |
| `run-artifact.sh` | Entry point for evaluators |
| `build-image.sh` | Builds the image locally instead of pulling it |
| `example.env` | Template for `.env` |
| `credentials.tar.gz.enc` | Encrypted temporary credentials (password via HotCRP) |
| `src/` | Framework source code and LLM prompts |
| `inputs/` | Fixed inputs for each task (CSV, models, labels, videos) |
| `tests/` | Offline unit tests (step 1) |
| `scripts/container/` | Helpers that run inside the container |
| `Dockerfile`, `requirements*.txt`, `vendor/` | The evaluation environment |


You can also build the image natively for your architecture, there is an known issue that the Arduino CLI has crashed under cross-architecture emulation.

To inspect things by hand inside the container:

```bash
docker run --rm -it -v "$PWD/.env:/artifact/.env:ro" noahwu/beaver-edge:middleware26 bash
```
If you built the image yourself with ./build-image.sh, use beaver-edge-artifact:middleware26 instead.

Inside the container, run `bash scripts/container/run-task.sh --provider openrouter data` for a single task, or `python -m pytest tests -q` for the offline tests.
