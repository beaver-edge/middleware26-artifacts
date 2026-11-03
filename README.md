# BEAVER-EDGE — Middleware 2026 artifact

Artifact for the paper **Autopilots Need Parachutes: Lessons Learned from LLM-Automated Embedded ML Pipelines**.

BEAVER-EDGE uses an LLM to generate, validate, and automatically repair code for five embedded-ML pipeline tasks. This package runs all five workflows in Docker with a single script and reports, for each one, the task result and, separately, the framework status. The paper claims this artifact supports are listed under [Major claims](#major-claims).

**Submitted to the Middleware 2026 artifact evaluation for the Artifacts Functional and Artifacts Available badges.** The package is publicly available and shows that the workflows install and run as the paper describes. It does not reproduce the paper's aggregate measurements, tables, or figures, which come from many repeated experiments.

## Abstract

This artifact accompanies *Autopilots Need Parachutes: Lessons Learned from LLM-Automated Embedded ML Pipelines*. It contains **BEAVER-EDGE**, the middleware "Autopilot" framework that the paper uses to study whether large language models (LLMs) can autonomously carry out the volatile, deployment-oriented stages of the embedded ML lifecycle. Given a task specification and a target-device profile, the framework builds stage-aware prompts, asks an LLM for an executable artifact, and runs that artifact against the real toolchain or runtime. When validation fails, it feeds the error and the faulty code back to the LLM and regenerates, up to a bounded repair budget. Every generated candidate, valid or invalid, is kept together with its logs and a manifest.

The artifact covers the five workflows evaluated in the paper: data processing (**DP**), model conversion with INT8 quantization (**MC**), Arduino sketch generation for a microcontroller (**ArdSG**), Python object-detection sketch generation for CPU execution (**Py-CPU**), and Edge TPU accelerated sketch generation on a Google Coral Dev Board (**Py-TPU**). Everything is packaged as a ready-to-use Docker image (which can also be built locally from the included Dockerfile) and driven by a single entry script. The script supports both an end-to-end run and individual task runs. For every task it reports two separate verdicts, backed by a per-run evidence directory: the task RESULT (PASS if a generated artifact was validated, FAIL if none was within the repair budget) and the FRAMEWORK status (OK if the framework ran the task to a recorded, verified outcome, ERROR otherwise). The four local workflows need only Docker and your own LLM API key, and Py-TPU additionally needs SSH access to a prepared Coral board. During the artifact evaluation we provided temporary access to both an API key and our Coral board; that access has ended. Two things differ from the paper's setup: Arduino sketches are compiled but not flashed (no Arduino board is needed), and Py-CPU runs in the container on the host CPU rather than on a Raspberry Pi 4.

The artifact is publicly available at https://github.com/beaver-edge/middleware26-artifacts (MIT license), and the prebuilt image is published on Docker Hub as noahwu/beaver-edge:middleware26. The paper's full source, datasets, and the execution traces and artifacts of the reported evaluation are at https://github.com/beaver-edge/middleware26; this repository packages the framework so that it can be run and checked. The goal is to show that the framework installs, runs, and behaves as described in the paper, including its validation-driven repair loop and its handling of runs that never converge. The Functional claim therefore rests on the FRAMEWORK status; a task FAIL is a legitimate result of the LLM under test, and the artifact reports it as FAIL. Because LLM outputs are non-deterministic, the aggregate success rates, token costs, and figures reported in the paper are outside the scope of this artifact and are not expected to be reproduced. This version calls the LLM API directly; the LangChain/Langfuse tracing layer described in §3.2 is not included. Instead, every LLM request is appended to `logs/llm_calls.jsonl` in the evidence directory: the full prompt, the raw response, finish reason, token usage, and latency (see "Evidence" below).

## Requirements
- **Software:** Docker Engine (Linux) or Docker Desktop (macOS/Windows), on `amd64` or `arm64`. All dependencies are in the image.
- **Hardware:** 4 CPUs, 8 GB RAM, and 20 GB free disk are suggested (estimates, not measured minimums). 
  - **Coral board (Py-TPU only):** a prepared Google Coral Dev Board that you can reach over SSH (see [Py-TPU details](#py-tpu-details)). Our board was available only during the artifact evaluation.
- **Network:** internet access for the image pull and the LLM API calls.
- **LLM access:** your own API key (see [Credentials](#credentials)). The runs use default `qwen/qwen3.8-27b` through OpenRouter. An Ollama Cloud configuration is also supported as a fallback or for trying other models (`--provider ollama`). Locally hosted LLMs work likewise with OpenAI-compatible APIs.
- **Time:** typically under an hour, up to about 3 hours depending on the LLM provider's load. The exact duration depends   on the host's speed and on the LLM provider's load.

## Credentials

Create `.env` from the template and fill in your LLM API key:

```bash
cp example.env .env      # set OPENROUTER_API_KEY, or OLLAMA_API_KEY for --provider ollama
```

An existing `.env` is used as is. Py-TPU also needs SSH access to a Coral board; see [Py-TPU details](#py-tpu-details).

During the Middleware 2026 artifact evaluation, the repository's `credentials.tar.gz.enc` held temporary access for the reviewers: a `.env` with an API key and board settings, and an `eval-ssh/` key that could only reach our Coral board. Its password was sent through HotCRP and the keys have been revoked, so the file no longer gives access. It stays in the repository as part of the evaluated version; `run-artifact.sh` unpacks it only when `.env` is missing and `BEAVER_EDGE_CREDENTIALS_PASSWORD` is set.

## Quick start

```bash
cp example.env .env      # add your LLM API key first (see Credentials)
./run-artifact.sh
```

This pulls the prebuilt image and runs all checks and the four local tasks (DP, MC, ArdSG, Py-CPU). Py-TPU runs only when you give the script access to a Coral board with `--tpu DIR` (see [Py-TPU details](#py-tpu-details)).

> **The image is published on Docker Hub**, but you can build it locally (slow; it downloads Conda, TensorFlow, and the Arduino toolchain):
>
> ```bash
> ./build-image.sh
> ./run-artifact.sh --local-image
> ```

## Expected output (Illustrative)

The script ends with a summary like this (from a real run of the published image; task results and timings vary between runs):

```text
=========================== BEAVER-EDGE evaluation summary ===========================
STEP / TASK          RESULT   FRAMEWORK  DETAILS
0 setup              PASS     -          image ready: noahwu/beaver-edge:middleware26@sha256:...
0 credentials        PASS     -          using the existing .env
1 self-test          PASS     -          offline unit tests: 109 passed, 2 skipped in 3.96s
2 api-check          PASS     -          openrouter API reachable and returns parseable JSON
DP (data)            FAIL     OK         no valid artifact: repair budget exhausted; attempts kept: 6 valid, 8 invalid, 15 LLM calls, 30188 tokens [1m42s]
MC (convert)         PASS     OK         artifact generated and verified; attempts kept: 1 valid, 2 invalid, 3 LLM calls, 7194 tokens [0m24s]
ArdSG (ardsketch)    PASS     OK         artifact generated and verified; attempts kept: 1 valid, 3 invalid, 5 LLM calls, 20609 tokens [2m31s]
Py-CPU (pysketch)    PASS     OK         artifact generated and verified; attempts kept: 1 valid, 2 invalid, 3 LLM calls, 14524 tokens [1m4s]
--------------------------------------------------------------------------------------
CHECKS:    PASS
TASKS:     3 PASS, 1 FAIL   (task results depend on the LLM and vary between runs)
FRAMEWORK: OK - every task that ran reached a recorded, verified outcome.
```

The summary is also saved to `artifact-runs/<timestamp>/SUMMARY.txt`. With a Coral board (`--tpu DIR`), it also has a `2b board-check` row and a Py-TPU row.

### How to read RESULT and FRAMEWORK

Each task gets two independent verdicts, because they judge different things:

- **RESULT** judges the *task*: the code the LLM generated inside the framework. **PASS** means a generated artifact passed validation and its outputs were verified. **FAIL** means no candidate passed validation within the bounded repair budget.
- **FRAMEWORK** judges *BEAVER-EDGE itself*. **OK** means the framework ran the task to a recorded outcome and the evidence is complete and consistent. **ERROR** means a crash, an uncaught exception, a missing dependency, an unreachable API or board, or missing, stale, or inconsistent evidence.

| What happened | RESULT | FRAMEWORK |
| --- | --- | --- |
| Generated code passed validation, and its outputs were verified | **PASS** | **OK** |
| Every attempt failed until the repair budget ran out; all generated candidates were kept | **FAIL** | **OK** |
| The framework itself broke (crash, missing dependency, unreachable service, bad evidence) | `-` (unknown) | **ERROR** |

A task FAIL with FRAMEWORK OK means the framework stopped cleanly within its repair budget and kept every generated candidate as evidence, one of the "parachutes" the paper argues for. It shows the framework working as designed, but the task still failed and is reported as FAIL. LLM output is non-deterministic, so a rerun can turn a task FAIL into a PASS or the reverse. The checks (steps 0-2b) test the environment and have a single PASS/FAIL verdict.

Exit status:

| Exit | Meaning |
| --- | --- |
| `0` | All checks passed, FRAMEWORK OK, and every task PASS |
| `3` | All checks passed and FRAMEWORK OK, but at least one task FAIL |
| `1` | A check failed, or FRAMEWORK ERROR on at least one task |
| `2` | Usage error |

If a check FAIL or a FRAMEWORK ERROR comes from the environment (for example a network error), rerun only the affected task, e.g. `./run-artifact.sh ardsketch`.

## What the script runs

| Step | What it checks |
| --- | --- |
| 0 setup | Docker is running and the image is available |
| 0 credentials | `.env` exists; for Py-TPU, board access is given (`--tpu DIR` or an `eval-ssh/` directory) |
| 1 self-test | Offline unit tests in the container (no API cost) |
| 2 api-check | One small request to the LLM provider; if it fails, the tasks are skipped |
| 2b board-check | Py-TPU only: SSH to the Coral board, Edge TPU visible, board inputs present; if it fails, only Py-TPU is skipped |
| Tasks | Each task runs in a fresh container; its outcome and evidence are then verified, and RESULT and FRAMEWORK are reported |

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
├── ENVIRONMENT.txt                     image id/digest/revision, checkout commit, Docker, host, model
├── 0-setup.log  1-self-test.log  2-api-check.log
└── <task>/
    ├── workflow.log                    full console output
    ├── logs/                           outcome (*.outcome.json), status, resource usage,
    │                                   llm_calls.jsonl (every LLM request and response)
    ├── work/                           what the container leaves behind: DP dataset state,
    │                                   model_quant.tflite, CPU output video (a hard link
    │                                   to the archived copy, so it takes no extra space)
    ├── work-links.txt                  which work/ files are such hard links (when any)
    └── artifacts/<task>/.../<run>/     valid/ and invalid/ code, outputs/, manifest.json
```

Every generated candidate is kept, whether it passed validation or not. [`artifacts/README.md`](artifacts/README.md) explains this layout; a copy is included in each run as `HOW-TO-READ-ARTIFACTS.md`. `*.resources.txt` records elapsed time, CPU, and peak memory of the workflow in the container. It does not include resources used on the board or by the API provider; the token counts, and the provider's cost when it returns one, are in `llm_calls.jsonl` and in each `*.outcome.json`.

## Major claims

### 1. A working middleware framework that automates the embedded ML pipeline end to end (paper §3, Figs. 2–4).
The paper claims an implemented framework that coordinates LLM-driven data processing, model optimization/conversion, and device-specific code generation across MCU- and MPU-class targets, with no human involvement after the task is specified.

*Artifact support:* `run-artifact.sh` runs DP, MC, ArdSG, and Py-CPU inside the container. It also runs Py-TPU on a Coral board that you give it access to with `--tpu <ssh-dir>` (during the artifact evaluation, it ran automatically on our remote board). Each task consumes the fixed inputs in `inputs/<task>/` and, when successful, yields a validated output: a transformed CSV (DP), a `model_quant.tflite` that the TFLite interpreter loads (MC), an Arduino sketch that compiles with `arduino-cli` for the Nano 33 BLE Sense (ArdSG; compiled only, as no Arduino board is provided), and an annotated output video with a readable frame (Py-CPU, run in the container on the host CPU instead of the Raspberry Pi 4 used in the paper; Py-TPU, run on a Coral Dev Board).

### 2. Stage-aware, hierarchical prompting (paper §3.3, Fig. 3).
The paper claims that prompts are split into modules (programming guidelines, application specification, sketch generation, error handling) and that device and dataset context is injected into them per stage.

*Artifact support:* The prompt templates are in `src/prompt_templates/`, one file per stage. `ard_sketch_prompts.py` implements the full hierarchy of Fig. 3 (programming guidelines, application specification, sketch generation, error handling); the other stages use the same pattern of stage-specific context, a generation prompt, and an error-handling prompt. The processors in `src/processors/` select the template for each stage and fill it with the device profile, dataset information, and, on repair attempts, the previous code and its error.

### 3. Validation-driven refinement is essential, and failures are first-class outcomes (paper §3.1, Fig. 4, §4.3 Table 3).
The paper claims that generated artifacts are compiled or executed, that errors go back to the LLM for repair within a bounded retry budget, and that the system is built to assume failure: it stops cleanly and records all evidence when convergence does not happen ("autopilots need parachutes").

*Artifact support:* In the evidence directory `artifact-runs/<timestamp>/<task>/artifacts/...`, every candidate is saved as `valid_attempt-*` or `invalid_attempt-*`, with a JSON sidecar recording its validation result and error, alongside an ordered `manifest.json`; the authoritative `*.outcome.json` is in `<task>/logs/`. A run that uses up its repair budget ends with `artifact_status=not_generated` and is reported as task RESULT FAIL with FRAMEWORK OK: the task failed, and the framework stopped cleanly and kept every generated candidate with its error. This bounded, evidence-preserving stop is one of the "parachutes" the paper argues for, together with stage-aware prompting and validation loops. Crashes, unreachable services, or inconsistent evidence are reported separately as FRAMEWORK ERROR. A run may show task FAILs on any stage; which stages fail varies between runs.

### 4. LLM-agnostic design (paper §3.2, §4.1).
The paper claims the core engine supports various LLM families (closed and open-weight) through a common compatible API.

*Artifact support:* The engine talks to an OpenAI-compatible Chat Completions endpoint. The provider and model are chosen through `.env` (OpenRouter or Ollama settings), so you can rerun any task with a different model without changing code. The base URL is also set in `.env` (`OPENROUTER_BASE_URL`, `OLLAMA_BASE_URL`), so another OpenAI-compatible vendor can be used by pointing the `OLLAMA_*` settings at it and passing `--provider ollama`.

### 5. Empirical findings on feasibility and dependability (paper §4).
The paper reports the following:

1. Sketch generation is the dominant point of failure on MCUs.<br>
2. Success improves on less constrained MPU targets.<br>
3. For most models, the standardized Edge TPU workflow converges more reliably than the more open-ended CPU workflow.<br>
4. Prompt layout alone can affect success rates.<br>
5. Relaxing prompt constraints lowers success.<br>
6. Execution success does not guarantee semantic fidelity.

*Artifact scope:* These results come from many independent runs (30 per task per LLM configuration) on specific hardware and model versions. The artifact's default model (`qwen/qwen3.8-27b` via OpenRouter) is not one of the models evaluated in Table 2. The Functional artifact does not reproduce these aggregate numbers, and exact values are not expected to match because LLM outputs are stochastic. What the artifact provides is the instrument that produced them: the exact pipeline, prompts, validators, and per-attempt logging. Individual runs show the qualitative behavior behind these findings, such as repairs that converge after a few attempts, runs that exhaust the repair budget, and validated code whose output may still be semantically wrong, but a single run is not evidence for the aggregate rates.

## Py-TPU details

The container generates the script, copies it to the board over SSH, runs it there, and copies the output video back. It runs automatically when an `eval-ssh/` directory exists next to `run-artifact.sh` (the layout of the sealed evaluation credentials) or when you pass `--tpu DIR`; `./run-artifact.sh --tpu DIR tpusketch` runs it alone. The SSH directory is mounted read-only into the container.

To use your own Coral Dev Board, prepare it as described in [`docs/CORAL_REMOTE_SETUP.md`](docs/CORAL_REMOTE_SETUP.md) and pass `--tpu /absolute/path/to/ssh-dir`. The runner may stop another process that holds the TPU, so use a dedicated board.

## Repository contents

| Path | Purpose |
| --- | --- |
| `run-artifact.sh` | Entry point: runs the checks, then the tasks |
| `build-image.sh` | Builds the image locally instead of pulling it |
| `example.env` | Template for `.env` |
| `src/` | Framework source code and LLM prompts |
| `inputs/` | Fixed inputs for each task (CSV, models, labels, videos) |
| `tests/` | Offline unit tests (step 1) |
| `scripts/container/` | Helpers that run inside the container |
| `Dockerfile`, `requirements*.txt`, `vendor/` | The container environment |


You can also build the image natively for your architecture, there is an known issue that the Arduino CLI has crashed under cross-architecture emulation.

To inspect things by hand inside the container:

```bash
docker run --rm -it -v "$PWD/.env:/artifact/.env:ro" noahwu/beaver-edge:middleware26 bash
```
If you built the image yourself with ./build-image.sh, use beaver-edge-artifact:middleware26 instead.

Inside the container, run `bash scripts/container/run-task.sh --provider openrouter data` for a single task, or `python -m pytest tests -q` for the offline tests.
