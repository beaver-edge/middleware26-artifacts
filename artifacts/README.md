# Generated artifact evidence

This directory is the canonical, evaluator-facing result of running the five
framework tasks. Immutable source material lives under `inputs/<task>/`, while
disposable generated-code and output workspaces live under `work/<task>/`.
Neither location is the authoritative evidence tree.

Every run uses this layout:

```text
artifacts/<task>/<session>/run-<number>-<trace-id>/
├── valid/
│   ├── valid_attempt-001_<label>_<model>.py
│   └── valid_attempt-001_<label>_<model>.py.json
├── invalid/
│   ├── invalid_attempt-002_<label>_<model>.py
│   └── invalid_attempt-002_<label>_<model>.py.json
├── outputs/
│   ├── valid/valid_<task output copied for local inspection>
│   └── invalid/invalid_<rejected output, when one exists>
└── manifest.json
```

Arduino candidates use `.ino`. A run may have only `valid/` when its first
candidate passes, or only `invalid/` when all generated candidates fail. A DP
run can have both, because DP validates one operation at a time. The task
RESULT is FAIL whenever the `outcome` in `manifest.json` has
`"artifact_status": "not_generated"`. An attempt whose LLM response contains no
code block produces no candidate; its error appears only in `workflow.log`.

The distinction is deliberately redundant:

- `GENERATION_VALIDATED: status=VALID ...` is printed and logged, and the code
  is saved under `valid/` with a `valid_...` filename.
- `GENERATION_VALIDATION_FAILED: status=INVALID ...` is printed and logged, and
  the code is saved under `invalid/` with an `invalid_...` filename.
- `manifest.json` records the validation result, attempt label, executor,
  command/return code, error, and exact local path for every candidate.

For Py-TPU, both failed and successful remotely executed scripts are classified
in this local tree. A successful board-side output video is copied back into the
run's `outputs/valid/` directory. A fresh partial output left by a failed local
or TPU attempt is retained under `outputs/invalid/`. The framework clears the
configured remote output before each TPU validation, preventing an old video
from falsely validating new code. The complete evidence is retained on the
host under `artifact-runs/<timestamp>/<task>/artifacts/`, which
`./run-artifact.sh` bind-mounts so it updates live.

## LLM call log, work directory, and environment

Besides the per-attempt code, each task keeps three more things on the host:

- `<task>/logs/llm_calls.jsonl`: one JSON object per LLM request, in order.
  `call_index` is the line's number. It holds the provider, model, the full
  request (`messages`, sampling parameters), the raw response (`content` before
  any `<think>` stripping, `finish_reason`, reasoning fields when present), the
  provider's `usage` (token counts, and `cost` when the provider returns it),
  `latency_seconds`, and the `error` of a failed request. A failed request is
  recorded too, including a response that has only reasoning and no answer.
  `latency_seconds` covers the whole call, including the SDK's automatic retries.
- The `llm_call` field of every `valid_*/invalid_*` sidecar `.json` and of every
  entry in `manifest.json` is the `call_index` of the request that produced that
  candidate, so a candidate, its prompt, and its raw response can be joined.
  The outcome JSON adds `llm_calls`, `prompt_tokens`, `completion_tokens`,
  `total_tokens`, and `cost` (only when reported).
- `<task>/work/`: the container's working directory, kept after the run. When a
  DP run exhausts its repair budget, `work/data/` still holds the dataset as
  left by the last completed operation. Per-attempt scratch directories are
  removed by the framework as before, so compiled Arduino build output is not
  kept (the compiler report is in `workflow.log`).
  A `work/` file whose bytes are identical to an archived `outputs/valid/` file
  (the CPU output video) is a hard link to that archived copy: same path, same
  content, one copy on disk. `work-links.txt` lists them. Editing such a file
  in place changes the archived copy too, and copying the directory with a tool
  that does not preserve hard links (`cp -r`, `rsync` without `-H`) simply
  writes two ordinary copies.

`ENVIRONMENT.txt` in the run directory records the image id, repo digests and
`org.opencontainers.image.revision` label, the checkout commit, Docker and host
versions, the command line, and the model name. It never contains the API key.
