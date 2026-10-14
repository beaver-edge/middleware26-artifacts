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
