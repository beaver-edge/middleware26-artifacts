# Workflow inputs

This directory contains immutable, evaluator-facing inputs grouped by CLI task.
Tasks write only to `work/<task>/`; generated code and validated outputs are
retained separately under `artifacts/`.

| Directory | Contents |
| --- | --- |
| `data/` | Fruit CSV consumed by Data Processing. |
| `convert/` | Source Keras model and representative calibration array. |
| `ardsketch/` | Arduino model header and fruit sample used for prompt context. |
| `pysketch/` | CPU TFLite model, labels, and input video. |
| `tpusketch/` | Edge TPU model, labels, and the matching board input video. |
