# THEIA — APT Research and Raw-Event SLM

THEIA is a research project for analyzing raw security-event telemetry using
feature engineering, machine-learning baselines, a custom sequence model,
anomaly scoring, and evidence-grounded LLM-assisted investigation.

## Repository layout

- `THEIA/notebooks/` — original research notebooks (preserved)
- `THEIA/src/notebook_exports/` — Python exports generated from notebooks
- `THEIA/src/models/` — extracted model architecture source
- `THEIA/src/data/` — data pipeline module (to be extracted)
- `THEIA/src/training/` — training/checkpoint module (to be extracted)
- `THEIA/src/evaluation/` — validation and test code (to be extracted)
- `THEIA/src/anomaly_detection/` — anomaly scoring (to be extracted)
- `THEIA/src/inference/` — inference utilities (to be extracted)
- `THEIA/src/ollama/` — local Ollama integration notes
- `THEIA/results/` — small, shareable evaluation reports
- `THEIA/docs/` — methodology, limitations, and reproducibility notes

## Important model-source distinction

`THEIA/src/models/theia_slm.py` is extracted from cell 44 of
`THEIA_APT_Research_FULL_SLM_INTERACTIVE_12HFINAL.ipynb`. It defines the
`THEIASLM` RoPE/GQA/SwiGLU prototype.

It has **not** been verified as the exact class used by the final
34,510,699-parameter checkpoint. The notebook contains multiple model
experiments/configurations. Match the checkpoint's state-dict keys and
configuration before loading it. Do not overwrite or retrain the checkpoint.

## Dataset and evidence limitations

The raw event schema contains timestamps, sequence/thread identifiers,
event/operation codes, size, presence flags, and source indices. It does not
provide process names, file paths, IP addresses, malware names, threat-actor
identities, or ground-truth APT labels. Reports must not invent these fields
or claim confirmed APT detection.

## Ollama status

The notebooks demonstrate Python-to-Ollama API integration for Qwen-based
investigation. This does not mean the custom THEIA `.pt` checkpoint can be
loaded natively with `ollama run theia`.

## Setup

Create a Python 3.12 environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The requirements are a starting list, not a fully pinned reproducibility lock.
Verify package versions against the working environment before claiming
reproducible installation.

## Safety

Do not commit raw datasets, tokenized shards, model checkpoints, access
tokens, `.env` files, or local virtual environments. Large model artifacts
are maintained separately.
