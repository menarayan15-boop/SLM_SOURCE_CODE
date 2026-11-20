# Extraction status

## Completed
- Five original notebooks copied into `THEIA/notebooks/`.
- Five notebook-to-Python exports created in `THEIA/src/notebook_exports/`.
- `.gitignore` added at repository root.
- Python package folders and `__init__.py` files created.
- `THEIASLM` prototype architecture extracted from primary notebook cell 44.

## Not yet extracted into standalone modules
- Final checkpoint-compatible model class (must be identified from checkpoint
  state-dict keys/configuration before extraction).
- Raw-event preprocessing and feature conversion.
- Sequential training and checkpoint recovery.
- Full validation/test evaluation.
- Anomaly scoring and threshold evaluation.
- Ollama investigator and RAG orchestration.

## Known notebook export issue
`THEIA_CODE.ipynb` contains a saved error output with a non-standard
`errorDetails` field. `nbconvert` still wrote `THEIA_CODE.py`; preserve the
original notebook and treat the export as a best-effort conversion.

## Publication checklist
- Inspect notebook outputs for credentials and private paths.
- Keep large datasets, `.pt` shards, checkpoints, and secrets out of Git.
- Verify source code against the final checkpoint before documenting it as
  the production model architecture.
- Do not claim APT detection accuracy without ground-truth labels.
