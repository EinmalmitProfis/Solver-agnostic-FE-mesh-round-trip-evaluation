# Paper results (archived)

This directory contains the artefacts exported by the deterministic evaluation pipeline.

## Files
- `converter_evaluation_paper_summary.csv`  
  Aggregated per-conversion summary table used for the paper.
- `paper_key_metrics.csv`  
  Compact table of key fidelity metrics.
- `paper_attention.csv`  
  Additional compact table highlighting target-dependent identifier/name effects.
- `timings_raw.csv`  
  Raw timing log (per run, per stage; includes warmup).
- `run_metadata.json`  
  Snapshot of effective configuration, environment/package versions, and SHA-256 fingerprints
  of the evaluation script and converter binary.

## Folders
- `paper_outputs/`  
  Converted outputs (public cases) and change-report artefacts (public + proprietary cases).
- `json_ref/`, `json_roundtrip/`  
  Canonicalised JSON artefacts used for comparisons (Z88 pivot).
- `explain/`  
  Debug artefacts emitted only when mismatches occur.
