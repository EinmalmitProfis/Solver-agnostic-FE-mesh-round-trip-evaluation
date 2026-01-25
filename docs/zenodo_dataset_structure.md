# Zenodo dataset
The paper evaluates 23 models in total, but only a public subset can be redistributed.

```
reproducibility_package/
- `inputs_public/`  
  Public FE input decks used in the benchmark (13 of 23 models). (including the Benchmark inventory.txt)

- `paper_results/`  
  runtime and metric results from the paper (CSV + `run_metadata.json`).

- `json_ref/`, `json_roundtrip/`  
  Canonicalized JSON artefacts (Z88 pivot) used for solver-agnostic comparisons (public models only).
  Including the change reports made by the parser and writer modules.

- `paper_outputs/`  
  Converted solver decks and writer change reports (public models only).
  Including the change reports made by the parser and writer modules.

- `CHECKSUMS.sha256`  
  File checksums for integrity.
```
