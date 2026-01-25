[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18367107.svg)](https://doi.org/10.5281/zenodo.18367107)

# Solver-agnostic FE mesh round-trip evaluation (paper companion)

This repository contains the **evaluation implementation** accompanying the manuscript:

> **Cross-solver finite-element mesh conversion via a neutral hierarchical intermediate representation: Abaqus, ANSYS Mechanical, MSC NASTRAN, and Z88**  
> Peter R. Grohmann, Stephan Tremmel (University of Bayreuth)  

The purpose of this repo is **not** to provide a full deck converter.  
Instead, it provides an **auditable, solver-agnostic mesh-level evaluation pipeline** (evaluation_script.py) that:
1. canonicalizes FE models via a neutral pivot representation (Z88 → JSON), and
2. computes the paper’s fidelity metrics (geometry/connectivity/sets/refpoints + integrity diagnostics + runtimes).

---

## What this repository is useful for

### A) Paper transparency & reproducibility
- Inspect **exact metric definitions and configuration** used in the paper (constants in the `USER SETTINGS` block).
- Verify the reported evaluation artefacts (CSV + diagnostics) from the accompanying Zenodo dataset.
- Audit environment + configuration via the automatically written `run_metadata.json` snapshot.

### B) A reusable methodology for mesh-level FE model comparison
- A reference implementation of **tolerance-aware coordinate hashing** and **order-/ID-invariant element signatures**.
- Capability-aware evaluation of **NodeSets/ElementSets** (membership vs. set names) and **RefPoint candidates**.
- Practical integrity sentinels (hash collisions, invalid references, orientation/Jacobian checks).

---

## What this repository is *not*

- ❌ **No converter binary**: the deck-converter used in the paper (`Konverter.exe`) is **not** part of this public repo (IP/third‑party restrictions).
- ❌ **No redistribution of proprietary benchmark models**: only the public subset is published on Zenodo.
- ❌ **No solver-result equivalence claim**: the evaluation scope is **mesh-level structural data** (nodes/elements/sets/refpoint candidates and limited hierarchy metadata), not analysis semantics (materials, loads, BCs, contacts, etc.).

---

## Zenodo records (recommended starting point)

The repository alone is intentionally minimal; the **data & precomputed results** on Zenodo:

- **Software (this repository, versioned)**: 
- **Dataset (public input decks + evaluation artefacts / results)**: DOI `10.5281/zenodo.18367107`

