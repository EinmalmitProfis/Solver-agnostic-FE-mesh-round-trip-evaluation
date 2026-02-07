"""
  "Cross-solver finite-element mesh conversion via a neutral hierarchical
  intermediate representation: Abaqus, ANSYS Mechanical, MSC NASTRAN, and Z88"
  Engineering Design and CAD, University of Bayreuth (preprint submitted to Elsevier).

Paper mapping:
- Evaluation methodology: Section 5 (Evaluation objectives and methodology).
- Pivot canonicalization operator C_Z88: Section 5.6.2, Eq. (2)–(6).
- Key evaluation settings: Appendix B, Table 6 (EPS_COORD, HASH_BITS, NODE_BASIS,
  REPEATS, WARMUP, RefNode↔mesh collision policy, multiscale eps factors).
- Fidelity metrics: Sections 5.2.2 and 5.6.4–5.6.8.

Pipeline overview (paper terminology):
- Canonical reference artifact:
    M_ref := C_Z88(f_src)
- For each target format τ:
    f_τ := W_τ(R(f_src))
    M_τ_rt := C_Z88(f_τ)
  where the Z88 pivot canonicalization operator is:
    C_Z88 := W_JSON ∘ R_Z88 ∘ W_Z88 ∘ R

Implemented here:
- `canonicalize_to_json()` implements C_Z88 by running:
    input deck -> Z88 pivot (z88i1.txt bundle) -> JSON (model.json)
- `run_all_tests()` iterates over all benchmark models and targets, computes the
  paper-defined fidelity metrics and runtime measurements, and writes CSV output.

Metrics implemented (paper Sections 5.6.4–5.6.8):
- Connected-node geometry fidelity (NODE_BASIS="connected"; Section 5.6.4).
- Element-incidence connectivity fidelity using order-invariant signatures σ(e)
  (Section 5.6.5). Additionally, an order-sensitive sentinel (`elem_f1_ordered`)
  is computed to monitor orientation regressions (Section 5.6.5).
- Set membership metrics (name-agnostic) and capability-aware set-name Name-F1
  (Section 5.6.6).
- Reference-point candidate retention (capability-aware; Section 5.6.8).
- Integrity diagnostics (Table 3): coordinate-key collision sentinel, invalid
  element-node references, and (tetra) Jacobian-sign monitoring.
- Robustness check at ε/2 and 2ε (multiscale factors 0.5, 1.0, 2.0; Section 5.6.7).

Reproducibility / audit trail (paper Section 5.2.2):
- The "USER SETTINGS" block intentionally keeps all experiment parameters as
  constants (no CLI) so the manuscript configuration is visible in one file.
- A `run_metadata.json` snapshot is written to EVAL_ROOT with:
    - effective configuration,
    - environment/package versions,
    - SHA-256 fingerprints of this script and the converter binary.
"""

# =============================================================================
# References / Attribution
# =============================================================================
#
# Algorithmic / specification attributions:
#
# - SplitMix64 (integer mixing used for tolerance-aware coordinate bin hashing):
#     The function `splitmix64()` follows the standard SplitMix64 64-bit mixing
#     function by Sebastiano Vigna (xoroshiro/splitmix family). It is used here
#     purely as a fast deterministic mixer for integer coordinate bins; it is NOT
#     cryptographically secure and is not used for security purposes.
#
# - BLAKE2b (fixed-width hashing of byte sequences for tokens/signatures):
#     This script uses BLAKE2b via Python's standard-library `hashlib.blake2b`.
#     BLAKE2 is specified in:
#       RFC 7693 — "The BLAKE2 Cryptographic Hash and Message Authentication Code (MAC)"
#       (J.-P. Aumasson, S. Neves, M. J. O. Saarinen), IETF.
#
# - SHA-256 (reproducibility fingerprints / audit trail):
#     Reproducibility fingerprints are computed using Python's standard-library
#     `hashlib.sha256`.
#
#
# Third-party libraries used (runtime dependencies):
#
# - Python standard library (hashlib, json, subprocess, etc.)
# - NumPy (token arrays, vectorized operations)
# - pandas (CSV exports)
# =============================================================================


from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import datetime
import platform
import shutil
import subprocess
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("numpy fehlt. Installiere mit: pip install numpy") from e

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas fehlt. Installiere mit: pip install pandas") from e


# =============================================================================
# manuscript defaults (Appendix B, Table 6)
# =============================================================================
# NOTE: Changing these values deviates from the paper's reported evaluation setup.
# Defaults match Appendix B / Table 6:
#   EPS_COORD = 0.005
#   HASH_BITS_CFG = 128
#   NODE_BASIS = "connected"
#   REPEATS = 5, WARMUP = 1 (median over r-1 after warm-up)
#   SET_REFNODE_COLLISION_POLICY = "collision"
#   MULTISCALE_EPS_FACTORS = [0.5, 1.0, 2.0]


def _default_project_root() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd().resolve()

PROJECT_ROOT = _default_project_root()

EXE_HINT = ""

TESTS_ROOT = PROJECT_ROOT / "Testcases"
EVAL_ROOT = PROJECT_ROOT / "Auswertung"

TARGETS = ["abaqus", "ansys", "nastran", "z88"]

MANIFEST_FILE = ""  # z.B. "manifest.json"

# Epsilon für Koordinaten-Quantisierung
EPS_COORD = 5e-3

# Hash-Bitbreite für Tokens/Signaturen (64 oder 128)
HASH_BITS_CFG = 128

# Timing: Wiederholungen / Warmup (Median der Messläufe)
REPEATS = 5
WARMUP = 1

# Node fidelity basis: "connected" (nur referenzierte Knoten) oder "all" (alle Mesh-Knoten)
NODE_BASIS = "connected"

# Output-Verzeichnisse vor jedem Lauf säubern?
CLEAN_OUTPUT_DIRS = True

# Ausgabe auf Konsole?
VERBOSE = True

# Multiscale-Ausgabe: "always" | "on-mismatch" | "off"
MULTISCALE_MODE = "always"
MULTISCALE_EPS_FACTORS = [0.5, 1.0, 2.0]

# Set-Name-Metrik: "auto" | "always" | "never"
SET_NAME_METRIC_MODE = "auto"

# Jacobian Sign Check: "tet" | "off"
JACOBIAN_CHECK = "tet"
JACOBIAN_VOL_TOL_REL = 1e-15

# Optional: entferne RefPoint-Artefakte aus Node-Fidelity (meist False)
FILTER_REFPOINTS_FROM_NODE_FIDELITY = False

# Explain/Debug
EXPLAIN_ON_MISMATCH = True
EXPLAIN_WRITE_DIAG_JSON = True
EXPLAIN_MAX_TOKENS = 50000
EXPLAIN_NODE_SAMPLES_PER_TOKEN = 1500
EXPLAIN_ELEM_SAMPLES_PER_SIG = 1500
EXPLAIN_NEIGHBOR_PROBE = True
NEIGHBOR_Q_RADIUS = 1
BOUNDARY_TOL_REL = 1e-3
MAX_NODE_PROBES_TOTAL = 500

EXPLAIN_WRITE_ELEMS_DIFF_AGNOSTIC = True
EXPLAIN_WRITE_ELEM_TYPES_DIFF = True

EXPLAIN_WRITE_SETS_DEBUG_CSV = True
EXPLAIN_SETS_DEBUG_MAX_SETS_PER_SIG = 8
EXPLAIN_SETS_DEBUG_SAMPLE_IDS = 12

# Dedup-Mode für Sets: "membership" | "exact" | "off"
SET_DEDUP_MODE = "membership"

# Set-Instance-Eval: "auto" | "always" | "never"
SET_INSTANCE_EVAL_MODE = "auto"

# RefNode<->Mesh-ID-Kollisionen in NodeSets: "collision" (empfohlen) | "mesh" | "ref"
# - "collision": nur bei IDs, die gleichzeitig Mesh-Node *und* RefNode sind, wird für Set-Metriken der RefNode-Token genutzt
# - "mesh": altes Verhalten (Mesh gewinnt immer) – kann NodeSet-Metriken bei RefNodes zerstören
# - "ref": RefNodes gewinnen immer, falls vorhanden (aggressiver)
SET_REFNODE_COLLISION_POLICY = "collision"

# Diagnose: elem_type_agnostic
DIAG_ELEM_TYPE_AGNOSTIC = True

# Diagnose: bbox-scale hint
DIAG_BBOX_SCALE_CHECK = True
BBOX_SCALE_WARN_TOL_REL = 0.05

# Dump kleine token-diffs auf stdout
DUMP_MISMATCH_EXAMPLES = True
MISMATCH_EXAMPLE_LIMIT = 250

# Toleranz für mismatch-Entscheidung (numerisch)
MISMATCH_TOL = 1e-10

# CSV: NaN explizit schreiben (statt leer)
CSV_NA_REP = "NaN"


# =============================================================================
# Konstanten / Format-Caps
# =============================================================================

FORMAT_CFG: Dict[str, Dict[str, str]] = {
    "abaqus": {"cli": "inp", "ext": "inp"},
    "ansys": {"cli": "dat", "ext": "dat"},
    "nastran": {"cli": "bdf", "ext": "bdf"},
    "z88": {"cli": "z88", "ext": "txt"},
}
SUPPORTED_TARGETS = sorted(FORMAT_CFG.keys())


# Capability matrix for capability-aware reporting (paper Table 1, Section 5.6.6, 5.6.8).
# - hierarchy: whether part/instance hierarchy is represented and evaluated (Abaqus, Z88).
# - set_names: whether human-readable set names are representable; Name-F1 is only reported
#   if both source and target support names in the evaluated subset (Section 5.6.6).
# - refpoints: this script reports RefPoint-candidate retention for all targets; for targets
#   without a native reference-point entity in the evaluated subset (ANSYS/NASTRAN), this
#   refers to writer-derived proxy tracking as described in Section 5.6.8 (not solver semantics).
FORMAT_CAPS: Dict[str, Dict[str, bool]] = {
    "abaqus": {"refpoints": True,  "hierarchy": True,  "set_names": True},
    "z88":    {"refpoints": True,  "hierarchy": True,  "set_names": True},
    "ansys":  {"refpoints": True, "hierarchy": False, "set_names": True},
    "nastran":{"refpoints": True, "hierarchy": False, "set_names": False},
}

MASK64 = (1 << 64) - 1
# =============================================================================
# Manifest (Testliste) – Pfade relativ zu TESTS_ROOT
# =============================================================================
TESTCASE_MANIFEST: List[Tuple[str, str]] = [
    ("#1/SUB40270rs28kN.inp", "abaqus"),
    ("#2/Makro_RLK_402_70_T2_21kN.inp", "abaqus"),
    ("#3/Modell-3.inp", "abaqus"),
    ("#4/Gehaeuse.inp", "abaqus"),
    ("#5/Craigbampton.inp", "abaqus"),
    ("#6/e4.inp", "abaqus"),
    ("#7/e6.inp", "abaqus"),
    ("#8/s2a.inp", "abaqus"),
    ("#9/FourBlocks.cdb", "ansys"),
    ("#10/Elastomerzahnkranz__Multimesh.dat", "ansys"),
    ("#11/academic_rotor.cdb", "ansys"),
    ("#12/HexBeam.cdb", "ansys"),
    ("#13/TetBeam.cdb", "ansys"),
    ("#14/hexlinz88i1.txt", "z88"),
    ("#15/z88i1.txt", "z88"),
    ("#16/z88i1.txt", "z88"),
    ("#17/z88i1.txt", "z88"),
    ("#18/z88i1.txt", "z88"),
    ("#19/WB-Housing.bdf", "nastran"),
    ("#20/out.bdf", "nastran"),
    ("#21/solid_bending.bdf", "nastran"),
    ("#22/nsc01a_n.dat", "nastran"),
    ("#23/cbush_test.bdf", "nastran"),
]

# =============================================================================
# Hashing: 64-bit vs 128-bit (default: 128)
# =============================================================================

HASH_BITS: int = 128
HASH_BYTES: int = 16
TOKEN_DTYPE: Any = np.dtype([("lo", "<u8"), ("hi", "<u8")])
TOKEN_VOID_DTYPE: Any = np.dtype(("V", 16))

_TYPE_CACHE: Dict[str, Any] = {}

def configure_hashing(bits: int) -> None:
    """Globales Hash-Setup (64/128-bit)."""
    global HASH_BITS, HASH_BYTES, TOKEN_DTYPE, TOKEN_VOID_DTYPE, _TYPE_CACHE
    bits = int(bits)
    if bits not in (64, 128):
        raise ValueError("HASH_BITS_CFG muss 64 oder 128 sein")
    HASH_BITS = bits
    HASH_BYTES = 8 if bits == 64 else 16
    if bits == 64:
        TOKEN_DTYPE = np.dtype("<u8")
        TOKEN_VOID_DTYPE = np.dtype(("V", 8))
    else:
        TOKEN_DTYPE = np.dtype([("lo", "<u8"), ("hi", "<u8")])
        TOKEN_VOID_DTYPE = np.dtype(("V", 16))
    _TYPE_CACHE = {}

def token_is_zero_scalar(tok: Any) -> bool:
    if HASH_BITS == 64:
        try:
            return int(tok) == 0
        except Exception:
            return False
    try:
        return int(tok["lo"]) == 0 and int(tok["hi"]) == 0
    except Exception:
        try:
            lo, hi = tok
            return int(lo) == 0 and int(hi) == 0
        except Exception:
            return False

def token_nonzero_mask(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros((0,), dtype=bool)
    if HASH_BITS == 64:
        return arr != 0
    return (arr["lo"] != 0) | (arr["hi"] != 0)


def digest_to_token(d: bytes) -> Any:
    """Digest -> Token (uint64 oder struct(lo,hi))."""
    if HASH_BITS == 64:
        v = int.from_bytes(d[:8], "little") & MASK64
        v = (v + 1) & MASK64
        return np.uint64(v)
    lo = int.from_bytes(d[:8], "little") & MASK64
    hi = int.from_bytes(d[8:16], "little") & MASK64
    if lo == 0 and hi == 0:
        lo = 1
    return np.array((lo, hi), dtype=TOKEN_DTYPE)[()]


def token_to_hex(tok: Any) -> str:
    if HASH_BITS == 64:
        try:
            return f"{int(tok) & MASK64:016x}"
        except Exception:
            return "0" * 16
    try:
        lo = int(tok["lo"]) & MASK64
        hi = int(tok["hi"]) & MASK64
        return f"{hi:016x}{lo:016x}"
    except Exception:
        try:
            lo, hi = tok
            lo = int(lo) & MASK64
            hi = int(hi) & MASK64
            return f"{hi:016x}{lo:016x}"
        except Exception:
            return "0" * 32

def token_xor(a: Any, b: Any) -> Any:
    if HASH_BITS == 64:
        return np.uint64((int(a) ^ int(b)) & MASK64)
    alo, ahi = int(a["lo"]) & MASK64, int(a["hi"]) & MASK64
    blo, bhi = int(b["lo"]) & MASK64, int(b["hi"]) & MASK64
    return np.array((alo ^ blo, ahi ^ bhi), dtype=TOKEN_DTYPE)[()]


# =============================================================================
# Token-Array Helpers (64/128) via VOID-View
# =============================================================================

def _as_void_1d(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.asarray(arr, dtype=TOKEN_VOID_DTYPE).reshape(-1)
    return np.asarray(arr).view(TOKEN_VOID_DTYPE).reshape(-1)


def token_array_sort(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    out = np.array(arr, copy=True)
    if HASH_BITS == 64:
        out.sort()
        return out
    v = _as_void_1d(out)
    v.sort()
    return v.view(TOKEN_DTYPE)


def token_unique(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.asarray(arr, dtype=TOKEN_DTYPE)
    if HASH_BITS == 64:
        return np.unique(arr)
    v = _as_void_1d(np.asarray(arr, dtype=TOKEN_DTYPE))
    vu = np.unique(v)
    return vu.view(TOKEN_DTYPE)


def token_unique_with_counts(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if arr.size == 0:
        return np.asarray([], dtype=TOKEN_DTYPE), np.asarray([], dtype=np.int64)
    if HASH_BITS == 64:
        return np.unique(arr, return_counts=True)
    v = _as_void_1d(np.asarray(arr, dtype=TOKEN_DTYPE))
    vu, c = np.unique(v, return_counts=True)
    return vu.view(TOKEN_DTYPE), c


def token_unique_with_index_counts(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if arr.size == 0:
        return (
            np.asarray([], dtype=TOKEN_DTYPE),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
        )
    if HASH_BITS == 64:
        vu, idx, c = np.unique(arr, return_index=True, return_counts=True)
        return vu, idx, c
    v = _as_void_1d(np.asarray(arr, dtype=TOKEN_DTYPE))
    vu, idx, c = np.unique(v, return_index=True, return_counts=True)
    return vu.view(TOKEN_DTYPE), idx, c


def token_intersect_unique(a_unique_sorted: np.ndarray, b_unique_sorted: np.ndarray) -> np.ndarray:
    if a_unique_sorted.size == 0 or b_unique_sorted.size == 0:
        return np.asarray([], dtype=TOKEN_DTYPE)
    if HASH_BITS == 64:
        return np.intersect1d(a_unique_sorted, b_unique_sorted, assume_unique=True)
    va = _as_void_1d(np.asarray(a_unique_sorted, dtype=TOKEN_DTYPE))
    vb = _as_void_1d(np.asarray(b_unique_sorted, dtype=TOKEN_DTYPE))
    vi = np.intersect1d(va, vb, assume_unique=True)
    return vi.view(TOKEN_DTYPE)


def token_searchsorted(a_unique_sorted: np.ndarray, values_sorted: np.ndarray) -> np.ndarray:
    if a_unique_sorted.size == 0 or values_sorted.size == 0:
        return np.asarray([], dtype=np.int64)
    if HASH_BITS == 64:
        return np.searchsorted(a_unique_sorted, values_sorted)
    va = _as_void_1d(np.asarray(a_unique_sorted, dtype=TOKEN_DTYPE))
    vv = _as_void_1d(np.asarray(values_sorted, dtype=TOKEN_DTYPE))
    return np.searchsorted(va, vv)


# =============================================================================
# Dataclasses: TestCase, SetEntry, ModelFeatures, Config
# =============================================================================

@dataclass(frozen=True)
class TestCase:
    name: str
    input_file: Path
    source_format: str
    tests_root: Path

    @property
    def rel_path(self) -> Path:
        try:
            return self.input_file.relative_to(self.tests_root)
        except ValueError:
            return self.input_file

    @property
    def group(self) -> str:
        parts = list(self.rel_path.parts)
        return parts[0] if parts else "ungrouped"

    @property
    def base_name(self) -> str:
        return self.input_file.stem


@dataclass
class SetEntry:
    name: str
    instance: str
    ids: List[int]


@dataclass
class ModelFeatures:
    model: Dict[str, Any]

    node_keys_mesh: np.ndarray
    node_keys_full: np.ndarray
    # Keys speziell für Set-Auswertung (Policy-gesteuert; Standard: collisions -> RefNode-Token)
    node_keys_sets: np.ndarray
    node_xyz_mesh: np.ndarray

    # ReferenceNodes (RefPoints/RPs): Token/XYZ nach ID (für collision-aware Set-Auflösung + Diagnose)
    refnode_keys_by_id: np.ndarray
    refnode_xyz_by_id: np.ndarray
    refnode_collision_mask: np.ndarray

    node_all: np.ndarray
    node_conn: np.ndarray

    elem_sigs: np.ndarray
    elem_sig_by_id: np.ndarray

    elem_sigs_ordered: np.ndarray
    elem_sig_by_id_ordered: np.ndarray

    rp: np.ndarray
    bbox: Optional[Dict[str, float]]

    parts: int
    instances: int
    refpoints_count: int

    nodesets_raw: List[SetEntry]
    elemsets_raw: List[SetEntry]

    nodesets_dedup_removed: int
    elemsets_dedup_removed: int

    coord_token_collisions: int
    coord_token_collision_examples: List[Dict[str, Any]]

    elem_total_node_refs: int
    elem_invalid_node_refs: int
    elem_zero_node_tokens: int

    refnode_id_collisions: int


@dataclass
class Config:
    project_root: Path
    exe: Path
    tests_root: Path
    eval_root: Path

    targets: List[str]

    manifest_file: Optional[Path] = None

    eps_coord: float = 5e-3
    hash_bits: int = 128

    repeats: int = 5
    warmup: int = 1
    node_basis: str = "connected"

    clean_output_dirs: bool = True
    verbose: bool = True

    json_ref_dir: Path = Path()
    json_roundtrip_dir: Path = Path()
    paper_outputs_dir: Path = Path()
    explain_dir: Path = Path()

    explain_on_mismatch: bool = True
    explain_max_tokens: int = 50
    explain_node_samples_per_token: int = 15
    explain_elem_samples_per_sig: int = 15
    explain_write_diag_json: bool = True

    mismatch_tol: float = 1e-10

    explain_neighbor_probe: bool = True
    neighbor_q_radius: int = 1
    boundary_tol_rel: float = 1e-3
    max_node_probes_total: int = 50

    explain_write_elems_diff_agnostic: bool = True
    explain_write_elem_types_diff: bool = True

    explain_write_sets_debug_csv: bool = True
    explain_sets_debug_max_sets_per_sig: int = 8
    explain_sets_debug_sample_ids: int = 12

    use_manifest: bool = True
    dump_mismatch_examples: bool = True
    mismatch_example_limit: int = 25

    set_dedup_mode: str = "membership"  # off|exact|membership
    set_instance_eval_mode: str = "auto"  # auto|always|never
    set_name_metric_mode: str = "auto"  # auto|always|never
    set_refnode_collision_policy: str = "collision"  # mesh|collision|ref|auto

    diag_elem_type_agnostic: bool = True
    multiscale_eps_factors: Optional[List[float]] = None
    multiscale_mode: str = "always"  # always|on-mismatch|off

    diag_bbox_scale_check: bool = True
    bbox_scale_warn_tol_rel: float = 0.05

    filter_refpoints_from_node_fidelity: bool = False

    jacobian_check: str = "tet"  # off|tet
    jacobian_vol_tol_rel: float = 1e-15

    def __post_init__(self):
        if self.multiscale_eps_factors is None:
            self.multiscale_eps_factors = [0.5, 1.0, 2.0]
        self.json_ref_dir = self.eval_root / "json_ref"
        self.json_roundtrip_dir = self.eval_root / "json_roundtrip"
        self.paper_outputs_dir = self.eval_root / "paper_outputs"
        self.explain_dir = self.eval_root / "explain"


# =============================================================================
# Utils
# =============================================================================
def append_timing_rows(
        timing_rows: List[Dict[str, Any]],
        *,
        testcase: str,
        testcase_id: str,
        source_format: str,
        target_format: str,
        phase: str,          # z.B. "ref" oder "roundtrip"
        stage: str,          # z.B. "src_to_z88", "z88_to_json", "src_to_target", ...
        run_dts: List[float],
        warmup: int,
        repeats: int,
) -> None:
    for i, dt in enumerate(run_dts):
        timing_rows.append({
            "testcase": testcase,
            "testcase_id": testcase_id,
            "source_format": source_format,
            "target_format": target_format,
            "phase": phase,
            "stage": stage,
            "run_idx": int(i),
            "is_warmup": bool(i < warmup),
            "used_for_stats": bool(i >= warmup),
            "dt_s": float(dt),
            "repeats": int(repeats),
            "warmup": int(warmup),
        })

def log(cfg: Config, *args: Any) -> None:
    if cfg.verbose:
        print(*args, flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_rmtree(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _is_finite(x: Any) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False


def bbox_diag(b: Optional[Dict[str, float]]) -> Optional[float]:
    if not b:
        return None
    dx = _safe_float(b.get("dx"))
    dy = _safe_float(b.get("dy"))
    dz = _safe_float(b.get("dz"))
    if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(dz)):
        return None
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


# =============================================================================
# Reproducibility metadata (paper / peer review)
# =============================================================================

def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    """
    Compute a SHA-256 fingerprint of a file.

    This is used purely for reproducibility metadata (it does not influence any metrics).
    Returns None if the file cannot be read.
    """
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _file_stat(path: Path) -> Dict[str, Any]:
    """Lightweight file info for metadata exports (never raises)."""
    try:
        st = path.stat()
        return {
            "exists": True,
            "size_bytes": int(st.st_size),
            "mtime_utc": datetime.datetime.utcfromtimestamp(st.st_mtime).replace(microsecond=0).isoformat() + "Z",
        }
    except Exception as exc:
        return {"exists": bool(path and Path(path).exists()), "error": str(exc)}


def _to_jsonable(obj: Any) -> Any:
    """
    Convert common types (Path, numpy scalars) into JSON-serialisable equivalents.
    """
    if isinstance(obj, Path):
        return str(obj)
    # numpy scalars / dtypes
    try:
        import numpy as _np  # local import to avoid circularity in edge cases
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def collect_run_metadata(cfg: "Config") -> Dict[str, Any]:
    """
    Collect a reproducibility snapshot: environment, config, and file fingerprints.

    The goal is to make paper runs independently auditable (peer review / archival).
    """
    now_utc = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    script_path = None
    try:
        script_path = Path(__file__).resolve()
    except Exception:
        script_path = None

    meta: Dict[str, Any] = {
        "created_utc": now_utc,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            "numpy": getattr(np, "__version__", None),
            "pandas": getattr(pd, "__version__", None),
        },
        "script": {
            "path": str(script_path) if script_path else None,
            "sha256": _sha256_file(script_path) if script_path else None,
        },
        "converter_exe": _to_jsonable({
            "path": cfg.exe,
            **_file_stat(cfg.exe),
            "sha256": _sha256_file(cfg.exe) if getattr(cfg, "exe", None) else None,
        }),
        "config": _to_jsonable(getattr(cfg, "__dict__", {})),
        # Keep the per-format definitions in the record so reviewers can verify capability gating.
        "format_cfg": _to_jsonable(FORMAT_CFG),
        "format_caps": _to_jsonable(FORMAT_CAPS),
    }
    return meta


def write_run_metadata(cfg: "Config") -> Optional[Path]:
    """
    Write run_metadata.json to EVAL_ROOT.

    The evaluation must never fail just because metadata cannot be written.
    """
    try:
        ensure_dir(cfg.eval_root)
        out = cfg.eval_root / "run_metadata.json"
        meta = collect_run_metadata(cfg)
        with out.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return out
    except Exception:
        return None


# =============================================================================
# Converter-Exe Detection + Preflight
# =============================================================================

def _detect_converter_exe(project_root: Path, exe_hint: Optional[str]) -> Path:
    if exe_hint:
        return Path(exe_hint).expanduser().resolve()

    candidates = [
        project_root / "Konverter.exe",

        ]
    for c in candidates:
        if c.is_file():
            return c
    # fallback: erstes Candidate-Target zurückgeben, damit Fehler aussagekräftig ist
    return candidates[0]


def preflight(cfg: Config) -> None:
    log(cfg, "PROJECT_ROOT:", cfg.project_root)
    log(cfg, "Konverter:", cfg.exe, "exists:", cfg.exe.is_file())
    log(cfg, "TESTS_ROOT:", cfg.tests_root, "exists:", cfg.tests_root.is_dir())
    log(cfg, "EVAL_ROOT:", cfg.eval_root)
    log(cfg, "  json_ref:", cfg.json_ref_dir)
    log(cfg, "  paper_outputs:", cfg.paper_outputs_dir)
    log(cfg, "  json_roundtrip:", cfg.json_roundtrip_dir)
    log(cfg, "  explain:", cfg.explain_dir)
    log(cfg, "TARGETS:", cfg.targets)
    log(cfg, "EPS_COORD:", cfg.eps_coord, "NODE_BASIS:", cfg.node_basis, "REPEATS:", cfg.repeats, "WARMUP:", cfg.warmup)
    log(cfg, "HASH_BITS:", cfg.hash_bits)
    log(cfg, "MULTISCALE:", cfg.multiscale_mode, "factors:", cfg.multiscale_eps_factors)
    log(cfg, "SET_NAME_METRIC:", cfg.set_name_metric_mode)
    log(cfg, "SET_REFNODE_COLLISION_POLICY:", cfg.set_refnode_collision_policy)
    log(cfg, "JACOBIAN_CHECK:", cfg.jacobian_check, "jacobian_vol_tol_rel:", cfg.jacobian_vol_tol_rel)
    if cfg.manifest_file:
        log(cfg, "MANIFEST_FILE:", cfg.manifest_file, "exists:", cfg.manifest_file.is_file())

    if not cfg.exe.is_file():
        raise RuntimeError(f"Konverter nicht gefunden: {cfg.exe}\nSetze EXE_HINT (Pfad zur Binary) oder baue den Konverter.")
    if not cfg.tests_root.is_dir():
        raise RuntimeError(f"Testcases-Ordner nicht gefunden: {cfg.tests_root}\nPasse TESTS_ROOT an.")


def _load_manifest_file(path: Path) -> List[Tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: List[Tuple[str, str]] = []
    if isinstance(raw, list):
        for it in raw:
            if isinstance(it, list) and len(it) >= 2:
                out.append((str(it[0]), str(it[1])))
            elif isinstance(it, dict):
                p = it.get("path") or it.get("file") or it.get("rel")
                f = it.get("format") or it.get("source_format")
                if p and f:
                    out.append((str(p), str(f)))
    return out


def discover_test_cases(cfg: Config) -> List[TestCase]:
    if not cfg.use_manifest:
        raise RuntimeError("Auto-Discovery ist deaktiviert. Bitte Manifest nutzen.")
    manifest = TESTCASE_MANIFEST
    if cfg.manifest_file and cfg.manifest_file.is_file():
        manifest = _load_manifest_file(cfg.manifest_file)

    cases: List[TestCase] = []
    for rel, fmt in manifest:
        p = cfg.tests_root / rel
        if not p.is_file():
            log(cfg, f"[WARN] fehlt oder keine Datei: {p}")
            continue
        fmt = str(fmt).lower().strip()
        cases.append(TestCase(name=str(rel).replace("\\", "/"), input_file=p, source_format=fmt, tests_root=cfg.tests_root))
    cases.sort(key=lambda tc: tc.name.lower())
    return cases


# Paper mapping: runtime protocol (paper Section 5.6.3):
# - Each stage is repeated r = REPEATS times.
# - First run is warm-up (WARMUP = 1), median is computed over remaining runs.
# - We also log raw per-run timings to timings_raw.csv for reproducibility.
def run_conversion_timed_with_runs(
        cfg: Config,
        input_file: Path,
        output_file: Path,
        output_format: str
) -> Tuple[float, List[float], str, str]:
    repeats = cfg.repeats
    warmup = cfg.warmup

    if repeats <= 0:
        raise ValueError("repeats muss >= 1 sein")
    if warmup < 0:
        raise ValueError("warmup muss >= 0 sein")
    if warmup >= repeats:
        raise ValueError("warmup muss < repeats sein")

    all_runs: List[float] = []
    last_out = ""
    last_err = ""

    for _ in range(repeats):
        dt, out, err = _run_conversion_once(cfg, input_file, output_file, output_format)
        all_runs.append(float(dt))
        last_out, last_err = out, err

    measured = all_runs[warmup:]  # nur diese gehen in die Statistik
    median_dt = float(np.median(np.asarray(measured, dtype=np.float64))) if measured else float("nan")
    return median_dt, all_runs, last_out, last_err

def _run_conversion_once(cfg: Config, input_file: Path, output_file: Path, output_format: str) -> Tuple[float, str, str]:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Robustheit: alte Outputs entfernen, damit "stale file" nicht fälschlich als Erfolg zählt
    try:
        if output_file.exists() and output_file.is_file():
            output_file.unlink()
    except Exception:
        pass

    cmd = [str(cfg.exe), "-i", str(input_file), "-o", str(output_file), "-f", output_format]

    start = time.perf_counter()
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(cfg.project_root),
    )
    dt = time.perf_counter() - start

    if cfg.verbose:
        log(cfg, "CMD:", " ".join(cmd))
        log(cfg, "RET:", res.returncode)
        if res.stdout.strip():
            log(cfg, "STDOUT:", res.stdout)
        if res.stderr.strip():
            log(cfg, "STDERR:", res.stderr)

    if res.returncode != 0:
        raise RuntimeError(
            "Konvertierung fehlgeschlagen\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{res.stdout}\n"
            f"STDERR:\n{res.stderr}"
        )

    if output_format in {"json", "inp", "dat", "bdf", "nas", "z88"} and not output_file.is_file():
        raise FileNotFoundError(f"Ausgabedatei nicht erzeugt: {output_file}")

    return dt, res.stdout, res.stderr


def run_conversion_timed(cfg: Config, input_file: Path, output_file: Path, output_format: str) -> Tuple[float, str, str]:
    median_dt, _runs, last_out, last_err = run_conversion_timed_with_runs(cfg, input_file, output_file, output_format)
    return median_dt, last_out, last_err


# Paper mapping: implements the Z88 pivot canonicalization operator C_Z88
# (paper Section 5.6.2, Eq. (2)–(4)):
#   C_Z88 := W_JSON ∘ R_Z88 ∘ W_Z88 ∘ R
# Here realised by:
#   (1) input -> Z88 pivot bundle (z88i1.txt)   [W_Z88 ∘ R]
#   (2) Z88 pivot -> JSON (model.json)          [W_JSON ∘ R_Z88]

def canonicalize_to_json(
        cfg: Config,
        input_file: Path,
        out_dir: Path,
        *,
        timing_rows: Optional[List[Dict[str, Any]]] = None,
        timing_meta: Optional[Dict[str, str]] = None,
        stage_prefix: str = "",
) -> Tuple[Path, float, float]:
    if cfg.clean_output_dirs:
        safe_rmtree(out_dir)
    ensure_dir(out_dir)

    z88_i1 = out_dir / "z88i1.txt"
    json_out = out_dir / "model.json"

    t1, runs1, _, _ = run_conversion_timed_with_runs(cfg, input_file, z88_i1, "z88")
    if timing_rows is not None and timing_meta is not None:
        append_timing_rows(
            timing_rows,
            testcase=timing_meta["testcase"],
            testcase_id=timing_meta["testcase_id"],
            source_format=timing_meta["source_format"],
            target_format=timing_meta["target_format"],
            phase=timing_meta["phase"],
            stage=f"{stage_prefix}src_to_z88",
            run_dts=runs1,
            warmup=cfg.warmup,
            repeats=cfg.repeats,
        )

    t2, runs2, _, _ = run_conversion_timed_with_runs(cfg, z88_i1, json_out, "json")
    if timing_rows is not None and timing_meta is not None:
        append_timing_rows(
            timing_rows,
            testcase=timing_meta["testcase"],
            testcase_id=timing_meta["testcase_id"],
            source_format=timing_meta["source_format"],
            target_format=timing_meta["target_format"],
            phase=timing_meta["phase"],
            stage=f"{stage_prefix}z88_to_json",
            run_dts=runs2,
            warmup=cfg.warmup,
            repeats=cfg.repeats,
        )

    return json_out, t1, t2


# =============================================================================
# JSON Laden + Schema Helper
# =============================================================================

def load_json_model(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON fehlt: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "Model" in data and isinstance(data["Model"], dict):
        data = data["Model"]

    if not isinstance(data, dict):
        raise ValueError(f"Unerwartete JSON-Wurzel: {type(data).__name__}")

    data = dict(data)
    data.pop("comment", None)
    return data


def iter_parts(model: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    parts_root = model.get("Parts")
    if not isinstance(parts_root, list):
        return
    for item in parts_root:
        if not isinstance(item, dict):
            continue
        if "Part" in item and isinstance(item["Part"], list):
            for p in item["Part"]:
                if isinstance(p, dict):
                    yield p
        else:
            yield item


def iter_mesh_nodes(model: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    nodes = model.get("Nodes")
    if isinstance(nodes, list):
        for n in nodes:
            if isinstance(n, dict):
                yield n

    for p in iter_parts(model):
        mesh = p.get("Mesh")
        if isinstance(mesh, dict):
            pn = mesh.get("Nodes")
            if isinstance(pn, list):
                for n in pn:
                    if isinstance(n, dict):
                        yield n


def iter_mesh_elements(model: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    elems = model.get("Elements")
    if isinstance(elems, list):
        for e in elems:
            if isinstance(e, dict):
                yield e

    for p in iter_parts(model):
        mesh = p.get("Mesh")
        if isinstance(mesh, dict):
            pe = mesh.get("Elements")
            if isinstance(pe, list):
                for e in pe:
                    if isinstance(e, dict):
                        yield e


def iter_reference_nodes(model: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    asm = model.get("Assembly")
    if not isinstance(asm, dict):
        return
    rns = asm.get("ReferenceNodes")
    if not isinstance(rns, list):
        return
    for n in rns:
        if isinstance(n, dict):
            yield n


def count_parts_instances_refpoints(model: Dict[str, Any]) -> Tuple[int, int, int]:
    parts = sum(1 for _ in iter_parts(model))
    asm = model.get("Assembly")
    instances = 0
    refpoints = 0
    if isinstance(asm, dict):
        pis = asm.get("PartInstances")
        if isinstance(pis, list):
            instances = sum(1 for x in pis if isinstance(x, dict))
        rns = asm.get("ReferenceNodes")
        if isinstance(rns, list):
            refpoints = sum(1 for x in rns if isinstance(x, dict) and ("id" in x or "Id" in x))
    return parts, instances, refpoints


def bbox_mesh_nodes(model: Dict[str, Any]) -> Optional[Dict[str, float]]:
    xs, ys, zs = [], [], []
    for n in iter_mesh_nodes(model):
        if not all(k in n for k in ("x", "y", "z")):
            continue
        try:
            x = float(n.get("x"))
            y = float(n.get("y"))
            z = float(n.get("z"))
        except Exception:
            continue
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        xs.append(x); ys.append(y); zs.append(z)
    if not xs:
        return None
    xmin, xmax = float(min(xs)), float(max(xs))
    ymin, ymax = float(min(ys)), float(max(ys))
    zmin, zmax = float(min(zs)), float(max(zs))
    return {
        "xmin": xmin, "xmax": xmax,
        "ymin": ymin, "ymax": ymax,
        "zmin": zmin, "zmax": zmax,
        "dx": xmax - xmin,
        "dy": ymax - ymin,
        "dz": zmax - zmin,
    }


# =============================================================================
# Sets: Rekursive Extraktion + Dedupe
# =============================================================================

def extract_all_sets(model: Dict[str, Any]) -> Tuple[List[SetEntry], List[SetEntry]]:
    nodesets: List[SetEntry] = []
    elemsets: List[SetEntry] = []

    def add_nodesets(container: Any) -> None:
        if not isinstance(container, list):
            return
        for s in container:
            if not isinstance(s, dict):
                continue
            name = s.get("Name")
            ids = s.get("NodeIds")
            if not isinstance(name, str) or not isinstance(ids, list):
                continue
            inst = s.get("InstanceName")
            inst_name = inst if isinstance(inst, str) else ""
            try:
                nodesets.append(SetEntry(name=name, instance=inst_name, ids=[int(x) for x in ids]))
            except Exception:
                continue

    def add_elemsets(container: Any) -> None:
        if not isinstance(container, list):
            return
        for s in container:
            if not isinstance(s, dict):
                continue
            name = s.get("Name")
            ids = s.get("ElementIds")
            if not isinstance(name, str) or not isinstance(ids, list):
                continue
            inst = s.get("InstanceName")
            inst_name = inst if isinstance(inst, str) else ""
            try:
                elemsets.append(SetEntry(name=name, instance=inst_name, ids=[int(x) for x in ids]))
            except Exception:
                continue

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if "NodeSets" in obj:
                add_nodesets(obj.get("NodeSets"))
            if "ElementSets" in obj:
                add_elemsets(obj.get("ElementSets"))
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    walk(model)
    return nodesets, elemsets


def dedupe_sets(sets: List[SetEntry], mode: str, ignore_instance: bool = False) -> Tuple[List[SetEntry], int]:
    mode = str(mode).lower().strip()
    if mode == "off":
        return sets, 0

    out: List[SetEntry] = []
    seen = set()
    removed = 0

    for s in sets:
        inst = "" if ignore_instance else s.instance
        if mode == "exact":
            key = (inst, s.name, tuple(s.ids))
        else:
            key = (inst, s.name, tuple(sorted(set(s.ids))))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(s)

    return out, removed


# =============================================================================
# Hashing / Signatures used for paper metrics (Section 5.6.5)
# =============================================================================
# Paper mapping:
# - Node keys k(n): tolerance-aware coordinate quantization + fixed-width hashing (Eq. (7)).
# - Element signatures σ(e): type token + multiset of incident node keys (Section 5.6.5).
#
# Provenance note:
# - splitmix64() below follows the standard SplitMix64 mixing function from the
#   xoroshiro/splitmix family (Sebastiano Vigna). Included for fast deterministic mixing
#   of integer coordinate bins; not cryptographically secure.
# - BLAKE2b hashing is via Python's standard library hashlib.blake2b (fixed digest_size).
# =============================================================================

def splitmix64(x: int) -> int:
    """SplitMix64 mixing function (standard algorithm; attribution: S. Vigna).

    Used here as a fast deterministic 64-bit mixer for integer coordinate bins.
    Not intended as a cryptographic hash.
    """

    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & MASK64
    z = z ^ (z >> 31)
    return z & MASK64


def _round_half_up_to_int(v: float) -> int:
    if v >= 0.0:
        return int(math.floor(v + 0.5))
    return int(math.ceil(v - 0.5))


def coord_token(x: float, y: float, z: float, eps: float) -> Any:
    """Compute a tolerance-aware node key k(n) (paper Section 5.6.5, Eq. (7)).

    Steps:
    1) Quantize coordinates using eps_coord (ε_coord) with round-half-up.
    2) Mix/hash the integer triplet into a fixed-width token (64- or 128-bit).
    """
    qx = _round_half_up_to_int(x / eps)
    qy = _round_half_up_to_int(y / eps)
    qz = _round_half_up_to_int(z / eps)

    if HASH_BITS == 64:
        h = splitmix64(qx & MASK64)
        h = splitmix64(h ^ (qy & MASK64))
        h = splitmix64(h ^ (qz & MASK64))
        return np.uint64((h + 1) & MASK64)

    lo = splitmix64(qx & MASK64)
    lo = splitmix64(lo ^ (qy & MASK64))
    lo = splitmix64(lo ^ (qz & MASK64))

    hi = splitmix64((qx ^ 0xA5A5A5A5A5A5A5A5) & MASK64)
    hi = splitmix64(hi ^ ((qy ^ 0x3C3C3C3C3C3C3C3C) & MASK64))
    hi = splitmix64(hi ^ ((qz ^ 0x5A5A5A5A5A5A5A5A) & MASK64))

    lo = splitmix64(lo ^ hi)
    hi = splitmix64(hi ^ lo)

    if lo == 0 and hi == 0:
        lo = 1
    return np.array((lo, hi), dtype=TOKEN_DTYPE)[()]


def type_token(type_str: str) -> Any:
    t = str(type_str)
    if t in _TYPE_CACHE:
        return _TYPE_CACHE[t]
    d = hashlib.blake2b(t.encode("utf-8"), digest_size=HASH_BYTES).digest()
    tok = digest_to_token(d)
    _TYPE_CACHE[t] = tok
    return tok


def token_mix_init(include_type: bool, elem_type: str) -> Any:
    if include_type:
        return type_token(elem_type)
    d = hashlib.blake2b(b"TYPE_AGNOSTIC", digest_size=HASH_BYTES).digest()
    return digest_to_token(d)


def token_mix_step(h: Any, x: Any) -> Any:
    if HASH_BITS == 64:
        hh = splitmix64(int(h) & MASK64)
        hh = splitmix64(hh ^ (int(x) & MASK64))
        hh &= MASK64
        if hh == 0:
            hh = 1
        return np.uint64(hh)

    hlo, hhi = int(h["lo"]) & MASK64, int(h["hi"]) & MASK64
    xlo, xhi = int(x["lo"]) & MASK64, int(x["hi"]) & MASK64

    hlo = splitmix64(hlo)
    hlo = splitmix64(hlo ^ xlo)
    hlo = splitmix64(hlo ^ xhi)

    hhi = splitmix64(hhi)
    hhi = splitmix64(hhi ^ xhi)
    hhi = splitmix64(hhi ^ xlo)

    hlo = splitmix64(hlo ^ hhi)
    hhi = splitmix64(hhi ^ hlo)

    if hlo == 0 and hhi == 0:
        hlo = 1
    return np.array((hlo, hhi), dtype=TOKEN_DTYPE)[()]


# =============================================================================
# Node Keys + Coordinates + Collision Sentinel
# =============================================================================

def quantize_triplet(x: float, y: float, z: float, eps: float) -> Tuple[int, int, int]:
    return (
        _round_half_up_to_int(x / eps),
        _round_half_up_to_int(y / eps),
        _round_half_up_to_int(z / eps),
    )


def detect_coord_token_collisions(model: Dict[str, Any], eps: float, max_examples: int = 5) -> Tuple[int, List[Dict[str, Any]]]:
    seen: Dict[str, Tuple[int, int, int]] = {}
    collisions = 0
    examples: List[Dict[str, Any]] = []
    for n in iter_mesh_nodes(model):
        if not all(k in n for k in ("x", "y", "z")):
            continue
        try:
            x = float(n["x"]); y = float(n["y"]); z = float(n["z"])
        except Exception:
            continue
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        q = quantize_triplet(x, y, z, eps)
        tok = coord_token(x, y, z, eps)
        if token_is_zero_scalar(tok):
            continue
        key = token_to_hex(tok)
        prev = seen.get(key)
        if prev is None:
            seen[key] = q
        else:
            if prev != q:
                collisions += 1
                if len(examples) < max_examples:
                    examples.append({"token_hex": key, "q_prev": prev, "q_new": q})
    return int(collisions), examples


def build_node_key_maps_and_xyz(model: Dict[str, Any], eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    mesh_nodes = list(iter_mesh_nodes(model))
    ref_nodes = list(iter_reference_nodes(model))

    max_id = 0
    for n in mesh_nodes:
        try:
            nid = int(n.get("id", 0))
            max_id = max(max_id, nid)
        except Exception:
            continue
    for n in ref_nodes:
        try:
            nid = int(n.get("id", 0))
            max_id = max(max_id, nid)
        except Exception:
            continue

    node_keys_mesh = np.zeros((max_id,), dtype=TOKEN_DTYPE)
    node_keys_full = np.zeros((max_id,), dtype=TOKEN_DTYPE)
    node_xyz_mesh = np.full((max_id, 3), np.nan, dtype=np.float64)

    # RefNodes separat mappen (auch wenn IDs kollidieren)
    refnode_keys_by_id = np.zeros((max_id,), dtype=TOKEN_DTYPE)
    refnode_xyz_by_id = np.full((max_id, 3), np.nan, dtype=np.float64)

    refnode_id_collisions = 0

    def add_mesh(nodes: List[Dict[str, Any]]) -> None:
        for d in nodes:
            if not all(k in d for k in ("id", "x", "y", "z")):
                continue
            try:
                nid = int(d["id"])
                if nid <= 0 or nid > max_id:
                    continue
                x, y, z = float(d["x"]), float(d["y"]), float(d["z"])
                tok = coord_token(x, y, z, eps)
                node_keys_mesh[nid - 1] = tok
                node_keys_full[nid - 1] = tok
                node_xyz_mesh[nid - 1, :] = (x, y, z)
            except Exception:
                continue

    def add_ref(nodes: List[Dict[str, Any]]) -> None:
        nonlocal refnode_id_collisions
        for d in nodes:
            if not all(k in d for k in ("id", "x", "y", "z")):
                continue
            try:
                nid = int(d["id"])
                if nid <= 0 or nid > max_id:
                    continue
                x, y, z = float(d["x"]), float(d["y"]), float(d["z"])
                tok = coord_token(x, y, z, eps)

                refnode_keys_by_id[nid - 1] = tok
                refnode_xyz_by_id[nid - 1, :] = (x, y, z)

                # node_keys_full bleibt Mesh-first (Geometrie-Fidelity unverändert)
                if token_is_zero_scalar(node_keys_mesh[nid - 1]):
                    # Nur belegen, wenn dort noch nichts steht
                    if token_is_zero_scalar(node_keys_full[nid - 1]):
                        node_keys_full[nid - 1] = tok
                else:
                    # Mesh-ID belegt -> NICHT überschreiben, aber zählen
                    refnode_id_collisions += 1
            except Exception:
                continue

    add_mesh(mesh_nodes)
    add_ref(ref_nodes)
    return (
        node_keys_mesh,
        node_keys_full,
        node_xyz_mesh,
        refnode_keys_by_id,
        refnode_xyz_by_id,
        int(refnode_id_collisions),
    )


def node_multiset_array(node_keys: np.ndarray) -> np.ndarray:
    arr = np.asarray(node_keys, dtype=TOKEN_DTYPE)
    m = token_nonzero_mask(arr)
    return arr[m]

# Paper mapping: connected-node basis (paper Section 5.6.4).
# Node fidelity is evaluated only on nodes referenced by at least one element
# to avoid route-dependent treatment of auxiliary/unconnected nodes.
def connected_node_mask(model: Dict[str, Any], node_count: int) -> np.ndarray:
    used = np.zeros(node_count, dtype=bool)
    for e in iter_mesh_elements(model):
        nids = e.get("node_ids")
        if not isinstance(nids, list):
            continue
        for nid in nids:
            try:
                i = int(nid) - 1
                if 0 <= i < used.size:
                    used[i] = True
            except Exception:
                continue
    return used


def connected_node_multiset(model: Dict[str, Any], node_keys_mesh: np.ndarray) -> np.ndarray:
    used = connected_node_mask(model, int(node_keys_mesh.size))
    arr = np.asarray(node_keys_mesh, dtype=TOKEN_DTYPE)
    arr = arr[used]
    m = token_nonzero_mask(arr)
    return arr[m]


def build_element_signatures(
        model: Dict[str, Any],
        node_keys_mesh: np.ndarray,
        order_invariant: bool,
        include_type: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    elems = list(iter_mesh_elements(model))
    """Build element signatures σ(e) for connectivity fidelity (paper Section 5.6.5).

    Order-invariant signature (order_invariant=True):
        σ(e) := h_128( t(e), sort([k(n1),...,k(nm)]) )
    This is invariant to global renumbering and local node ordering.

    Ordered sentinel (order_invariant=False):
        uses the same mixing but without sorting to monitor orientation regressions
        (paper: order-sensitive sentinel / Jacobian diagnostics).
    """


    max_eid = 0
    for e in elems:
        try:
            max_eid = max(max_eid, int(e.get("id", 0)))
        except Exception:
            continue

    elem_sig_by_id = np.zeros((max_eid,), dtype=TOKEN_DTYPE)
    sigs_list: List[Any] = []

    for d in elems:
        if not (isinstance(d, dict) and "id" in d and "type" in d and "node_ids" in d and isinstance(d.get("node_ids"), list)):
            continue
        try:
            eid = int(d["id"])
            if eid <= 0 or eid > max_eid:
                continue

            et = str(d["type"])
            toks: List[Any] = []
            for nid in d["node_ids"]:
                ni = int(nid)
                if 0 < ni <= int(node_keys_mesh.size):
                    toks.append(node_keys_mesh[ni - 1])
                else:
                    toks.append(np.zeros((), dtype=TOKEN_DTYPE)[()])

            arr = np.asarray(toks, dtype=TOKEN_DTYPE)
            if order_invariant:
                arr = token_array_sort(arr)

            h = token_mix_init(include_type, et)
            for t in arr:
                h = token_mix_step(h, t)

            elem_sig_by_id[eid - 1] = h
            sigs_list.append(h)
        except Exception:
            continue

    sigs = np.asarray(sigs_list, dtype=TOKEN_DTYPE)
    m = token_nonzero_mask(sigs)
    return sigs[m], elem_sig_by_id


def build_elem_sigs_agnostic(model: Dict[str, Any], node_keys_mesh: np.ndarray, order_invariant: bool = True) -> np.ndarray:
    sigs, _ = build_element_signatures(model, node_keys_mesh, order_invariant=order_invariant, include_type=False)
    return sigs


def element_type_counts(model: Dict[str, Any]) -> Dict[str, int]:
    c = Counter()
    for e in iter_mesh_elements(model):
        if isinstance(e, dict) and "type" in e:
            c[str(e["type"])] += 1
    return dict(c)


def element_node_ref_stats(model: Dict[str, Any], node_keys_mesh: np.ndarray) -> Dict[str, int]:
    nmax = int(node_keys_mesh.size)
    total = 0
    invalid = 0
    zero_tok = 0
    for e in iter_mesh_elements(model):
        nids = e.get("node_ids")
        if not isinstance(nids, list):
            continue
        for nid in nids:
            total += 1
            try:
                ni = int(nid)
            except Exception:
                invalid += 1
                continue
            if ni <= 0 or ni > nmax:
                invalid += 1
                continue
            k = node_keys_mesh[ni - 1]
            if token_is_zero_scalar(k):
                zero_tok += 1
    return {"total": int(total), "invalid": int(invalid), "zero_token": int(zero_tok)}


def refpoint_keys(model: Dict[str, Any], eps: float) -> np.ndarray:
    keys: List[Any] = []
    for rn in iter_reference_nodes(model):
        if not all(k in rn for k in ("x", "y", "z")):
            continue
        try:
            keys.append(coord_token(float(rn["x"]), float(rn["y"]), float(rn["z"]), eps))
        except Exception:
            continue
    arr = np.asarray(keys, dtype=TOKEN_DTYPE)
    m = token_nonzero_mask(arr)
    return arr[m]


def rp_as_nodes_coverage(rp_tokens: np.ndarray, node_tokens: np.ndarray) -> Tuple[float, int, int, int]:
    rp_u = token_unique(rp_tokens[token_nonzero_mask(rp_tokens)])
    nd_u = token_unique(node_tokens[token_nonzero_mask(node_tokens)])

    rp_unique = int(rp_u.size)
    nodes_unique = int(nd_u.size)

    if rp_unique == 0:
        return 1.0, 0, 0, nodes_unique

    inter = int(token_intersect_unique(rp_u, nd_u).size)
    coverage = inter / rp_unique
    return float(coverage), inter, rp_unique, nodes_unique


def filter_remove_refpoint_artifacts_from_nodes(node_rt_raw: np.ndarray, node_ref_mesh: np.ndarray, rp_ref: np.ndarray) -> Tuple[np.ndarray, int]:
    if node_rt_raw.size == 0 or rp_ref.size == 0:
        return node_rt_raw, 0

    vref, cref = token_unique_with_counts(node_ref_mesh)
    ref_count_map = {token_to_hex(v): int(c) for v, c in zip(vref.tolist(), cref.tolist())}

    vrp, crp = token_unique_with_counts(rp_ref)
    rp_count_map = {token_to_hex(v): int(c) for v, c in zip(vrp.tolist(), crp.tolist())}
    rp_token_set = set(rp_count_map.keys())

    rt_sorted = token_array_sort(np.array(node_rt_raw, copy=True))

    vals, starts, counts = token_unique_with_index_counts(rt_sorted)

    keep = np.ones(rt_sorted.shape[0], dtype=bool)
    removed = 0

    for v_u, start, cnt in zip(vals.tolist(), starts.tolist(), counts.tolist()):
        key = token_to_hex(v_u)
        if key not in rp_token_set:
            continue
        allowed = ref_count_map.get(key, 0)
        extra = cnt - allowed
        if extra <= 0:
            continue
        rp_mult = rp_count_map.get(key, 0)
        to_remove = min(extra, rp_mult)
        if to_remove <= 0:
            continue
        end = start + cnt
        keep[end - to_remove:end] = False
        removed += to_remove

    return rt_sorted[keep], int(removed)


# =============================================================================
# Metrics (Multiset, Set)
# =============================================================================

def multiset_intersection_count(a: np.ndarray, b: np.ndarray) -> int:
    if a.size == 0 or b.size == 0:
        return 0
    va, ca = token_unique_with_counts(a)
    vb, cb = token_unique_with_counts(b)
    common = token_intersect_unique(va, vb)
    if common.size == 0:
        return 0
    ia = token_searchsorted(va, common)
    ib = token_searchsorted(vb, common)
    return int(np.minimum(ca[ia], cb[ib]).sum())


def multiset_prf(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    na = int(a.size)
    nb = int(b.size)

    if na == 0 and nb == 0:
        return 1.0, 1.0, 1.0
    if na == 0 and nb > 0:
        return 0.0, 1.0, 0.0
    if na > 0 and nb == 0:
        return 0.0, 0.0, 0.0

    inter = multiset_intersection_count(a, b)
    prec = inter / nb if nb > 0 else 0.0
    rec = inter / na if na > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def refpoints_prf(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    na = int(a.size)
    nb = int(b.size)
    if na == 0 and nb == 0:
        return 1.0, 1.0, 1.0
    if na > 0 and nb == 0:
        return 0.0, 0.0, 0.0
    if na == 0 and nb > 0:
        return 0.0, 0.0, 0.0
    return multiset_prf(a, b)


def set_prf(a_tokens: np.ndarray, b_tokens: np.ndarray) -> Tuple[float, float, float, float, int]:
    a = token_unique(a_tokens[token_nonzero_mask(a_tokens)])
    b = token_unique(b_tokens[token_nonzero_mask(b_tokens)])
    na = int(a.size)
    nb = int(b.size)

    if na == 0 and nb == 0:
        return 1.0, 1.0, 1.0, 1.0, 0
    if na == 0 and nb > 0:
        return 0.0, 1.0, 0.0, 0.0, 0
    if na > 0 and nb == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    inter = token_intersect_unique(a, b)
    i = int(inter.size)
    prec = i / nb if nb > 0 else 0.0
    rec = i / na if na > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    union = na + nb - i
    jacc = (i / union) if union > 0 else 1.0
    return float(prec), float(rec), float(f1), float(jacc), i


def multiset_diff_table(a: np.ndarray, b: np.ndarray, limit: int) -> List[Tuple[str, int, int, int, int]]:
    a = a[token_nonzero_mask(a)]
    b = b[token_nonzero_mask(b)]

    va, ca = token_unique_with_counts(a) if a.size else (np.asarray([], dtype=TOKEN_DTYPE), np.asarray([], dtype=np.int64))
    vb, cb = token_unique_with_counts(b) if b.size else (np.asarray([], dtype=TOKEN_DTYPE), np.asarray([], dtype=np.int64))

    ref = {token_to_hex(k): int(v) for k, v in zip(va.tolist(), ca.tolist())}
    rt = {token_to_hex(k): int(v) for k, v in zip(vb.tolist(), cb.tolist())}

    keys = set(ref.keys()) | set(rt.keys())
    rows: List[Tuple[str, int, int, int, int]] = []
    for k in keys:
        rc = ref.get(k, 0)
        pc = rt.get(k, 0)
        miss = max(0, rc - pc)
        extra = max(0, pc - rc)
        if miss or extra:
            rows.append((k, rc, pc, miss, extra))

    rows.sort(key=lambda t: (t[3] + t[4], t[3], t[4]), reverse=True)
    return rows[:limit]


def _dump_small_multiset_diff(cfg: Config, a: np.ndarray, b: np.ndarray, label: str) -> None:
    if not cfg.dump_mismatch_examples:
        return
    inter = multiset_intersection_count(a, b)
    if inter == int(a.size) == int(b.size):
        return

    va, ca = token_unique_with_counts(a)
    vb, cb = token_unique_with_counts(b)

    ref_map = {token_to_hex(v): int(c) for v, c in zip(va.tolist(), ca.tolist())}
    rt_map  = {token_to_hex(v): int(c) for v, c in zip(vb.tolist(), cb.tolist())}
    keys = set(ref_map.keys()) | set(rt_map.keys())

    missing = [(k, ref_map.get(k, 0) - rt_map.get(k, 0)) for k in keys if ref_map.get(k, 0) > rt_map.get(k, 0)]
    extra   = [(k, rt_map.get(k, 0) - ref_map.get(k, 0)) for k in keys if rt_map.get(k, 0) > ref_map.get(k, 0)]
    missing.sort(key=lambda x: -x[1])
    extra.sort(key=lambda x: -x[1])

    log(cfg, f"[DIFF:{label}] inter={inter} ref={a.size} rt={b.size}")
    log(cfg, "  missing(top):", missing[:cfg.mismatch_example_limit])
    log(cfg, "  extra(top):  ", extra[:cfg.mismatch_example_limit])


# =============================================================================
# Sets: Membership Signatures + optional name-metric
# =============================================================================

def include_instance_for_set_eval(cfg: Config, source_format: str, target_format: str) -> bool:
    mode = str(cfg.set_instance_eval_mode).lower().strip()
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool(FORMAT_CAPS.get(source_format, {}).get("hierarchy", False)) and \
        bool(FORMAT_CAPS.get(target_format, {}).get("hierarchy", False))

# Paper mapping: Sets (paper Section 5.6.6).
# - Membership is evaluated name-agnostically using signatures derived from member tokens.
# - Name-F1 is only reported where both formats support human-readable set names in the
#   evaluated subset (capability-aware reporting; else N/A).
def include_set_names_for_eval(cfg: Config, source_format: str, target_format: str) -> bool:
    mode = str(cfg.set_name_metric_mode).lower().strip()
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool(FORMAT_CAPS.get(source_format, {}).get("set_names", False)) and \
        bool(FORMAT_CAPS.get(target_format, {}).get("set_names", False))


_SETNAME_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060"}  # zwsp/zwnj/zwj/wj

# Paper mapping: set-name normalization (paper Section 5.6.6).
# Goal: avoid spurious mismatches due to whitespace/control chars/unicode variants
# when computing Name-F1 on routes where names are representable.
def _normalize_set_name(name: str) -> str:
    s = "" if name is None else str(name)

    try:
        s = unicodedata.normalize("NFKC", s)
    except Exception:
        pass

    s = s.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2007", " ")
    s = s.replace("\ufeff", "")
    for zw in _SETNAME_ZERO_WIDTH:
        s = s.replace(zw, "")

    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    s = " ".join(s.split())
    return s.casefold()


def _set_sig_token(sorted_unique_tokens: np.ndarray, instance: str = "", include_instance: bool = True) -> Any:
    h = hashlib.blake2b(digest_size=HASH_BYTES)
    if include_instance:
        h.update(str(instance).encode("utf-8"))
        h.update(b"\0")
    if sorted_unique_tokens.size == 0:
        h.update(b"EMPTY")
        return digest_to_token(h.digest())
    h.update(sorted_unique_tokens.tobytes())
    return digest_to_token(h.digest())


def _set_sig_token_with_name(sorted_unique_tokens: np.ndarray, name: str, instance: str = "", include_instance: bool = True) -> Any:
    h = hashlib.blake2b(digest_size=HASH_BYTES)
    if include_instance:
        h.update(str(instance).encode("utf-8"))
        h.update(b"\0")
    h.update(_normalize_set_name(name).encode("utf-8"))
    h.update(b"\0")
    if sorted_unique_tokens.size == 0:
        h.update(b"EMPTY")
        return digest_to_token(h.digest())
    h.update(sorted_unique_tokens.tobytes())
    return digest_to_token(h.digest())


def nodeset_membership_sigs(sets: List[SetEntry], node_keys_full: np.ndarray, include_instance: bool) -> Tuple[np.ndarray, int]:
    sigs_list: List[Any] = []
    invalid = 0
    for s in sets:
        toks: List[Any] = []
        for nid in s.ids:
            if 0 < nid <= int(node_keys_full.size):
                k = node_keys_full[nid - 1]
                if not token_is_zero_scalar(k):
                    toks.append(k)
            else:
                invalid += 1
        arr = np.asarray(toks, dtype=TOKEN_DTYPE)
        arr = arr[token_nonzero_mask(arr)]
        arr = token_unique(arr)
        sigs_list.append(_set_sig_token(arr, instance=s.instance, include_instance=include_instance))
    return np.asarray(sigs_list, dtype=TOKEN_DTYPE), int(invalid)


def elemset_membership_sigs(sets: List[SetEntry], elem_sig_by_id: np.ndarray, include_instance: bool) -> Tuple[np.ndarray, int]:
    sigs_list: List[Any] = []
    invalid = 0
    for s in sets:
        toks: List[Any] = []
        for eid in s.ids:
            if 0 < eid <= int(elem_sig_by_id.size):
                k = elem_sig_by_id[eid - 1]
                if not token_is_zero_scalar(k):
                    toks.append(k)
            else:
                invalid += 1
        arr = np.asarray(toks, dtype=TOKEN_DTYPE)
        arr = arr[token_nonzero_mask(arr)]
        arr = token_unique(arr)
        sigs_list.append(_set_sig_token(arr, instance=s.instance, include_instance=include_instance))
    return np.asarray(sigs_list, dtype=TOKEN_DTYPE), int(invalid)


def nodeset_named_membership_sigs(sets: List[SetEntry], node_keys_full: np.ndarray, include_instance: bool) -> Tuple[np.ndarray, int]:
    sigs_list: List[Any] = []
    invalid = 0
    for s in sets:
        toks: List[Any] = []
        for nid in s.ids:
            if 0 < nid <= int(node_keys_full.size):
                k = node_keys_full[nid - 1]
                if not token_is_zero_scalar(k):
                    toks.append(k)
            else:
                invalid += 1
        arr = np.asarray(toks, dtype=TOKEN_DTYPE)
        arr = arr[token_nonzero_mask(arr)]
        arr = token_unique(arr)
        sigs_list.append(_set_sig_token_with_name(arr, name=s.name, instance=s.instance, include_instance=include_instance))
    return np.asarray(sigs_list, dtype=TOKEN_DTYPE), int(invalid)


def elemset_named_membership_sigs(sets: List[SetEntry], elem_sig_by_id: np.ndarray, include_instance: bool) -> Tuple[np.ndarray, int]:
    sigs_list: List[Any] = []
    invalid = 0
    for s in sets:
        toks: List[Any] = []
        for eid in s.ids:
            if 0 < eid <= int(elem_sig_by_id.size):
                k = elem_sig_by_id[eid - 1]
                if not token_is_zero_scalar(k):
                    toks.append(k)
            else:
                invalid += 1
        arr = np.asarray(toks, dtype=TOKEN_DTYPE)
        arr = arr[token_nonzero_mask(arr)]
        arr = token_unique(arr)
        sigs_list.append(_set_sig_token_with_name(arr, name=s.name, instance=s.instance, include_instance=include_instance))
    return np.asarray(sigs_list, dtype=TOKEN_DTYPE), int(invalid)


def nodeset_union_keys(sets: List[SetEntry], node_keys_full: np.ndarray) -> Tuple[np.ndarray, int]:
    invalid = 0
    toks: List[Any] = []
    for s in sets:
        for nid in s.ids:
            if 0 < nid <= int(node_keys_full.size):
                k = node_keys_full[nid - 1]
                if not token_is_zero_scalar(k):
                    toks.append(k)
            else:
                invalid += 1
    arr = np.asarray(toks, dtype=TOKEN_DTYPE)
    arr = arr[token_nonzero_mask(arr)]
    arr = token_unique(arr)
    return arr, int(invalid)


def elemset_union_sigs(sets: List[SetEntry], elem_sig_by_id: np.ndarray) -> Tuple[np.ndarray, int]:
    invalid = 0
    toks: List[Any] = []
    for s in sets:
        for eid in s.ids:
            if 0 < eid <= int(elem_sig_by_id.size):
                k = elem_sig_by_id[eid - 1]
                if not token_is_zero_scalar(k):
                    toks.append(k)
            else:
                invalid += 1
    arr = np.asarray(toks, dtype=TOKEN_DTYPE)
    arr = arr[token_nonzero_mask(arr)]
    arr = token_unique(arr)
    return arr, int(invalid)


# =============================================================================
# Explain: CSV Writer (na_rep)
# =============================================================================

def _fmt_node_samples(samples: List[Tuple[int, float, float, float]]) -> str:
    if not samples:
        return ""
    return " | ".join([f"id={nid} ({x:.6g},{y:.6g},{z:.6g})" for nid, x, y, z in samples])


def _fmt_elem_samples(samples: List[Tuple[int, str, List[int]]]) -> str:
    if not samples:
        return ""
    return " | ".join([f"eid={eid} type={t} nids={nids}" for eid, t, nids in samples])


def index_nodes_by_token_hex(
        model: Dict[str, Any],
        eps: float,
        connected_only: bool,
        connected_mask_opt: Optional[np.ndarray],
        max_samples_per_token: int,
) -> Dict[str, List[Tuple[int, float, float, float]]]:
    idx: Dict[str, List[Tuple[int, float, float, float]]] = defaultdict(list)

    for n in iter_mesh_nodes(model):
        if not all(k in n for k in ("id", "x", "y", "z")):
            continue
        try:
            nid = int(n["id"])
            if nid <= 0:
                continue
            if connected_only and connected_mask_opt is not None:
                i = nid - 1
                if i < 0 or i >= connected_mask_opt.size or not bool(connected_mask_opt[i]):
                    continue
            x, y, z = float(n["x"]), float(n["y"]), float(n["z"])
            tok = coord_token(x, y, z, eps)
            if token_is_zero_scalar(tok):
                continue
            key = token_to_hex(tok)
            lst = idx[key]
            if len(lst) < max_samples_per_token:
                lst.append((nid, x, y, z))
        except Exception:
            continue

    return dict(idx)


def build_element_sig(
        elem_type: str,
        node_ids: List[Any],
        node_keys_mesh: np.ndarray,
        include_type: bool,
        order_invariant: bool,
) -> Any:
    toks: List[Any] = []
    for nid in node_ids:
        ni = int(nid)
        if 0 < ni <= int(node_keys_mesh.size):
            toks.append(node_keys_mesh[ni - 1])
        else:
            toks.append(np.zeros((), dtype=TOKEN_DTYPE)[()])
    arr = np.asarray(toks, dtype=TOKEN_DTYPE)
    if order_invariant:
        arr = token_array_sort(arr)
    h = token_mix_init(include_type, elem_type)
    for t in arr:
        h = token_mix_step(h, t)
    return h


def index_elements_by_sig_hex(
        model: Dict[str, Any],
        node_keys_mesh: np.ndarray,
        include_type: bool,
        order_invariant: bool,
        max_samples_per_sig: int,
) -> Dict[str, List[Tuple[int, str, List[int]]]]:
    idx: Dict[str, List[Tuple[int, str, List[int]]]] = defaultdict(list)
    for e in iter_mesh_elements(model):
        if not isinstance(e, dict):
            continue
        if "id" not in e or "type" not in e or "node_ids" not in e:
            continue
        if not isinstance(e.get("node_ids"), list):
            continue
        try:
            eid = int(e["id"])
            et = str(e["type"])
            nids = [int(x) for x in e["node_ids"]]
            sig = build_element_sig(et, nids, node_keys_mesh, include_type=include_type, order_invariant=order_invariant)
            key = token_to_hex(sig)
            lst = idx[key]
            if len(lst) < max_samples_per_sig:
                lst.append((eid, et, nids))
        except Exception:
            continue
    return dict(idx)


def write_diff_csv_nodes(
        out_dir: Path,
        label: str,
        ref_tokens: np.ndarray,
        rt_tokens: np.ndarray,
        idx_ref: Dict[str, List[Tuple[int, float, float, float]]],
        idx_rt: Dict[str, List[Tuple[int, float, float, float]]],
        limit: int,
) -> None:
    rows = multiset_diff_table(ref_tokens, rt_tokens, limit=limit)
    if not rows:
        return
    data = []
    for tok_hex, rc, pc, miss, extra in rows:
        data.append({
            "token_hex": tok_hex,
            "ref_count": rc,
            "rt_count": pc,
            "missing": miss,
            "extra": extra,
            "ref_samples": _fmt_node_samples(idx_ref.get(tok_hex, [])),
            "rt_samples": _fmt_node_samples(idx_rt.get(tok_hex, [])),
        })
    pd.DataFrame(data).to_csv(out_dir / f"{label}.csv", index=False, na_rep=CSV_NA_REP)


def write_diff_csv_elems(
        out_dir: Path,
        label: str,
        ref_sigs: np.ndarray,
        rt_sigs: np.ndarray,
        idx_ref: Dict[str, List[Tuple[int, str, List[int]]]],
        idx_rt: Dict[str, List[Tuple[int, str, List[int]]]],
        limit: int,
) -> None:
    rows = multiset_diff_table(ref_sigs, rt_sigs, limit=limit)
    if not rows:
        return
    data = []
    for sig_hex, rc, pc, miss, extra in rows:
        data.append({
            "sig_hex": sig_hex,
            "ref_count": rc,
            "rt_count": pc,
            "missing": miss,
            "extra": extra,
            "ref_samples": _fmt_elem_samples(idx_ref.get(sig_hex, [])),
            "rt_samples": _fmt_elem_samples(idx_rt.get(sig_hex, [])),
        })
    pd.DataFrame(data).to_csv(out_dir / f"{label}.csv", index=False, na_rep=CSV_NA_REP)


def write_elem_types_diff_csv(out_dir: Path, ref_counts: Dict[str, int], rt_counts: Dict[str, int], limit: int = 200) -> None:
    keys = set(ref_counts.keys()) | set(rt_counts.keys())
    rows = []
    for k in keys:
        a = int(ref_counts.get(k, 0))
        b = int(rt_counts.get(k, 0))
        if a != b:
            rows.append({"type": k, "ref": a, "rt": b, "delta": b - a})
    rows.sort(key=lambda r: abs(int(r["delta"])), reverse=True)
    if rows:
        pd.DataFrame(rows[:limit]).to_csv(out_dir / "elem_types_diff.csv", index=False, na_rep=CSV_NA_REP)


def boundary_distance_abs(v: float, eps: float) -> float:
    r = v / eps
    frac = r - math.floor(r)
    return abs(frac - 0.5) * eps


def build_qgrid_index(
        model: Dict[str, Any],
        eps: float,
        connected_only: bool,
        connected_mask_opt: Optional[np.ndarray],
        max_samples_per_cell: int = 2,
) -> Dict[Tuple[int, int, int], List[Tuple[int, float, float, float]]]:
    grid: Dict[Tuple[int, int, int], List[Tuple[int, float, float, float]]] = defaultdict(list)

    for n in iter_mesh_nodes(model):
        if not all(k in n for k in ("id", "x", "y", "z")):
            continue
        try:
            nid = int(n["id"])
            if nid <= 0:
                continue
            if connected_only and connected_mask_opt is not None:
                i = nid - 1
                if i < 0 or i >= connected_mask_opt.size or not bool(connected_mask_opt[i]):
                    continue
            x, y, z = float(n["x"]), float(n["y"]), float(n["z"])
            cell = quantize_triplet(x, y, z, eps)
            lst = grid[cell]
            if len(lst) < max_samples_per_cell:
                lst.append((nid, x, y, z))
        except Exception:
            continue

    return dict(grid)


def probe_nearest_in_grid(
        sample: Tuple[int, float, float, float],
        grid: Dict[Tuple[int, int, int], List[Tuple[int, float, float, float]]],
        eps: float,
        radius: int,
) -> Dict[str, Any]:
    nid, x, y, z = sample
    qx, qy, qz = quantize_triplet(x, y, z, eps)

    best = None
    best_d2 = float("inf")
    best_cell = None

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                cell = (qx + dx, qy + dy, qz + dz)
                cand = grid.get(cell)
                if not cand:
                    continue
                for cnid, cx, cy, cz in cand:
                    d2 = (cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best = (cnid, cx, cy, cz)
                        best_cell = cell

    out = {
        "src_nid": nid,
        "src_x": x, "src_y": y, "src_z": z,
        "src_qx": qx, "src_qy": qy, "src_qz": qz,
        "min_boundary_abs": min(
            boundary_distance_abs(x, eps),
            boundary_distance_abs(y, eps),
            boundary_distance_abs(z, eps),
        ),
        "best_found": False,
        "best_nid": "",
        "best_x": "", "best_y": "", "best_z": "",
        "best_dx": "", "best_dy": "", "best_dz": "",
        "best_dist": "",
        "best_dq": "",
    }

    if best is None or best_cell is None:
        return out

    bnid, bx, by, bz = best
    dq = (best_cell[0] - qx, best_cell[1] - qy, best_cell[2] - qz)

    out.update({
        "best_found": True,
        "best_nid": bnid,
        "best_x": bx, "best_y": by, "best_z": bz,
        "best_dx": bx - x, "best_dy": by - y, "best_dz": bz - z,
        "best_dist": math.sqrt(best_d2),
        "best_dq": str(dq),
    })
    return out


def write_nodes_probe_csv(
        cfg: Config,
        out_dir: Path,
        ref_tokens: np.ndarray,
        rt_tokens: np.ndarray,
        idx_ref: Dict[str, List[Tuple[int, float, float, float]]],
        idx_rt: Dict[str, List[Tuple[int, float, float, float]]],
        ref_grid: Dict[Tuple[int, int, int], List[Tuple[int, float, float, float]]],
        rt_grid: Dict[Tuple[int, int, int], List[Tuple[int, float, float, float]]],
        eps: float,
) -> Dict[str, Any]:
    rows = multiset_diff_table(ref_tokens, rt_tokens, limit=cfg.explain_max_tokens)
    if not rows:
        return {"written": False}

    tol_abs = float(cfg.boundary_tol_rel) * float(eps)
    out_rows: List[Dict[str, Any]] = []

    for tok_hex, rc, pc, miss, extra in rows:
        if miss > 0:
            for s in idx_ref.get(tok_hex, []):
                pr = probe_nearest_in_grid(s, rt_grid, eps, cfg.neighbor_q_radius)
                pr.update({
                    "token_hex": tok_hex,
                    "direction": "missing_in_rt",
                    "ref_count": rc,
                    "rt_count": pc,
                    "boundary_tol_abs": tol_abs,
                    "boundary_suspect": bool(pr["min_boundary_abs"] <= tol_abs),
                })
                out_rows.append(pr)
                if len(out_rows) >= cfg.max_node_probes_total:
                    break
        if len(out_rows) >= cfg.max_node_probes_total:
            break

        if extra > 0:
            for s in idx_rt.get(tok_hex, []):
                pr = probe_nearest_in_grid(s, ref_grid, eps, cfg.neighbor_q_radius)
                pr.update({
                    "token_hex": tok_hex,
                    "direction": "extra_in_rt",
                    "ref_count": rc,
                    "rt_count": pc,
                    "boundary_tol_abs": tol_abs,
                    "boundary_suspect": bool(pr["min_boundary_abs"] <= tol_abs),
                })
                out_rows.append(pr)
                if len(out_rows) >= cfg.max_node_probes_total:
                    break
        if len(out_rows) >= cfg.max_node_probes_total:
            break

    if not out_rows:
        return {"written": False}

    dfp = pd.DataFrame(out_rows)
    dfp.to_csv(out_dir / "nodes_probe.csv", index=False, na_rep=CSV_NA_REP)

    best_found = dfp["best_found"].astype(bool) if "best_found" in dfp.columns else pd.Series([], dtype=bool)
    bf_rate = float(best_found.mean()) if len(best_found) else 0.0

    bnd = dfp["boundary_suspect"].astype(bool) if "boundary_suspect" in dfp.columns else pd.Series([], dtype=bool)
    bnd_rate = float(bnd.mean()) if len(bnd) else 0.0

    def _is_nonzero_dq(s: Any) -> bool:
        try:
            t = str(s)
            return t not in {"(0, 0, 0)", "(0,0,0)"}
        except Exception:
            return False

    dq_nonzero = dfp["best_dq"].apply(_is_nonzero_dq) if "best_dq" in dfp.columns else pd.Series([], dtype=bool)
    dq_nonzero_rate = float(dq_nonzero.mean()) if len(dq_nonzero) else 0.0

    dists = pd.to_numeric(dfp.get("best_dist", pd.Series([], dtype=float)), errors="coerce")
    dists = dists[np.isfinite(dists)]
    dist_median = float(dists.median()) if len(dists) else None
    dist_max = float(dists.max()) if len(dists) else None

    return {
        "written": True,
        "rows": int(len(dfp)),
        "best_found_rate": bf_rate,
        "boundary_suspect_rate": bnd_rate,
        "neighbor_cell_rate": dq_nonzero_rate,
        "best_dist_median": dist_median,
        "best_dist_max": dist_max,
        "boundary_tol_abs": float(tol_abs),
        "q_radius": int(cfg.neighbor_q_radius),
    }


def _truncate(items: List[str], max_items: int) -> str:
    if max_items <= 0:
        return ""
    if len(items) <= max_items:
        return "; ".join(items)
    rest = len(items) - max_items
    return "; ".join(items[:max_items]) + f"; ...(+{rest})"


def _fmt_ids_sample(ids: List[int], max_n: int) -> str:
    if max_n <= 0:
        return ""
    s = ids[:max_n]
    return "[" + ",".join(str(x) for x in s) + (",..." if len(ids) > max_n else "") + "]"


def _name_codepoints_short(name: str, max_chars: int = 48) -> str:
    s = "" if name is None else str(name)
    s = s[:max_chars]
    parts = []
    for ch in s:
        cp = f"U+{ord(ch):04X}"
        cat = unicodedata.category(ch)
        parts.append(f"{cp}({cat})")
    if len(str(name)) > max_chars:
        parts.append("...")
    return " ".join(parts)


def index_nodesets_by_sig_hex(cfg: Config, sets: List[SetEntry], node_keys_full: np.ndarray, include_instance: bool) -> Dict[str, List[Dict[str, Any]]]:
    idx: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in sets:
        toks: List[Any] = []
        invalid = 0
        for nid in s.ids:
            if 0 < nid <= int(node_keys_full.size):
                k = node_keys_full[nid - 1]
                if not token_is_zero_scalar(k):
                    toks.append(k)
            else:
                invalid += 1
        arr = np.asarray(toks, dtype=TOKEN_DTYPE)
        arr = arr[token_nonzero_mask(arr)]
        arr = token_unique(arr)
        sig = _set_sig_token(arr, instance=s.instance, include_instance=include_instance)
        key = token_to_hex(sig)
        idx[key].append({
            "instance": s.instance,
            "name": s.name,
            "n_ids_raw": int(len(s.ids)),
            "n_invalid_ids": int(invalid),
            "n_membership_unique": int(arr.size),
            "ids_sample": _fmt_ids_sample([int(x) for x in s.ids], cfg.explain_sets_debug_sample_ids),
        })
    return dict(idx)


def index_elemsets_by_sig_hex(cfg: Config, sets: List[SetEntry], elem_sig_by_id: np.ndarray, include_instance: bool) -> Dict[str, List[Dict[str, Any]]]:
    idx: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in sets:
        toks: List[Any] = []
        invalid = 0
        for eid in s.ids:
            if 0 < eid <= int(elem_sig_by_id.size):
                k = elem_sig_by_id[eid - 1]
                if not token_is_zero_scalar(k):
                    toks.append(k)
            else:
                invalid += 1
        arr = np.asarray(toks, dtype=TOKEN_DTYPE)
        arr = arr[token_nonzero_mask(arr)]
        arr = token_unique(arr)
        sig = _set_sig_token(arr, instance=s.instance, include_instance=include_instance)
        key = token_to_hex(sig)
        idx[key].append({
            "instance": s.instance,
            "name": s.name,
            "n_ids_raw": int(len(s.ids)),
            "n_invalid_ids": int(invalid),
            "n_membership_unique": int(arr.size),
            "ids_sample": _fmt_ids_sample([int(x) for x in s.ids], cfg.explain_sets_debug_sample_ids),
        })
    return dict(idx)


def index_nodesets_by_name_sig_hex(cfg: Config, sets: List[SetEntry], node_keys_full: np.ndarray, include_instance: bool) -> Dict[str, List[Dict[str, Any]]]:
    idx: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in sets:
        toks: List[Any] = []
        invalid = 0
        for nid in s.ids:
            if 0 < nid <= int(node_keys_full.size):
                k = node_keys_full[nid - 1]
                if not token_is_zero_scalar(k):
                    toks.append(k)
            else:
                invalid += 1
        arr = np.asarray(toks, dtype=TOKEN_DTYPE)
        arr = arr[token_nonzero_mask(arr)]
        arr = token_unique(arr)
        sig = _set_sig_token_with_name(arr, name=s.name, instance=s.instance, include_instance=include_instance)
        key = token_to_hex(sig)
        name_raw = s.name
        name_norm = _normalize_set_name(name_raw)
        idx[key].append({
            "instance": s.instance,
            "name_raw": name_raw,
            "name_norm": name_norm,
            "name_repr": repr(name_raw),
            "name_codepoints": _name_codepoints_short(name_raw),
            "n_ids_raw": int(len(s.ids)),
            "n_invalid_ids": int(invalid),
            "n_membership_unique": int(arr.size),
            "ids_sample": _fmt_ids_sample([int(x) for x in s.ids], cfg.explain_sets_debug_sample_ids),
        })
    return dict(idx)


def index_elemsets_by_name_sig_hex(cfg: Config, sets: List[SetEntry], elem_sig_by_id: np.ndarray, include_instance: bool) -> Dict[str, List[Dict[str, Any]]]:
    idx: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in sets:
        toks: List[Any] = []
        invalid = 0
        for eid in s.ids:
            if 0 < eid <= int(elem_sig_by_id.size):
                k = elem_sig_by_id[eid - 1]
                if not token_is_zero_scalar(k):
                    toks.append(k)
            else:
                invalid += 1
        arr = np.asarray(toks, dtype=TOKEN_DTYPE)
        arr = arr[token_nonzero_mask(arr)]
        arr = token_unique(arr)
        sig = _set_sig_token_with_name(arr, name=s.name, instance=s.instance, include_instance=include_instance)
        key = token_to_hex(sig)
        name_raw = s.name
        name_norm = _normalize_set_name(name_raw)
        idx[key].append({
            "instance": s.instance,
            "name_raw": name_raw,
            "name_norm": name_norm,
            "name_repr": repr(name_raw),
            "name_codepoints": _name_codepoints_short(name_raw),
            "n_ids_raw": int(len(s.ids)),
            "n_invalid_ids": int(invalid),
            "n_membership_unique": int(arr.size),
            "ids_sample": _fmt_ids_sample([int(x) for x in s.ids], cfg.explain_sets_debug_sample_ids),
        })
    return dict(idx)


def write_sets_debug_csv(
        cfg: Config,
        out_dir: Path,
        nodeset_rows: List[Tuple[str, int, int, int, int]],
        elemset_rows: List[Tuple[str, int, int, int, int]],
        idx_ref_nodesets: Dict[str, List[Dict[str, Any]]],
        idx_rt_nodesets: Dict[str, List[Dict[str, Any]]],
        idx_ref_elemsets: Dict[str, List[Dict[str, Any]]],
        idx_rt_elemsets: Dict[str, List[Dict[str, Any]]],
) -> bool:
    data: List[Dict[str, Any]] = []

    def add(kind: str, row: Tuple[str, int, int, int, int],
            idx_ref: Dict[str, List[Dict[str, Any]]],
            idx_rt: Dict[str, List[Dict[str, Any]]]) -> None:
        sig_hex, rc, pc, miss, extra = row
        ref_entries = idx_ref.get(sig_hex, [])
        rt_entries = idx_rt.get(sig_hex, [])

        def fmt_entries(entries: List[Dict[str, Any]]) -> str:
            parts: List[str] = []
            for e in entries:
                inst = e.get("instance", "")
                name = e.get("name", "")
                nm = int(e.get("n_membership_unique", 0))
                nr = int(e.get("n_ids_raw", 0))
                inv = int(e.get("n_invalid_ids", 0))
                ids_sample = str(e.get("ids_sample", ""))
                parts.append(f"inst='{inst}' name='{name}' m_unique={nm} ids={nr} invalid={inv} sample={ids_sample}")
            return _truncate(parts, cfg.explain_sets_debug_max_sets_per_sig)

        data.append({
            "kind": kind,
            "sig_hex": sig_hex,
            "ref_count": int(rc),
            "rt_count": int(pc),
            "missing": int(miss),
            "extra": int(extra),
            "ref_sets": fmt_entries(ref_entries),
            "rt_sets": fmt_entries(rt_entries),
            "ref_sets_total_for_sig": int(len(ref_entries)),
            "rt_sets_total_for_sig": int(len(rt_entries)),
        })

    for r in nodeset_rows:
        add("nodeset", r, idx_ref_nodesets, idx_rt_nodesets)
    for r in elemset_rows:
        add("elemset", r, idx_ref_elemsets, idx_rt_elemsets)

    if not data:
        return False

    pd.DataFrame(data).to_csv(out_dir / "sets_debug.csv", index=False, na_rep=CSV_NA_REP)
    return True


def write_sets_name_debug_csv(
        cfg: Config,
        out_dir: Path,
        nodeset_name_rows: List[Tuple[str, int, int, int, int]],
        elemset_name_rows: List[Tuple[str, int, int, int, int]],
        idx_ref_nodesets_name: Dict[str, List[Dict[str, Any]]],
        idx_rt_nodesets_name: Dict[str, List[Dict[str, Any]]],
        idx_ref_elemsets_name: Dict[str, List[Dict[str, Any]]],
        idx_rt_elemsets_name: Dict[str, List[Dict[str, Any]]],
) -> bool:
    data: List[Dict[str, Any]] = []

    def add(kind: str, row: Tuple[str, int, int, int, int],
            idx_ref: Dict[str, List[Dict[str, Any]]],
            idx_rt: Dict[str, List[Dict[str, Any]]]) -> None:
        sig_hex, rc, pc, miss, extra = row
        ref_entries = idx_ref.get(sig_hex, [])
        rt_entries = idx_rt.get(sig_hex, [])

        def fmt_entries(entries: List[Dict[str, Any]]) -> str:
            parts: List[str] = []
            for e in entries:
                inst = e.get("instance", "")
                name_raw = e.get("name_raw", "")
                name_norm = e.get("name_norm", "")
                name_repr = e.get("name_repr", "")
                cps = e.get("name_codepoints", "")
                nm = int(e.get("n_membership_unique", 0))
                nr = int(e.get("n_ids_raw", 0))
                inv = int(e.get("n_invalid_ids", 0))
                ids_sample = str(e.get("ids_sample", ""))
                parts.append(
                    f"inst='{inst}' name_raw='{name_raw}' name_norm='{name_norm}' "
                    f"repr={name_repr} cps={cps} m_unique={nm} ids={nr} invalid={inv} sample={ids_sample}"
                )
            return _truncate(parts, cfg.explain_sets_debug_max_sets_per_sig)

        data.append({
            "kind": kind,
            "sig_hex": sig_hex,
            "ref_count": int(rc),
            "rt_count": int(pc),
            "missing": int(miss),
            "extra": int(extra),
            "ref_sets": fmt_entries(ref_entries),
            "rt_sets": fmt_entries(rt_entries),
            "ref_sets_total_for_sig": int(len(ref_entries)),
            "rt_sets_total_for_sig": int(len(rt_entries)),
        })

    for r in nodeset_name_rows:
        add("nodeset_name", r, idx_ref_nodesets_name, idx_rt_nodesets_name)
    for r in elemset_name_rows:
        add("elemset_name", r, idx_ref_elemsets_name, idx_rt_elemsets_name)

    if not data:
        return False

    pd.DataFrame(data).to_csv(out_dir / "sets_name_debug.csv", index=False, na_rep=CSV_NA_REP)
    return True


# =============================================================================
# Diagnose: bbox-scale hint + multiscale + root-cause
# =============================================================================

def bbox_scale_hint(cfg: Config, bref: Optional[Dict[str, float]], brt: Optional[Dict[str, float]]) -> Dict[str, Any]:
    if not bref or not brt:
        return {"ok": False, "reason": "no_bbox"}
    dxr, dyr, dzr = bref.get("dx", 0.0), bref.get("dy", 0.0), bref.get("dz", 0.0)
    dxt, dyt, dzt = brt.get("dx", 0.0), brt.get("dy", 0.0), brt.get("dz", 0.0)

    ratios = []
    for r, t in [(dxr, dxt), (dyr, dyt), (dzr, dzt)]:
        if r > 0 and t > 0:
            ratios.append(t / r)
    if not ratios:
        return {"ok": False, "reason": "zero_extent"}

    rmin, rmax = min(ratios), max(ratios)
    rel_spread = (rmax - rmin) / max(1e-12, abs(sum(ratios) / len(ratios)))
    mean_ratio = sum(ratios) / len(ratios)
    uniform = rel_spread <= cfg.bbox_scale_warn_tol_rel
    return {
        "ok": True,
        "ratios": ratios,
        "mean_ratio": float(mean_ratio),
        "rel_spread": float(rel_spread),
        "uniform_scale_suspected": bool(uniform and abs(mean_ratio - 1.0) > cfg.bbox_scale_warn_tol_rel),
    }


def multiscale_check_f1(cfg: Config, ref_model: Dict[str, Any], rt_model: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in cfg.multiscale_eps_factors or [1.0]:
        eps = cfg.eps_coord * float(f)

        nk_ref_mesh, _, _, _, _, _ = build_node_key_maps_and_xyz(ref_model, eps)
        nk_rt_mesh, _, _, _, _, _ = build_node_key_maps_and_xyz(rt_model, eps)

        if cfg.node_basis == "connected":
            a_nodes = connected_node_multiset(ref_model, nk_ref_mesh)
            b_nodes = connected_node_multiset(rt_model, nk_rt_mesh)
        else:
            a_nodes = node_multiset_array(nk_ref_mesh)
            b_nodes = node_multiset_array(nk_rt_mesh)

        a_elems, _ = build_element_signatures(ref_model, nk_ref_mesh, order_invariant=True, include_type=True)
        b_elems, _ = build_element_signatures(rt_model, nk_rt_mesh, order_invariant=True, include_type=True)

        _, _, nf1 = multiset_prf(a_nodes, b_nodes)
        _, _, ef1 = multiset_prf(a_elems, b_elems)

        out[str(float(f))] = {"eps": eps, "node_f1": float(nf1), "elem_f1": float(ef1)}
    return out


def classify_root_cause(cfg: Config, diag: Dict[str, Any]) -> str:
    m = diag.get("metrics", {}) if isinstance(diag.get("metrics"), dict) else {}
    node_f1 = _safe_float(m.get("node_f1"))
    elem_f1 = _safe_float(m.get("elem_f1"))
    elem_f1_agn = _safe_float(m.get("elem_f1_type_agnostic")) if m.get("elem_f1_type_agnostic") is not None else float("nan")

    bbox_hint = diag.get("bbox_scale_hint", {}) if isinstance(diag.get("bbox_scale_hint"), dict) else {}
    if bbox_hint.get("uniform_scale_suspected") is True:
        return "unit_or_scale_suspected"

    if math.isfinite(elem_f1) and elem_f1 < 1.0 - cfg.mismatch_tol and math.isfinite(elem_f1_agn) and abs(elem_f1_agn - 1.0) <= cfg.mismatch_tol:
        return "element_type_mapping_only"

    ns_f1 = _safe_float(m.get("nodesets_mem_f1"))
    es_f1 = _safe_float(m.get("elemsets_mem_f1"))
    if abs(node_f1 - 1.0) <= cfg.mismatch_tol and abs(elem_f1 - 1.0) <= cfg.mismatch_tol:
        if (math.isfinite(ns_f1) and ns_f1 < 1.0 - cfg.mismatch_tol) or (math.isfinite(es_f1) and es_f1 < 1.0 - cfg.mismatch_tol):
            return "sets_only_loss"

    probe = diag.get("nodes_probe_summary", {}) if isinstance(diag.get("nodes_probe_summary"), dict) else {}
    if probe.get("written") is True:
        bnd = _safe_float(probe.get("boundary_suspect_rate"))
        neigh = _safe_float(probe.get("neighbor_cell_rate"))
        if (math.isfinite(node_f1) and node_f1 < 1.0 - cfg.mismatch_tol) and (bnd >= 0.4 and neigh >= 0.4):
            return "rounding_boundary_suspected"

    ms = diag.get("multiscale", {}) if isinstance(diag.get("multiscale"), dict) else {}
    if "2.0" in ms and isinstance(ms["2.0"], dict):
        nf1_2 = _safe_float(ms["2.0"].get("node_f1"))
        ef1_2 = _safe_float(ms["2.0"].get("elem_f1"))
        if (math.isfinite(nf1_2) and nf1_2 > 0.999999) and (math.isfinite(ef1_2) and ef1_2 > 0.999999):
            return "eps_too_strict_or_rounding_noise"

    return "topology_or_data_loss"


# =============================================================================
# Jacobian Sign Check (Tetra)
# =============================================================================

def _is_tet_like(elem_type: str, node_ids: List[Any]) -> bool:
    t = str(elem_type).lower()
    n = len(node_ids)
    if "tet" in t or "tetra" in t:
        return True
    return n in (4, 10)

# Paper mapping: orientation/validity monitoring (paper Section 5.6.5; Table 3).
# Because σ(e) is order-invariant, tetra orientation is additionally monitored via
# Jacobian sign statistics (negative volumes indicate potential orientation regressions).
def tet_jacobian_sign_stats(cfg: Config, feat: ModelFeatures) -> Dict[str, Any]:
    if str(cfg.jacobian_check).lower().strip() != "tet":
        return {"checked": 0, "neg": 0, "pos": 0, "degenerate": 0, "neg_frac": math.nan}

    bdiag = bbox_diag(feat.bbox)
    if bdiag is None or not math.isfinite(bdiag) or bdiag <= 0:
        tol = 0.0
    else:
        tol = float(cfg.jacobian_vol_tol_rel) * float(bdiag ** 3) * 6.0

    checked = 0
    neg = 0
    pos = 0
    deg = 0

    xyz = feat.node_xyz_mesh
    nmax = int(xyz.shape[0])

    for e in iter_mesh_elements(feat.model):
        if not isinstance(e, dict):
            continue
        if "type" not in e or "node_ids" not in e:
            continue
        nids = e.get("node_ids")
        if not isinstance(nids, list) or len(nids) < 4:
            continue
        et = str(e.get("type", ""))

        if not _is_tet_like(et, nids):
            continue

        try:
            ids4 = [int(nids[0]), int(nids[1]), int(nids[2]), int(nids[3])]
        except Exception:
            continue

        if any(i <= 0 or i > nmax for i in ids4):
            continue

        a = xyz[ids4[0] - 1, :]
        b = xyz[ids4[1] - 1, :]
        c = xyz[ids4[2] - 1, :]
        d = xyz[ids4[3] - 1, :]

        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.all(np.isfinite(c)) or not np.all(np.isfinite(d)):
            continue

        ab = b - a
        ac = c - a
        ad = d - a
        vol6 = float(np.dot(np.cross(ab, ac), ad))

        if abs(vol6) <= tol:
            deg += 1
            continue

        checked += 1
        if vol6 < 0:
            neg += 1
        else:
            pos += 1

    neg_frac = (neg / checked) if checked > 0 else math.nan
    return {"checked": int(checked), "neg": int(neg), "pos": int(pos), "degenerate": int(deg), "neg_frac": float(neg_frac) if math.isfinite(neg_frac) else math.nan}


# =============================================================================
# Feature-Build
# =============================================================================

def build_features(cfg: Config, model: Dict[str, Any]) -> ModelFeatures:
    parts, inst, refpts = count_parts_instances_refpoints(model)

    coll_count, coll_examples = detect_coord_token_collisions(model, cfg.eps_coord, max_examples=5)

    nk_mesh, nk_full, xyz_mesh, rk_by_id, rk_xyz, refnode_id_collisions = build_node_key_maps_and_xyz(model, cfg.eps_coord)

    rk_nonzero = token_nonzero_mask(rk_by_id)
    collision_mask = token_nonzero_mask(nk_mesh) & rk_nonzero

    # NodeKeys für Set-Auswertung (Policy-gesteuert)
    policy = str(cfg.set_refnode_collision_policy).lower().strip()
    node_keys_sets = np.array(nk_full, copy=True)
    if policy in {"collision", "auto"}:
        if collision_mask.any():
            node_keys_sets[collision_mask] = rk_by_id[collision_mask]
    elif policy == "ref":
        if rk_nonzero.any():
            node_keys_sets[rk_nonzero] = rk_by_id[rk_nonzero]
    else:
        # "mesh" oder unbekannt -> altes Verhalten
        node_keys_sets = nk_full

    node_all = node_multiset_array(nk_mesh)
    node_conn = connected_node_multiset(model, nk_mesh)

    elem_sigs, elem_sig_by_id = build_element_signatures(model, nk_mesh, order_invariant=True, include_type=True)
    elem_sigs_ordered, elem_sig_by_id_ordered = build_element_signatures(model, nk_mesh, order_invariant=False, include_type=True)

    rp = refpoint_keys(model, cfg.eps_coord)
    bbox = bbox_mesh_nodes(model)

    nodesets_raw, elemsets_raw = extract_all_sets(model)
    _, ns_removed = dedupe_sets(nodesets_raw, cfg.set_dedup_mode, ignore_instance=False)
    _, es_removed = dedupe_sets(elemsets_raw, cfg.set_dedup_mode, ignore_instance=False)

    er = element_node_ref_stats(model, nk_mesh)

    return ModelFeatures(
        model=model,
        node_keys_mesh=nk_mesh,
        node_keys_full=nk_full,
        node_keys_sets=node_keys_sets,
        node_xyz_mesh=xyz_mesh,

        refnode_keys_by_id=rk_by_id,
        refnode_xyz_by_id=rk_xyz,
        refnode_collision_mask=collision_mask,

        node_all=node_all,
        node_conn=node_conn,

        elem_sigs=elem_sigs,
        elem_sig_by_id=elem_sig_by_id,

        elem_sigs_ordered=elem_sigs_ordered,
        elem_sig_by_id_ordered=elem_sig_by_id_ordered,

        rp=rp,
        bbox=bbox,

        parts=parts,
        instances=inst,
        refpoints_count=refpts,

        nodesets_raw=nodesets_raw,
        elemsets_raw=elemsets_raw,

        nodesets_dedup_removed=ns_removed,
        elemsets_dedup_removed=es_removed,

        coord_token_collisions=coll_count,
        coord_token_collision_examples=coll_examples,

        elem_total_node_refs=int(er["total"]),
        elem_invalid_node_refs=int(er["invalid"]),
        elem_zero_node_tokens=int(er["zero_token"]),
        refnode_id_collisions=int(refnode_id_collisions),
    )


def pick_node_basis(cfg: Config, feat: ModelFeatures) -> np.ndarray:
    return feat.node_conn if cfg.node_basis == "connected" else feat.node_all


def compute_sets_metrics(cfg: Config, ref: ModelFeatures, rt: ModelFeatures, include_inst: bool, include_names: bool) -> Dict[str, Any]:
    ns_ref, _ = dedupe_sets(ref.nodesets_raw, cfg.set_dedup_mode, ignore_instance=not include_inst)
    es_ref, _ = dedupe_sets(ref.elemsets_raw, cfg.set_dedup_mode, ignore_instance=not include_inst)
    ns_rt, _  = dedupe_sets(rt.nodesets_raw,  cfg.set_dedup_mode, ignore_instance=not include_inst)
    es_rt, _  = dedupe_sets(rt.elemsets_raw,  cfg.set_dedup_mode, ignore_instance=not include_inst)

    ns_ref_sigs, ns_ref_invalid = nodeset_membership_sigs(ns_ref, ref.node_keys_sets, include_inst)
    ns_rt_sigs,  ns_rt_invalid  = nodeset_membership_sigs(ns_rt,  rt.node_keys_sets, include_inst)

    es_ref_sigs, es_ref_invalid = elemset_membership_sigs(es_ref, ref.elem_sig_by_id, include_inst)
    es_rt_sigs,  es_rt_invalid  = elemset_membership_sigs(es_rt,  rt.elem_sig_by_id, include_inst)

    ns_pres = multiset_intersection_count(ns_ref_sigs, ns_rt_sigs)
    es_pres = multiset_intersection_count(es_ref_sigs, es_rt_sigs)

    ns_mprec, ns_mrec, ns_mf1 = multiset_prf(ns_ref_sigs, ns_rt_sigs)
    es_mprec, es_mrec, es_mf1 = multiset_prf(es_ref_sigs, es_rt_sigs)

    ns_ref_union, _ = nodeset_union_keys(ns_ref, ref.node_keys_sets)
    ns_rt_union,  _ = nodeset_union_keys(ns_rt,  rt.node_keys_sets)

    es_ref_union, _ = elemset_union_sigs(es_ref, ref.elem_sig_by_id)
    es_rt_union,  _ = elemset_union_sigs(es_rt,  rt.elem_sig_by_id)

    ns_uprec, ns_urec, ns_uf1, ns_uj, ns_ui = set_prf(ns_ref_union, ns_rt_union)
    es_uprec, es_urec, es_uf1, es_uj, es_ui = set_prf(es_ref_union, es_rt_union)

    if include_names:
        ns_ref_name_sigs, _ = nodeset_named_membership_sigs(ns_ref, ref.node_keys_sets, include_inst)
        ns_rt_name_sigs, _  = nodeset_named_membership_sigs(ns_rt,  rt.node_keys_sets, include_inst)
        es_ref_name_sigs, _ = elemset_named_membership_sigs(es_ref, ref.elem_sig_by_id, include_inst)
        es_rt_name_sigs, _  = elemset_named_membership_sigs(es_rt,  rt.elem_sig_by_id, include_inst)
        ns_nprec, ns_nrec, ns_nf1 = multiset_prf(ns_ref_name_sigs, ns_rt_name_sigs)
        es_nprec, es_nrec, es_nf1 = multiset_prf(es_ref_name_sigs, es_rt_name_sigs)
    else:
        ns_nprec = ns_nrec = ns_nf1 = math.nan
        es_nprec = es_nrec = es_nf1 = math.nan

    return {
        "nodesets_ref_eval": ns_ref,
        "elemsets_ref_eval": es_ref,
        "nodesets_rt_eval": ns_rt,
        "elemsets_rt_eval": es_rt,
        "nodesets_preserved_exact": ns_pres,
        "nodesets_ref_total": int(ns_ref_sigs.size),
        "nodesets_mem_precision": ns_mprec,
        "nodesets_mem_recall": ns_mrec,
        "nodesets_mem_f1": ns_mf1,
        "nodesets_invalid_ids_ref": ns_ref_invalid,
        "nodesets_invalid_ids_rt": ns_rt_invalid,
        "nodesets_union_size_ref": int(ns_ref_union.size),
        "nodesets_union_size_rt": int(ns_rt_union.size),
        "nodesets_union_intersection": ns_ui,
        "nodesets_union_precision": ns_uprec,
        "nodesets_union_recall": ns_urec,
        "nodesets_union_f1": ns_uf1,
        "nodesets_union_jaccard": ns_uj,
        "elemsets_preserved_exact": es_pres,
        "elemsets_ref_total": int(es_ref_sigs.size),
        "elemsets_mem_precision": es_mprec,
        "elemsets_mem_recall": es_mrec,
        "elemsets_mem_f1": es_mf1,
        "elemsets_invalid_ids_ref": es_ref_invalid,
        "elemsets_invalid_ids_rt": es_rt_invalid,
        "elemsets_union_size_ref": int(es_ref_union.size),
        "elemsets_union_size_rt": int(es_rt_union.size),
        "elemsets_union_intersection": es_ui,
        "elemsets_union_precision": es_uprec,
        "elemsets_union_recall": es_urec,
        "elemsets_union_f1": es_uf1,
        "elemsets_union_jaccard": es_uj,

        "nodesets_name_precision": ns_nprec,
        "nodesets_name_recall": ns_nrec,
        "nodesets_name_f1": ns_nf1,
        "elemsets_name_precision": es_nprec,
        "elemsets_name_recall": es_nrec,
        "elemsets_name_f1": es_nf1,
    }


# =============================================================================
# Explain on mismatch
# =============================================================================
def write_explain_if_needed(
        cfg: Config,
        explain_dir: Path,
        tc: TestCase,
        target: str,
        ref: ModelFeatures,
        rt: ModelFeatures,
        include_inst_sets: bool,
        include_names_sets: bool,
        metrics: Dict[str, Any],
        node_arr_ref: np.ndarray,
        node_arr_rt_eval: np.ndarray,
        elem_f1_type_agnostic: float,
) -> None:
    mismatch = (
            (not math.isfinite(metrics["node_f1"])) or (not math.isfinite(metrics["elem_f1"])) or
            (abs(metrics["node_f1"] - 1.0) > cfg.mismatch_tol) or (abs(metrics["elem_f1"] - 1.0) > cfg.mismatch_tol) or
            (_is_finite(metrics.get("nodesets_mem_f1")) and abs(float(metrics["nodesets_mem_f1"]) - 1.0) > cfg.mismatch_tol) or
            (_is_finite(metrics.get("elemsets_mem_f1")) and abs(float(metrics["elemsets_mem_f1"]) - 1.0) > cfg.mismatch_tol)
    )

    if include_names_sets:
        nsn = metrics.get("nodesets_name_f1")
        esn = metrics.get("elemsets_name_f1")
        if (_is_finite(nsn) and abs(float(nsn) - 1.0) > cfg.mismatch_tol) or (_is_finite(esn) and abs(float(esn) - 1.0) > cfg.mismatch_tol):
            mismatch = True

    if not (cfg.explain_on_mismatch and mismatch):
        return

    if cfg.clean_output_dirs:
        safe_rmtree(explain_dir)
    ensure_dir(explain_dir)

    use_connected = (cfg.node_basis == "connected")
    ref_conn_mask = connected_node_mask(ref.model, int(ref.node_keys_mesh.size))
    rt_conn_mask  = connected_node_mask(rt.model,  int(rt.node_keys_mesh.size))

    idx_ref_nodes = index_nodes_by_token_hex(
        ref.model, cfg.eps_coord,
        connected_only=use_connected,
        connected_mask_opt=ref_conn_mask if use_connected else None,
        max_samples_per_token=cfg.explain_node_samples_per_token
    )
    idx_rt_nodes = index_nodes_by_token_hex(
        rt.model, cfg.eps_coord,
        connected_only=use_connected,
        connected_mask_opt=rt_conn_mask if use_connected else None,
        max_samples_per_token=cfg.explain_node_samples_per_token
    )

    idx_ref_elems = index_elements_by_sig_hex(ref.model, ref.node_keys_mesh, include_type=True, order_invariant=True, max_samples_per_sig=cfg.explain_elem_samples_per_sig)
    idx_rt_elems  = index_elements_by_sig_hex(rt.model,  rt.node_keys_mesh,  include_type=True, order_invariant=True, max_samples_per_sig=cfg.explain_elem_samples_per_sig)

    write_diff_csv_nodes(explain_dir, "nodes_diff", node_arr_ref, node_arr_rt_eval, idx_ref_nodes, idx_rt_nodes, cfg.explain_max_tokens)
    write_diff_csv_elems(explain_dir, "elems_diff", ref.elem_sigs, rt.elem_sigs, idx_ref_elems, idx_rt_elems, cfg.explain_max_tokens)

    idx_ref_elems_ord = index_elements_by_sig_hex(ref.model, ref.node_keys_mesh, include_type=True, order_invariant=False, max_samples_per_sig=cfg.explain_elem_samples_per_sig)
    idx_rt_elems_ord  = index_elements_by_sig_hex(rt.model,  rt.node_keys_mesh,  include_type=True, order_invariant=False, max_samples_per_sig=cfg.explain_elem_samples_per_sig)
    write_diff_csv_elems(explain_dir, "elems_diff_ordered", ref.elem_sigs_ordered, rt.elem_sigs_ordered, idx_ref_elems_ord, idx_rt_elems_ord, cfg.explain_max_tokens)

    if cfg.explain_write_elems_diff_agnostic:
        ref_agn = build_elem_sigs_agnostic(ref.model, ref.node_keys_mesh, order_invariant=True)
        rt_agn  = build_elem_sigs_agnostic(rt.model,  rt.node_keys_mesh,  order_invariant=True)
        idx_ref_elems_agn = index_elements_by_sig_hex(ref.model, ref.node_keys_mesh, include_type=False, order_invariant=True, max_samples_per_sig=cfg.explain_elem_samples_per_sig)
        idx_rt_elems_agn  = index_elements_by_sig_hex(rt.model,  rt.node_keys_mesh,  include_type=False, order_invariant=True, max_samples_per_sig=cfg.explain_elem_samples_per_sig)
        write_diff_csv_elems(explain_dir, "elems_diff_agnostic", ref_agn, rt_agn, idx_ref_elems_agn, idx_rt_elems_agn, cfg.explain_max_tokens)

    if cfg.explain_write_elem_types_diff:
        write_elem_types_diff_csv(explain_dir, element_type_counts(ref.model), element_type_counts(rt.model))

    ns_ref_eval = metrics["nodesets_ref_eval"]
    ns_rt_eval  = metrics["nodesets_rt_eval"]
    es_ref_eval = metrics["elemsets_ref_eval"]
    es_rt_eval  = metrics["elemsets_rt_eval"]

    ns_ref_sigs, _ = nodeset_membership_sigs(ns_ref_eval, ref.node_keys_sets, include_inst_sets)
    ns_rt_sigs,  _ = nodeset_membership_sigs(ns_rt_eval,  rt.node_keys_sets, include_inst_sets)
    es_ref_sigs, _ = elemset_membership_sigs(es_ref_eval, ref.elem_sig_by_id, include_inst_sets)
    es_rt_sigs,  _ = elemset_membership_sigs(es_rt_eval,  rt.elem_sig_by_id, include_inst_sets)

    set_rows_ns = multiset_diff_table(ns_ref_sigs, ns_rt_sigs, limit=cfg.explain_max_tokens)
    set_rows_es = multiset_diff_table(es_ref_sigs, es_rt_sigs, limit=cfg.explain_max_tokens)

    sdata = []
    for sig_hex, rc, pc, miss, extra in set_rows_ns:
        sdata.append({"kind": "nodeset", "sig_hex": sig_hex, "ref_count": rc, "rt_count": pc, "missing": miss, "extra": extra})
    for sig_hex, rc, pc, miss, extra in set_rows_es:
        sdata.append({"kind": "elemset", "sig_hex": sig_hex, "ref_count": rc, "rt_count": pc, "missing": miss, "extra": extra})
    if sdata:
        pd.DataFrame(sdata).to_csv(explain_dir / "sets_diff.csv", index=False, na_rep=CSV_NA_REP)

    set_rows_ns_n: List[Tuple[str, int, int, int, int]] = []
    set_rows_es_n: List[Tuple[str, int, int, int, int]] = []
    if include_names_sets:
        ns_ref_name_sigs, _ = nodeset_named_membership_sigs(ns_ref_eval, ref.node_keys_sets, include_inst_sets)
        ns_rt_name_sigs, _  = nodeset_named_membership_sigs(ns_rt_eval,  rt.node_keys_sets, include_inst_sets)
        es_ref_name_sigs, _ = elemset_named_membership_sigs(es_ref_eval, ref.elem_sig_by_id, include_inst_sets)
        es_rt_name_sigs, _  = elemset_named_membership_sigs(es_rt_eval,  rt.elem_sig_by_id, include_inst_sets)

        set_rows_ns_n = multiset_diff_table(ns_ref_name_sigs, ns_rt_name_sigs, limit=cfg.explain_max_tokens)
        set_rows_es_n = multiset_diff_table(es_ref_name_sigs, es_rt_name_sigs, limit=cfg.explain_max_tokens)
        sdata2 = []
        for sig_hex, rc, pc, miss, extra in set_rows_ns_n:
            sdata2.append({"kind": "nodeset_name", "sig_hex": sig_hex, "ref_count": rc, "rt_count": pc, "missing": miss, "extra": extra})
        for sig_hex, rc, pc, miss, extra in set_rows_es_n:
            sdata2.append({"kind": "elemset_name", "sig_hex": sig_hex, "ref_count": rc, "rt_count": pc, "missing": miss, "extra": extra})
        if sdata2:
            pd.DataFrame(sdata2).to_csv(explain_dir / "sets_diff_name.csv", index=False, na_rep=CSV_NA_REP)

    sets_debug_written = False
    sets_name_debug_written = False

    if cfg.explain_write_sets_debug_csv and (set_rows_ns or set_rows_es):
        idx_ref_ns_dbg = index_nodesets_by_sig_hex(cfg, ns_ref_eval, ref.node_keys_sets, include_inst_sets)
        idx_rt_ns_dbg  = index_nodesets_by_sig_hex(cfg, ns_rt_eval,  rt.node_keys_sets, include_inst_sets)
        idx_ref_es_dbg = index_elemsets_by_sig_hex(cfg, es_ref_eval, ref.elem_sig_by_id, include_inst_sets)
        idx_rt_es_dbg  = index_elemsets_by_sig_hex(cfg, es_rt_eval,  rt.elem_sig_by_id, include_inst_sets)

        sets_debug_written = write_sets_debug_csv(
            cfg, explain_dir,
            nodeset_rows=set_rows_ns,
            elemset_rows=set_rows_es,
            idx_ref_nodesets=idx_ref_ns_dbg,
            idx_rt_nodesets=idx_rt_ns_dbg,
            idx_ref_elemsets=idx_ref_es_dbg,
            idx_rt_elemsets=idx_rt_es_dbg,
        )

    if cfg.explain_write_sets_debug_csv and include_names_sets and (set_rows_ns_n or set_rows_es_n):
        idx_ref_ns_name_dbg = index_nodesets_by_name_sig_hex(cfg, ns_ref_eval, ref.node_keys_sets, include_inst_sets)
        idx_rt_ns_name_dbg  = index_nodesets_by_name_sig_hex(cfg, ns_rt_eval,  rt.node_keys_sets, include_inst_sets)
        idx_ref_es_name_dbg = index_elemsets_by_name_sig_hex(cfg, es_ref_eval, ref.elem_sig_by_id, include_inst_sets)
        idx_rt_es_name_dbg  = index_elemsets_by_name_sig_hex(cfg, es_rt_eval,  rt.elem_sig_by_id, include_inst_sets)

        sets_name_debug_written = write_sets_name_debug_csv(
            cfg, explain_dir,
            nodeset_name_rows=set_rows_ns_n,
            elemset_name_rows=set_rows_es_n,
            idx_ref_nodesets_name=idx_ref_ns_name_dbg,
            idx_rt_nodesets_name=idx_rt_ns_name_dbg,
            idx_ref_elemsets_name=idx_ref_es_name_dbg,
            idx_rt_elemsets_name=idx_rt_es_name_dbg,
        )

    nodes_probe_summary: Dict[str, Any] = {"written": False}
    if cfg.explain_neighbor_probe:
        ref_grid = build_qgrid_index(ref.model, cfg.eps_coord, use_connected, ref_conn_mask if use_connected else None, max_samples_per_cell=2)
        rt_grid  = build_qgrid_index(rt.model,  cfg.eps_coord, use_connected, rt_conn_mask  if use_connected else None, max_samples_per_cell=2)
        nodes_probe_summary = write_nodes_probe_csv(cfg, explain_dir, node_arr_ref, node_arr_rt_eval, idx_ref_nodes, idx_rt_nodes, ref_grid, rt_grid, cfg.eps_coord)

    if cfg.explain_write_diag_json:
        diag_obj: Dict[str, Any] = {
            "testcase": tc.name,
            "group": tc.group,
            "base": tc.base_name,
            "target": target,
            "eps_coord": cfg.eps_coord,
            "hash_bits": cfg.hash_bits,
            "node_basis": cfg.node_basis,
            "sets_include_instance": bool(include_inst_sets),
            "sets_include_names": bool(include_names_sets),
            "sets_refnode_collision_policy": str(cfg.set_refnode_collision_policy),
            "metrics": {
                "node_f1": float(metrics["node_f1"]),
                "elem_f1": float(metrics["elem_f1"]),
                "elem_f1_ordered": float(metrics.get("elem_f1_ordered")) if _is_finite(metrics.get("elem_f1_ordered")) else None,
                "elem_f1_type_agnostic": float(elem_f1_type_agnostic) if _is_finite(elem_f1_type_agnostic) else None,
                "nodesets_mem_f1": float(metrics.get("nodesets_mem_f1")) if _is_finite(metrics.get("nodesets_mem_f1")) else None,
                "elemsets_mem_f1": float(metrics.get("elemsets_mem_f1")) if _is_finite(metrics.get("elemsets_mem_f1")) else None,
                "nodesets_name_f1": float(metrics.get("nodesets_name_f1")) if _is_finite(metrics.get("nodesets_name_f1")) else None,
                "elemsets_name_f1": float(metrics.get("elemsets_name_f1")) if _is_finite(metrics.get("elemsets_name_f1")) else None,
                "refpoints_f1": float(metrics.get("refpoints_f1")) if _is_finite(metrics.get("refpoints_f1")) else None,
                "refpoints_as_nodes_coverage": float(metrics.get("refpoints_as_nodes_coverage")) if _is_finite(metrics.get("refpoints_as_nodes_coverage")) else None,
            },
            "counts": {
                "n_nodes_ref": int(metrics["n_nodes_ref"]),
                "n_nodes_rt": int(metrics["n_nodes_rt"]),
                "n_elems_ref": int(metrics["n_elems_ref"]),
                "n_elems_rt": int(metrics["n_elems_rt"]),
                "nodesets_ref": int(metrics["nodesets_ref"]),
                "nodesets_rt": int(metrics["nodesets_rt"]),
                "elemsets_ref": int(metrics["elemsets_ref"]),
                "elemsets_rt": int(metrics["elemsets_rt"]),
                "nodesets_dedup_removed_ref": int(ref.nodesets_dedup_removed),
                "nodesets_dedup_removed_rt": int(rt.nodesets_dedup_removed),
                "elemsets_dedup_removed_ref": int(ref.elemsets_dedup_removed),
                "elemsets_dedup_removed_rt": int(rt.elemsets_dedup_removed),

                "coord_token_collisions_ref": int(ref.coord_token_collisions),
                "coord_token_collisions_rt": int(rt.coord_token_collisions),
                "refnode_id_collisions_ref": int(ref.refnode_id_collisions),
                "refnode_id_collisions_rt": int(rt.refnode_id_collisions),
                "refnode_collision_ids_ref": int(np.count_nonzero(ref.refnode_collision_mask)),
                "refnode_collision_ids_rt": int(np.count_nonzero(rt.refnode_collision_mask)),
                "elem_total_node_refs_ref": int(ref.elem_total_node_refs),
                "elem_invalid_node_refs_ref": int(ref.elem_invalid_node_refs),
                "elem_zero_node_tokens_ref": int(ref.elem_zero_node_tokens),
                "elem_total_node_refs_rt": int(rt.elem_total_node_refs),
                "elem_invalid_node_refs_rt": int(rt.elem_invalid_node_refs),
                "elem_zero_node_tokens_rt": int(rt.elem_zero_node_tokens),
            },
            "coord_token_collision_examples_ref": ref.coord_token_collision_examples,
            "coord_token_collision_examples_rt": rt.coord_token_collision_examples,
            "nodes_probe_summary": nodes_probe_summary,
            "sets_probe_summary": {
                "membership_debug_written": bool(sets_debug_written),
                "name_debug_written": bool(sets_name_debug_written),
            },
            "elem_type_counts_ref": element_type_counts(ref.model),
            "elem_type_counts_rt": element_type_counts(rt.model),
        }

        if cfg.diag_bbox_scale_check:
            diag_obj["bbox_ref"] = ref.bbox
            diag_obj["bbox_rt"] = rt.bbox
            diag_obj["bbox_scale_hint"] = bbox_scale_hint(cfg, ref.bbox, rt.bbox)
            diag_obj["bbox_diag_ref"] = bbox_diag(ref.bbox)
            diag_obj["bbox_diag_rt"] = bbox_diag(rt.bbox)

        if cfg.multiscale_mode in {"always", "on-mismatch"}:
            diag_obj["multiscale"] = multiscale_check_f1(cfg, ref.model, rt.model)

        diag_obj["root_cause_guess"] = classify_root_cause(cfg, diag_obj)

        with (explain_dir / "diag.json").open("w", encoding="utf-8") as f:
            json.dump(diag_obj, f, indent=2, ensure_ascii=False)


# =============================================================================
# Main Evaluation
# =============================================================================

def run_all_tests(cfg: Config) -> None:
    preflight(cfg)
    rows: List[Dict[str, Any]] = []
    timing_rows: List[Dict[str, Any]] = []

    ensure_dir(cfg.eval_root)
    ensure_dir(cfg.json_ref_dir)
    ensure_dir(cfg.json_roundtrip_dir)
    ensure_dir(cfg.paper_outputs_dir)
    ensure_dir(cfg.explain_dir)

    # Reproducibility snapshot (config + environment + file fingerprints)
    _meta_path = write_run_metadata(cfg)
    if _meta_path is not None:
        log(cfg, f"[META] wrote {_meta_path}")

    cases = discover_test_cases(cfg)
    if not cases:
        raise RuntimeError("Keine Testfälle gefunden (Manifest prüfen).")

    log(cfg, "\nTestfälle:")
    for tc in cases:
        log(cfg, f"  - {tc.source_format:7s}  {tc.name}")

    rows: List[Dict[str, Any]] = []

    for idx, tc in enumerate(cases, start=1):
        log(cfg, "\n" + "=" * 80)
        log(cfg, f"[{idx}/{len(cases)}] {tc.name} (source={tc.source_format})")
        log(cfg, "=" * 80)

        group = tc.group
        base = tc.base_name

        ref_dir = cfg.json_ref_dir / group / base
        try:
            ref_json, t_ref_s_to_z88, t_ref_z88_to_json = canonicalize_to_json(
                cfg, tc.input_file, ref_dir,
                timing_rows=timing_rows,
                timing_meta={
                    "testcase": tc.name,
                    "testcase_id": group,
                    "source_format": tc.source_format,
                    "target_format": "REF",
                    "phase": "ref",
                },
                stage_prefix="ref_",
            )

            ref_model = load_json_model(ref_json)
            ref_feat = build_features(cfg, ref_model)

            node_arr_ref_all = ref_feat.node_all
            node_arr_ref_conn = ref_feat.node_conn
            elem_arr_ref = ref_feat.elem_sigs

            n_nodes_ref_all = int(node_arr_ref_all.size)
            n_nodes_ref_conn = int(node_arr_ref_conn.size)
            node_arr_ref = pick_node_basis(cfg, ref_feat)
            n_nodes_ref = int(node_arr_ref.size)
            n_elems_ref = int(elem_arr_ref.size)

            rp_ref_unique = int(token_unique(ref_feat.rp).size)

            ref_bbox_diag = bbox_diag(ref_feat.bbox)
            ref_eps_over = (cfg.eps_coord / ref_bbox_diag) if (ref_bbox_diag is not None and ref_bbox_diag > 0) else math.nan

            ref_tet_jac = tet_jacobian_sign_stats(cfg, ref_feat)

            log(
                cfg,
                f"[REF] nodes(all)={n_nodes_ref_all}, nodes(conn)={n_nodes_ref_conn}, elems={n_elems_ref}, "
                f"nodesets={len(ref_feat.nodesets_raw)}(-{ref_feat.nodesets_dedup_removed}), "
                f"elemsets={len(ref_feat.elemsets_raw)}(-{ref_feat.elemsets_dedup_removed}), "
                f"parts={ref_feat.parts}, inst={ref_feat.instances}, refpts={ref_feat.refpoints_count} "
                f"(rp_keys={int(ref_feat.rp.size)}, rp_unique={rp_ref_unique}), node_basis={cfg.node_basis}, "
                f"bbox_diag={ref_bbox_diag}, eps/bbox_diag={ref_eps_over}, "
                f"tet_jac(checked={ref_tet_jac.get('checked')}, neg_frac={ref_tet_jac.get('neg_frac')}), "
                f"coord_token_collisions={ref_feat.coord_token_collisions}, "
                f"refnode_id_collisions={ref_feat.refnode_id_collisions}, "
                f"refnode_collision_ids={int(np.count_nonzero(ref_feat.refnode_collision_mask))}, "
                f"elem_node_refs(total={ref_feat.elem_total_node_refs}, invalid={ref_feat.elem_invalid_node_refs}, zeroTok={ref_feat.elem_zero_node_tokens})"
            )

        except Exception as exc:
            rows.append({
                "testcase": tc.name,
                "testcase_id": group,
                "source_format": tc.source_format,
                "target_format": "",
                "status": "error_ref_cz88",
                "error_message": str(exc),
            })
            log(cfg, "[FEHLER] Referenz (C_Z88) fehlgeschlagen:", exc)
            continue

        t_ref_total = t_ref_s_to_z88 + t_ref_z88_to_json

        for target in cfg.targets:
            if target not in SUPPORTED_TARGETS:
                raise ValueError(f"Unbekanntes target '{target}'. Erlaubt: {SUPPORTED_TARGETS}")

            log(cfg, f"\n--- Target: {target} ---")
            status = "ok"
            err = ""

            out_t_dir = cfg.paper_outputs_dir / group / base / target
            if cfg.clean_output_dirs:
                safe_rmtree(out_t_dir)
            ensure_dir(out_t_dir)

            if target == "z88":
                t_file = out_t_dir / "z88i1.txt"
            else:
                t_file = out_t_dir / f"{base}.{FORMAT_CFG[target]['ext']}"

            rt_dir = cfg.json_roundtrip_dir / group / base / target
            explain_dir = cfg.explain_dir / group / base / target
            if cfg.explain_on_mismatch and cfg.clean_output_dirs:
                safe_rmtree(explain_dir)

            try:
                t_s_to_t, runs_s_to_t, _, _ = run_conversion_timed_with_runs(
                    cfg, tc.input_file, t_file, FORMAT_CFG[target]["cli"]
                )

                append_timing_rows(
                    timing_rows,
                    testcase=tc.name,
                    testcase_id=group,
                    source_format=tc.source_format,
                    target_format=target,
                    phase="roundtrip",
                    stage="src_to_target",
                    run_dts=runs_s_to_t,
                    warmup=cfg.warmup,
                    repeats=cfg.repeats,
                )

                rt_json, t_t_to_z88, t_z88_to_json = canonicalize_to_json(
                    cfg, t_file, rt_dir,
                    timing_rows=timing_rows,
                    timing_meta={
                        "testcase": tc.name,
                        "testcase_id": group,
                        "source_format": tc.source_format,
                        "target_format": target,
                        "phase": "roundtrip",
                    },
                    stage_prefix="rt_",
                )


                rt_model = load_json_model(rt_json)
                rt_feat = build_features(cfg, rt_model)

                node_arr_rt_all = rt_feat.node_all
                node_arr_rt_conn = rt_feat.node_conn

                node_arr_rt_raw = pick_node_basis(cfg, rt_feat)
                node_arr_rt_eval = node_arr_rt_raw
                removed_rp_tokens = 0

                if cfg.filter_refpoints_from_node_fidelity and ref_feat.rp.size > 0:
                    node_arr_rt_eval, removed_rp_tokens = filter_remove_refpoint_artifacts_from_nodes(
                        node_rt_raw=node_arr_rt_raw,
                        node_ref_mesh=node_arr_ref,
                        rp_ref=ref_feat.rp,
                    )

                node_prec, node_rec, node_f1 = multiset_prf(node_arr_ref, node_arr_rt_eval)
                elem_prec, elem_rec, elem_f1 = multiset_prf(ref_feat.elem_sigs, rt_feat.elem_sigs)

                elem_prec_o, elem_rec_o, elem_f1_ordered = multiset_prf(ref_feat.elem_sigs_ordered, rt_feat.elem_sigs_ordered)

                _dump_small_multiset_diff(cfg, node_arr_ref, node_arr_rt_eval, f"nodes:{target}")
                _dump_small_multiset_diff(cfg, ref_feat.elem_sigs, rt_feat.elem_sigs, f"elems:{target}")
                _dump_small_multiset_diff(cfg, ref_feat.elem_sigs_ordered, rt_feat.elem_sigs_ordered, f"elems_ordered:{target}")

                elem_f1_type_agnostic = math.nan
                if cfg.diag_elem_type_agnostic:
                    ref_agn = build_elem_sigs_agnostic(ref_feat.model, ref_feat.node_keys_mesh, order_invariant=True)
                    rt_agn = build_elem_sigs_agnostic(rt_feat.model, rt_feat.node_keys_mesh, order_invariant=True)
                    _, _, elem_f1_type_agnostic = multiset_prf(ref_agn, rt_agn)

                caps = FORMAT_CAPS.get(target, {"refpoints": False, "hierarchy": False, "set_names": False})
                if caps.get("refpoints", False):
                    rp_prec, rp_rec, rp_f1 = refpoints_prf(ref_feat.rp, rt_feat.rp)
                else:
                    rp_prec = rp_rec = rp_f1 = math.nan

                rp_as_nodes_cov, rp_as_nodes_inter, rp_as_nodes_rp_unique, rp_as_nodes_nodes_unique = rp_as_nodes_coverage(ref_feat.rp, node_arr_rt_all)
                rp_as_nodes_present_all = 1.0 if rp_as_nodes_cov >= 1.0 else 0.0

                rp_as_node_prec, rp_as_node_rec, rp_as_node_f1, rp_as_node_j, _ = set_prf(ref_feat.rp, node_arr_rt_all)

                include_inst_sets = include_instance_for_set_eval(cfg, tc.source_format, target)
                include_names_sets = include_set_names_for_eval(cfg, tc.source_format, target)

                setm = compute_sets_metrics(cfg, ref_feat, rt_feat, include_inst_sets, include_names_sets)

                rt_tet_jac = tet_jacobian_sign_stats(cfg, rt_feat)

                t_t_to_json_total = t_t_to_z88 + t_z88_to_json
                t_total = t_ref_total + t_s_to_t + t_t_to_json_total

                mismatch_now = (
                        (not math.isfinite(node_f1)) or (not math.isfinite(elem_f1)) or
                        (abs(node_f1 - 1.0) > cfg.mismatch_tol) or (abs(elem_f1 - 1.0) > cfg.mismatch_tol) or
                        (_is_finite(setm.get("nodesets_mem_f1")) and abs(float(setm["nodesets_mem_f1"]) - 1.0) > cfg.mismatch_tol) or
                        (_is_finite(setm.get("elemsets_mem_f1")) and abs(float(setm["elemsets_mem_f1"]) - 1.0) > cfg.mismatch_tol)
                )

                ms_out: Dict[str, Any] = {}
                if cfg.multiscale_mode == "always" or (cfg.multiscale_mode == "on-mismatch" and mismatch_now):
                    ms_out = multiscale_check_f1(cfg, ref_feat.model, rt_feat.model)

                def _ms_colname(prefix: str, factor: float) -> str:
                    s = str(factor).replace(".", "p").replace("-", "m")
                    return f"ms_{prefix}_{s}"

                ms_cols: Dict[str, Any] = {}
                for f in cfg.multiscale_eps_factors or [1.0]:
                    k = str(float(f))
                    if k in ms_out:
                        ms_cols[_ms_colname("node_f1", f)] = ms_out[k].get("node_f1", math.nan)
                        ms_cols[_ms_colname("elem_f1", f)] = ms_out[k].get("elem_f1", math.nan)
                        ms_cols[_ms_colname("eps", f)] = ms_out[k].get("eps", math.nan)
                    else:
                        ms_cols[_ms_colname("node_f1", f)] = math.nan
                        ms_cols[_ms_colname("elem_f1", f)] = math.nan
                        ms_cols[_ms_colname("eps", f)] = math.nan

                ref_bbox_diag = bbox_diag(ref_feat.bbox)
                rt_bbox_diag = bbox_diag(rt_feat.bbox)
                ref_eps_over = (cfg.eps_coord / ref_bbox_diag) if (ref_bbox_diag is not None and ref_bbox_diag > 0) else math.nan
                rt_eps_over = (cfg.eps_coord / rt_bbox_diag) if (rt_bbox_diag is not None and rt_bbox_diag > 0) else math.nan

                metrics_for_explain = {
                    "node_f1": float(node_f1),
                    "elem_f1": float(elem_f1),
                    "elem_f1_ordered": float(elem_f1_ordered),
                    "n_nodes_ref": int(node_arr_ref.size),
                    "n_nodes_rt": int(node_arr_rt_eval.size),
                    "n_elems_ref": int(ref_feat.elem_sigs.size),
                    "n_elems_rt": int(rt_feat.elem_sigs.size),
                    "nodesets_ref": int(len(ref_feat.nodesets_raw)),
                    "nodesets_rt": int(len(rt_feat.nodesets_raw)),
                    "elemsets_ref": int(len(ref_feat.elemsets_raw)),
                    "elemsets_rt": int(len(rt_feat.elemsets_raw)),
                    "refpoints_f1": rp_f1,
                    "refpoints_as_nodes_coverage": rp_as_nodes_cov,
                    "nodesets_mem_f1": setm["nodesets_mem_f1"],
                    "elemsets_mem_f1": setm["elemsets_mem_f1"],
                    "nodesets_name_f1": setm.get("nodesets_name_f1", math.nan),
                    "elemsets_name_f1": setm.get("elemsets_name_f1", math.nan),
                    "nodesets_ref_eval": setm["nodesets_ref_eval"],
                    "nodesets_rt_eval": setm["nodesets_rt_eval"],
                    "elemsets_ref_eval": setm["elemsets_ref_eval"],
                    "elemsets_rt_eval": setm["elemsets_rt_eval"],
                }

                write_explain_if_needed(
                    cfg=cfg,
                    explain_dir=explain_dir,
                    tc=tc,
                    target=target,
                    ref=ref_feat,
                    rt=rt_feat,
                    include_inst_sets=include_inst_sets,
                    include_names_sets=include_names_sets,
                    metrics=metrics_for_explain,
                    node_arr_ref=node_arr_ref,
                    node_arr_rt_eval=node_arr_rt_eval,
                    elem_f1_type_agnostic=elem_f1_type_agnostic,
                )

                row = {
                    "testcase": tc.name,
                    "testcase_id": group,
                    "source_format": tc.source_format,
                    "target_format": target,
                    "status": status,
                    "error_message": err,

                    "eps_coord": cfg.eps_coord,
                    "hash_bits": cfg.hash_bits,
                    "repeats": cfg.repeats,
                    "warmup_runs": cfg.warmup,
                    "node_fidelity_basis": cfg.node_basis,

                    "sets_include_instance": bool(include_inst_sets),
                    "sets_include_names": bool(include_names_sets),
                    "sets_refnode_collision_policy": str(cfg.set_refnode_collision_policy),

                    "t_ref_s_to_z88_s": t_ref_s_to_z88,
                    "t_ref_z88_to_json_s": t_ref_z88_to_json,
                    "t_ref_total_s": t_ref_total,

                    "t_s_to_t_s": t_s_to_t,
                    "t_t_to_z88_s": t_t_to_z88,
                    "t_z88_to_json_s": t_z88_to_json,
                    "t_t_to_json_total_s": t_t_to_json_total,
                    "t_total_s": t_total,

                    "n_nodes_ref_all": int(n_nodes_ref_all),
                    "n_nodes_ref_connected": int(n_nodes_ref_conn),
                    "n_nodes_ref": int(node_arr_ref.size),
                    "n_elems_ref": int(ref_feat.elem_sigs.size),

                    "n_nodes_rt_all": int(node_arr_rt_all.size),
                    "n_nodes_rt_connected": int(node_arr_rt_conn.size),
                    "n_nodes_rt": int(node_arr_rt_eval.size),
                    "n_elems_rt": int(rt_feat.elem_sigs.size),

                    "node_precision": node_prec,
                    "node_recall": node_rec,
                    "node_f1": node_f1,
                    "elem_precision": elem_prec,
                    "elem_recall": elem_rec,
                    "elem_f1": elem_f1,

                    "elem_precision_ordered": elem_prec_o,
                    "elem_recall_ordered": elem_rec_o,
                    "elem_f1_ordered": elem_f1_ordered,

                    "elem_f1_type_agnostic": elem_f1_type_agnostic,

                    "bbox_diag_ref": ref_bbox_diag if ref_bbox_diag is not None else math.nan,
                    "bbox_diag_rt": rt_bbox_diag if rt_bbox_diag is not None else math.nan,
                    "eps_over_bbox_diag_ref": ref_eps_over,
                    "eps_over_bbox_diag_rt": rt_eps_over,

                    "tet_jacobian_checked_ref": ref_tet_jac.get("checked", 0),
                    "tet_jacobian_neg_ref": ref_tet_jac.get("neg", 0),
                    "tet_jacobian_neg_frac_ref": ref_tet_jac.get("neg_frac", math.nan),
                    "tet_jacobian_checked_rt": rt_tet_jac.get("checked", 0),
                    "tet_jacobian_neg_rt": rt_tet_jac.get("neg", 0),
                    "tet_jacobian_neg_frac_rt": rt_tet_jac.get("neg_frac", math.nan),

                    "refpoints_ref_count": int(ref_feat.rp.size),
                    "refpoints_ref_unique": int(token_unique(ref_feat.rp).size),
                    "refpoints_rt_count": int(rt_feat.rp.size),
                    "refpoints_rt_unique": int(token_unique(rt_feat.rp).size),
                    "refpoints_precision": rp_prec,
                    "refpoints_recall": rp_rec,
                    "refpoints_f1": rp_f1,

                    "refpoints_as_nodes_coverage": rp_as_nodes_cov,
                    "refpoints_as_nodes_intersection": rp_as_nodes_inter,
                    "refpoints_as_nodes_rp_unique": rp_as_nodes_rp_unique,
                    "refpoints_as_nodes_nodes_unique": rp_as_nodes_nodes_unique,
                    "refpoints_as_nodes_present_all": rp_as_nodes_present_all,

                    "refpoints_as_nodes_precision": rp_as_node_prec,
                    "refpoints_as_nodes_recall": rp_as_node_rec,
                    "refpoints_as_nodes_f1": rp_as_node_f1,
                    "refpoints_as_nodes_jaccard": rp_as_node_j,

                    "nodesets_ref": len(ref_feat.nodesets_raw),
                    "nodesets_rt": len(rt_feat.nodesets_raw),
                    "nodesets_preserved_exact": setm["nodesets_preserved_exact"],
                    "nodesets_ref_total": setm["nodesets_ref_total"],
                    "nodesets_mem_precision": setm["nodesets_mem_precision"],
                    "nodesets_mem_recall": setm["nodesets_mem_recall"],
                    "nodesets_mem_f1": setm["nodesets_mem_f1"],
                    "nodesets_invalid_ids_ref": setm["nodesets_invalid_ids_ref"],
                    "nodesets_invalid_ids_rt": setm["nodesets_invalid_ids_rt"],
                    "nodesets_union_size_ref": setm["nodesets_union_size_ref"],
                    "nodesets_union_size_rt": setm["nodesets_union_size_rt"],
                    "nodesets_union_intersection": setm["nodesets_union_intersection"],
                    "nodesets_union_precision": setm["nodesets_union_precision"],
                    "nodesets_union_recall": setm["nodesets_union_recall"],
                    "nodesets_union_f1": setm["nodesets_union_f1"],
                    "nodesets_union_jaccard": setm["nodesets_union_jaccard"],
                    "nodesets_dedup_removed_ref": ref_feat.nodesets_dedup_removed,
                    "nodesets_dedup_removed_rt": rt_feat.nodesets_dedup_removed,

                    "elemsets_ref": len(ref_feat.elemsets_raw),
                    "elemsets_rt": len(rt_feat.elemsets_raw),
                    "elemsets_preserved_exact": setm["elemsets_preserved_exact"],
                    "elemsets_ref_total": setm["elemsets_ref_total"],
                    "elemsets_mem_precision": setm["elemsets_mem_precision"],
                    "elemsets_mem_recall": setm["elemsets_mem_recall"],
                    "elemsets_mem_f1": setm["elemsets_mem_f1"],
                    "elemsets_invalid_ids_ref": setm["elemsets_invalid_ids_ref"],
                    "elemsets_invalid_ids_rt": setm["elemsets_invalid_ids_rt"],
                    "elemsets_union_size_ref": setm["elemsets_union_size_ref"],
                    "elemsets_union_size_rt": setm["elemsets_union_size_rt"],
                    "elemsets_union_intersection": setm["elemsets_union_intersection"],
                    "elemsets_union_precision": setm["elemsets_union_precision"],
                    "elemsets_union_recall": setm["elemsets_union_recall"],
                    "elemsets_union_f1": setm["elemsets_union_f1"],
                    "elemsets_union_jaccard": setm["elemsets_union_jaccard"],
                    "elemsets_dedup_removed_ref": ref_feat.elemsets_dedup_removed,
                    "elemsets_dedup_removed_rt": rt_feat.elemsets_dedup_removed,

                    "nodesets_name_precision": setm.get("nodesets_name_precision", math.nan),
                    "nodesets_name_recall": setm.get("nodesets_name_recall", math.nan),
                    "nodesets_name_f1": setm.get("nodesets_name_f1", math.nan),
                    "elemsets_name_precision": setm.get("elemsets_name_precision", math.nan),
                    "elemsets_name_recall": setm.get("elemsets_name_recall", math.nan),
                    "elemsets_name_f1": setm.get("elemsets_name_f1", math.nan),

                    "parts_ref": ref_feat.parts,
                    "instances_ref": ref_feat.instances,
                    "parts_rt": rt_feat.parts,
                    "instances_rt": rt_feat.instances,
                    "refpoints_ref": ref_feat.refpoints_count,
                    "refpoints_rt": rt_feat.refpoints_count,

                    "removed_rp_tokens": removed_rp_tokens,

                    "coord_token_collisions_ref": ref_feat.coord_token_collisions,
                    "coord_token_collisions_rt": rt_feat.coord_token_collisions,
                    "refnode_id_collisions_ref": ref_feat.refnode_id_collisions,
                    "refnode_id_collisions_rt": rt_feat.refnode_id_collisions,
                    "refnode_collision_ids_ref": int(np.count_nonzero(ref_feat.refnode_collision_mask)),
                    "refnode_collision_ids_rt": int(np.count_nonzero(rt_feat.refnode_collision_mask)),
                    "elem_total_node_refs_ref": ref_feat.elem_total_node_refs,
                    "elem_invalid_node_refs_ref": ref_feat.elem_invalid_node_refs,
                    "elem_zero_node_tokens_ref": ref_feat.elem_zero_node_tokens,
                    "elem_total_node_refs_rt": rt_feat.elem_total_node_refs,
                    "elem_invalid_node_refs_rt": rt_feat.elem_invalid_node_refs,
                    "elem_zero_node_tokens_rt": rt_feat.elem_zero_node_tokens,
                }

                row.update(ms_cols)
                rows.append(row)

            except Exception as exc:
                status = "error_roundtrip"
                err = str(exc)
                log(cfg, "[FEHLER] Roundtrip fehlgeschlagen:", exc)

                rows.append({
                    "testcase": tc.name,
                    "testcase_id": group,
                    "source_format": tc.source_format,
                    "target_format": target,
                    "status": status,
                    "error_message": err,
                    "hash_bits": cfg.hash_bits,
                    "eps_coord": cfg.eps_coord,
                })

    df = pd.DataFrame(rows)
    out_csv = cfg.eval_root / "converter_evaluation_paper_summary.csv"
    df.to_csv(out_csv, index=False, na_rep=CSV_NA_REP)
    log(cfg, "\nFertig. CSV:", out_csv)


    # Rohzeiten extra
    if timing_rows:
        out_timing = cfg.eval_root / "timings_raw.csv"
        pd.DataFrame(timing_rows).to_csv(out_timing, index=False, na_rep=CSV_NA_REP)
        log(cfg, "Rohzeiten CSV:", out_timing)


    write_paper_extras(cfg, df)


# =============================================================================
# Paper Extra CSVs (key metrics + attention)
# =============================================================================

def write_paper_extras(cfg: Config, df: pd.DataFrame) -> None:
    PAPER_F1_TOL = 0.0

    def _is_one(x: Any) -> bool:
        try:
            if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
                return False
            return abs(float(x) - 1.0) <= PAPER_F1_TOL
        except Exception:
            return False

    def _safe_ratio(num: Any, den: Any) -> float:
        try:
            n = float(num); d = float(den)
            if d <= 0 or math.isnan(n) or math.isnan(d):
                return math.nan
            return n / d
        except Exception:
            return math.nan

    def _caps_refpoints(target: str) -> bool:
        return bool(FORMAT_CAPS.get(str(target), {}).get("refpoints", False))

    def _caps_hierarchy(target: str) -> bool:
        return bool(FORMAT_CAPS.get(str(target), {}).get("hierarchy", False))

    df2 = df.copy()
    if "testcase_id" in df2.columns:
        df2["model_id"] = df2["testcase_id"].astype(str).str.extract(r"(\d+)")[0].fillna("-1").astype(int)
    else:
        df2["model_id"] = -1

    if "nodesets_preserved_exact" in df2.columns and "nodesets_ref_total" in df2.columns:
        df2["nodesets_exact_ratio"] = [_safe_ratio(a, b) for a, b in zip(df2["nodesets_preserved_exact"], df2["nodesets_ref_total"])]
    if "elemsets_preserved_exact" in df2.columns and "elemsets_ref_total" in df2.columns:
        df2["elemsets_exact_ratio"] = [_safe_ratio(a, b) for a, b in zip(df2["elemsets_preserved_exact"], df2["elemsets_ref_total"])]

    key_cols = [
        "model_id", "testcase", "source_format", "target_format", "status",
        "hash_bits", "eps_coord", "bbox_diag_ref", "bbox_diag_rt", "eps_over_bbox_diag_ref", "eps_over_bbox_diag_rt",
        "n_nodes_ref", "n_elems_ref", "nodesets_ref", "elemsets_ref", "refpoints_ref_unique",
        "node_f1", "elem_f1", "elem_f1_ordered", "elem_f1_type_agnostic",
        "nodesets_mem_f1", "elemsets_mem_f1", "nodesets_exact_ratio", "elemsets_exact_ratio",
        "nodesets_union_f1", "elemsets_union_f1",
        "nodesets_name_f1", "elemsets_name_f1",
        "refpoints_f1", "refpoints_as_nodes_coverage",
        "tet_jacobian_checked_rt", "tet_jacobian_neg_frac_rt",
        "parts_ref", "instances_ref", "parts_rt", "instances_rt",
        "t_ref_total_s", "t_s_to_t_s", "t_t_to_json_total_s", "t_total_s",
        "sets_include_instance", "sets_include_names", "sets_refnode_collision_policy",
        "coord_token_collisions_ref", "coord_token_collisions_rt",
        "refnode_id_collisions_ref", "refnode_id_collisions_rt",
        "refnode_collision_ids_ref", "refnode_collision_ids_rt",
        "elem_total_node_refs_ref", "elem_invalid_node_refs_ref", "elem_zero_node_tokens_ref",
        "elem_total_node_refs_rt", "elem_invalid_node_refs_rt", "elem_zero_node_tokens_rt",
    ]
    for f in cfg.multiscale_eps_factors or []:
        s = str(float(f)).replace(".", "p").replace("-", "m")
        key_cols.extend([f"ms_node_f1_{s}", f"ms_elem_f1_{s}", f"ms_eps_{s}"])

    key_cols = [c for c in key_cols if c in df2.columns]

    df_key = df2[key_cols].copy()
    if all(c in df_key.columns for c in ["model_id", "source_format", "target_format"]):
        df_key.sort_values(["model_id", "source_format", "target_format"], inplace=True)

    out_key = cfg.eval_root / "paper_key_metrics.csv"
    df_key.to_csv(out_key, index=False, na_rep=CSV_NA_REP)

    def _row_assessment(r: pd.Series) -> Tuple[str, str]:
        if str(r.get("status", "")) != "ok":
            return "FAIL", "status!=ok"

        target = str(r.get("target_format", ""))
        geom_ok = _is_one(r.get("node_f1")) and _is_one(r.get("elem_f1"))

        ns_ref = int(r.get("nodesets_ref", 0)) if not pd.isna(r.get("nodesets_ref", 0)) else 0
        es_ref = int(r.get("elemsets_ref", 0)) if not pd.isna(r.get("elemsets_ref", 0)) else 0
        nodesets_ok = True if ns_ref == 0 else _is_one(r.get("nodesets_mem_f1"))
        elemsets_ok = True if es_ref == 0 else _is_one(r.get("elemsets_mem_f1"))

        hier_ok = True
        if _caps_hierarchy(target):
            hier_ok = (r.get("parts_ref") == r.get("parts_rt")) and (r.get("instances_ref") == r.get("instances_rt"))

        rp_unique = int(r.get("refpoints_ref_unique", 0)) if not pd.isna(r.get("refpoints_ref_unique", 0)) else 0
        rp_ok = True
        rp_warn = False
        if rp_unique > 0:
            if _caps_refpoints(target):
                rp_ok = _is_one(r.get("refpoints_f1"))
            else:
                cov = r.get("refpoints_as_nodes_coverage")
                try:
                    rp_warn = (not pd.isna(cov)) and (float(cov) < 1.0)
                except Exception:
                    rp_warn = True

        jac_warn = False
        try:
            nj = float(r.get("tet_jacobian_neg_frac_rt"))
            if math.isfinite(nj) and nj > 0.0:
                jac_warn = True
        except Exception:
            jac_warn = False

        name_warn = False
        try:
            if bool(r.get("sets_include_names", False)):
                nsn = r.get("nodesets_name_f1")
                esn = r.get("elemsets_name_f1")
                if _is_finite(nsn) and (abs(float(nsn) - 1.0) > PAPER_F1_TOL):
                    name_warn = True
                if _is_finite(esn) and (abs(float(esn) - 1.0) > PAPER_F1_TOL):
                    name_warn = True
        except Exception:
            name_warn = False

        reasons = []
        if not geom_ok:
            reasons.append("geom(node/elem)!=1")
        if not nodesets_ok:
            reasons.append("nodesets!=1")
        if not elemsets_ok:
            reasons.append("elemsets!=1")
        if not hier_ok:
            reasons.append("hierarchy!=preserved")
        if not rp_ok:
            reasons.append("refpoints!=1 (supported target)")
        if rp_warn:
            reasons.append("WARN: refpoints coverage<1 (unsupported target)")
        if jac_warn:
            reasons.append("WARN: tet_jacobian_neg_frac_rt>0")
        if name_warn:
            reasons.append("WARN: set_names!=1")

        if (geom_ok and nodesets_ok and elemsets_ok and hier_ok and rp_ok):
            if rp_warn or jac_warn or name_warn:
                return "WARN", "; ".join(reasons)
            return "OK", ""
        return "FAIL", "; ".join(reasons)

    assessments = df2.apply(_row_assessment, axis=1, result_type="expand")
    df2["paper_assessment"] = assessments[0]
    df2["paper_reason"] = assessments[1]

    df_attention = df2[df2["paper_assessment"] != "OK"].copy()
    att_cols = [
        "model_id", "testcase", "source_format", "target_format", "status",
        "paper_assessment", "paper_reason",
        "hash_bits", "eps_coord", "bbox_diag_ref", "eps_over_bbox_diag_ref",
        "node_f1", "elem_f1", "elem_f1_ordered", "elem_f1_type_agnostic",
        "nodesets_mem_f1", "elemsets_mem_f1",
        "nodesets_name_f1", "elemsets_name_f1",
        "refpoints_ref_unique", "refpoints_f1", "refpoints_as_nodes_coverage",
        "tet_jacobian_checked_rt", "tet_jacobian_neg_frac_rt",
        "parts_ref", "instances_ref", "parts_rt", "instances_rt",
        "t_total_s", "sets_include_instance", "sets_include_names", "sets_refnode_collision_policy",
        "coord_token_collisions_ref", "coord_token_collisions_rt",
        "refnode_id_collisions_ref", "refnode_id_collisions_rt",
        "refnode_collision_ids_ref", "refnode_collision_ids_rt",
        "elem_total_node_refs_rt", "elem_invalid_node_refs_rt", "elem_zero_node_tokens_rt",
    ]
    att_cols = [c for c in att_cols if c in df_attention.columns]
    df_attention = df_attention[att_cols]

    if all(c in df_attention.columns for c in ["paper_assessment", "model_id", "source_format", "target_format"]):
        df_attention = df_attention.sort_values(["paper_assessment", "model_id", "source_format", "target_format"])
    out_attention = cfg.eval_root / "paper_attention.csv"
    df_attention.to_csv(out_attention, index=False, na_rep=CSV_NA_REP)
    log(cfg, "\nZusatz-CSV erzeugt:")
    log(cfg, "  -", out_key)
    log(cfg, "  -", out_attention)


def make_config(
        project_root: Path = PROJECT_ROOT,
        exe_hint: str = EXE_HINT,
        tests_root: Path = TESTS_ROOT,
        eval_root: Path = EVAL_ROOT,
        targets: Optional[List[str]] = None,
        manifest_file: str = MANIFEST_FILE,
        eps_coord: float = EPS_COORD,
        hash_bits: int = HASH_BITS_CFG,
        repeats: int = REPEATS,
        warmup: int = WARMUP,
        node_basis: str = NODE_BASIS,
        clean_output_dirs: bool = CLEAN_OUTPUT_DIRS,
        verbose: bool = VERBOSE,
        use_manifest: bool = True,
) -> Config:
    project_root = Path(project_root).expanduser().resolve()
    tests_root = Path(tests_root).expanduser().resolve()
    eval_root = Path(eval_root).expanduser().resolve()
    exe_hint_s = str(exe_hint or "").strip() or None
    exe = _detect_converter_exe(project_root, exe_hint_s)
    tlist = list(targets) if targets is not None else list(TARGETS)
    tlist = [str(t).lower().strip() for t in tlist if str(t).strip()]
    if not tlist:
        tlist = list(TARGETS)
    bad = [t for t in tlist if t not in SUPPORTED_TARGETS]
    if bad:
        raise ValueError(f"Unbekannte targets: {bad}. Erlaubt: {SUPPORTED_TARGETS}")
    mf = str(manifest_file or "").strip()
    manifest_path = Path(mf).expanduser().resolve() if mf else None

    # ---------------------------------------------------------------------
    # Validate + normalise key scientific parameters (paper defaults).
    # This is defensive only; it does not change results for the manuscript defaults.
    # ---------------------------------------------------------------------
    eps_coord_f = float(eps_coord)
    if (not math.isfinite(eps_coord_f)) or eps_coord_f <= 0.0:
        raise ValueError(f"eps_coord must be a positive finite float, got: {eps_coord!r}")

    hb = int(hash_bits)
    if hb not in (64, 128):
        raise ValueError(f"hash_bits must be 64 or 128, got: {hash_bits!r}")

    rep = int(repeats)
    wu = int(warmup)
    if rep <= 0:
        raise ValueError(f"repeats must be >= 1, got: {repeats!r}")
    if wu < 0 or wu >= rep:
        raise ValueError(f"warmup must satisfy 0 <= warmup < repeats (measurement runs). Got warmup={warmup!r}, repeats={repeats!r}")

    nb = str(node_basis).lower().strip()
    if nb not in ("connected", "all"):
        raise ValueError(f"node_basis must be 'connected' or 'all', got: {node_basis!r}")

    cfg = Config(
        project_root=project_root,
        exe=exe,
        tests_root=tests_root,
        eval_root=eval_root,
        targets=tlist,
        manifest_file=manifest_path,
        eps_coord=eps_coord_f,
        hash_bits=hb,
        repeats=rep,
        warmup=wu,
        node_basis=nb,
        clean_output_dirs=bool(clean_output_dirs),
        verbose=bool(verbose),
        multiscale_mode=str(MULTISCALE_MODE),
        multiscale_eps_factors=list(MULTISCALE_EPS_FACTORS),
        set_name_metric_mode=str(SET_NAME_METRIC_MODE),
        jacobian_check=str(JACOBIAN_CHECK),
        jacobian_vol_tol_rel=float(JACOBIAN_VOL_TOL_REL),
        filter_refpoints_from_node_fidelity=bool(FILTER_REFPOINTS_FROM_NODE_FIDELITY),
        explain_on_mismatch=bool(EXPLAIN_ON_MISMATCH),
        explain_write_diag_json=bool(EXPLAIN_WRITE_DIAG_JSON),
        explain_max_tokens=int(EXPLAIN_MAX_TOKENS),
        explain_node_samples_per_token=int(EXPLAIN_NODE_SAMPLES_PER_TOKEN),
        explain_elem_samples_per_sig=int(EXPLAIN_ELEM_SAMPLES_PER_SIG),
        explain_neighbor_probe=bool(EXPLAIN_NEIGHBOR_PROBE),
        neighbor_q_radius=int(NEIGHBOR_Q_RADIUS),
        boundary_tol_rel=float(BOUNDARY_TOL_REL),
        max_node_probes_total=int(MAX_NODE_PROBES_TOTAL),
        explain_write_elems_diff_agnostic=bool(EXPLAIN_WRITE_ELEMS_DIFF_AGNOSTIC),
        explain_write_elem_types_diff=bool(EXPLAIN_WRITE_ELEM_TYPES_DIFF),
        explain_write_sets_debug_csv=bool(EXPLAIN_WRITE_SETS_DEBUG_CSV),
        explain_sets_debug_max_sets_per_sig=int(EXPLAIN_SETS_DEBUG_MAX_SETS_PER_SIG),
        explain_sets_debug_sample_ids=int(EXPLAIN_SETS_DEBUG_SAMPLE_IDS),
        set_dedup_mode=str(SET_DEDUP_MODE),
        set_instance_eval_mode=str(SET_INSTANCE_EVAL_MODE),
        set_refnode_collision_policy=str(SET_REFNODE_COLLISION_POLICY),
        diag_elem_type_agnostic=bool(DIAG_ELEM_TYPE_AGNOSTIC),
        diag_bbox_scale_check=bool(DIAG_BBOX_SCALE_CHECK),
        bbox_scale_warn_tol_rel=float(BBOX_SCALE_WARN_TOL_REL),
        dump_mismatch_examples=bool(DUMP_MISMATCH_EXAMPLES),
        mismatch_example_limit=int(MISMATCH_EXAMPLE_LIMIT),
        mismatch_tol=float(MISMATCH_TOL),

        use_manifest=bool(use_manifest),
    )

    configure_hashing(cfg.hash_bits)
    return cfg


if __name__ == "__main__":
    cfg = make_config()
    run_all_tests(cfg)