"""All-SDK Pauli Propagation benchmark orchestrator.

Runs PPS, cuPauliProp, Qiskit pauli-prop, and Julia PauliPropagation.jl on
the same 2D-grid Ising / QAOA-style circuits and produces a unified JSON
report covering compile time, evaluation time, memory usage, term counts,
and computed expectation values.

Engine "julia_surrogate" is PauliPropagation.jl's compile-once/evaluate-many
surrogate (same worker script as "julia", invoked with --surrogate): the
NodePathProperties graph is built once and zerofilter!-ed as the compile stage
(so evaluate! touches only |0>-contributing paths, as in PP.jl's own surrogate
example) and re-evaluated per repetition; its final_n_terms is the post-filter
evaluation set and n_terms_propagated the pre-filter graph size. When the installed PauliPropagation.jl is v0.8.0+, the
"julia" engine additionally times its native rewindgradient as t_fwdbwd_s
(expval+grad in one paired sweep; comparable to other engines' t_eval+t_bwd).

Each engine runs in its own subprocess worker for dependency/runtime isolation:
    * GPU engines (PPS, cuPauliProp) measure GPU VRAM via pynvml.
    * CPU engines (Qiskit, Julia) measure peak RAM via /proc/self/status
      (VmHWM in Python, Sys.maxrss in Julia).

Outputs (under ./results/):
    * results.json           — full per-test, per-engine records
    * results_partial.json   — incrementally updated checkpoint
    * summary.json           — flat summary table for quick comparison
    * summary.txt            — the same table, human-readable
    * run_log.txt            — human-readable execution log
    * per-call config/result JSONs under results/_work/

The summary column named setup/compile is engine-specific setup time. Only PPS
uses a reusable `compile_program(...)` object; the other engines time the
closest non-evaluation setup they perform before repeated propagation calls.
Embedding-batch cases use native PPS batching; SDKs without native batch support
run the same batch as a loop over single-sample propagations and report eval/s.

Run from this directory:
    python benchmark_all_sdks.py [--smoke-test]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def _repo_root(start: Path) -> Path:
    """Nearest ancestor holding .git, else the parent directory.

    This bundle used to live at <repo>/reproducibility/engine_benchmarks and
    hard-coded parents[1]; it now sits directly under the repo root, and the
    A100 data copy sits beside it. Locate the root instead of counting levels.
    """
    for cand in (start, *start.parents):
        if (cand / ".git").exists():
            return cand
    return start.parent


REPO_ROOT = _repo_root(SCRIPT_DIR)
ENGINES_DIR = SCRIPT_DIR / "engines"
# Results are device-scoped: <repo>/results_on_<DEVICE>/engine_benchmarks/results.
# The same benchmark run on an A100, an H100 and an MI300X then keeps three
# separate result sets from one code tree. See _outdir.py.
sys.path.insert(0, str(SCRIPT_DIR))
from _outdir import out_dir as _out_dir  # noqa: E402

RESULTS_DIR = Path(_out_dir("engine_benchmarks", "results"))
WORK_DIR = RESULTS_DIR / "_work"

PYTHON_BIN = os.environ.get("PPS_PYTHON", sys.executable)
JULIA_BIN = os.environ.get(
    "PPS_JULIA",
    shutil.which("julia") or os.path.expanduser("~/.juliaup/bin/julia"),
)

# ---------------------------------------------------------------------------
# Problem builder
# ---------------------------------------------------------------------------

WeightedEdge = Tuple[int, int, float]
Field = Tuple[int, float]


@dataclass(frozen=True)
class IsingProblem:
    name: str
    n_qubits: int
    edges: List[WeightedEdge]
    fields: List[Field]
    meta: Dict[str, Any]


def _canonical_edge(u: int, v: int) -> Tuple[int, int]:
    uu = int(u)
    vv = int(v)
    if uu == vv:
        raise ValueError("Self-loop is not allowed.")
    return (uu, vv) if uu < vv else (vv, uu)


def build_grid2d_problem(
    *,
    rows: int,
    cols: int,
    seed: int,
    nn_mean: float = 0.0,
    nn_std: float = 1.0,
    nnn_mean: float = 0.0,
    nnn_std: float = 0.3,
    include_nnn: bool = True,
    periodic: bool = False,
    field_mean: float = 0.0,
    field_std: float = 0.1,
) -> IsingProblem:
    r = int(rows)
    c = int(cols)
    if r < 2 or c < 2:
        raise ValueError("rows and cols must be >= 2")
    if float(nn_std) <= 0.0:
        raise ValueError("nn_std must be > 0")
    if bool(include_nnn) and float(nnn_std) <= 0.0:
        raise ValueError("nnn_std must be > 0 when include_nnn is enabled")

    rng = np.random.default_rng(int(seed))

    def qid(i: int, j: int) -> int:
        return int(i) * c + int(j)

    nn_pairs = set()
    nnn_pairs = set()
    for i in range(r):
        for j in range(c):
            u = qid(i, j)
            right_j = (j + 1) % c
            down_i = (i + 1) % r
            if periodic or j + 1 < c:
                nn_pairs.add(_canonical_edge(u, qid(i, right_j)))
            if periodic or i + 1 < r:
                nn_pairs.add(_canonical_edge(u, qid(down_i, j)))

            if bool(include_nnn):
                down_right_j = (j + 1) % c
                down_left_j = (j - 1) % c
                if periodic or (i + 1 < r and j + 1 < c):
                    nnn_pairs.add(_canonical_edge(u, qid(down_i, down_right_j)))
                if periodic or (i + 1 < r and j - 1 >= 0):
                    nnn_pairs.add(_canonical_edge(u, qid(down_i, down_left_j)))

    edges: List[WeightedEdge] = []
    for (u, v) in sorted(nn_pairs):
        j = float(rng.normal(loc=float(nn_mean), scale=float(nn_std)))
        edges.append((int(u), int(v), j))
    for (u, v) in sorted(nnn_pairs):
        j = float(rng.normal(loc=float(nnn_mean), scale=float(nnn_std)))
        edges.append((int(u), int(v), j))

    n_qubits = r * c
    fields: List[Field] = []
    if float(field_std) > 0.0:
        for q in range(int(n_qubits)):
            h = float(rng.normal(loc=float(field_mean), scale=float(field_std)))
            fields.append((int(q), h))

    return IsingProblem(
        name="grid2d",
        n_qubits=int(n_qubits),
        edges=edges,
        fields=fields,
        meta={
            "seed": int(seed),
            "rows": int(r),
            "cols": int(c),
            "periodic": bool(periodic),
            "distribution": "gaussian",
            "nn_mean": float(nn_mean),
            "nn_std": float(nn_std),
            "nnn_mean": float(nnn_mean),
            "nnn_std": float(nnn_std),
            "include_nnn": bool(include_nnn),
            "field_mean": float(field_mean),
            "field_std": float(field_std),
        },
    )

# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EngineSpec:
    name: str
    kind: str  # "gpu" or "cpu"
    runner: List[str]  # command prefix
    script: Path
    extra_args: Tuple[str, ...] = ()  # appended after <config.json> <result.json>
    enabled: bool = True


def _python_runner() -> List[str]:
    return [PYTHON_BIN]


ENGINES: Dict[str, EngineSpec] = {
    "cupauliprop": EngineSpec(
        name="cupauliprop", kind="gpu",
        runner=_python_runner(),
        script=ENGINES_DIR / "engine_cpp.py",
    ),
    "pps": EngineSpec(
        name="pps", kind="gpu",
        runner=_python_runner(),
        script=ENGINES_DIR / "engine_pps.py",
    ),
    "qiskit": EngineSpec(
        name="qiskit", kind="cpu",
        runner=_python_runner(),
        script=ENGINES_DIR / "engine_qiskit.py",
    ),
    "julia": EngineSpec(
        name="julia", kind="cpu",
        runner=[JULIA_BIN],
        script=ENGINES_DIR / "engine_julia.jl",
    ),
    # PP.jl's compile-once/evaluate-many surrogate (same worker, --surrogate).
    # Skips coeff_truncation cases (no coefficient-value truncation) with an
    # explicit "Unsupported" record.
    "julia_surrogate": EngineSpec(
        name="julia_surrogate", kind="cpu",
        runner=[JULIA_BIN],
        script=ENGINES_DIR / "engine_julia.jl",
        extra_args=("--surrogate",),
    ),
}

# Not an engine: the exact statevector value each engine's expval is scored
# against. Every engine truncates, and a given threshold does NOT retain the
# same terms in every engine, so speed at a nominally equal threshold means
# nothing without the accuracy that bought it. Run per case, never timed
# against the engines.
REFERENCE_SCRIPT = ENGINES_DIR / "reference_expval.py"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(path, "w", buffering=1)  # line-buffered
        self.log(f"=== Benchmark start @ {datetime.now().isoformat()} ===")

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._fp.write(line + "\n")

    def close(self) -> None:
        self.log(f"=== Benchmark end @ {datetime.now().isoformat()} ===")
        self._fp.close()


# ---------------------------------------------------------------------------
# Problem & circuit builders (one canonical definition for all engines)
# ---------------------------------------------------------------------------

def build_problem(rows: int, cols: int, seed: int = 42) -> Any:
    return build_grid2d_problem(
        rows=rows, cols=cols, seed=seed,
        nn_mean=0.0, nn_std=1.0, nnn_mean=0.0, nnn_std=0.3,
        include_nnn=True, field_mean=0.0, field_std=0.1,
    )


def build_circuit_def(problem: Any, p_layers: int,
                       include_embedding: bool = False
                       ) -> Tuple[List[Dict[str, Any]], int, int]:
    """Generate the canonical gate sequence:
        (optional embedding pre-layer) → Hadamards → p × (ZZ → Z → X)

    Returns:
        gate_defs, n_params, n_embedding
    """
    gate_defs: List[Dict[str, Any]] = []
    n_qubits = problem.n_qubits
    eidx = 0
    if include_embedding:
        for q in range(n_qubits):
            gate_defs.append({
                "type": "embedding", "pauli": "X",
                "qubits": [q], "eidx": eidx,
            })
            eidx += 1
    for q in range(n_qubits):
        gate_defs.append({
            "type": "clifford", "symbol": "H", "qubits": [q],
        })
    pidx = 0
    for _ in range(p_layers):
        for u, v, _ in problem.edges:
            gate_defs.append({
                "type": "rotation", "pauli": "ZZ",
                "qubits": [int(u), int(v)], "pidx": pidx,
            })
            pidx += 1
        for q, _ in problem.fields:
            gate_defs.append({
                "type": "rotation", "pauli": "Z",
                "qubits": [int(q)], "pidx": pidx,
            })
            pidx += 1
        for q in range(n_qubits):
            gate_defs.append({
                "type": "rotation", "pauli": "X",
                "qubits": [int(q)], "pidx": pidx,
            })
            pidx += 1
    return gate_defs, pidx, eidx


def problem_to_dict(problem: Any) -> Dict[str, Any]:
    return {
        "n_qubits": int(problem.n_qubits),
        "edges": [[int(u), int(v), float(w)] for u, v, w in problem.edges],
        "fields": [[int(q), float(h)] for q, h in problem.fields],
    }


# ---------------------------------------------------------------------------
# Test case datatypes
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    suite: str
    label: str
    problem: Any
    gate_defs: List[Dict[str, Any]]
    theta: np.ndarray
    embedding: Optional[np.ndarray] = None
    max_weight: Optional[int] = None
    min_abs_coeff: Optional[float] = None
    max_terms: Optional[int] = None
    n_reps: int = 5
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def test_id(self) -> str:
        return f"{self.suite}__{self.label}"

    def to_config(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "suite": self.suite,
            "label": self.label,
            "problem": problem_to_dict(self.problem),
            "gate_defs": self.gate_defs,
            "theta": [float(x) for x in self.theta],
            "embedding": (
                np.asarray(self.embedding, dtype=float).tolist()
                if self.embedding is not None else None
            ),
            "max_weight": self.max_weight,
            "min_abs_coeff": self.min_abs_coeff,
            "max_terms": self.max_terms,
            "n_reps": self.n_reps,
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# Test case generators
# ---------------------------------------------------------------------------

def make_full_scaling_cases(p_layers: int, smoke: bool, seed: int = 0) -> List[TestCase]:
    sizes = [
        ("2x2", 2, 2),
        ("2x3", 2, 3),
        ("3x3", 3, 3),
        ("2x5", 2, 5),
        ("3x4", 3, 4),
    ]
    if smoke:
        sizes = sizes[:2]
    rng = np.random.default_rng(seed)
    cases: List[TestCase] = []
    for label, r, c in sizes:
        prob = build_problem(r, c)
        gate_defs, n_p, _ = build_circuit_def(prob, p_layers)
        theta = rng.standard_normal(n_p).astype(np.float32) * 0.3
        cases.append(TestCase(
            suite="full_scaling", label=label, problem=prob,
            gate_defs=gate_defs, theta=theta,
            extra={"qubits": int(prob.n_qubits), "rows": r, "cols": c},
        ))
    return cases


def make_mw_truncation_cases(p_layers: int, smoke: bool, seed: int = 0) -> List[TestCase]:
    if smoke:
        prob = build_problem(2, 3)
        mw_values = [2, 3]
        label_qubits = 6
    else:
        prob = build_problem(4, 4)
        mw_values = [2, 3, 4, 5, 6]
        label_qubits = 16
    gate_defs, n_p, _ = build_circuit_def(prob, p_layers)
    rng = np.random.default_rng(seed + 1)
    theta = rng.standard_normal(n_p).astype(np.float32) * 0.3
    cases: List[TestCase] = []
    for mw in mw_values:
        cases.append(TestCase(
            suite="mw_truncation", label=f"mw={mw}", problem=prob,
            gate_defs=gate_defs, theta=theta,
            max_weight=int(mw),
            # Mirror PPS/Julia's max_weight as Qiskit's max_terms via an
            # over-budget proxy (sum_{k=0..mw} C(n,k) * 3^k). The orchestrator
            # records both so the report can flag the proxy.
            max_terms=_max_terms_proxy(prob.n_qubits, mw),
            extra={"qubits": label_qubits, "mw": int(mw)},
        ))
    return cases


def make_coeff_truncation_cases(p_layers: int, smoke: bool, seed: int = 0) -> List[TestCase]:
    if smoke:
        prob = build_problem(2, 3)
        co_values = [5e-3]
    else:
        prob = build_problem(4, 4)
        co_values = [1e-3, 5e-3, 1e-2]
    gate_defs, n_p, _ = build_circuit_def(prob, p_layers)
    rng = np.random.default_rng(seed + 2)
    theta = rng.standard_normal(n_p).astype(np.float32) * 0.3
    cases: List[TestCase] = []
    for co in co_values:
        cases.append(TestCase(
            suite="coeff_truncation", label=f"coeff={co:g}", problem=prob,
            gate_defs=gate_defs, theta=theta,
            min_abs_coeff=float(co),
            extra={"qubits": int(prob.n_qubits), "coeff": float(co)},
        ))
    return cases


def make_embedding_batch_cases(p_layers: int, smoke: bool, seed: int = 0) -> List[TestCase]:
    prob = build_problem(3, 3)
    gate_defs, n_p, n_e = build_circuit_def(prob, max(1, min(p_layers, 2)), include_embedding=True)
    batch_sizes = [10] if smoke else [10, 100, 300]
    rng = np.random.default_rng(seed + 3)
    theta = rng.standard_normal(n_p).astype(np.float32) * 0.3
    cases: List[TestCase] = []
    for bs in batch_sizes:
        embedding = rng.standard_normal((bs, n_e)).astype(np.float32) * 0.3
        cases.append(TestCase(
            suite="embedding_batch",
            label=f"bs={bs}",
            problem=prob,
            gate_defs=gate_defs,
            theta=theta,
            embedding=embedding,
            extra={
                "qubits": int(prob.n_qubits),
                "batch_size": int(bs),
                "batch_axis": "embedding_idx",
                "native_batch": {"pps": True, "cupauliprop": False, "qiskit": False,
                                 "julia": False, "julia_surrogate": False},
            },
        ))
    return cases


def _max_terms_proxy(n_qubits: int, max_weight: int) -> int:
    """Number of Pauli strings on n qubits with weight in [0, max_weight]."""
    from math import comb
    return int(sum(comb(n_qubits, k) * (3 ** k) for k in range(max_weight + 1)))


# ---------------------------------------------------------------------------
# Exact reference
# ---------------------------------------------------------------------------

def compute_reference(tc: TestCase, logger: RunLogger,
                      timeout_s: float, grad_check_k: int = 0) -> Dict[str, Any]:
    """Exact statevector expval for one case, in its own subprocess.

    Returned as a case-level field, not an engine row: it is the yardstick the
    engines are measured against, not a competitor.
    """
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = WORK_DIR / f"cfg_reference_{tc.test_id}.json"
    res_path = WORK_DIR / f"res_reference_{tc.test_id}.json"
    with open(cfg_path, "w") as f:
        json.dump(tc.to_config(), f)

    cmd = [PYTHON_BIN, str(REFERENCE_SCRIPT), str(cfg_path), str(res_path),
           str(int(grad_check_k))]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, cwd=str(REPO_ROOT))
    except subprocess.TimeoutExpired:
        logger.log(f"    ⏱ reference TIMEOUT after {timeout_s}s")
        return {"status": "Timeout", "reference_expval": None}
    if not res_path.exists():
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        logger.log("    ✗ reference produced no result file")
        for line in tail:
            logger.log(f"      stderr> {line}")
        return {"status": "NoResult", "reference_expval": None}
    with open(res_path, "r") as f:
        res = json.load(f)
    if res.get("status") == "Success":
        n_g = len(res.get("reference_grad_indices") or [])
        logger.log(f"    · reference (exact) expval={res['reference_expval']:+.6e}"
                   + (f", {n_g} exact grad components" if n_g else ""))
    else:
        logger.log(f"    · reference {res.get('status')}: {res.get('error')}")
    return res


# ---------------------------------------------------------------------------
# Engine invocation
# ---------------------------------------------------------------------------

def invoke_engine(engine: EngineSpec, tc: TestCase, logger: RunLogger,
                   timeout_s: float) -> Dict[str, Any]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = WORK_DIR / f"cfg_{engine.name}_{tc.test_id}.json"
    res_path = WORK_DIR / f"res_{engine.name}_{tc.test_id}.json"
    with open(cfg_path, "w") as f:
        json.dump(tc.to_config(), f)

    cmd = [*engine.runner, str(engine.script), str(cfg_path), str(res_path),
           *engine.extra_args]
    wall_start = time.perf_counter()
    logger.log(f"  → {engine.name} | {tc.test_id} ... ")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            cwd=str(REPO_ROOT),
        )
        wall = time.perf_counter() - wall_start
        if proc.returncode != 0:
            logger.log(
                f"    ✗ {engine.name} returncode={proc.returncode} "
                f"(wall {wall:.2f}s)"
            )
            stderr_tail = (proc.stderr or "").strip().splitlines()[-15:]
            for line in stderr_tail:
                logger.log(f"      stderr> {line}")
        else:
            logger.log(f"    ✓ {engine.name} (wall {wall:.2f}s)")
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - wall_start
        logger.log(f"    ⏱ {engine.name} TIMEOUT after {wall:.1f}s")
        return {
            "engine": engine.name,
            "test_id": tc.test_id,
            "status": "Timeout",
            "wall_s": wall,
            "error": f"timed out after {timeout_s}s",
            "history": [],
        }
    except FileNotFoundError as e:
        logger.log(f"    ✗ {engine.name} BIN MISSING: {e}")
        return {
            "engine": engine.name,
            "test_id": tc.test_id,
            "status": "MissingBinary",
            "error": str(e),
            "history": [],
        }

    # Read result file
    if res_path.exists():
        try:
            with open(res_path, "r") as f:
                result = json.load(f)
        except Exception as e:
            logger.log(f"    ✗ {engine.name} bad result JSON: {e}")
            result = {
                "engine": engine.name,
                "status": "BadJSON",
                "error": str(e),
            }
    else:
        result = {
            "engine": engine.name,
            "status": "NoResult",
            "error": "result file not produced by worker",
        }

    result["test_id"] = tc.test_id
    result["wall_s"] = float(wall)
    result["cmd"] = " ".join(cmd)
    if proc.returncode != 0:
        if "stderr_tail" not in result:
            result["stderr_tail"] = (proc.stderr or "").strip().splitlines()[-30:]
        if "status" not in result or result.get("status") == "Success":
            result["status"] = "Error"
    return result


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _terms_cell(res: Dict[str, Any], s: Dict[str, Any]) -> str:
    """Term count as 'propagated→final' when the engine zero-filters.

    PADO and the PP.jl surrogate evaluate a filtered subset; everyone else
    evaluates everything they propagated. Printing one number for both hides a
    factor of ~10 (41 vs 693 at mw=3 is the same physics counted twice over).
    """
    final = s.get("final_n_terms")
    prop = res.get("n_terms_propagated")
    if final is None:
        return "—"
    if res.get("n_terms_kind") != "post_zero_filter":
        return f"{int(final)}"
    if prop is None:
        return f"?→{int(final)}"
    return f"{int(prop)}→{int(final)}"


def _abs_err(expval: Any, reference: Any) -> Any:
    """|engine expval - exact expval|, or None when either side is missing."""
    if expval is None or reference is None:
        return None
    try:
        err = abs(float(expval) - float(reference))
    except (TypeError, ValueError):
        return None
    return None if err != err else err  # NaN in -> None out


def _grad_abs_err(res: Dict[str, Any], reference: Dict[str, Any]) -> Any:
    """Largest deviation of an engine's gradient from the exact one.

    Only the components the reference actually checked are compared. None when
    either side has nothing to compare — an engine without a gradient API, or
    a run with the gradient check disabled.
    """
    grad = res.get("grad_row0")
    idx = (reference or {}).get("reference_grad_indices") or []
    vals = (reference or {}).get("reference_grad_values") or []
    if not grad or not idx:
        return None
    worst = 0.0
    for i, exact in zip(idx, vals):
        if i >= len(grad):
            return None
        worst = max(worst, abs(float(grad[i]) - float(exact)))
    return worst


def _occupancy_from_history(res: Dict[str, Any], key: str) -> Any:
    """Mean absolute end-of-rep reading over all reps, from the raw history."""
    h = res.get("history") or []
    vals = [float(x[key]) for x in h if isinstance(x, dict) and x.get(key) is not None]
    return (sum(vals) / len(vals)) if len(vals) == len(h) and vals else None


def _flat_summary_row(tc: TestCase, engine_name: str, res: Dict[str, Any],
                      reference_expval: Any = None,
                      reference: Dict[str, Any] | None = None) -> Dict[str, Any]:
    s = res.get("summary") or {}
    return {
        "suite": tc.suite,
        "label": tc.label,
        "engine": engine_name,
        "engine_kind": ENGINES[engine_name].kind,
        "qubits": tc.extra.get("qubits"),
        "max_weight": tc.max_weight,
        "min_abs_coeff": tc.min_abs_coeff,
        "max_terms": tc.max_terms,
        "status": res.get("status"),
        "compile_time_s": res.get("compile_time_s"),
        "compile_vram_MB": res.get("compile_vram_MB"),
        "compile_ram_MB": res.get("compile_ram_MB"),
        "setup_kind": (res.get("config_echo") or {}).get("setup_kind"),
        "t_eval_steady_s": s.get("t_eval_s_steady"),
        "t_eval_first_s": s.get("t_eval_s_first"),
        "t_bwd_steady_s": s.get("t_bwd_s_steady"),
        "t_bwd_first_s": s.get("t_bwd_s_first"),
        # Julia rewindgradient: expval+grad in ONE paired sweep. Comparable to
        # other engines' t_eval + t_bwd, not to t_bwd alone.
        "t_fwdbwd_steady_s": s.get("t_fwdbwd_s_steady"),
        "t_fwdbwd_first_s": s.get("t_fwdbwd_s_first"),
        # The case knows its own batch size even when the engine produced no
        # summary — a timed-out row must still say which case timed out.
        "batch_size": (s.get("batch_size") if s.get("batch_size") is not None
                       else tc.extra.get("batch_size")),
        "batch_throughput_evals_s": s.get("batch_throughput_evals_s_steady"),
        "batch_throughput_train_s": s.get("batch_throughput_train_s_steady"),
        "native_batch": s.get("native_batch"),
        "vram_steady_MB": s.get("step_vram_MB_steady"),
        "vram_first_MB": s.get("step_vram_MB_first"),
        # Mean absolute occupancy over reps 0-4 (see summarize_history). A
        # worker that builds its own summary (the Julia one) may omit the
        # mean; fall back to computing it from the recorded history.
        "vram_occupancy_MB": (s.get("vram_used_end_MB_mean")
                              or _occupancy_from_history(res, "vram_used_end_MB")),
        "ram_occupancy_MB": (s.get("ram_used_end_MB_mean")
                             or _occupancy_from_history(res, "ram_used_end_MB")),
        "ram_steady_MB": s.get("step_ram_MB_steady"),
        "ram_first_MB": s.get("step_ram_MB_first"),
        # Term counts are NOT the same quantity in every engine: PPS and the
        # PP.jl surrogate report the post-zero-filter set they actually
        # evaluate, while cuPauliProp / Qiskit / plain PP.jl report everything
        # they propagated (no zero filter exists there). n_terms_kind says
        # which one final_n_terms is, so the two are never silently compared.
        "n_terms_propagated": res.get("n_terms_propagated"),
        "final_n_terms": s.get("final_n_terms"),
        "n_terms_kind": res.get("n_terms_kind"),
        "final_expval": s.get("final_expval"),
        # Accuracy, without which the timings above are not comparable across
        # engines: each engine's truncation retains a different set of terms.
        "reference_expval": reference_expval,
        "abs_err": _abs_err(s.get("final_expval"), reference_expval),
        "bwd_supported": res.get("bwd_supported"),
        # The backward pass was previously timed but never checked: every
        # engine computed a gradient and discarded it. Now each keeps row 0's
        # and it is scored against exact central differences.
        "grad_abs_err": _grad_abs_err(res, reference or {}),
        "grad_kind": res.get("grad_kind"),
        "wall_s": res.get("wall_s"),
    }


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _fmt_num(x: Any, kind: str) -> str:
    if x is None:
        return "—"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(f):
        return "nan"
    if kind == "time":
        return f"{f:.4f}s"
    if kind == "mem":
        return f"{f:.2f}MB"
    if kind == "ev":
        return f"{f:+.5e}"
    if kind == "int":
        return f"{int(f):d}"
    if kind == "rate":
        return f"{f:.2f}/s"
    return f"{f:g}"


def _format_summary_table(results: Dict[str, List[Dict[str, Any]]],
                          engines: List[str]) -> str:
    lines: List[str] = []
    for suite_name, entries in results.items():
        if not entries:
            continue
        lines.append(f"=== Summary: {suite_name} ===")
        header = (
            f"{'label':<22} {'engine':<12} {'kind':<4} {'status':<10} "
            f"{'setup':<10} {'eval(steady)':<14} {'bwd(steady)':<14} "
            f"{'fwdbwd(steady)':<15} "
            f"{'vram(MB)':<10} {'ram(MB)':<10} {'occ(MB)':<10} "
            f"{'throughput':<14} {'n_terms(prop→final)':<22} "
            f"{'expval':<14} {'abs_err':<12} {'grad_err':<12}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for entry in entries:
            ref = entry.get("reference_expval")
            lines.append(
                f"  exact reference: "
                + (f"{ref:+.6e}" if isinstance(ref, (int, float)) else "unavailable")
            )
            for engine in engines:
                res = entry["engines"].get(engine, {})
                s = res.get("summary", {})
                lines.append(
                    f"{entry['label']:<22} {engine:<12} "
                    f"{ENGINES[engine].kind:<4} "
                    f"{str(res.get('status','—')):<10} "
                    f"{_fmt_num(res.get('compile_time_s'),'time'):<10} "
                    f"{_fmt_num(s.get('t_eval_s_steady'),'time'):<14} "
                    f"{_fmt_num(s.get('t_bwd_s_steady'),'time'):<14} "
                    f"{_fmt_num(s.get('t_fwdbwd_s_steady'),'time'):<15} "
                    f"{_fmt_num(s.get('step_vram_MB_steady'),'mem'):<10} "
                    f"{_fmt_num(s.get('step_ram_MB_steady'),'mem'):<10} "
                    f"{_fmt_num(s.get('vram_used_end_MB_mean', s.get('ram_used_end_MB_mean')),'mem'):<10} "
                    f"{_fmt_num(s.get('batch_throughput_evals_s_steady'),'rate'):<14} "
                    f"{_terms_cell(res, s):<22} "
                    f"{_fmt_num(s.get('final_expval'),'ev'):<14} "
                    f"{_fmt_num(_abs_err(s.get('final_expval'), ref),'ev'):<12} "
                    f"{_fmt_num(_grad_abs_err(res, entry.get('reference') or {}),'ev'):<12}"
                )
            lines.append("")
    return "\n".join(lines)


def _write_summary_txt(path: Path, results: Dict[str, List[Dict[str, Any]]],
                       engines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _format_summary_table(results, engines)
    with open(path, "w") as f:
        f.write(text + "\n")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny configuration for quick correctness check")
    parser.add_argument("--engines", type=str,
                        default="cupauliprop,pps,qiskit,julia,julia_surrogate",
                        help="Comma-separated subset of engines to run")
    parser.add_argument("--suites", type=str,
                        default="full_scaling,mw_truncation,coeff_truncation,embedding_batch",
                        help="Comma-separated subset of suites to run")
    parser.add_argument("--p-layers", type=int, default=3,
                        help="Number of QAOA layers")
    parser.add_argument("--n-reps", type=int, default=5,
                        help="Steady-state evaluation repetitions per case")
    parser.add_argument("--timeout-cpu", type=float, default=900.0,
                        help="Per-engine wall-time limit for CPU engines (s)")
    parser.add_argument("--timeout-gpu", type=float, default=600.0,
                        help="Per-engine wall-time limit for GPU engines (s)")
    parser.add_argument("--timeout-reference", type=float, default=1800.0,
                        help="Wall-time limit for the exact reference run (s)")
    parser.add_argument("--grad-check", type=int, default=4,
                        help="Exact gradient components (central difference) to "
                             "score each engine's backward pass against. 0 "
                             "disables. Costs 2*K exact simulations per case.")
    parser.add_argument("--no-reference", action="store_true",
                        help="Skip the exact statevector reference. The engines' "
                             "timings then carry no accuracy, and truncated "
                             "suites become uninterpretable — for quick "
                             "smoke runs only.")
    parser.add_argument("--keep-work", action="store_true",
                        help="Keep _work/ intermediate JSONs (default: keep)")
    parser.add_argument("--no-keep-work", dest="keep_work", action="store_false")
    parser.set_defaults(keep_work=True)
    args = parser.parse_args()

    selected_engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    selected_suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    for e in selected_engines:
        if e not in ENGINES:
            print(f"Unknown engine: {e}", file=sys.stderr)
            return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(RESULTS_DIR / "run_log.txt")
    logger.log(f"engines: {selected_engines}")
    logger.log(f"suites:  {selected_suites}")
    logger.log(f"p_layers={args.p_layers}, n_reps={args.n_reps}, "
               f"smoke={args.smoke_test}")
    logger.log(
        "metric note: setup column is engine-specific; only PPS is a reusable "
        "padopauli compile_program compile."
    )
    logger.log(
        "truncation note: Qiskit uses max_terms/atol semantics, not PPS/CPP/Julia "
        "max_weight/build_min_abs semantics."
    )
    logger.log(
        "batch note: PPS uses native embedding batch; other SDKs loop over single-sample "
        "propagations and report throughput."
    )
    logger.log(f"results dir: {RESULTS_DIR}")

    # ---- Build all test cases up front ---------------------------------
    cases_by_suite: Dict[str, List[TestCase]] = {}
    if "full_scaling" in selected_suites:
        cases = make_full_scaling_cases(args.p_layers, args.smoke_test)
        for c in cases:
            c.n_reps = args.n_reps
        cases_by_suite["full_scaling"] = cases
    if "mw_truncation" in selected_suites:
        cases = make_mw_truncation_cases(args.p_layers, args.smoke_test)
        for c in cases:
            c.n_reps = args.n_reps
        cases_by_suite["mw_truncation"] = cases
    if "coeff_truncation" in selected_suites:
        cases = make_coeff_truncation_cases(args.p_layers, args.smoke_test)
        for c in cases:
            c.n_reps = args.n_reps
        cases_by_suite["coeff_truncation"] = cases
    if "embedding_batch" in selected_suites:
        cases = make_embedding_batch_cases(args.p_layers, args.smoke_test)
        for c in cases:
            c.n_reps = args.n_reps
        cases_by_suite["embedding_batch"] = cases

    total_cases = sum(len(v) for v in cases_by_suite.values())
    logger.log(f"total test cases: {total_cases} × engines {len(selected_engines)} "
               f"= {total_cases * len(selected_engines)} runs")

    # ---- Storage ------------------------------------------------------
    results: Dict[str, List[Dict[str, Any]]] = {s: [] for s in cases_by_suite}
    flat_rows: List[Dict[str, Any]] = []
    had_engine_error = False

    def _save_partial() -> None:
        _save_json(RESULTS_DIR / "results_partial.json", {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "engines": selected_engines,
                "p_layers": args.p_layers,
                "n_reps": args.n_reps,
                "smoke_test": args.smoke_test,
            },
            "results": results,
            "summary_rows": flat_rows,
        })

    # ---- Run engines suite-by-suite -----------------------------------
    # Order: GPU engines first (less RAM pressure later for CPU runs)
    engine_run_order = [e for e in selected_engines if ENGINES[e].kind == "gpu"]
    engine_run_order += [e for e in selected_engines if ENGINES[e].kind == "cpu"]

    # An engine that times out on one case of a suite times out harder on the
    # bigger ones (the suites grow monotonically: batch size, weight, size).
    # Running them anyway burns a full timeout each and yields no data, so the
    # rest of that suite is recorded as SkippedAfterTimeout instead.
    timed_out: set = set()

    for suite_name in selected_suites:
        cases = cases_by_suite.get(suite_name, [])
        if not cases:
            continue
        logger.log(f"\n=== SUITE: {suite_name} ({len(cases)} cases) ===")

        for tc in cases:
            entry: Dict[str, Any] = {
                "suite": suite_name,
                "label": tc.label,
                "test_id": tc.test_id,
                "qubits": tc.extra.get("qubits"),
                "max_weight": tc.max_weight,
                "min_abs_coeff": tc.min_abs_coeff,
                "max_terms": tc.max_terms,
                "engines": {},
            }

            # Exact value first: it is what makes the timings below comparable.
            ref_expval = None
            ref_res: Dict[str, Any] = {}
            if not args.no_reference:
                ref_res = compute_reference(tc, logger,
                                            timeout_s=args.timeout_reference,
                                            grad_check_k=args.grad_check)
                entry["reference"] = ref_res
                ref_expval = ref_res.get("reference_expval")
            entry["reference_expval"] = ref_expval

            for engine_name in engine_run_order:
                eng = ENGINES[engine_name]
                timeout_s = args.timeout_gpu if eng.kind == "gpu" else args.timeout_cpu
                if (engine_name, suite_name) in timed_out:
                    logger.log(f"  ⏭ {engine_name} | {tc.test_id} "
                               f"skipped (timed out earlier in this suite)")
                    res = {
                        "engine": engine_name,
                        "test_id": tc.test_id,
                        "status": "SkippedAfterTimeout",
                        "error": "an earlier, smaller case in this suite timed out",
                        "history": [],
                    }
                else:
                    res = invoke_engine(eng, tc, logger, timeout_s=timeout_s)
                    if res.get("status") == "Timeout":
                        timed_out.add((engine_name, suite_name))
                # "Unsupported" is a deliberate per-engine skip (e.g. the PP.jl
                # surrogate cannot truncate on coefficient values), not a failure.
                # A timeout and the skips that follow it are a recorded outcome
                # (the engine cannot do this case in the budget), not an error.
                if res.get("status") not in ("Success", "Unsupported",
                                             "Timeout", "SkippedAfterTimeout"):
                    had_engine_error = True
                entry["engines"][engine_name] = res
                flat_rows.append(_flat_summary_row(tc, engine_name, res,
                                                   reference_expval=ref_expval,
                                                   reference=ref_res))
                _save_partial()
            results[suite_name].append(entry)
            _save_partial()

    # ---- Finalize ----------------------------------------------------
    _save_json(RESULTS_DIR / "results.json", {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "engines": selected_engines,
            "p_layers": args.p_layers,
            "n_reps": args.n_reps,
            "smoke_test": args.smoke_test,
        },
        "results": results,
    })
    _save_json(RESULTS_DIR / "summary.json", {
        "rows": flat_rows,
        "columns": [
            "suite", "label", "engine", "engine_kind", "qubits",
            "max_weight", "min_abs_coeff", "max_terms",
            "status", "compile_time_s",
            "compile_vram_MB", "compile_ram_MB",
            "t_eval_steady_s", "t_eval_first_s",
            "t_bwd_steady_s", "t_bwd_first_s",
            "t_fwdbwd_steady_s", "t_fwdbwd_first_s",
            "vram_steady_MB", "vram_first_MB",
            "vram_occupancy_MB", "ram_occupancy_MB",
            "ram_steady_MB", "ram_first_MB",
            "batch_size", "batch_throughput_evals_s", "batch_throughput_train_s",
            "native_batch", "n_terms_propagated", "final_n_terms", "n_terms_kind",
            "final_expval", "reference_expval", "abs_err",
            "bwd_supported", "grad_abs_err", "grad_kind", "wall_s",
        ],
    })

    # ---- Console summary table -----------------------------------------
    _write_summary_txt(RESULTS_DIR / "summary.txt", results, selected_engines)
    logger.log("")
    logger.log(_format_summary_table(results, selected_engines))

    logger.log(f"\nWrote {RESULTS_DIR / 'results.json'}")
    logger.log(f"Wrote {RESULTS_DIR / 'summary.json'}")
    logger.log(f"Wrote {RESULTS_DIR / 'summary.txt'}")
    logger.log(f"Wrote {RESULTS_DIR / 'run_log.txt'}")

    if not args.keep_work:
        shutil.rmtree(WORK_DIR, ignore_errors=True)
        logger.log("Removed _work/")

    logger.close()
    return 1 if had_engine_error else 0


if __name__ == "__main__":
    sys.exit(main())
