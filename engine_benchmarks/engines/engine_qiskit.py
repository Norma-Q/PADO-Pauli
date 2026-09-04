"""Qiskit pauli-prop subprocess worker.

Usage:
    python engine_qiskit.py <config.json> <result.json>

Runs Pauli propagation through a circuit using qiskit's `pauli_prop` package
on CPU. Returns timing, RAM usage, and term counts in JSON form.

Truncation:
    - `atol` is Qiskit's per-propagation coefficient cutoff. It is related
      to PPS/Julia/CPP coefficient truncation, but it is not guaranteed to be
      numerically identical because pruning order and internal representations
      differ by backend.
    - `max_terms` maps to a hard cap on retained Pauli strings.
    - Qiskit has no native max_weight; for the mw_truncation suite the
      orchestrator passes `max_weight` for bookkeeping only — Qiskit ignores it
      and runs with a permissive max_terms instead.
"""
from __future__ import annotations

import gc
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple

from _common import (
    package_versions,
    load_config,
    read_peak_rss_mb,
    read_current_rss_mb,
    reset_peak_rss,
    save_result,
    summarize_history,
)



def _build_qiskit_circuit(n_qubits: int, gate_defs: List[Dict[str, Any]],
                           theta: List[float], embedding: List[float] | None):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_qubits)
    for g in gate_defs:
        t = g["type"]
        if t == "clifford":
            sym = g["symbol"]
            qubits = g["qubits"]
            if sym == "H":
                for q in qubits:
                    qc.h(q)
            elif sym == "S":
                for q in qubits:
                    qc.s(q)
            elif sym == "SDG":
                for q in qubits:
                    qc.sdg(q)
            elif sym == "X":
                for q in qubits:
                    qc.x(q)
            elif sym == "Y":
                for q in qubits:
                    qc.y(q)
            elif sym == "Z":
                for q in qubits:
                    qc.z(q)
            elif sym == "CX" and len(qubits) == 2:
                qc.cx(qubits[0], qubits[1])
            else:
                raise ValueError(f"Unsupported Clifford symbol {sym} for qubits {qubits}")
        elif t == "rotation":
            angle = float(theta[g["pidx"]])
            pauli = g["pauli"]
            qubits = g["qubits"]
            _apply_pauli_rotation(qc, pauli, qubits, angle)
        elif t == "embedding":
            if embedding is None:
                raise ValueError("embedding gate present but no embedding vector supplied")
            angle = float(embedding[g["eidx"]])
            pauli = g["pauli"]
            qubits = g["qubits"]
            _apply_pauli_rotation(qc, pauli, qubits, angle)
        else:
            raise ValueError(f"Unsupported gate type: {t}")
    return qc


def _apply_pauli_rotation(qc, pauli: str, qubits: List[int], angle: float) -> None:
    if pauli == "X":
        qc.rx(angle, qubits[0])
    elif pauli == "Y":
        qc.ry(angle, qubits[0])
    elif pauli == "Z":
        qc.rz(angle, qubits[0])
    elif pauli == "ZZ":
        qc.rzz(angle, qubits[0], qubits[1])
    elif pauli == "XX":
        qc.rxx(angle, qubits[0], qubits[1])
    elif pauli == "YY":
        qc.ryy(angle, qubits[0], qubits[1])
    else:
        raise ValueError(f"Unsupported Pauli rotation: {pauli}")


def _build_observable(n_qubits: int, edges: List[Tuple[int, int, float]],
                      fields: List[Tuple[int, float]]):
    from qiskit.quantum_info import SparsePauliOp
    terms: List[Tuple[str, float]] = []
    for u, v, w in edges:
        s = ["I"] * n_qubits
        s[u] = "Z"
        s[v] = "Z"
        terms.append(("".join(reversed(s)), float(w)))
    for q, h in fields:
        s = ["I"] * n_qubits
        s[q] = "Z"
        terms.append(("".join(reversed(s)), float(h)))
    if not terms:
        raise ValueError("observable has no terms")
    return SparsePauliOp.from_list(terms)


def _run_embedding_batch(config: Dict[str, Any], embedding_batch) -> Dict[str, Any]:
    from pauli_prop import (
        circuit_to_rotation_gates,
        evolve_through_cliffords,
        propagate_through_rotation_gates,
    )

    problem = config["problem"]
    gate_defs = config["gate_defs"]
    theta = config["theta"]
    n_reps = int(config.get("n_reps", 5))

    atol = float(config.get("min_abs_coeff") or 0.0)
    atol = max(atol, 1e-14)
    max_terms_cfg = config.get("max_terms")
    max_terms = 10 ** 8 if max_terms_cfg is None or max_terms_cfg <= 0 else int(max_terms_cfg)

    n_qubits = int(problem["n_qubits"])
    edges = [tuple(e) for e in problem["edges"]]
    fields = [tuple(f) for f in problem["fields"]]
    batch_size = int(embedding_batch.shape[0])

    gc.collect()
    reset_peak_rss()
    rss_start = read_peak_rss_mb()
    t0 = time.perf_counter()
    obs = _build_observable(n_qubits, edges, fields)
    compile_time = time.perf_counter() - t0
    compile_mem = max(0.0, read_peak_rss_mb() - rss_start)

    def eval_one(embed_row):
        qc = _build_qiskit_circuit(n_qubits, gate_defs, theta, embed_row)
        clifford, non_clifford = evolve_through_cliffords(qc)
        rot_gates = circuit_to_rotation_gates(non_clifford)
        evolved, trunc_err = propagate_through_rotation_gates(
            obs, rot_gates, max_terms=max_terms, atol=atol, frame="h"
        )
        evolved.paulis = evolved.paulis.evolve(clifford, frame="h")
        mask = ~evolved.paulis.x.any(axis=1)
        expval = float(evolved.coeffs[mask].real.sum())
        return evolved, expval, trunc_err

    try:
        eval_one(embedding_batch[0])
    except Exception as e:  # pragma: no cover
        return {
            "engine": "qiskit",
            "status": "Error",
            "error": f"batch warmup failed: {e}",
            "traceback": traceback.format_exc(),
            "compile_time_s": compile_time,
            "compile_mem_MB": compile_mem,
            "history": [],
        }

    history: List[Dict[str, Any]] = []
    for step in range(n_reps):
        gc.collect()
        reset_peak_rss()
        mem_baseline = read_peak_rss_mb()
        # Absolute readings alongside the windowed-HWM delta: the delta alone
        # cannot answer "how much RAM does the engine hold during the loop".
        rss_start = read_current_rss_mb()
        t0 = time.perf_counter()
        expval = float("nan")
        trunc_err = 0.0
        n_terms = 0
        for i in range(batch_size):
            evolved, sample_expval, sample_trunc_err = eval_one(embedding_batch[i])
            if i == 0:
                expval = sample_expval
            trunc_err += float(sample_trunc_err)
            n_terms = int(len(evolved))
        t_eval = time.perf_counter() - t0
        step_mem = max(0.0, read_peak_rss_mb() - mem_baseline)
        rss_end = read_current_rss_mb()
        history.append({
            "step": step,
            "t_eval_s": t_eval,
            "step_ram_MB": step_mem,
            "ram_used_start_MB": rss_start,
            "ram_used_end_MB": rss_end,
            "n_terms": n_terms,
            "expval": expval,
            "trunc_err_l1": trunc_err,
            "batch_size": batch_size,
            "throughput_evals_s": float(batch_size / t_eval) if t_eval > 0 else float("inf"),
            "native_batch": False,
        })

    summary = summarize_history(history, time_key="t_eval_s", mem_key="step_ram_MB")
    return {
        "engine": "qiskit",
        "versions": package_versions("qiskit", "pauli-prop"),
        "status": "Success",
        # No zero filter and no gradient API in qiskit pauli-prop: the term
        # count is everything propagated, and the bwd column is empty by
        # definition rather than by a failed measurement.
        "n_terms_kind": "propagated",
        "bwd_supported": False,
        "compile_time_s": compile_time,
        "compile_ram_MB": compile_mem,
        "history": history,
        "summary": summary,
        "config_echo": {
            "max_terms_used": max_terms,
            "atol_used": atol,
            "n_reps": n_reps,
            "device": "cpu",
            "memory_kind": "cpu_ram_highwater_delta_MB (VmHWM reset per step)",
            "setup_kind": "build observable; per-sample circuit/split inside batch loop",
            "batch_support": "single-sample loop",
            "truncation_note": "atol is Qiskit pauli-prop's own coefficient cutoff; not PPS build_min_abs",
        },
    }


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    from pauli_prop import (
        circuit_to_rotation_gates,
        evolve_through_cliffords,
        propagate_through_rotation_gates,
    )

    problem = config["problem"]
    gate_defs = config["gate_defs"]
    theta = config["theta"]
    embedding = config.get("embedding")
    n_reps = int(config.get("n_reps", 5))
    embedding_arr = np.asarray(embedding, dtype=np.float32) if embedding is not None else None
    if embedding_arr is not None and embedding_arr.ndim == 2:
        return _run_embedding_batch(config, embedding_arr)

    atol = float(config.get("min_abs_coeff") or 0.0)
    atol = max(atol, 1e-14)
    max_terms_cfg = config.get("max_terms")
    if max_terms_cfg is None or max_terms_cfg <= 0:
        max_terms = 10 ** 8
    else:
        max_terms = int(max_terms_cfg)

    n_qubits = int(problem["n_qubits"])
    edges = [tuple(e) for e in problem["edges"]]
    fields = [tuple(f) for f in problem["fields"]]

    gc.collect()
    reset_peak_rss()
    rss_start = read_peak_rss_mb()

    # Setup stage: build circuit + observable + Clifford split. This is not a
    # reusable PPS-style compiled program; the propagated operator is rebuilt in
    # each evaluation below.
    t0 = time.perf_counter()
    qc = _build_qiskit_circuit(n_qubits, gate_defs, theta, embedding)
    obs = _build_observable(n_qubits, edges, fields)
    clifford, non_clifford = evolve_through_cliffords(qc)
    rot_gates = circuit_to_rotation_gates(non_clifford)
    compile_time = time.perf_counter() - t0
    compile_mem = max(0.0, read_peak_rss_mb() - rss_start)

    # Warmup (first call does Rust-side allocation that we want excluded from steady)
    try:
        propagate_through_rotation_gates(
            obs, rot_gates, max_terms=max_terms, atol=atol, frame="h"
        )
    except Exception as e:  # pragma: no cover
        return {
            "engine": "qiskit",
            "status": "Error",
            "error": f"warmup failed: {e}",
            "traceback": traceback.format_exc(),
            "compile_time_s": compile_time,
            "compile_mem_MB": compile_mem,
            "history": [],
        }

    history: List[Dict[str, Any]] = []
    for step in range(n_reps):
        gc.collect()
        reset_peak_rss()
        mem_baseline = read_peak_rss_mb()
        # Absolute readings alongside the windowed-HWM delta: the delta alone
        # cannot answer "how much RAM does the engine hold during the loop".
        rss_start = read_current_rss_mb()
        t0 = time.perf_counter()
        evolved, trunc_err = propagate_through_rotation_gates(
            obs, rot_gates, max_terms=max_terms, atol=atol, frame="h"
        )
        # Apply Clifford evolution to extracted Paulis (cheap, Heisenberg)
        evolved.paulis = evolved.paulis.evolve(clifford, frame="h")
        mask = ~evolved.paulis.x.any(axis=1)
        expval = float(evolved.coeffs[mask].real.sum())
        t_eval = time.perf_counter() - t0
        step_mem = max(0.0, read_peak_rss_mb() - mem_baseline)
        rss_end = read_current_rss_mb()
        history.append({
            "step": step,
            "t_eval_s": t_eval,
            "step_ram_MB": step_mem,
            "ram_used_start_MB": rss_start,
            "ram_used_end_MB": rss_end,
            "n_terms": int(len(evolved)),
            "expval": expval,
            "trunc_err_l1": float(trunc_err),
        })

    summary = summarize_history(history, time_key="t_eval_s", mem_key="step_ram_MB")
    return {
        "engine": "qiskit",
        "versions": package_versions("qiskit", "pauli-prop"),
        "status": "Success",
        # No zero filter and no gradient API in qiskit pauli-prop: the term
        # count is everything propagated, and the bwd column is empty by
        # definition rather than by a failed measurement.
        "n_terms_kind": "propagated",
        "bwd_supported": False,
        "compile_time_s": compile_time,
        "compile_ram_MB": compile_mem,
        "history": history,
        "summary": summary,
        "config_echo": {
            "max_terms_used": max_terms,
            "atol_used": atol,
            "n_reps": n_reps,
            "device": "cpu",
            "memory_kind": "cpu_ram_highwater_delta_MB (VmHWM reset per step)",
            "setup_kind": "build QuantumCircuit/observable and split Clifford/non-Clifford",
            "truncation_note": "atol is Qiskit pauli-prop's own coefficient cutoff; not PPS build_min_abs",
        },
    }


def main(argv: List[str]) -> int:
    config = load_config(argv)
    try:
        result = run(config)
    except Exception as e:
        result = {
            "engine": "qiskit",
            "status": "Error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "history": [],
        }
    result["test_id"] = config.get("test_id")
    save_result(argv, result)
    return 0 if result.get("status") == "Success" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
