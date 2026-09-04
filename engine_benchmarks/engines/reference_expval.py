"""Exact statevector reference for one benchmark case.

Usage:
    python reference_expval.py <config.json> <result.json>

Every engine in this suite truncates (max_weight / min_abs_coeff / max_terms),
and the engines do NOT agree on what a given threshold retains. Without an
untruncated reference, a speed comparison at a nominally equal threshold is
uninterpretable: the fastest engine may simply be the one that threw away the
most terms. This worker computes the exact expectation value of the SAME
circuit and observable with no truncation at all, so every engine's result can
be reported with its own accuracy.

It uses `padopauli.pennylane_expvals` — the PennyLane cross-check the
rest of the reproducibility bundle uses — so the gate conventions are the ones
the engine itself is validated against. The reference is exact statevector
simulation and is therefore capped at `max_qubits` (20 by default).

Status values:
    Success      reference_expval is the exact value
    Unsupported  too many qubits for exact simulation (no reference available)
    Error        something went wrong; see traceback
"""
from __future__ import annotations

import sys
import time
import traceback
from typing import Any, Dict

from _common import load_config, save_result

MAX_QUBITS = 20


def run(config: Dict[str, Any], grad_check_k: int = 0) -> Dict[str, Any]:
    import numpy as np
    import torch

    from padopauli import CliffordGate, PauliRotation, PauliSum
    from padopauli import pennylane_expvals

    problem = config["problem"]
    gate_defs = config["gate_defs"]
    n_qubits = int(problem["n_qubits"])
    if n_qubits > MAX_QUBITS:
        return {
            "engine": "reference",
            "status": "Unsupported",
            "error": f"{n_qubits} qubits exceeds the exact-simulation cap ({MAX_QUBITS})",
            "n_qubits": n_qubits,
            "history": [],
        }

    # Same observable the engines are given: H = sum w_uv Z_u Z_v + sum h_q Z_q.
    obs = PauliSum(n_qubits)
    for u, v, w in problem["edges"]:
        obs.add_from_str("ZZ", float(w), qubits=[int(u), int(v)])
    for q, h in problem["fields"]:
        obs.add_from_str("Z", float(h), qubits=[int(q)])

    circuit = []
    for g in gate_defs:
        t = g["type"]
        if t == "clifford":
            circuit.append(CliffordGate(g["symbol"], list(g["qubits"])))
        elif t == "embedding":
            circuit.append(PauliRotation(
                g["pauli"], list(g["qubits"]), embedding_idx=int(g["eidx"]),
            ))
        elif t == "rotation":
            circuit.append(PauliRotation(
                g["pauli"], list(g["qubits"]), param_idx=int(g["pidx"]),
            ))
        else:
            raise ValueError(f"unknown gate type: {t}")

    # float64 throughout: this is the value the engines are scored against.
    theta = torch.tensor(config["theta"], dtype=torch.float64)
    embedding = config.get("embedding")
    embed_row = None
    if embedding is not None:
        # The engines report the expval of the FIRST batch row; match that.
        embed_row = torch.tensor(
            np.asarray(embedding, dtype=float)[:1], dtype=torch.float64,
        )

    def exact(th: "torch.Tensor") -> float:
        vals = pennylane_expvals(
            circuit=circuit, observables=[obs], thetas=th,
            embedding=embed_row, n_qubits=n_qubits, max_qubits=MAX_QUBITS,
        )
        return float(np.asarray(vals.detach().cpu()).ravel()[0])

    t0 = time.perf_counter()
    ref_expval = exact(theta)
    elapsed = time.perf_counter() - t0

    # Exact gradient for the first few parameters, by central difference.
    # The reference path is not differentiable, and a full exact gradient would
    # cost 2*n_params exact simulations per case; a handful of components is
    # enough to catch a backward pass that is wrong rather than merely slow.
    grad_indices: list = []
    grad_values: list = []
    t_grad = 0.0
    k = max(0, min(int(grad_check_k), int(theta.numel())))
    if k:
        h = 1e-4
        t0 = time.perf_counter()
        for i in range(k):
            plus = theta.clone()
            plus[i] += h
            minus = theta.clone()
            minus[i] -= h
            grad_indices.append(i)
            grad_values.append((exact(plus) - exact(minus)) / (2.0 * h))
        t_grad = time.perf_counter() - t0

    return {
        "engine": "reference",
        "status": "Success",
        "reference_expval": ref_expval,
        "reference_grad_indices": grad_indices,
        "reference_grad_values": grad_values,
        "n_qubits": n_qubits,
        "t_reference_s": elapsed,
        "t_reference_grad_s": t_grad,
        "method": "exact statevector (padopauli.pennylane_expvals)",
        "grad_method": "central difference, h=1e-4" if k else None,
        "history": [],
    }


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        return 2
    grad_check_k = int(sys.argv[3]) if len(sys.argv) == 4 else 0
    config = load_config(sys.argv)
    try:
        result = run(config, grad_check_k=grad_check_k)
    except Exception as exc:  # noqa: BLE001
        result = {
            "engine": "reference",
            "status": "Error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "history": [],
        }
    result["test_id"] = config.get("test_id")
    save_result(sys.argv, result)
    return 0 if result.get("status") in ("Success", "Unsupported") else 1


if __name__ == "__main__":
    raise SystemExit(main())
