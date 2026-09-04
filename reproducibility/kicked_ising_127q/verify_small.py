#!/usr/bin/env python
"""Exact cross-check of the kicked-Ising gate/angle conventions and the
IBM Eagle heavy-hex edge construction, on a small connected heavy-hex patch.

The circuit is built from a sub-patch of the actual ``ibmeagletopology`` edge
list used at 127 qubits (with at least one degree-3 branch vertex), relabelled
to a contiguous range so an exact PennyLane statevector is feasible. This
catches an edge-list or 0/1-indexing error as well as a convention error.
Results are saved to results/verify_small_heavyhex.json.
"""
import json
from pathlib import Path

import numpy as np
import torch

from padopauli import Circuit
from padopauli import pennylane_expvals
from run_kicked_ising import IBMEAGLE_TOPOLOGY_1IDX, RZZ_ANGLE

HERE = Path(__file__).resolve().parent
# Output location is device-aware: <repo>/results_on_<DEVICE>/{exp}/...
# so the same code can run on an A100, an H100 and an MI300X without one run
# overwriting another. See reproducibility/_outdir.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _outdir import out_dir as _out_dir

RESULTS = Path(_out_dir("kicked_ising_127q", "results"))


def heavyhex_patch(node_set):
    """0-indexed sub-patch of the actual ibmeagletopology induced on `node_set`,
    relabelled to a contiguous range. Returns (n, edges, relabel)."""
    full = [(u - 1, v - 1) for (u, v) in IBMEAGLE_TOPOLOGY_1IDX]   # 0-indexed
    S = sorted(node_set)
    relabel = {q: i for i, q in enumerate(S)}
    edges = [(relabel[u], relabel[v]) for (u, v) in full if u in S and v in S]
    return len(S), edges, relabel


def build(n, edges, steps):
    qc = Circuit(n)
    for _ in range(steps):
        for q in range(n):
            qc.rx(q, param_idx=0)
        for (u, v) in edges:
            qc.rzz(u, v, param_idx=1)
    return qc


def main():
    # A small connected heavy-hex patch from the real Eagle topology.
    # 0-indexed node 4 has neighbours {3,5,15} (degree-3 branch); node 0 has {1,14}.
    node_set = {0, 1, 2, 3, 4, 5, 6, 14, 15}
    n, edges, relabel = heavyhex_patch(node_set)
    center = relabel[4]                     # the degree-3 branch vertex
    steps = 5
    qc = build(n, edges, steps)
    degree = {}
    for u, v in edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    print(f"heavy-hex patch: n={n} edges={len(edges)} center={center} "
          f"(orig qubit 4, degree {degree[center]}) steps={steps}")
    print(f"  edges (relabelled): {edges}")

    rows = []
    max_dev = 0.0
    for h in [np.pi / 32, np.pi / 8, np.pi / 4, np.pi / 3]:
        thetas = torch.tensor([h, RZZ_ANGLE], dtype=torch.float64)
        prog = qc.compile(observables=[("Z", [center])], preset="cpu")
        pado = float(prog.expval(thetas, obs_index=0))
        ref = float(np.asarray(pennylane_expvals(
            circuit=qc.gates, observables=prog.observables, thetas=thetas,
            n_qubits=n)).reshape(-1)[0])
        dev = abs(pado - ref)
        max_dev = max(max_dev, dev)
        rows.append({"h": h, "h_over_pi": h / np.pi, "pado": pado,
                     "pennylane": ref, "abs_dev": dev})
        print(f"  h={h:.5f} ({h/np.pi:.4f}π)  pado={pado:+.8f}  "
              f"pennylane={ref:+.8f}  |dev|={dev:.2e}")

    out = {
        "patch": {"orig_nodes": sorted(node_set), "n": n, "edges": edges,
                  "center_orig": 4, "center_relabelled": center,
                  "center_degree": degree[center], "trotter_steps": steps},
        "rzz_angle": RZZ_ANGLE, "max_abs_dev": max_dev, "data": rows,
        "note": "exact PennyLane statevector vs PADO-Pauli (untruncated, float64) "
                "on a real ibmeagletopology sub-patch incl. a degree-3 branch vertex",
    }
    with open(RESULTS / "verify_small_heavyhex.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  max |pado - pennylane| = {max_dev:.2e}  -> "
          f"{'PASS' if max_dev < 1e-9 else 'CHECK'}")
    print(f"  -> wrote {RESULTS / 'verify_small_heavyhex.json'}")


if __name__ == "__main__":
    main()
