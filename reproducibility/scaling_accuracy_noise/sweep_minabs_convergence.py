"""Min-abs (coefficient) truncation convergence.

Fixed 16-qubit depth-3 ER MaxCut-QAOA circuit at a fixed parameter vector.
Sweep the compile-time min-abs threshold build_min_abs (only truncation active;
max_weight left effectively unbounded) and record the expectation value
<sum_{(i,j) in E} Z_i Z_j>. As the threshold decreases the surrogate converges
to the exact PennyLane statevector value.

Run:  python sweep_minabs_convergence.py
Outputs: results/minabs_convergence.{json,csv} and figures/fig_minabs_convergence.png
         (+ companion error plot fig_minabs_error.png)
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import er_edges, build_qaoa, build_zz_observable, zz_paulisum
from padopauli import pennylane_expvals

HERE = os.path.dirname(os.path.abspath(__file__))
# Output location is device-aware: <repo>/results_on_<DEVICE>/{exp}/...
# so the same code can run on an A100, an H100 and an MI300X without one run
# overwriting another. See reproducibility/_outdir.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _outdir import out_dir as _out_dir

RESULTS = _out_dir("scaling_accuracy_noise", "results")
FIGS = _out_dir("scaling_accuracy_noise", "figures")

# Fixed instance.
N_QUBITS = 16
DEPTH = 3
EDGE_PROB = 0.3
SEED = 42
PRESET = "hybrid"
MIN_ABS_LIST = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5, 1e-6]


def run():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    edges = er_edges(N_QUBITS, EDGE_PROB, SEED)
    qc, n_params = build_qaoa(N_QUBITS, edges, DEPTH)
    obs = build_zz_observable(edges)
    # Fixed, reproducible parameter vector.
    thetas = torch.tensor(0.2 * np.sin(np.arange(n_params)), dtype=torch.float64)

    exact = float(pennylane_expvals(
        circuit=qc.gates, observables=[zz_paulisum(N_QUBITS, edges)], thetas=thetas,
        n_qubits=N_QUBITS, max_qubits=20)[0])
    print(f"[exact] <sum ZZ> = {exact:.8f}  (n={N_QUBITS}, edges={len(edges)}, params={n_params})")

    rows = []
    for eps in MIN_ABS_LIST:
        prog = qc.compile(
            observables=[obs], preset=PRESET,
            max_weight=N_QUBITS,
            build_thetas=thetas, build_min_abs=eps)
        val = float(prog.expval(thetas, obs_index=0))
        retained = int(prog.psum_union.x_mask.shape[0])
        propagated = int(prog.compile_stats.get("summary", {}).get("propagated_terms", -1))
        rows.append({
            "min_abs": eps, "expval": val, "exact": exact,
            "abs_error": abs(val - exact),
            "propagated_terms": propagated, "retained_terms": retained,
        })
        print(f"  eps={eps:.0e}  expval={val:+.6f}  err={abs(val-exact):.3e}  retained={retained}")

    with open(os.path.join(RESULTS, "minabs_convergence.json"), "w") as fh:
        json.dump({"exact": exact, "rows": rows}, fh, indent=2)
    with open(os.path.join(RESULTS, "minabs_convergence.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    _plot_convergence(rows, exact)
    _plot_error(rows)


def _plot_convergence(rows, exact):
    eps = [r["min_abs"] for r in rows]
    val = [r["expval"] for r in rows]
    plt.figure(figsize=(4.0, 2.45))
    plt.axhline(exact, color="k", ls="--", lw=1.5, label=f"exact (statevector) = {exact:.4f}")
    plt.plot(eps, val, "o-", color="tab:blue", label=r"PADO-Pauli $\langle \sum Z_iZ_j\rangle$")
    plt.xscale("log")
    plt.gca().invert_xaxis()  # coarse (large eps) -> fine (small eps), converging rightward
    plt.xlabel(r"coefficient-truncation threshold $\epsilon$ (build_min_abs)")
    plt.ylabel(r"expectation value $\langle \sum_{(i,j)\in E} Z_iZ_j\rangle$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(FIGS, "fig_minabs_convergence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()


def _plot_error(rows):
    eps = [r["min_abs"] for r in rows]
    err = [max(r["abs_error"], 1e-12) for r in rows]
    plt.figure(figsize=(4.0, 2.45))
    plt.loglog(eps, err, "s-", color="tab:red", label="absolute error vs. exact")
    plt.gca().invert_xaxis()
    plt.xlabel(r"coefficient-truncation threshold $\epsilon$ (build_min_abs)")
    plt.ylabel(r"$|\langle O\rangle_\epsilon - \langle O\rangle_{\mathrm{exact}}|$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(FIGS, "fig_minabs_error.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()



if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv:
        with open(os.path.join(RESULTS, "minabs_convergence.json")) as fh:
            d = json.load(fh)
        _plot_convergence(d["rows"], d["exact"])
        _plot_error(d["rows"])
        print("DONE (plot only)")
        raise SystemExit(0)
    run()
