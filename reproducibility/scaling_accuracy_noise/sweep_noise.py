"""Noise experiments: (1) correctness vs PennyLane density-matrix emulator,
(2) noise-assisted truncation, (3) amplitude-damping correctness.

Fixed ER MaxCut-QAOA circuit; single-qubit depolarizing noise after each layer
with per-axis probability p/3 (matching qml.DepolarizingChannel(p)). PADO-Pauli's
DepolarizingNoise(q, px, py, pz) is validated against PennyLane `default.mixed`.

Run:  python sweep_noise.py
Outputs: results/noise.json + figures/fig_noise_validation.png, fig_noise_truncation.png,
         fig_noise_ampdamp.png
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml

from _common import er_edges
from padopauli import Circuit

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

# n=10 keeps the exact density-matrix (default.mixed, 2^n x 2^n) reference fast
# enough to sweep.
N_QUBITS = 10
DEPTH = 5
EDGE_PROB = 0.5
SEED = 42
# Term counts / expvals are preset-independent; CPU preset keeps the build simple.
PRESET = "cpu"

EDGES = er_edges(N_QUBITS, EDGE_PROB, SEED)
N_PARAMS = DEPTH * (len(EDGES) + N_QUBITS)
THETAS = torch.tensor(0.2 * np.sin(np.arange(N_PARAMS)), dtype=torch.float64)

NOISE_RATES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]          # experiment 1
AD_RATES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]              # experiment 3 (amplitude damping)
EPS_LIST = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5]  # experiment 2
EPS_NEAR_EXACT = 1e-6


def pado_circuit(p_noise: float):
    """PADO-Pauli Circuit; depolarizing px=py=pz=p_noise/3 after each layer."""
    qc = Circuit(N_QUBITS)
    for q in range(N_QUBITS):
        qc.h(q)
    k = 0
    for _ in range(DEPTH):
        for (u, v) in EDGES:
            qc.rzz(u, v, param_idx=k); k += 1
        for q in range(N_QUBITS):
            qc.rx(q, param_idx=k); k += 1
        if p_noise > 0:
            pa = p_noise / 3.0
            for q in range(N_QUBITS):
                qc.depolarizing(q, pa, pa, pa)
    return qc


def pado_expval(p_noise: float, min_abs: float):
    obs = [("ZZ", [u, v]) for (u, v) in EDGES]
    qc = pado_circuit(p_noise)
    prog = qc.compile(
        observables=[obs], preset=PRESET,
        max_weight=N_QUBITS,
        build_thetas=THETAS, build_min_abs=min_abs)
    val = float(prog.expval(THETAS, obs_index=0))
    retained = int(prog.psum_union.x_mask.shape[0])
    return val, retained


def pennylane_noisy_exact(p_noise: float) -> float:
    dev = qml.device("default.mixed", wires=N_QUBITS)
    params = THETAS.numpy()

    @qml.qnode(dev)
    def qn():
        for q in range(N_QUBITS):
            qml.Hadamard(q)
        k = 0
        for _ in range(DEPTH):
            for (u, v) in EDGES:
                qml.PauliRot(float(params[k]), "ZZ", wires=[u, v]); k += 1
            for q in range(N_QUBITS):
                qml.PauliRot(float(params[k]), "X", wires=[q]); k += 1
            if p_noise > 0:
                for q in range(N_QUBITS):
                    qml.DepolarizingChannel(p_noise, wires=q)
        return qml.expval(sum(qml.PauliZ(u) @ qml.PauliZ(v) for (u, v) in EDGES))

    return float(qn())


def pado_circuit_ad(gamma: float):
    """Same QAOA circuit; single-qubit amplitude damping (rate gamma) after each layer."""
    qc = Circuit(N_QUBITS)
    for q in range(N_QUBITS):
        qc.h(q)
    k = 0
    for _ in range(DEPTH):
        for (u, v) in EDGES:
            qc.rzz(u, v, param_idx=k); k += 1
        for q in range(N_QUBITS):
            qc.rx(q, param_idx=k); k += 1
        if gamma > 0:
            for q in range(N_QUBITS):
                qc.amplitude_damping(q, gamma)
    return qc


def pado_expval_ad(gamma: float, min_abs: float):
    obs = [("ZZ", [u, v]) for (u, v) in EDGES]
    qc = pado_circuit_ad(gamma)
    prog = qc.compile(
        observables=[obs], preset=PRESET,
        max_weight=N_QUBITS,
        build_thetas=THETAS, build_min_abs=min_abs)
    val = float(prog.expval(THETAS, obs_index=0))
    retained = int(prog.psum_union.x_mask.shape[0])
    return val, retained


def pennylane_ad_exact(gamma: float) -> float:
    dev = qml.device("default.mixed", wires=N_QUBITS)
    params = THETAS.numpy()

    @qml.qnode(dev)
    def qn():
        for q in range(N_QUBITS):
            qml.Hadamard(q)
        k = 0
        for _ in range(DEPTH):
            for (u, v) in EDGES:
                qml.PauliRot(float(params[k]), "ZZ", wires=[u, v]); k += 1
            for q in range(N_QUBITS):
                qml.PauliRot(float(params[k]), "X", wires=[q]); k += 1
            if gamma > 0:
                for q in range(N_QUBITS):
                    qml.AmplitudeDamping(gamma, wires=q)
        return qml.expval(sum(qml.PauliZ(u) @ qml.PauliZ(v) for (u, v) in EDGES))

    return float(qn())


def run():
    os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
    print(f"n={N_QUBITS} edges={len(EDGES)} params={N_PARAMS}")

    # ---- Experiment 1: correctness vs PennyLane noisy emulator ----
    exp1 = []
    for p in NOISE_RATES:
        pado, _ = pado_expval(p, EPS_NEAR_EXACT)
        ref = pennylane_noisy_exact(p)
        exp1.append({"p": p, "pado": pado, "pl_exact": ref, "abs_error": abs(pado - ref)})
        print(f"[exp1] p={p:.2f}  pado={pado:+.6f}  PL={ref:+.6f}  |dif|={abs(pado-ref):.2e}")

    # ---- Experiment 2: noise-assisted truncation ----
    exp2 = {}
    for p in (0.0, 0.1):
        ref = pennylane_noisy_exact(p)
        rows = []
        for eps in EPS_LIST:
            val, retained = pado_expval(p, eps)
            rows.append({"min_abs": eps, "expval": val, "retained": retained,
                         "abs_error": abs(val - ref)})
            print(f"[exp2] p={p:.2f} eps={eps:.0e}  err={abs(val-ref):.2e}  retained={retained}")
        exp2[str(p)] = {"pl_exact": ref, "rows": rows}

    # ---- Experiment 3: amplitude-damping correctness vs PennyLane noisy emulator ----
    exp3 = []
    for g in AD_RATES:
        pado, _ = pado_expval_ad(g, EPS_NEAR_EXACT)
        ref = pennylane_ad_exact(g)
        exp3.append({"gamma": g, "pado": pado, "pl_exact": ref, "abs_error": abs(pado - ref)})
        print(f"[exp3-AD] gamma={g:.2f}  pado={pado:+.6f}  PL={ref:+.6f}  |dif|={abs(pado-ref):.2e}")

    with open(os.path.join(RESULTS, "noise.json"), "w") as fh:
        json.dump({"exp1": exp1, "exp2": exp2, "exp3_ampdamp": exp3}, fh, indent=2)

    _plot_validation(exp1)
    _plot_truncation(exp2)
    _plot_ampdamp(exp3)


def _plot_validation(exp1):
    p = [r["p"] for r in exp1]
    pado = [r["pado"] for r in exp1]
    ref = [r["pl_exact"] for r in exp1]
    plt.figure(figsize=(3.3, 2.3))
    plt.plot(p, ref, "-", color="k", lw=1.5, label="exact (default.mixed)")
    plt.plot(p, pado, "o", color="tab:blue", ms=8, label="PADO-Pauli")
    plt.xlabel("depolarizing noise rate $p$")
    plt.ylabel(r"$\langle \sum_{(i,j)\in E} Z_iZ_j\rangle$")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=8, loc="upper right"); plt.tight_layout()
    out = os.path.join(FIGS, "fig_noise_validation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()


def _plot_ampdamp(exp3):
    g = [r["gamma"] for r in exp3]
    pado = [r["pado"] for r in exp3]
    ref = [r["pl_exact"] for r in exp3]
    plt.figure(figsize=(3.3, 2.3))
    plt.plot(g, ref, "-", color="k", lw=1.5, label="exact (default.mixed)")
    plt.plot(g, pado, "o", color="tab:green", ms=8, label="PADO-Pauli")
    plt.xlabel(r"amplitude-damping rate $\gamma$")
    plt.ylabel(r"$\langle \sum_{(i,j)\in E} Z_iZ_j\rangle$")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=8, loc="upper left"); plt.tight_layout()
    out = os.path.join(FIGS, "fig_noise_ampdamp.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()


def _plot_truncation(exp2):
    plt.figure(figsize=(3.3, 2.7))
    styles = {"0.0": ("tab:gray", "o", "noiseless ($p=0$)"),
              "0.1": ("tab:red", "s", "noisy ($p=0.1$)")}
    for key, (c, m, lab) in styles.items():
        rows = sorted(exp2[key]["rows"], key=lambda r: r["retained"])
        ret = [max(r["retained"], 1) for r in rows]
        err = [max(r["abs_error"], 1e-12) for r in rows]
        plt.loglog(ret, err, m + "-", color=c, label=lab)
    plt.xlabel("retained Pauli terms (cost)")
    plt.ylabel(r"$|\langle O\rangle_\epsilon - \langle O\rangle_{\mathrm{exact}}|$ (accuracy)")
    plt.grid(True, which="both", alpha=0.3); plt.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2); plt.tight_layout()
    out = os.path.join(FIGS, "fig_noise_truncation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()



if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv:
        with open(os.path.join(RESULTS, "noise.json")) as fh:
            d = json.load(fh)
        _plot_validation(d["exp1"])
        _plot_truncation(d["exp2"])
        _plot_ampdamp(d["exp3_ampdamp"])
        print("DONE (plot only)")
        raise SystemExit(0)
    run()
