"""Build the terms-vs-qubits, VRAM-vs-qubits and max-weight-growth figures
from the swept results (results/qubit_sweep.json, results/wmax_sweep.json).

Run after the sweeps finish:
  python make_figures.py
Outputs land in ./figures/.
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

PRESET = "gpu"  # GPU-resident program; we report the VRAM footprint


def _load(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


def fig_terms_vs_qubits(rows):
    rows = sorted([r for r in rows if r["preset"] == PRESET], key=lambda r: r["n_qubits"])
    N = [r["n_qubits"] for r in rows]
    prop = [r["propagated_terms"] for r in rows]
    zf = [r["zero_filtered_terms"] for r in rows]
    ub = [r["upper_bound"] for r in rows]

    plt.figure(figsize=(7, 4.2))
    plt.semilogy(N, ub, "k--", marker="x", label=r"weight-limited Pauli bound $\sum_w \binom{n}{w}3^w$")
    plt.semilogy(N, prop, "o-", color="tab:blue", label="unique propagated terms")
    plt.semilogy(N, zf, "s-", color="tab:green", label="zero-filtered diagonal terms")
    plt.xlabel("number of qubits $n$")
    plt.ylabel("term count")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIGS, "fig_terms_vs_qubits_p03.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out


def fig_vram_vs_qubits(rows):
    rows = sorted([r for r in rows if r["preset"] == PRESET], key=lambda r: r["n_qubits"])
    N = np.array([r["n_qubits"] for r in rows], dtype=float)
    vram = np.array([r["vram_peak_reserved_gb"] for r in rows], dtype=float)

    plt.figure(figsize=(7, 4.2))
    plt.plot(N, vram, "o-", color="tab:red", label="measured peak GPU VRAM")
    plt.xlabel("number of qubits $n$")
    plt.ylabel(r"peak GPU-VRAM (GB)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIGS, "fig_vram_vs_qubits_p03.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out


def fig_wmax_growth(rows):
    rows = sorted([r for r in rows if r["preset"] == PRESET], key=lambda r: r["w_max"])
    W = [r["w_max"] for r in rows]
    prop = [r["propagated_terms"] for r in rows]
    zf = [r["zero_filtered_terms"] for r in rows]

    x = np.arange(len(W)); width = 0.38
    plt.figure(figsize=(6.5, 4))
    plt.bar(x - width / 2, prop, width, color="tab:blue", label="unique propagated terms")
    plt.bar(x + width / 2, zf, width, color="tab:green", label="zero-filtered diagonal terms")
    plt.yscale("log")
    plt.xticks(x, [str(w) for w in W])
    plt.xlabel(r"max-weight threshold $w_{\max}$")
    plt.ylabel("term count")
    plt.grid(True, axis="y", which="both", alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIGS, "fig_wmax_growth_p03.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return out


def main():
    os.makedirs(FIGS, exist_ok=True)
    outs = []
    if os.path.exists(os.path.join(RESULTS, "qubit_sweep.json")):
        q = _load("qubit_sweep.json")
        outs.append(fig_terms_vs_qubits(q))
        outs.append(fig_vram_vs_qubits(q))
    if os.path.exists(os.path.join(RESULTS, "wmax_sweep.json")):
        w = _load("wmax_sweep.json")
        outs.append(fig_wmax_growth(w))

    for o in outs:
        print("wrote", o)


if __name__ == "__main__":
    main()
