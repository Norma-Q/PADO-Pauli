"""Two validation experiments.

(1) Autodiff vs finite-difference gradient agreement.
    For a fixed compiled surrogate, the PyTorch autodiff gradient of the
    expectation value must match a central finite-difference of the SAME
    surrogate. This is an internal-consistency check of the differentiation
    machinery (both vjp and autograd modes), independent of truncation
    error -- both sides act on the identical truncated operator program.

(2) Quasi-probability reconstruction vs exact PennyLane probabilities.
    Propagate WITHOUT weight truncation so the tracked Z-correlators are exact,
    then reconstruct q^(k)(x) at increasing Fourier order k and compare to the
    exact output distribution. Isolates the order-k reconstruction truncation.
    At k = n the Fourier sum is complete and the reconstruction must equal the
    exact distribution to machine precision.

Outputs: results/grad_quasi.json plus figures/fig_grad_check.pdf and
figures/fig_quasi_check.pdf.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

from _common import er_edges, build_qaoa, build_zz_observable, require_gpu_for_preset  # noqa: E402

# Output location is device-aware: <repo>/results_on_<DEVICE>/{exp}/...
# so the same code can run on an A100, an H100 and an MI300X without one run
# overwriting another. See reproducibility/_outdir.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _outdir import out_dir as _out_dir

RESULTS_DIR = _out_dir("scaling_accuracy_noise", "results")
FIG_DIR = _out_dir("scaling_accuracy_noise", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

PRESET = "hybrid"
torch.manual_seed(0)


# ----------------------------------------------------------------------------
# Experiment 1: autodiff vs finite-difference
# ----------------------------------------------------------------------------
def exp_grad(n=12, depth=2, p=0.4, seed=7, w_max=5, h=1e-5):
    require_gpu_for_preset(PRESET)
    edges = er_edges(n, p, seed)
    qc, n_params = build_qaoa(n, edges, depth)
    obs = build_zz_observable(edges)

    prog = qc.compile(
        observables=[obs], preset=PRESET,
        max_weight=int(w_max),
    )

    rng = np.random.default_rng(123)
    theta0 = torch.tensor(rng.uniform(-np.pi, np.pi, size=n_params), dtype=torch.float64)

    out = {"n_qubits": n, "depth": depth, "n_params": int(n_params),
           "w_max": w_max, "fd_step": h, "modes": {}}

    for mode in ("vjp", "autograd"):
        th = theta0.clone().requires_grad_(True)
        E = prog.expvals(th, diff_mode=mode).reshape(())
        E.backward()
        g_ad = th.grad.detach().clone().cpu().numpy()

        # central finite difference of the same surrogate (no grad)
        g_fd = np.zeros(n_params)
        with torch.no_grad():
            for i in range(n_params):
                tp = theta0.clone(); tp[i] += h
                tm = theta0.clone(); tm[i] -= h
                Ep = float(prog.expvals(tp, diff_mode=mode).reshape(()))
                Em = float(prog.expvals(tm, diff_mode=mode).reshape(()))
                g_fd[i] = (Ep - Em) / (2.0 * h)

        abs_err = np.abs(g_ad - g_fd)
        scale = max(float(np.max(np.abs(g_fd))), 1e-30)
        out["modes"][mode] = {
            "max_abs_err": float(abs_err.max()),
            "mean_abs_err": float(abs_err.mean()),
            "max_rel_err": float(abs_err.max() / scale),
            "grad_scale": scale,
            "g_ad": g_ad.tolist(),
            "g_fd": g_fd.tolist(),
        }
        print(f"[grad/{mode}] max|Δ|={abs_err.max():.3e}  "
              f"max rel={abs_err.max()/scale:.3e}  scale={scale:.3e}")

    return out


# ----------------------------------------------------------------------------
# Experiment 2: quasi-probability vs exact PennyLane
# ----------------------------------------------------------------------------
def build_born_circuit(n, layers, seed):
    """H init + L layers of (per-qubit Z-rot, per-qubit X-rot, ZZ ring); returns (Circuit, n_params)."""
    from padopauli import Circuit
    qc = Circuit(n)
    for q in range(n):
        qc.h(q)
    k = 0
    for _ in range(layers):
        for q in range(n):
            qc.rz(q, param_idx=k); k += 1
        for q in range(n):
            qc.rx(q, param_idx=k); k += 1
        for q in range(n):
            qc.rzz(q, (q + 1) % n, param_idx=k); k += 1
    return qc, k


def exp_quasi(n=10, layers=2, seed=5):
    from itertools import combinations
    from padopauli import build_quasi_sampler, pennylane_probs

    require_gpu_for_preset(PRESET)
    qc, n_params = build_born_circuit(n, layers, seed)

    max_order = n  # full Fourier order -> exact at k=n
    z_combos = [list(c) for kk in range(1, max_order + 1)
                for c in combinations(range(n), kk)]

    # No weight truncation: tracked correlators are exact, isolating order-k effect.
    sampler = build_quasi_sampler(
        n_qubits=n, circuit=qc.gates, z_combos=z_combos, max_order=max_order,
        preset=PRESET, preset_overrides={"max_weight": 10**9},
    )

    rng = np.random.default_rng(2024)
    thetas = torch.tensor(rng.uniform(-np.pi, np.pi, size=n_params), dtype=torch.float64)

    # exact reference distribution (little-endian bit i = qubit i)
    p_exact = pennylane_probs(
        circuit=qc.gates, thetas=thetas, n_qubits=n, max_qubits=20, bit_order="le"
    ).numpy().astype(np.float64)

    # all bitstrings, row code -> bits (little-endian)
    dim = 1 << n
    x_all = np.array([[(code >> i) & 1 for i in range(n)] for code in range(dim)],
                     dtype=np.int64)
    x_batch = torch.tensor(x_all, dtype=torch.float64)

    moments = sampler.compute_moments(thetas)

    orders = list(range(1, max_order + 1))
    tv, neg_mass, neg_frac, l2 = [], [], [], []
    q_at = {}
    for k in orders:
        q = sampler.quasi_prob_from_moments(x_batch, moments, order=k).cpu().numpy().astype(np.float64)
        tv.append(0.5 * float(np.abs(q - p_exact).sum()))
        neg_mass.append(float(-q[q < 0].sum()))
        neg_frac.append(float((q < 0).mean()))
        l2.append(float(np.sqrt(((q - p_exact) ** 2).sum())))
        if k in (3, 6, 8):
            q_at[k] = q.tolist()
        print(f"[quasi] k={k:2d}  TV={tv[-1]:.3e}  neg_mass={neg_mass[-1]:.3e}  "
              f"neg_frac={neg_frac[-1]:.3f}")

    return {
        "n_qubits": n, "layers": layers, "n_params": int(n_params),
        "orders": orders, "tv": tv, "neg_mass": neg_mass, "neg_frac": neg_frac,
        "l2": l2, "p_exact": p_exact.tolist(), "q_at": q_at,
    }


# ----------------------------------------------------------------------------
def make_figs(grad, quasi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- figure: gradient agreement (vjp) ----
    modes = grad["modes"]
    m = modes.get("manual_vjp", modes.get("vjp"))  # results/grad_quasi.json records the mode as manual_vjp
    g_ad = np.array(m["g_ad"]); g_fd = np.array(m["g_fd"])
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    lim = max(np.abs(g_fd).max(), np.abs(g_ad).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=1.0, zorder=0)
    ax.scatter(g_fd, g_ad, s=22, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_xlabel(r"finite-difference gradient $\partial E/\partial\theta_i$")
    ax.set_ylabel(r"autodiff gradient $\partial E/\partial\theta_i$")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    # Spell the two figures out rather than calling them "max|Delta|" / "max rel":
    # the manuscript already uses Delta for the discarded coefficient mass, and a
    # bare "rel" hides that the denominator is one global scale, not per component.
    ax.text(0.05, 0.95,
            f"$n={grad['n_qubits']}$, {grad['n_params']} params\n"
            f"max$\\,|$autodiff$\\,-\\,$FD$\\,|={m['max_abs_err']:.1e}$\n"
            f"rel. to max$\\,|$FD$\\,|={m['max_rel_err']:.1e}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    fig.tight_layout()
    p1 = os.path.join(FIG_DIR, "fig_grad_check.pdf")
    fig.savefig(p1, bbox_inches="tight"); plt.close(fig)
    print("wrote", p1)

    # ---- figure: quasi-prob reconstruction ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.7))
    orders = quasi["orders"]
    axL.semilogy(orders, quasi["tv"], "o-", label="total variation")
    axL.semilogy(orders, quasi["neg_mass"], "s--", label="negative mass")
    axL.axvline(quasi["n_qubits"], color="0.6", ls=":", lw=1.0)
    axL.set_xlabel("reconstruction order $k$")
    axL.set_ylabel(r"distance to exact / neg. mass")
    axL.legend(fontsize=8); axL.grid(alpha=0.3)

    k_show = 8
    q = np.array(quasi["q_at"][str(k_show)] if str(k_show) in quasi["q_at"]
                 else quasi["q_at"][k_show])
    p = np.array(quasi["p_exact"])
    lim = max(p.max(), q.max()) * 1.1
    lo = min(0.0, q.min()) * 1.1
    axR.plot([lo, lim], [lo, lim], color="0.6", lw=1.0, zorder=0)
    axR.scatter(p, q, s=14, alpha=0.7, edgecolor="k", linewidth=0.2)
    axR.axhline(0.0, color="0.8", lw=0.8)
    axR.set_xlabel(r"exact probability $\Pr(x)$")
    axR.set_ylabel(rf"$q^{{({k_show})}}(x)$")
    axR.text(0.05, 0.95, f"TV$={quasi['tv'][k_show-1]:.2e}$",
             transform=axR.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    fig.tight_layout()
    p2 = os.path.join(FIG_DIR, "fig_quasi_check.pdf")
    fig.savefig(p2, bbox_inches="tight"); plt.close(fig)
    print("wrote", p2)


if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv:
        # redraw both figures from the recorded results without re-running
        with open(os.path.join(RESULTS_DIR, "grad_quasi.json")) as fh:
            d = json.load(fh)
        make_figs(d["grad"], d["quasi"])
        print("DONE (plot only)")
        raise SystemExit(0)
    grad = exp_grad()
    quasi = exp_quasi()
    with open(os.path.join(RESULTS_DIR, "grad_quasi.json"), "w") as fh:
        json.dump({"grad": grad, "quasi": quasi}, fh, indent=2)
    make_figs(grad, quasi)
    print("DONE")
