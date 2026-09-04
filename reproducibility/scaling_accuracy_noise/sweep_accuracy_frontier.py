#!/usr/bin/env python
"""Truncation accuracy-vs-cost frontier + at-scale apparent convergence.

Two studies, both on the same Erdos-Renyi MaxCut-QAOA family used elsewhere
(edge prob 0.3, seed 42, depth 3), evaluated at a fixed reproducible theta.

Part A -- frontier (exact baseline, n = 16, n <= 20 so default.qubit is exact):
    Sweep the two truncation knobs independently and record the TRUE absolute
    error against the exact statevector value as a function of the number of
    retained Pauli strings:
      * max-weight truncation: w_max = 1..6            (build_min_abs = None)
      * min-abs  truncation : eps = 1e-1..1e-6         (max_weight = n)
    Plotting error vs retained-term count puts both strategies on a common
    cost axis -> the accuracy-vs-term-count frontier.

Part B -- convergence at scale (n = 20, 24, 28, exact value from the
    GPU statevector, cached in <data root>/scaling_accuracy_noise/results/scale_exact_cache.json):
    For each n the surrogate <sum ZZ> is recomputed under progressively looser
    max-weight truncation and compared with the exact value.

The GPU statevector device follows the installed torch build:
lightning.gpu on CUDA, lightning.amdgpu on ROCm (which needs LD_PRELOAD, see README).

Run:  python sweep_accuracy_frontier.py
Outputs: results/accuracy_frontier.{json,csv}, results/accuracy_scale.{json,csv}
         figures/fig_accuracy_frontier.png, figures/fig_accuracy_scale.png
"""

from __future__ import annotations

import csv
import json
import os
import time

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
from _outdir import lightning_gpu_device as _lightning_gpu_device

RESULTS = _out_dir("scaling_accuracy_noise", "results")
FIGS = _out_dir("scaling_accuracy_noise", "figures")

DEPTH = 3
EDGE_PROB = 0.3
SEED = 42
PRESET = "hybrid"
# Large chunk so the heavy w_max=5 propagations run in few passes
# (hybrid keeps term storage on the host).
CHUNK_SIZE = 100_000_000

# Part A: fixed exact-baseline instance.
FRONTIER_N = 16
WMAX_LIST = [1, 2, 3, 4, 5, 6]
MINABS_LIST = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5, 1e-6]

# Part B: convergence to the exact value across system sizes. Sizes kept within
# the GPU statevector regime so an exact line is available for each n.
SCALE_N_LIST = [20, 24, 28]
SCALE_WMAX_LIST = [1, 2, 3, 4, 5]
SCALE_TIME_BUDGET_S = 200.0   # per-n time guard
EXACT_CACHE = os.path.join(RESULTS, "scale_exact_cache.json")


def _fixed_thetas(n_params: int) -> torch.Tensor:
    # Same reproducible angle pattern as sweep_minabs_convergence.py.
    return torch.tensor(0.2 * np.sin(np.arange(n_params)), dtype=torch.float64)


def _retained(prog) -> int:
    return int(prog.psum_union.x_mask.shape[0])


def _seed_suffix(seed: int) -> str:
    """Non-default seeds get their own filenames and cache keys, so a multi-seed
    run never overwrites or reuses the seed-42 data."""
    return "" if int(seed) == SEED else f"_s{int(seed)}"


def _exact_value(n: int, circuit, obs, thetas, seed: int = SEED) -> float:
    """Exact <sum ZZ> via the GPU statevector (cached by n and seed)."""
    key = f"{n}{_seed_suffix(seed)}"
    cache = {}
    if os.path.exists(EXACT_CACHE):
        with open(EXACT_CACHE) as fh:
            cache = json.load(fh)
    if key in cache:
        return float(cache[key])

    import pennylane as qml
    from padopauli import pennylane_apply_circuit, pennylane_observable

    op = pennylane_observable(obs, qml)
    dev = qml.device(_lightning_gpu_device(), wires=int(n))

    @qml.qnode(dev)
    def qn(params):
        pennylane_apply_circuit(circuit, params, qml)
        return qml.expval(op)

    val = float(qn(thetas.detach().cpu().numpy()))
    cache[key] = val
    with open(EXACT_CACHE, "w") as fh:
        json.dump(cache, fh, indent=2)
    return val


def run_frontier(seed: int = SEED):
    sfx = _seed_suffix(seed)
    edges = er_edges(FRONTIER_N, EDGE_PROB, seed)
    qc, n_params = build_qaoa(FRONTIER_N, edges, DEPTH)
    obs = build_zz_observable(edges)
    thetas = _fixed_thetas(n_params)

    exact = float(pennylane_expvals(
        circuit=qc.gates, observables=[zz_paulisum(FRONTIER_N, edges)], thetas=thetas,
        n_qubits=FRONTIER_N, max_qubits=20)[0])
    print(f"[frontier] n={FRONTIER_N} seed={seed} edges={len(edges)} params={n_params} "
          f"exact<sumZZ>={exact:.8f}")

    maxw_rows = []
    for w in WMAX_LIST:
        prog = qc.compile(
            observables=[obs], preset=PRESET,
            max_weight=int(w),
            build_thetas=thetas)
        val = float(prog.expval(thetas, obs_index=0))
        maxw_rows.append({"knob": "max_weight", "level": int(w),
                          "expval": val, "abs_error": abs(val - exact),
                          "retained_terms": _retained(prog)})
        print(f"  [maxw] w={w:>3}  val={val:+.6f}  err={abs(val-exact):.3e}  "
              f"retained={maxw_rows[-1]['retained_terms']}")
        prog.clear_cache()

    minabs_rows = []
    for eps in MINABS_LIST:
        prog = qc.compile(
            observables=[obs], preset=PRESET,
            max_weight=FRONTIER_N,
            build_thetas=thetas, build_min_abs=eps)
        val = float(prog.expval(thetas, obs_index=0))
        minabs_rows.append({"knob": "min_abs", "level": float(eps),
                            "expval": val, "abs_error": abs(val - exact),
                            "retained_terms": _retained(prog)})
        print(f"  [minabs] eps={eps:.0e}  val={val:+.6f}  err={abs(val-exact):.3e}  "
              f"retained={minabs_rows[-1]['retained_terms']}")
        prog.clear_cache()

    payload = {"n_qubits": FRONTIER_N, "depth": DEPTH, "edge_prob": EDGE_PROB,
               "seed": int(seed), "exact": exact,
               "max_weight": maxw_rows, "min_abs": minabs_rows}
    with open(os.path.join(RESULTS, f"accuracy_frontier{sfx}.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(os.path.join(RESULTS, f"accuracy_frontier{sfx}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["knob", "level", "expval", "abs_error", "retained_terms"])
        w.writeheader(); w.writerows(maxw_rows + minabs_rows)

    _plot_frontier(maxw_rows, minabs_rows, seed)
    return payload


def run_scale(seed: int = SEED):
    sfx = _seed_suffix(seed)
    all_rows = []
    series = {}
    for n in SCALE_N_LIST:
        edges = er_edges(n, EDGE_PROB, seed)
        qc, n_params = build_qaoa(n, edges, DEPTH)
        obs = build_zz_observable(edges)
        thetas = _fixed_thetas(n_params)
        exact = _exact_value(n, qc.gates, zz_paulisum(n, edges), thetas, seed)
        print(f"  [scale] n={n} exact<sumZZ>={exact:+.6f}")
        rows = []
        spent = 0.0
        for w in SCALE_WMAX_LIST:
            if spent > SCALE_TIME_BUDGET_S:
                print(f"  [scale] n={n}: time budget hit, stop at w<{w}")
                break
            t0 = time.time()
            try:
                prog = qc.compile(
                    observables=[obs], preset=PRESET,
                    max_weight=int(w), chunk_size=CHUNK_SIZE,
                    build_thetas=thetas)
                val = float(prog.expval(thetas, obs_index=0))
                ret = _retained(prog)
                prog.clear_cache()
            except Exception as exc:  # noqa: BLE001  (record the failure point, keep going)
                print(f"  [scale] n={n} w={w} FAILED: {type(exc).__name__}: {exc}")
                break
            dt = time.time() - t0
            spent += dt
            rows.append({"n_qubits": n, "w_max": int(w), "expval": val,
                         "retained_terms": ret, "compile_s": dt,
                         "exact": exact, "abs_error": abs(val - exact)})
            print(f"  [scale] n={n:>3} w={w}  val={val:+.6f}  err={abs(val-exact):.3e}  "
                  f"retained={ret}  ({dt:.1f}s)")
        # successive-difference convergence indicator
        for i in range(1, len(rows)):
            rows[i]["delta_vs_prev"] = abs(rows[i]["expval"] - rows[i - 1]["expval"])
        if rows:
            rows[0]["delta_vs_prev"] = float("nan")
        series[n] = {"exact": exact, "rows": rows}
        all_rows.extend(rows)

    with open(os.path.join(RESULTS, f"accuracy_scale{sfx}.json"), "w") as fh:
        json.dump({"depth": DEPTH, "edge_prob": EDGE_PROB, "seed": int(seed), "series": series}, fh, indent=2)
    with open(os.path.join(RESULTS, f"accuracy_scale{sfx}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["n_qubits", "w_max", "expval", "exact", "abs_error",
                                           "retained_terms", "compile_s", "delta_vs_prev"])
        w.writeheader()
        for r in all_rows:
            r.setdefault("delta_vs_prev", float("nan"))
            w.writerow(r)

    _plot_scale(series, seed)
    return series


def _plot_frontier(maxw_rows, minabs_rows, seed=SEED):
    def xyl(rows, labeler):
        # keep (retained_terms, error, knob-label) and sort along the cost axis.
        # Settings that retain no terms at all (w_max=1, eps=1e-1 here) have no
        # place on a log term-count axis: matplotlib drops the marker but still
        # draws a segment in from the axis edge, which reads as a flat left tail
        # that is not data. Drop those rows and say so in the caption instead.
        pts = sorted((r["retained_terms"], max(r["abs_error"], 1e-12), labeler(r))
                     for r in rows if r["retained_terms"] > 0)
        return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]

    xw, yw, lw = xyl(maxw_rows, lambda r: f"$w_{{\\max}}{{=}}{int(r['level'])}$")
    xe, ye, le = xyl(minabs_rows, lambda r: f"$\\epsilon{{=}}10^{{{int(round(np.log10(r['level'])))}}}$")

    plt.figure(figsize=(7.2, 4.7))
    plt.loglog(xw, yw, "o-", color="tab:green", label="max-weight truncation (annotated $w_{\\max}$)")
    plt.loglog(xe, ye, "s-", color="tab:blue", label=r"coefficient truncation (annotated $\epsilon$)")

    # annotate each marker with the truncation knob value that produced it,
    # so the (otherwise hidden) per-curve sweep parameter is readable. A fixed
    # offset put labels on top of each other where the two curves nearly meet
    # and on top of the curve at the turning points, so each label is placed by
    # hand relative to its own point: (dx, dy, horizontal alignment).
    WMAX_OFF = {2: (5, 8, "left"), 3: (5, 8, "left"), 4: (5, 8, "left"),
                5: (0, -15, "center"), 6: (5, 7, "left")}
    EPS_OFF = {-2: (-6, 3, "right"), -3: (0, -15, "center"), -4: (-6, 6, "right"),
               -5: (5, 7, "left"), -6: (5, 7, "left")}

    for r in maxw_rows:
        if r["retained_terms"] <= 0:
            continue
        w = int(r["level"])
        dx, dy, ha = WMAX_OFF.get(w, (5, 8, "left"))
        plt.annotate(fr"$w_{{\max}}{{=}}{w}$",
                     (r["retained_terms"], max(r["abs_error"], 1e-12)),
                     textcoords="offset points", xytext=(dx, dy),
                     fontsize=8, color="tab:green", ha=ha)
    # annotate only the decade thresholds, which are the readable landmarks
    for r in minabs_rows:
        if r["retained_terms"] <= 0:
            continue
        k = int(round(np.log10(r["level"])))
        if abs(np.log10(r["level"]) - k) > 1e-6 or k not in EPS_OFF:
            continue
        dx, dy, ha = EPS_OFF[k]
        plt.annotate(fr"$\epsilon{{=}}10^{{{k}}}$",
                     (r["retained_terms"], max(r["abs_error"], 1e-12)),
                     textcoords="offset points", xytext=(dx, dy),
                     fontsize=8, color="tab:blue", ha=ha)

    plt.xlabel("retained Pauli strings (term-count budget)")
    plt.ylabel(r"absolute error $|\langle O\rangle_{\mathrm{trunc}}-\langle O\rangle_{\mathrm{exact}}|$")
    plt.grid(True, which="both", alpha=0.3)
    plt.margins(x=0.12, y=0.18)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    # suffixed like the CSV/JSON so a non-default seed never overwrites the
    # default-seed figure
    out = os.path.join(FIGS, f"fig_accuracy_frontier{_seed_suffix(seed)}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()


def _plot_scale(series, seed: int = SEED):
    # small figure size with 9 pt text
    with plt.rc_context({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8,
                         "figure.dpi": 200, "savefig.dpi": 200}):
        _plot_scale_inner(series, seed)


def _plot_scale_inner(series, seed):
    plt.figure(figsize=(4.64, 3.1))
    cmap = plt.get_cmap("viridis")
    ns = sorted(series.keys(), key=lambda x: int(x))
    for k, n in enumerate(ns):
        entry = series[n]
        rows = entry["rows"] if isinstance(entry, dict) else entry
        if not rows:
            continue
        exact = entry["exact"] if isinstance(entry, dict) else rows[0].get("exact")
        color = cmap(k / max(1, len(ns) - 1))
        w = [r["w_max"] for r in rows]
        v = [r["expval"] for r in rows]
        # exact ideal line for this system size
        plt.axhline(exact, color=color, ls="--", lw=1.2, alpha=0.9)
        plt.plot(w, v, "o-", color=color, label=f"$n={n}$ surrogate")
        plt.annotate(f"exact $n{{=}}{n}$", (w[-1], exact), textcoords="offset points",
                     xytext=(6, 2), fontsize=8, color=color, ha="left", va="bottom")
    plt.xlabel(r"max-weight truncation threshold $w_{\max}$")
    plt.ylabel(r"surrogate $\langle \sum_{(i,j)\in E} Z_iZ_j\rangle$")
    plt.grid(True, which="both", alpha=0.3)
    # extra right headroom so the inline "exact n=NN" labels are not clipped
    plt.xlim(0.6, max(SCALE_WMAX_LIST) + 1.7)
    plt.legend(fontsize=9, title="dashed = exact statevector", title_fontsize=8)
    plt.tight_layout()
    # suffixed like the CSV/JSON so a non-default seed never overwrites the
    # default-seed figure
    out = os.path.join(FIGS, f"fig_accuracy_scale{_seed_suffix(seed)}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED,
                    help="Erdos-Renyi seed for the across-n scale study; non-default "
                         "seeds write accuracy_scale_s<seed>.{json,csv}")
    ap.add_argument("--scale-only", action="store_true",
                    help="skip the fixed-16q frontier")
    ap.add_argument("--frontier-seed", type=int, default=SEED,
                    help="Erdos-Renyi seed for the fixed-16q frontier; non-default "
                         "seeds write accuracy_frontier_s<seed>.{json,csv}")
    ap.add_argument("--frontier-only", action="store_true",
                    help="skip the across-n scale study")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figures from the recorded results/*.json without re-running")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    if args.plot_only:
        if not args.scale_only:
            with open(os.path.join(RESULTS, f"accuracy_frontier{_seed_suffix(args.frontier_seed)}.json")) as fh:
                d = json.load(fh)
            _plot_frontier(d["max_weight"], d["min_abs"], args.frontier_seed)
        if not args.frontier_only:
            with open(os.path.join(RESULTS, f"accuracy_scale{_seed_suffix(args.seed)}.json")) as fh:
                d = json.load(fh)
            _plot_scale(d["series"], args.seed)
        print("DONE (plot only)")
        raise SystemExit(0)
    if not args.scale_only:
        run_frontier(args.frontier_seed)
    if not args.frontier_only:
        run_scale(args.seed)
    print("DONE")
