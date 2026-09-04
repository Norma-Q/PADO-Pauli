#!/usr/bin/env python
"""Differentiation-backend benchmark: manual sparse-chain VJP vs PyTorch autograd.

The compiled surrogate exposes two differentiation modes through `diff_mode`
(CompiledProgram.expvals):

  - "vjp" (DEFAULT): a custom torch.autograd.Function whose forward runs
    detached and stores only the per-step adjoint vectors, and whose backward is a
    hand-written sparse-chain VJP (per-step cos/sin recomputed, fused reductions).
  - "autograd": let PyTorch trace the sparse-dense adjoint matmul chain and retain
    its full autograd graph.

Both return the same gradient. This script measures the forward+backward cost and
peak GPU memory of each mode on the ER MaxCut-QAOA family (reusing the builders in
../scaling_accuracy_noise/_common.py), sweeping max_weight at a fixed 16-qubit
depth-5 instance.

Term counts are angle-independent; timings/VRAM are hardware-dependent.
Run: python run_diff_mode_benchmark.py
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import sys
import time
from pathlib import Path

import torch

# Reuse the QAOA circuit family helpers from the sibling experiment folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scaling_accuracy_noise"))

from _common import er_edges, build_qaoa, build_zz_observable  # noqa: E402

# REPS is the number of evaluations timed against one compiled program.
N, DEPTH, P, SEED, REPS = 16, 5, 0.3, 42, 5
WMAXES = [4, 5, 6, 7]
# Output location is device-aware: <repo>/results_on_<DEVICE>/{exp}/...
# so the same code can run on an A100, an H100 and an MI300X without one run
# overwriting another. See reproducibility/_outdir.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _outdir import out_dir as _out_dir

OUT = Path(_out_dir("diff_mode_vjp", "results"))
OUT.mkdir(exist_ok=True)


def _fwd_bwd(prog, thetas, mode):
    if thetas.grad is not None:
        thetas.grad = None
    energy = prog.expvals(thetas, diff_mode=mode).sum()
    energy.backward()
    torch.cuda.synchronize()
    return thetas.grad.detach().clone()


def _measure(prog, k, mode):
    thetas = torch.linspace(-0.3, 0.3, k, dtype=torch.float64, device="cuda").requires_grad_(True)
    grad = _fwd_bwd(prog, thetas, mode)  # warmup + capture gradient for cross-mode check
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        _fwd_bwd(prog, thetas, mode)
        ts.append(time.perf_counter() - t0)
    ts_sorted = sorted(ts)
    return {
        "fwdbwd_s": ts_sorted[len(ts) // 2],   # median, matching run_zero_filter_ablation.py
        "fwdbwd_s_min": ts_sorted[0],
        "fwdbwd_s_max": ts_sorted[-1],
        "fwdbwd_reps_s": ts,
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024 ** 3,
    }, grad


def main():
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA GPU (gpu preset).")
    results = []
    for wmax in WMAXES:
        edges = er_edges(N, P, SEED)
        qc, k = build_qaoa(N, edges, DEPTH)
        obs = build_zz_observable(edges)
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        prog = qc.compile(observables=[obs], preset="gpu", dtype="float32", max_weight=wmax)
        prop = int(prog.psum_union.x_mask.shape[0])
        row = {"w_max": wmax, "n": N, "depth": DEPTH, "p": P, "seed": SEED,
               "n_params": k, "propagated_terms": prop}
        grads = {}
        for mode in ("vjp", "autograd"):
            try:
                row[mode], grads[mode] = _measure(prog, k, mode)
            except RuntimeError as exc:  # e.g. CUDA OOM at the largest size
                row[mode] = {"error": str(exc)[:160]}
                torch.cuda.empty_cache()
        if "vjp" in grads and "autograd" in grads:
            row["grad_max_abs_diff"] = float((grads["vjp"] - grads["autograd"]).abs().max().cpu())
        m, a = row.get("vjp", {}), row.get("autograd", {})
        if "fwdbwd_s" in m and "fwdbwd_s" in a:
            row["speedup_x"] = a["fwdbwd_s"] / m["fwdbwd_s"]
            row["manual_mem_fraction"] = m["peak_reserved_gb"] / a["peak_reserved_gb"]
        results.append(row)
        print(f"w={wmax} prop={prop:>9,} | vjp {m.get('fwdbwd_s', float('nan')):8.3f}s "
              f"{m.get('peak_reserved_gb', float('nan')):7.3f}GB | autograd "
              f"{a.get('fwdbwd_s', a.get('error', '?'))} {a.get('peak_reserved_gb', '')}"
              f" | speedup={row.get('speedup_x', 'NA')} dgrad={row.get('grad_max_abs_diff', 'NA')}",
              flush=True)
        del prog
        torch.cuda.empty_cache()

    payload = {
        "config": {"n": N, "depth": DEPTH, "p": P, "seed": SEED, "reps": REPS,
                   # compile pins dtype="float32" (heavy run); theta inputs are float64 and cast.
                   "w_max_sweep": WMAXES, "preset": "gpu", "dtype": "float32"},
        "results": results,
    }
    with open(OUT / "diff_mode_vjp.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT / 'diff_mode_vjp.json'}")


if __name__ == "__main__":
    main()
