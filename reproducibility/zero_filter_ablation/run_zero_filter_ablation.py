#!/usr/bin/env python
"""Zero-filter ablation: what exact zero-filtering buys at evaluation time.

compile_program exposes `zero_filter`: True (default)
runs the backprop pruning pass and shrinks the program to the terms that can
contribute to <0...0| expectations; False keeps every propagated term. Both
programs return identical expvals (the pruned terms contribute exactly zero),
so the ablation isolates the cost side: program size, resident GPU memory,
forward-eval latency (single theta and batched), and vjp fwd+bwd cost.

Instance: the 16-qubit depth-5 ER MaxCut-QAOA family (reusing
../scaling_accuracy_noise/_common.py), sweeping max_weight. gpu preset
(float32 compute).

Term counts and memory are deterministic; wall-clock is hardware-dependent.
Each measurement phase is individually OOM-guarded: a failure is recorded in
the row and the sweep continues.

Run: python run_zero_filter_ablation.py
Output: results/zero_filter_ablation.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

# Reuse the QAOA circuit family helpers from the sibling experiment folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scaling_accuracy_noise"))

from _common import er_edges, build_qaoa, build_zz_observable  # noqa: E402

N, DEPTH, P, SEED = 16, 5, 0.3, 42
WMAXES = [4, 5, 6, 7]
REPS = 5            # timed reps per phase, after one discarded warm-up
                    # (same protocol as run_diff_mode_benchmark.py)
BATCH = 200         # batched thetas: makes eval compute-bound instead of
                    # kernel-launch-bound, which is where the term count shows
# Output location is device-aware: <repo>/results_on_<DEVICE>/{exp}/...
# so the same code can run on an A100, an H100 and an MI300X without one run
# overwriting another. See reproducibility/_outdir.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _outdir import out_dir as _out_dir

OUT = Path(_out_dir("zero_filter_ablation", "results"))
OUT.mkdir(exist_ok=True)


# GB here means 2^30 bytes (the convention torch reports in).
def _gb(x: int) -> float:
    return x / 1024**3


def _timed(call, reps=REPS):
    """One discarded warm-up, then `reps` timed calls. Returns (median, min, max, all).

    The warm-up matters most for the backward pass: the first one builds the
    transposed-CSR cache the adjoint reads.
    """
    call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts_sorted = sorted(ts)
    return ts_sorted[len(ts) // 2], ts_sorted[0], ts_sorted[-1], ts


def measure(zf: bool, wmax: int, qc, obs, thetas, thetas_b):
    """Compile one program and run the guarded measurement phases."""
    row = {"w_max": wmax, "zero_filter": zf}
    val = grad = None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    try:
        t0 = time.perf_counter()
        prog = qc.compile(observables=[obs], preset="gpu", dtype="float32",
                          max_weight=wmax, zero_filter=zf)
        row["compile_s"] = time.perf_counter() - t0
        row["terms"] = int(prog.psum_union.x_mask.shape[0])
        # nonzeros held by the compiled step matrices
        row["step_nnz"] = sum(int(s.mat_const._nnz()) + int(s.mat_cos._nnz())
                              + int(s.mat_sin._nnz()) for s in prog.psum_union.steps)
        row["n_steps"] = len(prog.psum_union.steps)
    except RuntimeError as exc:
        row["compile_error"] = str(exc)[:160]
        torch.cuda.empty_cache()
        return row, val, grad

    try:  # single-theta forward
        val = prog.expvals(thetas).detach().cpu().double()
        torch.cuda.synchronize()
        row["resident_gb"] = _gb(torch.cuda.memory_allocated() - base)
        torch.cuda.reset_peak_memory_stats()
        med, lo, hi, all_ts = _timed(lambda: prog.expvals(thetas))
        row["eval_s"], row["eval_s_min"], row["eval_s_max"] = med, lo, hi
        row["eval_reps_s"] = all_ts
        row["eval_peak_gb"] = _gb(torch.cuda.max_memory_allocated())
    except RuntimeError as exc:
        row["eval_error"] = str(exc)[:160]
        torch.cuda.empty_cache()

    try:  # batched forward
        torch.cuda.reset_peak_memory_stats()
        med, lo, hi, all_ts = _timed(lambda: prog.expvals(thetas_b))
        row["batch_eval_s"], row["batch_eval_s_min"], row["batch_eval_s_max"] = med, lo, hi
        row["batch_eval_reps_s"] = all_ts
        row["batch_peak_gb"] = _gb(torch.cuda.max_memory_allocated())
    except RuntimeError as exc:
        row["batch_error"] = str(exc)[:160]
        torch.cuda.empty_cache()

    try:  # vjp fwd+bwd, single theta
        th = thetas.clone().requires_grad_(True)

        def _fwdbwd():
            th.grad = None
            prog.expvals(th).sum().backward()

        # the cold first call is reported separately: it is the one-time cost of
        # building the adjoint's transposed-CSR cache, not the steady-state pass
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        _fwdbwd()
        torch.cuda.synchronize()
        row["fwdbwd_first_s"] = time.perf_counter() - t0

        med, lo, hi, all_ts = _timed(_fwdbwd)
        row["fwdbwd_s"], row["fwdbwd_s_min"], row["fwdbwd_s_max"] = med, lo, hi
        row["fwdbwd_reps_s"] = all_ts
        row["fwdbwd_peak_gb"] = _gb(torch.cuda.max_memory_allocated())
        grad = th.grad.detach().cpu().double()
    except RuntimeError as exc:
        row["fwdbwd_error"] = str(exc)[:160]
        torch.cuda.empty_cache()

    prog.clear_cache()
    del prog
    torch.cuda.empty_cache()
    return row, val, grad


def main():
    if not torch.cuda.is_available():
        raise SystemExit("This ablation requires a GPU (gpu preset).")
    edges = er_edges(N, P, SEED)
    qc, k = build_qaoa(N, edges, DEPTH)
    obs = build_zz_observable(edges)
    thetas = torch.linspace(-0.3, 0.3, k, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    thetas_b = (torch.rand(BATCH, k, generator=g, dtype=torch.float64) - 0.5) * 0.6

    # Discarded warm-up: the first compile of the process pays CUDA context
    # creation, kernel autotune and allocator warm-up.
    measure(True, WMAXES[0], qc, obs, thetas, thetas_b)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    rows, parity = [], []
    for wmax in WMAXES:
        got = {}
        for zf in (True, False):
            row, val, grad = measure(zf, wmax, qc, obs, thetas, thetas_b)
            rows.append(row)
            got[zf] = (val, grad)
            print(f"w={wmax} zf={'on ' if zf else 'off'}: "
                  f"terms={format(row['terms'], '>12,') if 'terms' in row else '?':>12} "
                  f"resident={row.get('resident_gb', float('nan')):7.3f}GB "
                  f"eval={row.get('eval_s', float('nan'))*1e3:8.2f}ms "
                  f"batch{BATCH}={row.get('batch_eval_s', float('nan'))*1e3:9.2f}ms "
                  f"fwdbwd={row.get('fwdbwd_s', float('nan')):8.2f}s "
                  f"err={[key for key in row if key.endswith('_error')]}", flush=True)
        p = {"w_max": wmax}
        if got[True][0] is not None and got[False][0] is not None:
            p["expval_max_abs_diff"] = float((got[True][0] - got[False][0]).abs().max())
        if got[True][1] is not None and got[False][1] is not None:
            p["grad_max_abs_diff"] = float((got[True][1] - got[False][1]).abs().max())
        parity.append(p)
        print(f"w={wmax} parity: {p}", flush=True)

    payload = {
        "config": {"n": N, "depth": DEPTH, "p": P, "seed": SEED,
                   "w_max_sweep": WMAXES, "preset": "gpu", "dtype": "float32",
                   "reps": REPS, "batch": BATCH,
                   "timing": "median of REPS timed calls after one discarded warm-up",
                   "gpu": torch.cuda.get_device_name(0),
                   "torch": torch.__version__},
        "rows": rows,
        "parity": parity,
    }
    with open(OUT / "zero_filter_ablation.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT / 'zero_filter_ablation.json'}")


if __name__ == "__main__":
    main()
