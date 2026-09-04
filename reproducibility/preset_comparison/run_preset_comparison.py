#!/usr/bin/env python
"""Same-engine preset comparison: what the GPU execution path buys.

Compiles one circuit (the 16-qubit depth-5 ER MaxCut-QAOA instance, seed 42 --
the zero-filter-ablation instance) once per execution preset (cpu / hybrid / gpu,
all at their shared float64 default dtype) and measures, for each preset: compile
time, single-theta eval, batched eval (200 thetas), and warm vjp fwd+bwd. Same
engine, same circuit, same dtype, same truncation; the only variable is where
term storage and the matrix compute live.

Protocol matches run_zero_filter_ablation.py: one discarded warm-up call, then
the median of 5 timed reps. Wall-clock is hardware-dependent.

Run: python run_preset_comparison.py
Output: results/preset_comparison.json
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
WMAXES = [4, 5, 6]
PRESETS = ["cpu", "hybrid", "gpu"]
REPS = 5
BATCH = 200
# Output location is device-aware: <repo>/results_on_<DEVICE>/{exp}/...
# so the same code can run on an A100, an H100 and an MI300X without one run
# overwriting another. See reproducibility/_outdir.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _outdir import out_dir as _out_dir

OUT = Path(_out_dir("preset_comparison", "results"))
OUT.mkdir(exist_ok=True)


def _uses_cuda(preset: str) -> bool:
    return preset in ("hybrid", "gpu")


def _timed(call, sync, reps=REPS):
    """One discarded warm-up, then `reps` timed calls. Returns (median, min, max, all)."""
    call()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        call()
        sync()
        ts.append(time.perf_counter() - t0)
    ts_sorted = sorted(ts)
    return ts_sorted[len(ts) // 2], ts_sorted[0], ts_sorted[-1], ts


def measure(preset: str, wmax: int, qc, obs, thetas, thetas_b):
    """Compile one program under `preset` and run the guarded measurement phases."""
    row = {"preset": preset, "w_max": wmax}
    val = grad = None
    sync = torch.cuda.synchronize if _uses_cuda(preset) else (lambda: None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    try:
        t0 = time.perf_counter()
        prog = qc.compile(observables=[obs], preset=preset, max_weight=wmax)
        sync()
        row["compile_s"] = time.perf_counter() - t0
        row["terms"] = int(prog.psum_union.x_mask.shape[0])
        row["step_nnz"] = sum(int(s.mat_const._nnz()) + int(s.mat_cos._nnz())
                              + int(s.mat_sin._nnz()) for s in prog.psum_union.steps)
    except RuntimeError as exc:
        row["compile_error"] = str(exc)[:160]
        return row, val, grad

    try:  # single-theta forward
        val = prog.expvals(thetas).detach().cpu().double()
        med, lo, hi, all_ts = _timed(lambda: prog.expvals(thetas), sync)
        row["eval_s"], row["eval_s_min"], row["eval_s_max"] = med, lo, hi
        row["eval_reps_s"] = all_ts
    except RuntimeError as exc:
        row["eval_error"] = str(exc)[:160]

    try:  # batched forward
        med, lo, hi, all_ts = _timed(lambda: prog.expvals(thetas_b), sync)
        row["batch_eval_s"], row["batch_eval_s_min"], row["batch_eval_s_max"] = med, lo, hi
        row["batch_eval_reps_s"] = all_ts
    except RuntimeError as exc:
        row["batch_error"] = str(exc)[:160]

    try:  # vjp fwd+bwd, single theta (warm; the cold first call is reported separately)
        th = thetas.clone().requires_grad_(True)

        def _fwdbwd():
            th.grad = None
            prog.expvals(th).sum().backward()

        t0 = time.perf_counter()
        _fwdbwd()
        sync()
        row["fwdbwd_first_s"] = time.perf_counter() - t0

        med, lo, hi, all_ts = _timed(_fwdbwd, sync)
        row["fwdbwd_s"], row["fwdbwd_s_min"], row["fwdbwd_s_max"] = med, lo, hi
        row["fwdbwd_reps_s"] = all_ts
        grad = th.grad.detach().cpu().double()
    except RuntimeError as exc:
        row["fwdbwd_error"] = str(exc)[:160]

    prog.clear_cache()
    del prog
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row, val, grad


def main():
    if not torch.cuda.is_available():
        raise SystemExit("The hybrid/gpu presets require a GPU.")
    edges = er_edges(N, P, SEED)
    qc, k = build_qaoa(N, edges, DEPTH)
    obs = build_zz_observable(edges)
    thetas = torch.linspace(-0.3, 0.3, k, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    thetas_b = (torch.rand(BATCH, k, generator=g, dtype=torch.float64) - 0.5) * 0.6

    # Discarded warm-up: the first compile of the process pays CUDA context
    # creation, kernel autotune and allocator warm-up.
    measure("gpu", WMAXES[0], qc, obs, thetas, thetas_b)

    rows, parity = [], []
    for wmax in WMAXES:
        got = {}
        for preset in PRESETS:
            row, val, grad = measure(preset, wmax, qc, obs, thetas, thetas_b)
            rows.append(row)
            got[preset] = (val, grad)
            print(f"w={wmax} preset={preset:>6}: "
                  f"compile={row.get('compile_s', float('nan')):7.2f}s "
                  f"eval={row.get('eval_s', float('nan'))*1e3:8.2f}ms "
                  f"batch{BATCH}={row.get('batch_eval_s', float('nan'))*1e3:9.2f}ms "
                  f"fwdbwd={row.get('fwdbwd_s', float('nan'))*1e3:9.2f}ms "
                  f"err={[key for key in row if key.endswith('_error')]}", flush=True)
        # cross-preset parity against the gpu rows: same engine, same program,
        # so any disagreement beyond float64 roundoff is a bug
        p = {"w_max": wmax}
        ref_val, ref_grad = got["gpu"]
        for preset in ("cpu", "hybrid"):
            if got[preset][0] is not None and ref_val is not None:
                p[f"expval_max_abs_diff_{preset}_vs_gpu"] = float(
                    (got[preset][0] - ref_val).abs().max())
            if got[preset][1] is not None and ref_grad is not None:
                p[f"grad_max_abs_diff_{preset}_vs_gpu"] = float(
                    (got[preset][1] - ref_grad).abs().max())
        parity.append(p)
        print(f"w={wmax} parity: {p}", flush=True)

    payload = {
        "config": {"n": N, "depth": DEPTH, "p": P, "seed": SEED,
                   "w_max_sweep": WMAXES, "presets": PRESETS,
                   "dtype": "float64 (every preset's default)",
                   "reps": REPS, "batch": BATCH,
                   "timing": "median of REPS timed calls after one discarded warm-up",
                   "gpu": torch.cuda.get_device_name(0),
                   "torch": torch.__version__},
        "rows": rows,
        "parity": parity,
    }
    with open(OUT / "preset_comparison.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote", OUT / "preset_comparison.json")


if __name__ == "__main__":
    main()
