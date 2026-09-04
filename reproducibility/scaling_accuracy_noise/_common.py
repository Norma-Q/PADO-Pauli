"""Shared helpers for the scaling / accuracy / noise reproduction experiments.

Circuit family (fully specified for reproducibility):
  - Graph: Erdos-Renyi on N nodes, edge added iff np.random.default_rng(seed).random() < p.
           Unweighted MaxCut (each edge contributes Z_i Z_j with coefficient 1.0).
  - Ansatz: depth-D QAOA. One Hadamard init layer, then D blocks of
            (ZZ rotation per edge = cost) + (X rotation per qubit = mixer).
  - Observable: sum_{(i,j) in E} Z_i Z_j.

Term counts are structural under max-weight truncation, hence independent of the
rotation angles; no theta vector is needed to reproduce them.
"""

from __future__ import annotations

import math
import os
import time
import resource
import sys
from typing import Dict, List, Tuple

import numpy as np


def er_edges(n: int, p: float, seed: int) -> List[Tuple[int, int]]:
    """Erdos-Renyi edge list via numpy default_rng (reproducible)."""
    rng = np.random.default_rng(seed)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]


def build_qaoa(n: int, edges: List[Tuple[int, int]], depth: int):
    """H init + depth x (ZZ cost rotations + X mixer rotations); returns (Circuit, n_params)."""
    from padopauli import Circuit

    qc = Circuit(n)
    for q in range(n):
        qc.h(q)
    k = 0
    for _ in range(depth):
        for (u, v) in edges:
            qc.rzz(u, v, param_idx=k); k += 1
        for q in range(n):
            qc.rx(q, param_idx=k); k += 1
    return qc, k


def build_zz_observable(edges: List[Tuple[int, int]]):
    """sum_{(u,v) in E} Z_u Z_v as ONE multi-term tuple-spec observable."""
    return [("ZZ", [u, v]) for (u, v) in edges]


def zz_paulisum(n: int, edges: List[Tuple[int, int]]):
    """PauliSum form of the ZZ observable (the PennyLane reference API needs one)."""
    from padopauli import PauliSum

    obs = PauliSum(n)
    for (u, v) in edges:
        obs.add_from_str("ZZ", 1.0, qubits=[u, v])
    return obs


def weight_limited_upper_bound(n: int, w_max: int) -> int:
    """Number of Pauli strings of weight 1..w_max: sum_w C(n,w) 3^w."""
    return int(sum(math.comb(n, w) * (3 ** w) for w in range(1, w_max + 1)))


def _cur_rss_gb() -> float:
    """Current resident set size (GB) from /proc/self/status."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / (1024.0 * 1024.0)  # KB -> GB
    return float("nan")


def _peak_rss_gb() -> float:
    """Process high-water RSS (GB). ru_maxrss is KB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _gpu_used_mb() -> float:
    """Whole-GPU used memory (MiB) via NVML; cross-check on torch's per-process stats."""
    try:
        import pynvml

        if not getattr(_gpu_used_mb, "_init", False):
            pynvml.nvmlInit()
            _gpu_used_mb._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            _gpu_used_mb._init = True
        info = pynvml.nvmlDeviceGetMemoryInfo(_gpu_used_mb._handle)
        return float(info.used) / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def require_gpu_for_preset(preset: str) -> None:
    """Hard guard: if a GPU-using preset is requested, CUDA must be available.

    We never silently fall back to a CPU-only path.
    """
    import torch

    if preset in {"hybrid", "gpu"} and not torch.cuda.is_available():
        raise RuntimeError(
            f"preset='{preset}' requires CUDA, but torch.cuda.is_available() is False. "
            "Refusing to fall back to CPU. Fix the environment and rerun."
        )


def compile_and_measure(n: int, w_max: int, preset: str, depth: int, p: float, seed: int) -> Dict:
    """Compile one MaxCut-QAOA surrogate and measure term counts, time, RAM delta."""
    os.environ["PPS_COMPILE_PROFILE"] = "1"
    import torch  # noqa: F401  (ensures CUDA context initialized before baseline)

    require_gpu_for_preset(preset)

    edges = er_edges(n, p, seed)
    qc, n_params = build_qaoa(n, edges, depth)
    obs = build_zz_observable(edges)

    on_gpu = preset in {"hybrid", "gpu"}
    vram_baseline_nvml = float("nan")
    # Touch CUDA so its context memory is part of the baseline, not the delta.
    if on_gpu:
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        vram_baseline_nvml = _gpu_used_mb()

    rss_baseline = _cur_rss_gb()
    # float32 pinned: this helper feeds the heavy term-count/memory sweeps, where
    # the recorded results use the fp32 working set (no PennyLane comparison here).
    prog = qc.compile(observables=[obs], preset=preset, max_weight=int(w_max), dtype="float32")
    if on_gpu:
        torch.cuda.synchronize()

    rss_peak = _peak_rss_gb()
    # GPU VRAM footprint of the compiled program. torch's caching-allocator
    # high-water marks (reset above) capture the transient propagation peak
    # per-process; NVML used-delta is a whole-GPU cross-check. None on cpu preset.
    if on_gpu:
        vram_peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
        vram_peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024.0 ** 3)
        vram_nvml_delta_gb = (_gpu_used_mb() - vram_baseline_nvml) / 1024.0
    else:
        vram_peak_alloc_gb = None
        vram_peak_reserved_gb = None
        vram_nvml_delta_gb = None
    summary = prog.compile_stats.get("summary", {})

    # Per-query cost of the compiled program: one discarded warm-up then the
    # median of five calls (same protocol as run_zero_filter_ablation.py), on a
    # single parameter vector and on a batch of 200. step_nnz is the number of
    # nonzeros the evaluation streams.
    eval_s = batch_eval_s = step_nnz = None
    if on_gpu:
        try:
            dev = prog.psum_union.x_mask.device
            th = torch.linspace(-0.3, 0.3, n_params, dtype=torch.float64, device=dev)
            g = torch.Generator().manual_seed(0)
            th_b = ((torch.rand(200, n_params, generator=g, dtype=torch.float64) - 0.5)
                    * 0.6).to(dev)

            def _timed(call, reps=5):
                call(); torch.cuda.synchronize()
                ts = []
                for _ in range(reps):
                    t0 = time.perf_counter(); call(); torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                return sorted(ts)[len(ts) // 2]

            eval_s = _timed(lambda: prog.expvals(th))
            batch_eval_s = _timed(lambda: prog.expvals(th_b))
            step_nnz = sum(int(s.mat_const._nnz()) + int(s.mat_cos._nnz())
                           + int(s.mat_sin._nnz()) for s in prog.psum_union.steps)
        except RuntimeError:
            torch.cuda.empty_cache()

    xm = prog.psum_union.x_mask
    diag = int((xm == 0).sum()) if xm.dim() == 1 else int((xm == 0).all(dim=1).sum())

    return {
        "n_qubits": int(n),
        "w_max": int(w_max),
        "preset": preset,
        "depth": int(depth),
        "edge_prob": float(p),
        "seed": int(seed),
        "n_edges": int(len(edges)),
        "n_params": int(n_params),
        "propagated_terms": int(summary.get("propagated_terms", -1)),
        "zero_filtered_terms": int(summary.get("final_terms", prog.psum_union.x_mask.shape[0])),
        "diagonal_terms": diag,
        "upper_bound": weight_limited_upper_bound(n, int(w_max)),
        "propagate_total_s": float(summary.get("propagate_total_s", float("nan"))),
        "compile_total_s": float(summary.get("compile_total_s", float("nan"))),
        "rss_baseline_gb": rss_baseline,
        "rss_peak_gb": rss_peak,
        "rss_delta_gb": rss_peak - rss_baseline,
        "vram_peak_alloc_gb": vram_peak_alloc_gb,
        "vram_peak_reserved_gb": vram_peak_reserved_gb,
        "vram_nvml_delta_gb": vram_nvml_delta_gb,
        "eval_s": eval_s,
        "batch_eval_200_s": batch_eval_s,
        "step_nnz": step_nnz,
    }
