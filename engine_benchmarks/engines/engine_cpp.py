"""cuPauliProp (NVIDIA) subprocess worker.

Usage:
    python engine_cpp.py <config.json> <result.json>

Runs the cuPauliProp (cuQuantum) propagation pipeline and produces result
records in the unified format used by this benchmark suite.
"""
from __future__ import annotations

import gc
import sys
import time
import traceback
from typing import Any, Dict, List

from _common import (
    package_versions,
    load_config,
    save_result,
    summarize_history,
)



def _gpu_used_mb() -> float:
    import pynvml
    if not getattr(_gpu_used_mb, "_init", False):
        pynvml.nvmlInit()
        _gpu_used_mb._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        _gpu_used_mb._init = True
    info = pynvml.nvmlDeviceGetMemoryInfo(_gpu_used_mb._handle)
    return float(info.used) / (1024 ** 2)


def _cleanup() -> None:
    import torch
    import cupy as cp
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cp.get_default_memory_pool().free_all_blocks()


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    import torch
    import cupy as cp
    import cuquantum.pauliprop.experimental as ppe

    problem = config["problem"]
    gate_defs = config["gate_defs"]
    theta = np.asarray(config["theta"], dtype=np.float32)
    embedding_cfg = config.get("embedding")
    embedding = (
        np.asarray(embedding_cfg, dtype=np.float32)
        if embedding_cfg is not None else None
    )
    if embedding is None:
        embedding_rows = [None]
    elif embedding.ndim == 1:
        embedding_rows = [embedding]
    elif embedding.ndim == 2:
        embedding_rows = [embedding[i] for i in range(int(embedding.shape[0]))]
    else:
        raise ValueError(f"embedding must be 1D or 2D, got shape {embedding.shape}")
    is_embedding_batch = embedding is not None and embedding.ndim == 2
    batch_size = len(embedding_rows)
    n_reps = int(config.get("n_reps", 5))

    max_weight = config.get("max_weight")
    min_abs_coeff = config.get("min_abs_coeff")

    n_qubits = int(problem["n_qubits"])
    edges = problem["edges"]
    fields = problem["fields"]

    handle = ppe.LibraryHandle()

    # CUDA warmup before measurement
    _t = torch.zeros(1, device="cuda"); del _t
    cp.zeros(1)
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()

    _cleanup()

    # Build observable as PauliExpansion
    num_terms = len(edges) + len(fields)
    xz_size = ppe.get_num_packed_integers(n_qubits)
    xz_bits = cp.zeros((num_terms, 2 * xz_size), dtype=cp.uint64)
    coeffs = cp.zeros((num_terms,), dtype=cp.complex64)
    idx = 0
    for u, v, w in edges:
        xz_bits[idx, 1] = (1 << int(u)) | (1 << int(v))
        coeffs[idx] = float(w); idx += 1
    for q, h in fields:
        xz_bits[idx, 1] = (1 << int(q))
        coeffs[idx] = float(h); idx += 1

    mem_before = _gpu_used_mb()
    t0 = time.perf_counter()
    H_obs = ppe.PauliExpansion(handle, n_qubits, idx, xz_bits[:idx], coeffs[:idx])
    cp.cuda.Stream.null.synchronize()
    compile_time = time.perf_counter() - t0
    used_after_setup = _gpu_used_mb()
    compile_mem = max(0.0, used_after_setup - mem_before)

    trunc = ppe.Truncation(
        pauli_weight_cutoff=int(max_weight) if max_weight else None,
        pauli_coeff_cutoff=float(min_abs_coeff) if min_abs_coeff is not None else None,
    )

    grad_row0 = [0.0] * len(theta)

    history: List[Dict[str, Any]] = []
    for step in range(n_reps):
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()
        mem_baseline = _gpu_used_mb()

        t_fwd = 0.0
        t_bwd = 0.0
        expval = float("nan")
        n_terms = 0
        for row_idx, embed_row in enumerate(embedding_rows):
            t0 = time.perf_counter()
            active = H_obs
            obs_list = [active]
            for g in reversed(gate_defs):
                if g["type"] == "clifford":
                    gate = ppe.CliffordGate(g["symbol"], list(g["qubits"]))
                elif g["type"] == "embedding":
                    if embed_row is None:
                        raise ValueError("embedding gate present but no embedding vector supplied")
                    gate = ppe.PauliRotationGate(
                        float(embed_row[int(g["eidx"])]), g["pauli"], list(g["qubits"]),
                    )
                else:
                    gate = ppe.PauliRotationGate(
                        float(theta[int(g["pidx"])]), g["pauli"], list(g["qubits"]),
                    )
                active = active.apply_gate(gate, adjoint=True, truncation=trunc)
                obs_list.append(active)
            sig, ex = active.trace_with_zero_state()
            sample_expval = float((sig * (2.0 ** ex)).real)
            cp.cuda.Stream.null.synchronize()
            t_fwd += time.perf_counter() - t0
            if row_idx == 0:
                expval = sample_expval
            n_terms = int(active.num_terms)

            # Backward gradient (optional, kept consistent with PPS reporting).
            # Seed cotangents per the documented chain rule for the
            # (significand, exponent) trace pair, with upstream seed 1.0:
            #   d trace / d sig = 2**ex,   d trace / d ex = sig * 2**ex * ln 2
            t0 = time.perf_counter()
            record_grad = (step == 0 and row_idx == 0)
            cot_trace_sig = 1.0 * (2.0 ** ex) + 0.0j
            # cotangent_trace_exponent is a real scalar (the API stores it as
            # float64); the loss is the real part of the trace, so seed it with
            # the real part of sig * 2**ex * ln 2.
            cot_trace_ex = float((complex(sig) * (2.0 ** ex) * float(np.log(2.0))).real)
            cot_evolved = active.trace_with_zero_state_backward_diff(cot_trace_sig, cot_trace_ex)
            for i, g in enumerate(gate_defs):
                input_obs = obs_list[len(gate_defs) - i - 1]
                if g["type"] == "clifford":
                    gate = ppe.CliffordGate(g["symbol"], list(g["qubits"]))
                elif g["type"] == "embedding":
                    if embed_row is None:
                        raise ValueError("embedding gate present but no embedding vector supplied")
                    gate = ppe.PauliRotationGate(
                        float(embed_row[int(g["eidx"])]), g["pauli"], list(g["qubits"]),
                    )
                else:
                    gate = ppe.PauliRotationGate(
                        float(theta[int(g["pidx"])]), g["pauli"], list(g["qubits"]),
                    )
                # The second return value IS the gate-parameter gradient. It
                # used to be discarded, which left the backward pass timed but
                # unverified; keep row 0's so it can be checked.
                cot_evolved, param_grad = input_obs.apply_gate_backward_diff(
                    gate, cot_evolved.view(), adjoint=True, truncation=trunc,
                )
                if record_grad and g["type"] == "rotation":
                    val = param_grad
                    if hasattr(val, "get"):      # cupy -> host
                        val = val.get()
                    grad_row0[int(g["pidx"])] = float(np.real(np.ravel(val)[0]))
            cp.cuda.Stream.null.synchronize()
            t_bwd += time.perf_counter() - t0

        # Absolute readings are kept alongside the delta: the delta alone
        # cannot answer "how much VRAM does the engine hold during the loop"
        # (memory surviving into the baseline disappears from it).
        used_end = _gpu_used_mb()
        step_mem = max(0.0, used_end - mem_baseline)
        history.append({
            "step": step,
            "t_eval_s": t_fwd,
            "t_bwd_s": t_bwd,
            "step_vram_MB": step_mem,
            "vram_used_start_MB": mem_baseline,
            "vram_used_end_MB": used_end,
            "expval": expval,
            "n_terms": n_terms,
            "batch_size": batch_size,
            "throughput_evals_s": float(batch_size / t_fwd) if t_fwd > 0 else float("inf"),
            "throughput_train_s": float(batch_size / (t_fwd + t_bwd)) if (t_fwd + t_bwd) > 0 else float("inf"),
            "native_batch": False,
        })

    summary = summarize_history(history, time_key="t_eval_s", mem_key="step_vram_MB")
    return {
        "engine": "cupauliprop",
        "versions": package_versions("cupauliprop-cu12", "cuquantum-python-cu12", "cuquantum"),
        "status": "Success",
        # cuPauliProp has no zero filter: num_terms is everything propagated.
        "n_terms_kind": "propagated",
        "bwd_supported": True,
        "grad_row0": grad_row0,
        "grad_kind": "d(expval of batch row 0)/d theta",
        "compile_time_s": compile_time,
        "vram_used_before_setup_MB": mem_before,
        "vram_used_after_setup_MB": used_after_setup,
        "compile_vram_MB": compile_mem,
        "history": history,
        "summary": summary,
        "config_echo": {
            "max_weight_used": int(max_weight) if max_weight else None,
            "min_abs_coeff_used": float(min_abs_coeff) if min_abs_coeff is not None else None,
            "n_reps": n_reps,
            "device": "cuda",
            "memory_kind": "gpu_vram_peak_delta_MB",
            "setup_kind": "construct cuPauliProp PauliExpansion observable",
            "batch_support": "single-sample loop" if is_embedding_batch else "single sample",
        },
    }


def main(argv: List[str]) -> int:
    config = load_config(argv)
    try:
        result = run(config)
    except Exception as e:
        result = {
            "engine": "cupauliprop",
            "status": "Error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "history": [],
        }
    result["test_id"] = config.get("test_id")
    save_result(argv, result)
    return 0 if result.get("status") == "Success" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
