"""PPS (padopauli) subprocess worker.

Usage:
    python engine_pps.py <config.json> <result.json>

Runs the PPS (padopauli) compile/eval pipeline and produces result
records in the unified format used by this benchmark suite.

Memory is measured as GPU VRAM via pynvml (peak delta during forward eval).
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


def _build_circuit_and_obs(problem: Dict[str, Any],
                            gate_defs: List[Dict[str, Any]]):
    from padopauli import (
        CliffordGate as PPSCliffordGate,
        PauliRotation as PPSPauliRotation,
        PauliSum,
    )
    n_qubits = int(problem["n_qubits"])
    obs = PauliSum(n_qubits)
    for u, v, w in problem["edges"]:
        obs.add_from_str("ZZ", float(w), qubits=[int(u), int(v)])
    for q, h in problem["fields"]:
        obs.add_from_str("Z", float(h), qubits=[int(q)])
    circuit = []
    for g in gate_defs:
        t = g["type"]
        if t == "clifford":
            circuit.append(PPSCliffordGate(g["symbol"], list(g["qubits"])))
        elif t == "embedding":
            circuit.append(PPSPauliRotation(
                g["pauli"], list(g["qubits"]), embedding_idx=int(g["eidx"]),
            ))
        elif t == "rotation":
            circuit.append(PPSPauliRotation(
                g["pauli"], list(g["qubits"]), param_idx=int(g["pidx"]),
            ))
        else:
            raise ValueError(f"Unsupported gate type: {t}")
    return circuit, obs


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    import torch
    import cupy as cp
    from padopauli import compile_program

    problem = config["problem"]
    gate_defs = config["gate_defs"]
    theta = np.asarray(config["theta"], dtype=np.float32)
    embedding_cfg = config.get("embedding")
    embedding = (
        np.asarray(embedding_cfg, dtype=np.float32)
        if embedding_cfg is not None else None
    )
    is_embedding_batch = embedding is not None and embedding.ndim == 2
    batch_size = int(embedding.shape[0]) if is_embedding_batch else 1
    n_reps = int(config.get("n_reps", 5))

    max_weight = config.get("max_weight")
    min_abs_coeff = config.get("min_abs_coeff")

    n_qubits = int(problem["n_qubits"])

    # CUDA warmup before measurement
    _t = torch.zeros(1, device="cuda"); del _t
    cp.zeros(1)
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()

    _cleanup()
    circuit, obs = _build_circuit_and_obs(problem, gate_defs)

    build_thetas = (
        torch.tensor(theta, device="cuda") if min_abs_coeff is not None else None
    )

    mem_before = _gpu_used_mb()
    t0 = time.perf_counter()
    program = compile_program(
        circuit=circuit, observables=[obs], preset="gpu",
        preset_overrides={
            "max_weight": int(max_weight) if max_weight else n_qubits,
            "dtype": "float32",
        },
        build_thetas=build_thetas,
        build_min_abs=float(min_abs_coeff) if min_abs_coeff is not None else None,
    )
    torch.cuda.synchronize()
    compile_time = time.perf_counter() - t0
    used_after_compile = _gpu_used_mb()
    compile_mem = max(0.0, used_after_compile - mem_before)

    # Prepare batched theta / embedding for expvals
    theta_batch = theta[None, :]
    th = torch.tensor(theta_batch, device="cuda", requires_grad=True)
    if embedding is None:
        em = None
    elif embedding.ndim == 1:
        em = torch.tensor(embedding[None, :], device="cuda")
    elif embedding.ndim == 2:
        em = torch.tensor(embedding, device="cuda")
    else:
        raise ValueError(f"embedding must be 1D or 2D, got shape {embedding.shape}")

    history: List[Dict[str, Any]] = []
    for step in range(n_reps):
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()
        mem_baseline = _gpu_used_mb()

        t0 = time.perf_counter()
        res = program.expvals(th, embedding=em, diff_mode="vjp")
        torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t0

        # Optional backward — measured separately, not in t_eval_s
        t0 = time.perf_counter()
        res.sum().backward(retain_graph=False)
        torch.cuda.synchronize()
        t_bwd = time.perf_counter() - t0

        # Absolute readings are kept alongside the delta (see engine_cpp.py).
        used_end = _gpu_used_mb()
        step_vram = max(0.0, used_end - mem_baseline)
        expval = float(res.detach().cpu().numpy().flatten()[0])

        # Clear grads so backward can run again next step
        if th.grad is not None:
            th.grad.detach_()
            th.grad.zero_()

        history.append({
            "step": step,
            "t_eval_s": t_fwd,
            "t_bwd_s": t_bwd,
            "step_vram_MB": step_vram,
            "vram_used_start_MB": mem_baseline,
            "vram_used_end_MB": used_end,
            "expval": expval,
            "n_terms": int(
                getattr(program, "compile_stats", {}).get("summary", {}).get("final_terms", 0)
            ),
            "batch_size": batch_size,
            "throughput_evals_s": float(batch_size / t_fwd) if t_fwd > 0 else float("inf"),
            "throughput_train_s": float(batch_size / (t_fwd + t_bwd)) if (t_fwd + t_bwd) > 0 else float("inf"),
            "native_batch": bool(is_embedding_batch),
        })

    # Untimed: the gradient of the FIRST batch row, recorded so the backward
    # pass is verifiable rather than merely timed. The timed backward above
    # differentiates the batch sum; the other engines loop rows one at a time,
    # so row 0 is the one quantity every engine can produce.
    # The propagated (pre-zero-filter) count, so the report can state both
    # numbers for PADO the way the other engines' single number is stated.
    # Untimed and best-effort: an unfiltered compile keeps every term, so it is
    # allowed to fail (memory) without taking the measured run with it.
    n_terms_propagated = None
    try:
        unfiltered = compile_program(
            circuit=circuit, observables=[obs], preset="gpu",
            preset_overrides={
                "max_weight": int(max_weight) if max_weight else n_qubits,
                "dtype": "float32",
            },
            build_thetas=build_thetas,
            build_min_abs=float(min_abs_coeff) if min_abs_coeff is not None else None,
            zero_filter=False,
        )
        n_terms_propagated = int(unfiltered.psum_union.x_mask.shape[0])
        del unfiltered
        torch.cuda.empty_cache()
    except Exception:
        n_terms_propagated = None

    grad_row0 = None
    try:
        th_g = torch.tensor(theta_batch, device="cuda", requires_grad=True)
        res_g = program.expvals(th_g, embedding=em, diff_mode="vjp")
        res_g.reshape(res_g.shape[0], -1)[0].sum().backward()
        grad_row0 = [float(x) for x in th_g.grad.detach().cpu().numpy().reshape(-1)]
    except Exception:  # never let the gradient record break a timing run
        grad_row0 = None

    summary = summarize_history(history, time_key="t_eval_s", mem_key="step_vram_MB")
    final_terms = int(
        getattr(program, "compile_stats", {}).get("summary", {}).get(
            "final_terms",
            getattr(getattr(program, "psum_union", None), "x_mask", []).shape[0],
        )
    )
    summary["final_n_terms"] = final_terms

    return {
        "engine": "pps",
        "versions": package_versions("torch", "padopauli"),
        "status": "Success",
        # final_n_terms is the post-zero-filter set the program evaluates.
        # The pre-filter (propagated) count is not exposed by the public
        # API; the zero_filter_ablation experiment measures that instead.
        "n_terms_kind": "post_zero_filter",
        "n_terms_propagated": n_terms_propagated,
        "bwd_supported": True,
        "grad_row0": grad_row0,
        "grad_kind": "d(expval of batch row 0)/d theta",
        "compile_time_s": compile_time,
        "vram_used_before_compile_MB": mem_before,
        "vram_used_after_compile_MB": used_after_compile,
        "compile_vram_MB": compile_mem,
        "history": history,
        "summary": summary,
        "config_echo": {
            "max_weight_used": int(max_weight) if max_weight else n_qubits,
            "min_abs_coeff_used": float(min_abs_coeff) if min_abs_coeff is not None else None,
            "n_reps": n_reps,
            "device": "cuda",
            "memory_kind": "gpu_vram_peak_delta_MB",
            "setup_kind": "padopauli.compile_program",
            "batch_support": "native embedding batch" if is_embedding_batch else "single sample",
        },
    }


def main(argv: List[str]) -> int:
    config = load_config(argv)
    try:
        result = run(config)
    except Exception as e:
        result = {
            "engine": "pps",
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
