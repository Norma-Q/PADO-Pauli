# All-SDK Pauli Propagation Benchmark — Design Notes

This document records the design rationale behind the cross-engine benchmark.
The authoritative, runnable instructions live in `../README.md`; the
authoritative interface and schema are the shipped files themselves
(`benchmark_all_sdks.py`, `engines/_common.py`, and the per-engine workers).
Where this note states a concrete key or shape, it is described as the code
currently emits it — if in doubt, the code wins.

## Goal

Compare the PPS engine (`padopauli`, "PADO-Pauli") against other
Pauli-propagation implementations on identical circuit families, with
consistent timing and memory metrics.

## Engines and their workers

| Engine | Language | Device | Worker (`engines/`) |
|---|---|---|---|
| PPS (`padopauli`) | Python/C++ | GPU (CUDA) | `engine_pps.py` |
| cuPauliProp (cuQuantum) | Python/CUDA | GPU (CUDA) | `engine_cpp.py` |
| Qiskit `pauli-prop` | Python (Rust-accelerated) | CPU only | `engine_qiskit.py` |
| Julia `PauliPropagation.jl` | Julia | CPU only | `engine_julia.jl` |
| PP.jl surrogate (`julia_surrogate`) | Julia | CPU only | `engine_julia.jl --surrogate` |

The default engine set is `cupauliprop,pps,qiskit,julia,julia_surrogate`
(`benchmark_all_sdks.py`, `--engines`). `julia_surrogate` is PP.jl's
compile-once/evaluate-many NodePathProperties graph (build + `zerofilter!`
once = compile stage, `evaluate!` per repetition; the filter keeps only
|0⟩-contributing paths, as in PP.jl's own surrogate example, and is the
counterpart of PADO's zero filtering — the result records both
`n_terms_propagated`, pre-filter, and `final_n_terms`, post-filter); it
reports `Unsupported` on coeff_truncation cases because the surrogate cannot
truncate on numerical coefficient values. With PauliPropagation.jl v0.8.0+, the `julia` engine also
times the native `rewindgradient` (expval+grad in one paired sweep) as
`t_fwdbwd_s`; that is comparable to the other engines' `t_eval + t_bwd`, not
to `t_bwd` alone. Each worker reads a JSON config and
writes a JSON result; the exact SDK calls each worker makes are in its own
file (they are the reference, not reproduced here, so this note cannot drift
from them).

SDK documentation:
- cuPauliProp: https://docs.nvidia.com/cuda/cuquantum/latest/python/pauliprop.html
- Qiskit: https://github.com/Qiskit/pauli-prop
- Julia: https://github.com/MSRudolph/PauliPropagation.jl

## Orchestration: subprocess workers with file I/O

Each engine runs in its own subprocess for dependency/runtime isolation
(`benchmark_all_sdks.py`, `invoke_engine`):

```
benchmark_all_sdks.py
    ├─ write TestCase.to_config() to  results/_work/cfg_<engine>_<test_id>.json
    ├─ subprocess.run([*runner, script, cfg_path, res_path], capture_output=True, timeout=...)
    │      runner = [python]  for engine_pps/cpp/qiskit
    │      runner = [julia]   for engine_julia.jl
    │   worker: read config file → run propagation → write JSON to res_path
    └─ read  results/_work/res_<engine>_<test_id>.json  into the unified record
```

Worker stdout/stderr is captured only for logging (error tails on a nonzero
return code); results are exchanged through the result file, not stdout. The
orchestrator also records the subprocess wall time (`time.perf_counter()`)
for logging, separately from the in-worker steady-state timings below.

Rationale for subprocess isolation: it keeps each SDK's runtime (and, for
Julia, its JIT/precompile and GC) independent and out of the measured numbers,
and lets a worker that crashes or times out fail without taking the suite down.

## Config schema (`TestCase.to_config()`)

Each worker receives one JSON object of this shape (see
`benchmark_all_sdks.py`):

```json
{
    "test_id": "full_scaling__3x3",
    "suite": "full_scaling",
    "label": "3x3",
    "problem": {"n_qubits": 9, "edges": [[0, 1, 0.8]], "fields": [[0, 0.1]]},
    "gate_defs": [{"type": "rotation", "pauli": "ZZ", "qubits": [0, 1], "pidx": 0}],
    "theta": [0.0],
    "embedding": null,
    "max_weight": 4,
    "min_abs_coeff": 1e-8,
    "max_terms": null,
    "n_reps": 5,
    "extra": {}
}
```

The circuit is shipped **pre-built** as `gate_defs` (an ordered gate list the
orchestrator generates once from the QAOA problem), so every engine propagates
the identical circuit; there is no `p_layers` field in the config (`--p-layers`
is an orchestrator-side CLI knob used only when generating the cases).

## Suites

Suite ids (`--suites`, default = all four):

| Suite id | Sweep | PPS | cuPauliProp | Qiskit | Julia | Julia surrogate |
|---|---|---|---|---|---|---|
| `full_scaling` | qubit count | O | O | O | O | O |
| `mw_truncation` | `max_weight` | O | O | O\* | O | O |
| `coeff_truncation` | `min_abs_coeff` | O | O | O | O | Unsupported |
| `embedding_batch` | embedding-batch size | O (native) | O (looped) | O (looped) | O (looped) | O (looped, graph reused) |

\* Qiskit has no `max_weight` concept; it is driven by `max_terms`/`atol`
instead (see the truncation mapping below), so its `mw_truncation` points are
not a like-for-like weight cap.

All engines run on every suite by default. For `embedding_batch`, only PPS
evaluates a batch as one native operation; the cuPauliProp, Qiskit, and Julia
workers loop over the batch one sample at a time (`engine_cpp.py`,
`engine_qiskit.py`, `engine_julia.jl`). The Julia surrogate also loops, but
reuses the compiled, zerofiltered graph across samples (`evaluate!` per
sample).

## Timing and memory metrics

Steady-state numbers are the mean over `n_reps` measured repetitions with the
first (warm-up) repetition excluded. Timing and memory are measured *inside*
each worker; the GPU and CPU engines use different memory metrics, and the two
are never mixed in one column.

| Engine type | In-worker timing | Memory metric | `memory_kind` field |
|---|---|---|---|
| GPU (PPS, cuPauliProp) | `time.perf_counter()` | NVML GPU-used delta (MB) | `gpu_vram_peak_delta_MB` |
| CPU (Qiskit, Julia) | `time.perf_counter()` / `time_ns()` | windowed VmHWM delta (MB): peak-RSS counter reset per step via `/proc/self/clear_refs`, then `/proc/self/status` VmHWM read | `cpu_ram_highwater_delta_MB (VmHWM reset per step)` |

Without the per-step VmHWM reset, the kernel's peak-RSS counter is monotone
over the process lifetime, so CPU steady-state deltas after warm-up come out
structurally 0 (this was the case in runs before 2026-08). If the reset write
fails, the workers fall back to the old lifetime-peak behavior and say so in
`memory_kind`.

Per-step result keys: `step_vram_MB` / `compile_vram_MB` (GPU workers) and
`step_ram_MB` / `compile_ram_MB` (CPU workers). After aggregation, the rows in
`results/summary.json` carry `vram_first_MB` / `vram_steady_MB` (GPU) and
`ram_first_MB` / `ram_steady_MB` (CPU), alongside `t_eval_first_s` /
`t_eval_steady_s` and `final_n_terms` / `final_expval`. CPU rows leave the
VRAM columns empty and vice versa.

## Truncation parameter mapping

| Concept | PPS (`padopauli`) | cuPauliProp | Julia `PauliPropagation.jl` | Qiskit `pauli-prop` |
|---|---|---|---|---|
| Pauli-weight cap | `max_weight` | `max_weight` | `max_weight` | none (use `max_terms`) |
| Coefficient cutoff | `min_abs_coeff` | `min_abs` | `min_abs_coeff` | `atol` |
| Retained-term cap | — | — | — | `max_terms` |

The config carries `max_weight`, `min_abs_coeff`, and `max_terms`; each worker
maps the relevant fields onto its SDK's knobs. Because Qiskit lacks a weight
cap, its `mw_truncation` points are reported with that caveat.

## Environment

The Python engines run inside a CUDA environment carrying `pauli-prop`,
cuQuantum (`cupauliprop`), and PennyLane, alongside the installed `padopauli`
package. Julia is installed separately and needs
`PauliPropagation.jl` **and `JSON.jl`** in the active Julia project; the
`julia` binary is found on `PATH` or via the `PPS_JULIA` environment variable
(see `../README.md`).
