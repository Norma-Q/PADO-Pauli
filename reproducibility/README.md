# PADO-Pauli (PPS) — Paper Reproducibility Bundle

Scripts that reproduce the quantitative figures and tables of the PADO-Pauli
paper. This folder holds code only — every run writes its data into a sibling
`../results_on_<DEVICE>/` tree (see [DEVICE_RUNS.md](DEVICE_RUNS.md)). It is
self-contained except for the engine itself: it imports the installed
`padopauli` package (the wheel from the repository README).

## Layout

```
reproducibility/  (repo root also holds: tutorial/, examples/)
├── README.md                  ← this index
├── DEVICE_RUNS.md             ← running the same code on several GPUs (e.g. A100 / MI300X)
├── _outdir.py                 ← decides where a run writes: ../results_on_<DEVICE>/
├── _runmeta.py                ← writes <stem>.runmeta.json provenance sidecars
├── run_rerun_all.sh           ← re-runs every experiment below, on whatever GPU is present
├── scaling_accuracy_noise/    ← accuracy, truncation, noise, gradient, quasi-prob, scaling
│   ├── _common.py             ← shared circuit family + measurement helpers
│   ├── sweep_*.py             ← one experiment each (see table below)
│   ├── sweep_accuracy_frontier.py ← accuracy-vs-term-count frontier + across-n convergence
│   ├── make_figures.py        ← re-renders the scaling plots from the run's results/*.json
│   ├── run_all.sh             ← driver for the two structural sweeps
│                              (outputs go to ../results_on_<DEVICE>/scaling_accuracy_noise/)
├── random_circuit_statevector/      ← PPS vs exact statevector on the random-circuit stress test
│   ├── run_random_circuit_sweep.py      ← truncation errors + forward throughput vs lightning.qubit/lightning.amdgpu
│   ├── run_fwdbwd.py                    ← fwd+bwd Adam-loop timing vs the same two exact baselines
│   └── run_random_circuit_termcounts.py ← propagated/zero-filtered term counts
├── diff_mode_vjp/            ← differentiation backend: manual sparse-chain VJP vs PyTorch autograd
│   └── run_diff_mode_benchmark.py  ← fwd+bwd time + peak VRAM, vjp vs autograd, w_max sweep
├── zero_filter_ablation/     ← zero filtering on/off ablation
│   └── run_zero_filter_ablation.py ← compiles with zero_filter=True/False and compares evaluation cost
├── preset_comparison/        ← cpu / hybrid / gpu preset timing on one circuit
│   └── run_preset_comparison.py    ← same circuit and dtype under each preset
├── kicked_ising_127q/        ← the 127-qubit kicked-Ising bundle (its own copy, see below)
└── repeat_timing/            ← repeat-run driver for the wall-clock numbers
    ├── run_repeats.sh         ← re-runs the timing-sensitive experiments above under a fixed protocol
    └── summarize_repeats.py   ← medians + min-max ranges; writes repeat_summary.json
                                  into ../results_on_<DEVICE>/repeat_timing/
```

Every experiment writes into `../results_on_<DEVICE>/` rather than next to its
script, so the same code can be run on, for example, an A100 and an MI300X without
one run overwriting another — see [DEVICE_RUNS.md](DEVICE_RUNS.md).

The worked-example notebooks live in the repo's top-level `../tutorial/` and
`../examples/` folders. The 127-qubit kicked-Ising benchmark has its own copy
here in `kicked_ising_127q/` (its `README.md` documents the setup and the re-run
commands) so a device run never writes into the tracked `../examples/` tree.

## Environment

- Python: **the full setup is the Install section of the repository README**
  (torch first, then `padopauli`, then the extras) — follow it rather than the
  short form here. In brief this bundle needs, on top of the engine:
  `pennylane==0.45.1` + `pennylane-lightning==0.45.0`, the vendor's GPU lightning
  (`pennylane-lightning-gpu` on NVIDIA / `pennylane-lightning-amdgpu` on AMD, both
  `0.45.0` — the matched 0.45 line is required), `matplotlib`, and, on NVIDIA,
  `nvidia-ml-py`. torch, NumPy, SciPy and tqdm come in with the package.
- `nvidia-ml-py` is optional: without it every script still runs and
  `vram_peak_alloc_gb` / `vram_peak_reserved_gb` (from torch, vendor-neutral) are
  unaffected, but `vram_nvml_delta_gb` comes out `NaN` without an error.
- GPU: any CUDA- or ROCm-capable GPU; the tag is derived at runtime and each run
  fills its own `../results_on_<DEVICE>/` tree ([DEVICE_RUNS.md](DEVICE_RUNS.md)).
  Accuracy / term counts are hardware-independent; timing / VRAM are device-specific.
  The host, GPU and package versions behind any result file are in its
  `.runmeta.json` sidecar.
- The optional cross-engine benchmark is a separate bundle at the repo root,
  `../engine_benchmarks/`, and is CUDA-only (cuPauliProp has no ROCm build). It
  needs the three engines at the versions the paper benchmarked:
  `pip install cuquantum-python-cu12==26.6.0` (cuPauliProp 0.4.0),
  `pip install qiskit==2.5.2 pauli-prop==0.2.1`, and Julia 1.12.7 with
  `PauliPropagation.jl` 0.8.1 and `JSON.jl` in the active Julia project (the `julia`
  binary is found on `PATH`, then `~/.juliaup/bin/julia`, or via `PPS_JULIA`). It is
  not part of `run_rerun_all.sh` and has to be run separately on a CUDA machine.
- Exact baselines use PennyLane `default.qubit` (small-system statevector) and
  `default.mixed` (noisy density-matrix), both CPU. The stress test and the
  at-scale accuracy sweep additionally use `lightning.qubit` (CPU,
  `pennylane-lightning`) and the GPU statevector, whose device name
  `_outdir.lightning_gpu_device()` picks by vendor: `lightning.gpu`
  (`pennylane-lightning-gpu`, cuStateVec) on NVIDIA, `lightning.amdgpu`
  (`pennylane-lightning-amdgpu`, HIP build of the Kokkos backend) on AMD.
  `PPS_PL_GPU_DEVICE` overrides it.
- If you use `lightning.amdgpu`, set `LD_PRELOAD` to your system ROCm's
  `libhsa-runtime64.so.1` (e.g. `/opt/rocm-<ver>/lib/libhsa-runtime64.so.1`)
  before Python starts: torch bundles its own ROCm `libhsa-runtime64.so` under the
  same SONAME and claims it first, which otherwise breaks the device with
  `undefined symbol: hsa_amd_memory_get_preferred_copy_engine`. The drivers here
  do not set a default; they use `LD_PRELOAD` if it is set.

## Circuit families (fully specified)

- **ER MaxCut-QAOA** (`_common.py`): Erdős–Rényi graph, edge `(i,j)` added iff
  `numpy.random.default_rng(seed).random() < p`; unweighted cost
  `O = Σ_(i,j)∈E Z_iZ_j`. Ansatz: one Hadamard layer, then depth-`D` blocks of
  (per-edge `ZZ` cost rotation) + (per-qubit `X` mixer rotation).

## Re-running everything at once

`run_rerun_all.sh` drives every experiment that runs on an AMD (ROCm) GPU, cheapest
and most deterministic first, one process at a time (`python` on `PATH` must have
`padopauli` installed):

```bash
bash reproducibility/run_rerun_all.sh          # all jobs
JOBS="grad_quasi noise" bash reproducibility/run_rerun_all.sh   # a subset
DRYRUN=1 bash reproducibility/run_rerun_all.sh # just print the plan
```

It refuses to start if the env is missing a GPU or an importable `padopauli`
engine. Per-job logs and a status table land in
`../results_on_<DEVICE>/rerun_logs/`; a failed job is recorded and skipped rather than aborting the run,
and a job already marked `ok` is skipped unless `FORCE=1`.

The `qubit_seeds` and `frontier_seeds` jobs re-run the qubit sweep and the
across-n accuracy sweep with extra Erdős–Rényi seeds (`QUBIT_SEEDS`, default
`7 123`); every other experiment stays on the recorded seed 42. Extra seeds never
overwrite the recorded run — they write `qubit_sweep_s<seed>.{json,csv}` and
`accuracy_scale_s<seed>.*`.

Result files written through `run_rerun_all.sh` get a `<stem>.runmeta.json` sidecar recording what produced
it: GPU model and ROCm/CUDA backend, platform and Python version,
torch/PennyLane/NumPy versions, the `padopauli` version, and the `LD_PRELOAD` in
effect. Sidecars rather than a new key because several result files are JSON
lists at the top level. Inspect the current environment with `python -m reproducibility._runmeta`.

When comparing a re-run against the recorded data: *deterministic* quantities —
term counts, accuracies, expectation values — should not move at all (float32
round-off aside); *timing/memory* fields are hardware- and session-dependent and
are expected to differ.

## Paper artifact → script → data

Every path in the data columns below is relative to this bundle's output tree,
`../results_on_<DEVICE>/<experiment folder>/` — e.g. `results/noise.json` is
`../results_on_A100/scaling_accuracy_noise/results/noise.json` on an A100 run.

### `scaling_accuracy_noise/`

| Paper artifact | Script | Recorded data |
|---|---|---|
| Max-weight sweep numbers of Sec. 3.4.2 (15q, depth-5; quoted in the text, no table or figure) | `sweep_wmax_fixed_circuit.py` | `results/wmax_sweep.{json,csv}`, `point_w*_*.json` |
| Terms-vs-qubits figure, VRAM-vs-qubits figure, compile times (10–100q, fixed w_max=3, `gpu` preset) | `sweep_qubits_fixed_wmax.py` | `results/qubit_sweep.{json,csv}` (one full run); compile/propagate time medians come from the five runs in `repeat_timing/runs/` |
| Min-abs convergence figure (16q) | `sweep_minabs_convergence.py` | `results/minabs_convergence.{json,csv}` |
| Accuracy-vs-term-count frontier figure (16q) and across-n convergence figure (n = 20/24/28) | `sweep_accuracy_frontier.py` | `results/accuracy_frontier*.{json,csv}`, `results/accuracy_scale*.{json,csv}`, `results/scale_exact_cache.json` |
| Gradient-check and quasi-probability numbers of Sec. 3.3 and 3.5 (quoted in the text, no figure) | `sweep_grad_and_quasi.py` | `results/grad_quasi.json` |
| Noise validation, amplitude-damping and noise-assisted-truncation figures (10q, vs `default.mixed`) | `sweep_noise.py` | `results/noise.json` |

Each plotting script writes its PNG/PDF into
`../results_on_<DEVICE>/scaling_accuracy_noise/figures/`.

```bash
cd scaling_accuracy_noise
python sweep_minabs_convergence.py        # one figure-set per script
python sweep_accuracy_frontier.py
python sweep_grad_and_quasi.py
python sweep_noise.py
bash run_all.sh                        # the two structural sweeps (wmax + qubits)
```

### Cross-engine benchmark — moved out of this bundle

The comparison against cuPauliProp, Qiskit `pauli-prop` and Julia
`PauliPropagation.jl` is not a reproduction of this paper's own numbers: it
needs a different toolchain (CUDA-only SDKs plus Julia) and answers a different
question. It now lives beside this folder, at the repository root:

```
../engine_benchmarks/      orchestrator + per-engine workers (code only; a run
                           writes to ../results_on_<DEVICE>/engine_benchmarks/)
```

```bash
cd ../engine_benchmarks
PPS_JULIA=<path to julia> python benchmark_all_sdks.py \
    --engines cupauliprop,pps,qiskit,julia,julia_surrogate
```

### `random_circuit_statevector/`  (random-circuit stress test)

| Paper artifact | Source |
|---|---|
| Truncation-error table; forward-throughput table vs `lightning.qubit` (CPU, up to n=16) and `lightning.amdgpu` | `run_random_circuit_sweep.py` → `results/random_circuit_sweep.json` |
| fwd+bwd Adam-loop speedups vs the same two exact baselines | `run_fwdbwd.py` → `repeat_timing/runs/repeat1/fwdbwd/random_circuit_fwdbwd.json`, the run the manuscript quotes (`fwdbwd` is a ONCE_JOB, so repeat1 is its only repeat). `results/random_circuit_fwdbwd.json` is a separate same-protocol run and agrees within about 1%: 16.9x vs `lightning.qubit` at n=16 where repeat1 reads 16.8x (9.4x vs 9.3x against `lightning.amdgpu`) |
| Propagated / zero-filtered term-count columns of the truncation-error table | `run_random_circuit_termcounts.py` → `results/random_circuit_termcounts.json` |

Random quantum circuit `U(γ)+V(θ)`: random RX+CZ generative circuit `U(γ)` with
random angles followed by a trainable circuit `V(θ)` (construction from Hirviniemi,
Basheer, Cope, "Random Quantum Circuits as Seeds for Continuous Generative Models",
arXiv:2602.10049; hard for Pauli propagation). γ = per-sample embedding input
(non-trainable, passed as data), θ trainable via `thetas`; `n=12,16,20`,
`max_weight=9`, `gpu` preset.

### `diff_mode_vjp/`  (PyTorch integration and automatic differentiation)

| Paper artifact | Source |
|---|---|
| `vjp` vs `autograd` forward+backward time and peak VRAM, with the cross-mode gradient difference | `run_diff_mode_benchmark.py` → `results/diff_mode_vjp.json` |

Differentiation-backend comparison on the ER MaxCut-QAOA family (`n=16`, depth 5,
`gpu` preset), sweeping `max_weight ∈ {4,5,6,7}`. Both modes return the same
gradient; the script records each mode's forward+backward time, peak reserved
VRAM, and the cross-mode gradient difference. Timings/VRAM are hardware-dependent.

```bash
python reproducibility/diff_mode_vjp/run_diff_mode_benchmark.py
```

### `zero_filter_ablation/`  (zero-filtering ablation)

| Paper artifact | Source |
|---|---|
| Zero-filtering on/off ablation | `run_zero_filter_ablation.py` → `results/zero_filter_ablation.json` |

Compiles the same program with `zero_filter=True` (default) and `zero_filter=False`
(`compile_program(..., zero_filter=False)`) and records the evaluation-set
size and per-query cost of each, isolating what zero filtering alone contributes.

### `preset_comparison/`  (in-engine preset comparison)

| Paper artifact | Source |
|---|---|
| cpu/hybrid/gpu preset comparison | `run_preset_comparison.py` → `results/preset_comparison.json` |

Compiles the zero-filter-ablation instance once per execution preset (all three at
their shared `float64` default dtype) and times compile, single and batched evaluation,
and the warm vjp forward+backward pass, isolating what the GPU execution path buys
inside the engine itself, separately from any cross-engine comparison.

### `kicked_ising_127q/`

| Paper artifact | Source |
|---|---|
| 127q kicked-Ising `<Z_62>` vs RX angle figure (max-weight thresholds 3–8) | `run_kicked_ising.py --indices all --preset gpu --max-weight W --min-abs 1e-4 --tag wW` for W = 3…7 and the default `--tag full` for W = 8 (`run_sweep.sh` is the one-field-per-process driver for that `full` run) → `results/kicked_ising_127q_{w3,…,w7,full}.{json,csv}`; `make_convergence_figure.py` → `figures/fig_kicked_ising_127q_wmax.png`, the paper's figure. `make_figure.py` → `figures/fig_kicked_ising_127q.{png,pdf}` draws the `max_weight=8` curve alone |

The notebook version of this case is `../examples/01_kicked_ising_127q.ipynb`;
the scripts here are the paper run, and they write to the device tree rather
than into the tracked `../examples/` folder.

```bash
cd kicked_ising_127q
python run_kicked_ising.py --indices all --preset gpu --max-weight 8 --min-abs 1e-4   # -> _full
for W in 3 4 5 6 7; do
  python run_kicked_ising.py --indices all --preset gpu --max-weight $W --min-abs 1e-4 --tag w$W
done
python make_convergence_figure.py   # the paper's figure (all six thresholds)
python make_figure.py               # the max_weight=8 curve alone
```

### `repeat_timing/`  (the wall-clock numbers)

Wall-clock timings move between sessions while term counts, accuracies and VRAM
reproduce bit-identically, so the timing-sensitive experiments are repeated under
a fixed protocol (`run_repeats.sh`):

- Experiments that measure **compilation** — the 10–100q sweep and the 127q
  kicked-Ising sweep — are compiled **5 times**; the median with the min–max range
  is reported.
- Experiments that measure **evaluation** — `diff_mode_vjp`, the fwd+bwd Adam loop and
  the throughput sweep — are compiled **once** and evaluated 5 times inside the script
  (one extra discarded warm-up).

`repeat_timing/runs/repeat{1..5}/<job>/` holds the raw outputs of the repeat runs
and `repeat_timing/repeat_summary.json` the medians/ranges (`summarize_repeats.py`
regenerates it) — both inside `../results_on_<DEVICE>/`.
Each experiment's own `results/` folder there holds a same-protocol run: its deterministic
fields (term counts, accuracies, deviations, VRAM) match `runs/repeat1/`, while
wall-clock fields differ session to session and the fwd+bwd training
losses/speedups differ slightly (GPU-nondeterministic reduction order). The
`diff_mode_vjp` timings and the per-query evaluation fields of
`scaling_accuracy_noise/results/qubit_sweep.json` + `point_q*_gpu.json` are taken
from those `results/` files; the compile-time medians of the 10–100q sweep come
from `repeat_timing/runs/`.

```bash
bash repeat_timing/run_repeats.sh          # all five jobs
python repeat_timing/summarize_repeats.py  # medians + ranges
```

## Worked-example figures

These come from the runnable notebooks in the repo's top-level `../tutorial/` (API
walkthroughs) and `../examples/` (application cases) folders. Executing a notebook
writes its figure(s) to the `figures/` folder beside it (each plotting cell calls
`plt.savefig("figures/<name>.png")`).

| Figure | Notebook |
|---|---|
| embedding models: sin / linear / XOR (`fig_sin_regression.png`, `fig_class_linear.png`, `fig_class_xor.png`) | `../tutorial/04_embedding_batched_inputs_basics.ipynb` |
| VQE H2 dissociation (`fig_vqe_h2_dissociation.png`; also writes `fig_vqe_h2_convergence.png`) | `../examples/02_vqe_h2_molecule.ipynb` |
| QAOA cut histogram (`fig_qaoa_cut_hist.png`) | `../examples/03_maxcut_qaoa.ipynb` |
| SAFE ma-QAOA convergence (`fig_SAFE_ma-QAOA_convergence.png`; also writes `fig_SAFE_ma-QAOA_summary.png`) | `../examples/04_SAFE_ma-QAOA.ipynb` |

The kicked-Ising figure is not notebook-generated — it comes from the scripts in
`../examples/kicked_ising_127q/` (see the section above).
