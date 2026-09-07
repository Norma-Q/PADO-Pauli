# Running this bundle on more than one GPU

One code tree, one output tree per device. The same scripts run on, for example,
an NVIDIA A100 and an AMD MI300X, and each fills its own folder at the repo root:

```
reproducibility/                  the code (this folder) — device-agnostic
engine_benchmarks/                the cross-engine comparison — its own bundle
results_on_A100/
├── scaling_accuracy_noise/{results,figures}/
├── random_circuit_statevector/results/
├── diff_mode_vjp/results/
├── zero_filter_ablation/results/
├── preset_comparison/results/      (MI300X only)
├── kicked_ising_127q/{results,figures}/
├── repeat_timing/{runs,logs}/
├── engine_benchmarks/results/
└── rerun_logs/
results_on_MI300X/                same shape (no engine_benchmarks — CUDA only)
```

Nothing has to be passed on the command line: `_outdir.py` reads the GPU name
from torch and derives the tag (`A100`, `MI300X`, ...). Just run:

```bash
bash run_rerun_all.sh
```

Check where a run would write before starting it:

```bash
python _outdir.py          # prints the device tag and the data root
```

Overrides, when the default is not what you want:

| variable | effect |
|---|---|
| `PPS_DEVICE_TAG=A100_rerun` | writes to `results_on_A100_rerun/` |
| `PPS_DATA_ROOT=/scratch/run7` | writes there instead of the repo root |
| `PPS_PL_GPU_DEVICE=lightning.kokkos` | overrides the GPU statevector baseline |

## Vendor differences the code handles by itself

- **GPU statevector baseline.** PennyLane's CUDA device is `lightning.gpu`
  (cuStateVec); on ROCm it is `lightning.amdgpu` (Kokkos/HIP), the only one with
  ROCm wheels. `_outdir.lightning_gpu_device()` picks by `torch.version.hip`, so
  no script names a vendor.
- **The engine itself.** `padopauli` is vendor-neutral: `compute_device="cuda"`
  maps to the AMD GPU on a ROCm torch build.

Still manual on AMD: `lightning.amdgpu` needs `LD_PRELOAD` set to the system
ROCm's `libhsa-runtime64.so.1` before Python starts, because torch bundles its
own library under the same SONAME. The drivers honour `LD_PRELOAD` if it is set
but never set a default.

## What is comparable across devices, and what is not

Accuracy, term counts and expectation values are deterministic: they must match
across devices (float32 round-off aside). Timing, VRAM and
throughput are device measurements and are expected to differ — that is the
point of keeping the runs apart.

Each result file gets a `<stem>.runmeta.json` sidecar recording the GPU,
CUDA/ROCm backend, platform, Python and package versions, so a number can always
be traced to the software and hardware that produced it.

## The two platforms behind the recorded trees

The trees shipped in this repository were recorded on:

| tree | GPU | host | software |
|---|---|---|---|
| `results_on_MI300X/` | AMD Instinct MI300X VF (gfx942), 191.7 GB of device memory as reported by the driver (`/sys/class/drm/card*/device/mem_info_vram_total`, 205,822,885,888 bytes); amdgpu driver 6.16.13 | Intel Xeon Platinum 8568Y+, 20 vCPUs, 236 GB RAM, under a KVM hypervisor | PyTorch 2.11.0 (ROCm 7.2.4), Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1; PennyLane 0.45.1 for every exact statevector reference (the engine-only runs of 2026-08-07 record PennyLane 0.44.1 as installed but never import it; the `preset_comparison` sidecar predates the stamper and records torch only) |
| `results_on_A100/` | NVIDIA A100 80GB PCIe; NVIDIA driver 550.54.14 | Intel Xeon Gold 6338 at 2.00 GHz, 24 vCPUs, under a KVM hypervisor | PyTorch 2.11.0 (CUDA 12.6 runtime), Python 3.11.15, PennyLane 0.45.1 |

These are the values the paper's Code and Data Availability statement quotes.
GB means 2^30 bytes. The sidecars carry the package versions per result
file; the host figures above were read once with `lscpu`, `free`, `nvidia-smi`
and the driver files named. The sidecars published in the public repository omit hostnames and checkout
paths; in the run logs (`rerun_logs/`, `repeat_timing/logs/`,
`engine_benchmarks/results/run_log.txt`) and the engine-benchmark `cmd` fields
the recording checkout appears as `<repo>/` and its interpreters as plain
`python` / `julia`.

## History

- **2026-08-21** — this bundle was branched off the recorded MI300X run to be
  re-measured on an A100 (NVIDIA, torch 2.11+cu126, `padopauli` 2.0.0 from the
  release wheel). Several paths escaped the bundle and were rerouted; the
  cross-engine benchmark's cuPauliProp and Julia workers ran for the first time
  and needed fixes.
- **2026-08-24** — restructured to the layout above: code lives once at the repo
  root, data is device-scoped, and the device-specific pieces are resolved at
  runtime instead of being edited per machine. Before this, each experiment wrote
  next to its own script, so a second GPU could not be measured without
  overwriting the first.
