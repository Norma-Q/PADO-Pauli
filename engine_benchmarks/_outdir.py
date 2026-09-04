"""(Copy of reproducibility/_outdir.py — keep the two in sync.)

Where a run writes its data: `<repo>/results_on_<DEVICE>/<experiment>/...`.

One code tree, one output tree per GPU. Running the same script on an A100, an
H100 and an MI300X fills three sibling folders instead of overwriting one set of
files or needing three branches:

    results_on_A100/scaling_accuracy_noise/{results,figures}/
    results_on_H100/scaling_accuracy_noise/{results,figures}/
    results_on_MI300X/scaling_accuracy_noise/{results,figures}/

The device tag comes from the GPU torch reports, so nothing has to be passed on
the command line. Override either half when needed:

    PPS_DEVICE_TAG=A100_run2   ->  <repo>/results_on_A100_run2/...
    PPS_DATA_ROOT=/tmp/scratch ->  /tmp/scratch/...

`PPS_DATA_ROOT=.` restores the old in-place layout (`<experiment>/results/`).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Known accelerators, longest/most specific first. Anything else falls back to a
# sanitized form of the full name, which is ugly but never silently collides.
_KNOWN = [
    r"\bMI\d{3}[A-Z]*\b",                      # MI300X, MI250X
    r"\bRTX\s*\d{4}\s*(?:Ti|SUPER)?\b",       # RTX 4090, RTX 3090 Ti
    r"\b[AL]\d{4}\b",                          # A6000 (before the 3-digit rule)
    r"\b[AHVB]\d{3}\b",                        # A100, H100, V100, B200
    r"\bL\d{1,2}S?\b",                         # L4, L40S
]


def repo_root() -> Path:
    for cand in (_HERE, *_HERE.parents):
        if (cand / ".git").exists():
            return cand
    return _HERE.parent


def device_tag() -> str:
    """Short name of the GPU this run is using: A100, H100, MI300X, ..."""
    forced = os.environ.get("PPS_DEVICE_TAG")
    if forced:
        return forced
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        name = torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"
    for pattern in _KNOWN:
        m = re.search(pattern, name, flags=re.IGNORECASE)
        if m:
            return re.sub(r"\s+", "", m.group(0)).upper()
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "unknown"


def data_root() -> Path:
    root = os.environ.get("PPS_DATA_ROOT")
    if root:
        return Path(root).resolve()
    return repo_root() / f"results_on_{device_tag()}"


def out_dir(experiment: str, kind: str = "results", create: bool = True) -> str:
    """Directory for one experiment's `results` (or `figures`), as a string.

    `experiment` is the experiment folder's own name, e.g. "scaling_accuracy_noise".
    """
    base = data_root()
    path = (Path(experiment) / kind) if base.name != "." else Path(experiment) / kind
    full = base / path
    if create:
        full.mkdir(parents=True, exist_ok=True)
    return str(full)


if __name__ == "__main__":
    print(f"device tag : {device_tag()}")
    print(f"data root  : {data_root()}")
    print(f"example    : {out_dir('scaling_accuracy_noise', create=False)}")


# ---------------------------------------------------------------------------
# GPU statevector baseline
# ---------------------------------------------------------------------------

def lightning_gpu_device() -> str:
    """PennyLane's GPU statevector device for whichever torch build is installed.

    NVIDIA gets `lightning.gpu` (cuStateVec, pennylane-lightning-gpu); AMD gets
    `lightning.amdgpu` (the Kokkos/HIP build), which is the only one with ROCm
    wheels. Hard-coding either name pins the whole bundle to one vendor, and the
    scripts are otherwise vendor-neutral. Override with PPS_PL_GPU_DEVICE.
    """
    forced = os.environ.get("PPS_PL_GPU_DEVICE")
    if forced:
        return forced
    try:
        import torch

        if getattr(torch.version, "hip", None):
            return "lightning.amdgpu"
    except Exception:
        pass
    return "lightning.gpu"
