#!/usr/bin/env python
"""Record what produced a result file (platform, GPU, package versions).

The provenance is written *beside* the result as a <stem>.runmeta.json sidecar
rather than inside it: several result files are JSON lists at the top level, so
there is no key to inject, and readers of the result files need no new schema.

Only facts that can change a reported number are recorded. Machine identity
(hostname) and repository identity (checkout path, branch, commit) are
deliberately left out: a sidecar travels with the data it describes, and must
be publishable as-is without carrying anything about the machine or the tree it
was produced on.

    python -m reproducibility._runmeta                    # print the metadata
    python -m reproducibility._runmeta results/*.json     # stamp those files

`run_rerun_all.sh` stamps automatically after every job.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

SIDECAR_SUFFIX = ".runmeta.json"

# Packages whose version can change a reported number.
_WATCHED = (
    "torch", "pennylane", "pennylane_lightning",
    # whichever of the two GPU statevector plugins is installed
    "pennylane_lightning_gpu", "pennylane_lightning_amdgpu",
    "numpy", "scipy", "qiskit", "cupauliprop", "cuquantum",
)


def _padopauli() -> str | None:
    """Which engine version produced the number (installed padopauli)."""
    try:
        import padopauli
        return str(padopauli.__version__)
    except Exception:
        return None


def _pkg_versions() -> dict:
    out: dict[str, str] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:
        return out
    for name in _WATCHED:
        for cand in (name, name.replace("_", "-")):
            try:
                out[name] = version(cand)
                break
            except PackageNotFoundError:
                continue
            except Exception:
                break
    return out


def _gpu() -> dict:
    info: dict = {"name": None, "backend": None, "count": 0}
    try:
        import torch
    except Exception:
        return info
    try:
        if torch.cuda.is_available():
            info["name"] = torch.cuda.get_device_name(0)
            info["count"] = torch.cuda.device_count()
            # torch.version.hip is set on ROCm builds, torch.version.cuda on CUDA ones
            hip = getattr(torch.version, "hip", None)
            cuda = getattr(torch.version, "cuda", None)
            info["backend"] = f"rocm {hip}" if hip else (f"cuda {cuda}" if cuda else None)
    except Exception:
        pass
    return info


def run_meta() -> dict:
    """Everything needed to say where a number came from."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpu": _gpu(),
        "padopauli": _padopauli(),
        "packages": _pkg_versions(),
        # ROCm-only workaround; kept so A100 sidecars stay comparable to the MI300X ones.
        "ld_preload": os.environ.get("LD_PRELOAD"),
        "rocm_host": next((p.name.replace("rocm-", "")
                           for p in sorted(Path("/opt").glob("rocm-*"))), None),
    }


def sidecar_for(result_path: str | Path) -> Path:
    p = Path(result_path)
    return p.parent / (p.stem + SIDECAR_SUFFIX)


def stamp(result_path: str | Path, meta: dict | None = None) -> Path | None:
    """Write <stem>.runmeta.json next to a result file. Returns the sidecar path."""
    p = Path(result_path)
    if not p.is_file() or p.name.endswith(SIDECAR_SUFFIX):
        return None
    side = sidecar_for(p)
    payload = dict(meta or run_meta())
    payload["result_file"] = p.name
    side.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return side


def main(argv: list[str]) -> int:
    if not argv:
        print(json.dumps(run_meta(), indent=2, sort_keys=True))
        return 0
    meta = run_meta()          # one snapshot for the whole batch
    n = 0
    for a in argv:
        if stamp(a, meta) is not None:
            n += 1
    print(f"[runmeta] stamped {n} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
