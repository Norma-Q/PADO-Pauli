"""Fix the circuit, increase w_max: term counts and VRAM vs max-weight.

Fixed circuit: 15-qubit Erdos-Renyi MaxCut-QAOA, edge_prob=0.3, depth=5, seed=42.
Sweep w_max in {2,3,4,5}. Each (preset, w_max) point runs in its own subprocess.

Usage:
  python sweep_wmax_fixed_circuit.py                 # driver: full sweep
  python sweep_wmax_fixed_circuit.py --single ...    # one point (internal)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Output location is device-aware: <repo>/results_on_<DEVICE>/{exp}/...
# so the same code can run on an A100, an H100 and an MI300X without one run
# overwriting another. See reproducibility/_outdir.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _outdir import out_dir as _out_dir

RESULTS = _out_dir("scaling_accuracy_noise", "results")

# Fixed experimental settings.
N_QUBITS = 15
DEPTH = 5
EDGE_PROB = 0.3
SEED = 42
W_MAX_LIST = [2, 3, 4, 5]
PRESETS = ["gpu"]                     # GPU-resident program; track VRAM footprint


def run_single(w_max: int, preset: str, out_path: str) -> None:
    from _common import compile_and_measure
    rec = compile_and_measure(n=N_QUBITS, w_max=w_max, preset=preset,
                              depth=DEPTH, p=EDGE_PROB, seed=SEED)
    with open(out_path, "w") as fh:
        json.dump(rec, fh)
    vram = rec.get("vram_peak_reserved_gb")
    vram_s = f"{vram:.3f}GB" if vram is not None else "n/a"
    print(f"[done] preset={preset} w_max={w_max} "
          f"propagated={rec['propagated_terms']:,} zero_filtered={rec['zero_filtered_terms']:,} "
          f"VRAM_peak={vram_s} prop_s={rec['propagate_total_s']:.2f}")


def driver() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    combined = []
    for preset in PRESETS:
        for w in W_MAX_LIST:
            tag = f"w{w}_{preset}"
            pt_path = os.path.join(RESULTS, f"point_{tag}.json")
            print(f"\n=== launching preset={preset} w_max={w} ===", flush=True)
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--single",
                 "--w", str(w), "--preset", preset, "--out", pt_path],
                cwd=HERE,
            )
            if proc.returncode != 0 or not os.path.exists(pt_path):
                print(f"[WARN] point failed: preset={preset} w_max={w} (rc={proc.returncode}). Stopping this preset.")
                break
            with open(pt_path) as fh:
                combined.append(json.load(fh))
            _write_outputs(combined)
    print("\n[ALL DONE] points:", len(combined))


def _write_outputs(rows) -> None:
    with open(os.path.join(RESULTS, "wmax_sweep.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    if rows:
        keys = list(rows[0].keys())
        with open(os.path.join(RESULTS, "wmax_sweep.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--w", type=int)
    ap.add_argument("--preset", type=str)
    ap.add_argument("--out", type=str)
    args = ap.parse_args()
    if args.single:
        run_single(args.w, args.preset, args.out)
    else:
        driver()


if __name__ == "__main__":
    main()
