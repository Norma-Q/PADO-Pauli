"""Fix w_max, increase qubit count: term counts and VRAM vs n.

Each (preset, N) point runs in its own subprocess so that ru_maxrss is a clean
per-process high-water mark. Results stream into results/qubit_sweep.json|.csv.

Usage:
  python sweep_qubits_fixed_wmax.py                 # driver: full sweep
  python sweep_qubits_fixed_wmax.py --single ...    # one point (internal)
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
W_MAX = 3
DEPTH = 5
EDGE_PROB = 0.3
SEED = 42
QUBITS = list(range(10, 101, 10))     # 10, 20, ..., 100
PRESETS = ["gpu"]                     # GPU-resident program; track VRAM footprint


def run_single(n: int, preset: str, out_path: str, seed: int = SEED) -> None:
    from _common import compile_and_measure
    rec = compile_and_measure(n=n, w_max=W_MAX, preset=preset,
                              depth=DEPTH, p=EDGE_PROB, seed=seed)
    with open(out_path, "w") as fh:
        json.dump(rec, fh)
    vram = rec.get("vram_peak_reserved_gb")
    vram_s = f"{vram:.3f}GB" if vram is not None else "n/a"
    print(f"[done] preset={preset} N={n} "
          f"propagated={rec['propagated_terms']:,} zero_filtered={rec['zero_filtered_terms']:,} "
          f"VRAM_peak={vram_s} prop_s={rec['propagate_total_s']:.2f}")


def driver(seed: int = SEED) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    combined = []
    for preset in PRESETS:
        for n in QUBITS:
            tag = f"q{n:03d}_{preset}{_seed_suffix(seed)}"
            pt_path = os.path.join(RESULTS, f"point_{tag}.json")
            print(f"\n=== launching preset={preset} N={n} seed={seed} ===", flush=True)
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--single",
                 "--n", str(n), "--preset", preset, "--out", pt_path,
                 "--seed", str(seed)],
                cwd=HERE,
            )
            if proc.returncode != 0 or not os.path.exists(pt_path):
                print(f"[WARN] point failed: preset={preset} N={n} (rc={proc.returncode}). Stopping this preset.")
                break
            with open(pt_path) as fh:
                combined.append(json.load(fh))
            _write_outputs(combined, seed)
    print("\n[ALL DONE] points:", len(combined))


def _seed_suffix(seed: int) -> str:
    """Non-default seeds get their own filenames so a multi-seed sweep never
    overwrites the seed-42 run."""
    return "" if int(seed) == SEED else f"_s{int(seed)}"


def _write_outputs(rows, seed: int = SEED) -> None:
    sfx = _seed_suffix(seed)
    with open(os.path.join(RESULTS, f"qubit_sweep{sfx}.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    if rows:
        keys = list(rows[0].keys())
        with open(os.path.join(RESULTS, f"qubit_sweep{sfx}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--n", type=int)
    ap.add_argument("--preset", type=str)
    ap.add_argument("--out", type=str)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="Erdos-Renyi graph seed; non-default seeds write to "
                         "qubit_sweep_s<seed>.{json,csv} and point_*_s<seed>.json")
    args = ap.parse_args()
    if args.single:
        run_single(args.n, args.preset, args.out, args.seed)
    else:
        driver(args.seed)


if __name__ == "__main__":
    main()
