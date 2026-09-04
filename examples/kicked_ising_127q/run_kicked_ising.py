#!/usr/bin/env python
"""Reproduce the 127-qubit kicked-Ising <Z_62> vs RX-angle sweep with PADO-Pauli,
matching the prior Pauli-propagation study of Rudolph et al. (PauliPropagation.jl,
arXiv:2505.21606), example ``3-utility-example.ipynb``.

That study uses:
  * the IBM Eagle HEAVY-HEX topology (`ibmeagletopology`, 127 qubits, 144 edges),
  * per Trotter step: RX(h) on every qubit, then RZZ(-pi/2) on every edge (Clifford),
  * nl = 20 Trotter steps,
  * observable Z on qubit 63 (1-indexed) == qubit 62 (0-indexed),
  * field sweep h in LinRange(0, pi/2, 20),
  * truncation: max_weight = 8 and min_abs_coeff = 1e-4.

We replicate it verbatim in PADO-Pauli (no qiskit, no Julia): the heavy-hex edge
list below is the exact `ibmeagletopology` constant from PauliPropagation.jl,
converted from 1-indexed (Julia) to 0-indexed. PADO-Pauli's `max_weight` preset
override and `build_min_abs` are the analogues of `max_weight` / `min_abs_coeff`.

Expectation value: PADO-Pauli sums the surviving diagonal {I,Z} coefficients,
the same rule the Pauli-propagation references use.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from padopauli import Circuit

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

N = 127
CENTER = 62            # Z_62 (0-indexed) == Rudolph's PauliString(nq, :Z, 63) (1-indexed)
NUM_STEPS = 20
RZZ_ANGLE = -np.pi / 2
HS = list(np.linspace(0.0, np.pi / 2, 20))   # Rudolph: LinRange(0, pi/2, 20)

# Exact `ibmeagletopology` from PauliPropagation.jl (src/Circuits/topologies.jl),
# the IBM Eagle heavy-hex coupling on 127 qubits. Listed 1-indexed (as in Julia).
IBMEAGLE_TOPOLOGY_1IDX = [
    (1, 2), (1, 15), (2, 3), (3, 4), (4, 5), (5, 6), (5, 16), (6, 7), (7, 8), (8, 9), (9, 10), (9, 17),
    (10, 11), (11, 12), (12, 13), (13, 14), (13, 18), (15, 19), (16, 23), (17, 27), (18, 31), (19, 20),
    (20, 21), (21, 22), (21, 34), (22, 23), (23, 24), (24, 25), (25, 26), (25, 35), (26, 27), (27, 28),
    (28, 29), (29, 30), (29, 36), (30, 31), (31, 32), (32, 33), (33, 37), (34, 40), (35, 44), (36, 48),
    (37, 52), (38, 39), (38, 53), (39, 40), (40, 41), (41, 42), (42, 43), (42, 54), (43, 44), (44, 45),
    (45, 46), (46, 47), (46, 55), (47, 48), (48, 49), (49, 50), (50, 51), (50, 56), (51, 52), (53, 57),
    (54, 61), (55, 65), (56, 69), (57, 58), (58, 59), (59, 60), (59, 72), (60, 61), (61, 62), (62, 63),
    (63, 64), (63, 73), (64, 65), (65, 66), (66, 67), (67, 68), (67, 74), (68, 69), (69, 70), (70, 71),
    (71, 75), (72, 78), (73, 82), (74, 86), (75, 90), (76, 77), (76, 91), (77, 78), (78, 79), (79, 80),
    (80, 81), (80, 92), (81, 82), (82, 83), (83, 84), (84, 85), (84, 93), (85, 86), (86, 87), (87, 88),
    (88, 89), (88, 94), (89, 90), (91, 95), (92, 99), (93, 103), (94, 107), (95, 96), (96, 97), (97, 98),
    (97, 110), (98, 99), (99, 100), (100, 101), (101, 102), (101, 111), (102, 103), (103, 104), (104, 105),
    (105, 106), (105, 112), (106, 107), (107, 108), (108, 109), (109, 113), (110, 115), (111, 119), (112, 123),
    (113, 127), (114, 115), (115, 116), (116, 117), (117, 118), (118, 119), (119, 120), (120, 121), (121, 122),
    (122, 123), (123, 124), (124, 125), (125, 126), (126, 127),
]


def eagle_edges():
    """IBM Eagle heavy-hex edges, 0-indexed for PADO-Pauli."""
    return [(u - 1, v - 1) for (u, v) in IBMEAGLE_TOPOLOGY_1IDX]



def build_circuit(edges):
    """Per step: RX(h) on every qubit (param_idx 0), then RZZ(-pi/2) on every edge (param_idx 1)."""
    qc = Circuit(N)
    for _ in range(NUM_STEPS):
        for q in range(N):
            qc.rx(q, param_idx=0)
        for (u, v) in edges:
            qc.rzz(u, v, param_idx=1)
    return qc


def zero_filtered_terms(program):
    for name in ("psum_union",) + tuple(getattr(program, "__dict__", {}).keys()):
        obj = getattr(program, name, None)
        for attr in ("x_mask", "x", "x_masks"):
            xm = getattr(obj, attr, None)
            if xm is not None and hasattr(xm, "shape"):
                try:
                    return int(xm.shape[0])
                except Exception:
                    pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", default="all",
                    help='comma list of field indices 0..19, or "all"')
    ap.add_argument("--preset", default="gpu")
    ap.add_argument("--min-abs", type=float, default=1e-4)     # = min_abs_coeff
    ap.add_argument("--max-weight", type=int, default=8)       # = max_weight
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available() and args.preset in ("gpu", "hybrid")
    dev = "cuda" if use_cuda else "cpu"

    edges = eagle_edges()
    qc = build_circuit(edges)
    idxs = (list(range(len(HS))) if args.indices == "all"
            else [int(x) for x in args.indices.split(",")])

    print(f"[kicked-Ising 127q | heavy-hex]  N={N} edges={len(edges)} center={CENTER} "
          f"steps={NUM_STEPS} preset={args.preset} min_abs={args.min_abs} "
          f"max_weight={args.max_weight}")
    print(f"  gates/step = {N} RX + {len(edges)} RZZ ; total gates = "
          f"{NUM_STEPS * (N + len(edges))}", flush=True)

    meta = {
        "experiment": "127q kicked Ising, <Z_62> vs RX angle (heavy-hex)",
        "n_qubits": N, "center": CENTER, "num_trotter_steps": NUM_STEPS,
        "rzz_angle": RZZ_ANGLE, "n_edges": len(edges),
        "topology": "IBM Eagle heavy-hex (ibmeagletopology, PauliPropagation.jl)",
        "build_min_abs": args.min_abs, "max_weight": args.max_weight,
        "preset": args.preset, "hs": HS, "h_formula": "LinRange(0, pi/2, 20)",
        "reference": "Rudolph et al. PauliPropagation.jl arXiv:2505.21606, "
                     "example 3-utility-example.ipynb (max_weight=8, min_abs_coeff=1e-4)",
    }
    out_json = RESULTS / f"kicked_ising_127q_{args.tag}.json"
    out_csv = RESULTS / f"kicked_ising_127q_{args.tag}.csv"

    def save(rows):
        with open(out_json, "w") as f:
            json.dump({"meta": meta, "data": rows}, f, indent=2)
        with open(out_csv, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["i", "h", "h_over_pi", "ev", "time_s",
                          "zero_filtered_terms", "peak_vram_gb"])
            for r in rows:
                wtr.writerow([r["i"], r["h"], r["h_over_pi"], r["ev"],
                              r.get("time_s"), r.get("zero_filtered_terms"),
                              r.get("peak_vram_gb")])

    existing = {}
    if out_json.exists():
        try:
            existing = {r["i"]: r for r in json.load(open(out_json)).get("data", [])}
        except Exception:
            existing = {}

    for i in idxs:
        h = float(HS[i])
        thetas = torch.tensor([h, RZZ_ANGLE], dtype=torch.float64, device=dev)
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        st = time.perf_counter()
        program = qc.compile(
            observables=[("Z", [CENTER])], preset=args.preset,
            dtype="float32", max_weight=args.max_weight,
            build_thetas=thetas, build_min_abs=args.min_abs)
        ev = float(program.expval(thetas, obs_index=0))
        dt = time.perf_counter() - st
        nt = zero_filtered_terms(program)
        peak = float(torch.cuda.max_memory_allocated() / 1024**3) if use_cuda else None
        program.clear_cache()
        existing[i] = {"i": i, "h": h, "h_over_pi": h / np.pi, "ev": ev,
                       "time_s": dt, "zero_filtered_terms": nt, "peak_vram_gb": peak}
        save([existing[k] for k in sorted(existing)])
        print(f"  i={i:2d}  h={h:.5f} ({h/np.pi:.4f}π)  "
              f"<Z_{CENTER}>={ev:+.6f}  t={dt:.1f}s  terms={nt}  "
              f"peak={None if peak is None else round(peak,2)}GB", flush=True)

    print(f"  -> wrote {out_json}\n  -> wrote {out_csv}")


if __name__ == "__main__":
    main()
