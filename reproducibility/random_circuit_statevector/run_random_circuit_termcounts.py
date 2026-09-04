#!/usr/bin/env python
"""Record propagated/zero-filtered term counts for the random-circuit sweep instances.

Companion to run_random_circuit_sweep.py, which runs without PPS_COMPILE_PROFILE
(random_circuit_sweep.json has peak_terms/final_terms = null). This script rebuilds
the same circuits (SEED=42, identical construction) and compiles each (n, mw)
config with profiling enabled, recording only:
  - propagated_terms : unique Pauli terms after union-basis propagation
  - final_terms      : terms surviving zero-filter backprop (evaluation set)
  - compile_s, vram_gb

Configs: (12, 9), (16, 9), (16, 10), (20, 9) -- the rows of the random-circuit
accuracy table.

Run: PPS_COMPILE_PROFILE=1 python run_random_circuit_termcounts.py
(the script also sets the env var itself if unset).
"""
import gc, json, os, sys, time
from pathlib import Path

os.environ.setdefault("PPS_COMPILE_PROFILE", "1")

import numpy as np
import torch

from padopauli import Circuit  # noqa: E402

# Erdos-Renyi edge probability, held fixed across sizes (EDGE_PROB=log uses log(n)/n).
EDGE_PROB = os.environ.get("EDGE_PROB", "0.15")


# Output location is device-aware: <repo>/results_on_<DEVICE>/{exp}/...
# so the same code can run on an A100, an H100 and an MI300X without one run
# overwriting another. See reproducibility/_outdir.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _outdir import out_dir as _out_dir

OUT = Path(_out_dir("random_circuit_statevector", "results"))

GEN_LAYERS, V_LAYERS = 3, 2
SEED = 42
CONFIGS = [(12, 9), (16, 9), (16, 10), (20, 9)]


def build(n):
    # Identical to run_random_circuit_sweep.py:build (same SEED -> same graphs).
    rng = np.random.default_rng(SEED)
    ep = float(np.log(n) / n) if EDGE_PROB == "log" else float(EDGE_PROB)
    graphs = [[(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < ep]
              for _ in range(GEN_LAYERS)]
    def acz(qc, a, b): qc.h(b); qc.cnot(a, b); qc.h(b)
    def brick(l): s = l % 2; return [(i, i + 1) for i in range(s, n - 1, 2)]
    qc, e = Circuit(n), 0
    for l in range(GEN_LAYERS):
        for q in range(n): qc.rx(q, embedding_idx=e); e += 1
        for a, b in graphs[l]: acz(qc, a, b)
    for q in range(n): qc.rx(q, embedding_idx=e); e += 1
    for q in range(n): qc.ry(q, embedding_idx=e); e += 1
    p = 0
    for l in range(V_LAYERS):
        for q in range(n):
            qc.ry(q, param_idx=p); p += 1
            qc.rz(q, param_idx=p); p += 1
        for a, b in brick(l): acz(qc, a, b)
    groups = [(0,), (n - 2, n - 1)]
    return qc, groups


def make_obs(n, groups):
    return [("Z" * len(gp), list(gp)) for gp in groups]


results = {"config": {"gen_layers": GEN_LAYERS, "v_layers": V_LAYERS, "seed": SEED,
                      "dtype": "float32", "configs": [list(c) for c in CONFIGS]},
           "rows": []}

for n, mw in CONFIGS:
    print(f"\n########## n={n}, mw={mw} ##########")
    circ, groups = build(n)
    obs = make_obs(n, groups)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    prog = circ.compile(observables=obs, preset="gpu",
                        dtype="float32", max_weight=mw, chunk_size=25_000_000)
    cs = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    sm = prog.compile_stats.get("summary", {})
    row = {"n": n, "mw": mw,
           "propagated_terms": sm.get("propagated_terms"),
           "final_terms": sm.get("final_terms"),
           "compile_s": cs, "vram_gb": vram}
    results["rows"].append(row)
    print(f"n={n} mw={mw}: propagated={row['propagated_terms']:,} "
          f"zero_filtered={row['final_terms']:,} compile={cs:.1f}s VRAM={vram:.1f}GB")
    del prog, circ  # circ holds the program too
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

(OUT / "random_circuit_termcounts.json").write_text(json.dumps(results, indent=2))
print(f"\n[saved] {OUT/'random_circuit_termcounts.json'}")
