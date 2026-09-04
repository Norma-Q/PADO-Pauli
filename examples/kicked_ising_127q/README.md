# 127-qubit kicked-Ising magnetisation sweep (PADO-Pauli, heavy-hex)

Reproduces the single-site magnetisation curve of the 127-qubit kicked-Ising
"utility" benchmark with the PADO-Pauli engine, following the PauliPropagation.jl
example of Rudolph et al. (see Reference at the end). No qiskit and no Julia
dependency: the same circuit, topology, observable and truncation are built
directly in PADO-Pauli.

Output: `<Z_62>` of the kicked-Ising circuit as a function of the transverse-field
(RX) angle `theta_h`, the magnetisation-decay curve between successive Clifford
points.

## Setup

| | value |
|---|---|
| qubits | 127 |
| **topology** | **IBM Eagle heavy-hex** (`ibmeagletopology`, 144 edges) |
| field layer | `RX(h)` on every qubit |
| coupling layer | `RZZ(-pi/2)` on every edge (Clifford) |
| Trotter steps | 20 |
| observable | `Z_62` (0-indexed) = `PauliString(127, :Z, 63)` (1-indexed) |
| field sweep | `h = LinRange(0, pi/2, 20)` (20 points) |
| truncation | `max_weight = 8`, `min_abs_coeff = 1e-4` |

The exact `ibmeagletopology` edge list is taken verbatim from PauliPropagation.jl
(`src/Circuits/topologies.jl`) and converted from 1-indexed (Julia) to 0-indexed.
PADO-Pauli's `max_weight` preset override and `build_min_abs` are the analogues of
`max_weight` / `min_abs_coeff`.

## Method

`RZZ(-pi/2)` is Clifford, so the only branching comes from the `RX(h)` layers.
The observable is propagated backward (Heisenberg frame) with `max_weight = 8`
(the only term-count bound PADO-Pauli exposes) and a coefficient
threshold `build_min_abs = 1e-4`, evaluated per field. The expectation value is
the sum of the surviving diagonal `{I,Z}` coefficients (PADO-Pauli's exact
zero-filter). The endpoints are Clifford and exact by construction: `h=0` gives
`<Z>=1`; `h=pi/2` is a Clifford point.

## Files

```
kicked_ising_127q/
├── run_kicked_ising.py   ← heavy-hex topology + circuit, compiles per field, evaluates <Z_62>
│                            (embeds the exact ibmeagletopology edge list)
├── run_sweep.sh          ← optional driver: one process per field (robust to OOM); ends-inward
├── make_figure.py        ← renders the curve from results/ into figures/
├── make_convergence_figure.py ← <Z_62> vs max_weight (w3..w7 vs the full run) into figures/
├── verify_small.py       ← exact PennyLane cross-check on a small patch of the actual
│                            ibmeagletopology (incl. a degree-3 branch vertex), saving
│                            results/verify_small_heavyhex.json
├── results/              ← kicked_ising_127q_full.{json,csv} (curve + per-field timing,
│                            term counts, peak VRAM), kicked_ising_127q_w{3..7}.{json,csv},
│                            kicked_ising_127q_notebook.json (written by the notebook),
│                            verify_small_heavyhex.json, *.runmeta.json sidecars
└── figures/              ← fig_kicked_ising_127q.{png,pdf}, fig_kicked_ising_127q_wmax.png
```

## Reproduce

```bash
# full 20-field sweep (one process):
python run_kicked_ising.py --indices all --preset gpu --max-weight 8 --min-abs 1e-4

# or one process per field (robust to per-field OOM):
bash run_sweep.sh

# render the figure:
python make_figure.py            # figures/fig_kicked_ising_127q.{png,pdf}

# (optional) confirm conventions + heavy-hex edge construction vs exact statevector:
python verify_small.py            # -> results/verify_small_heavyhex.json (max |dev| ~ 1e-14)
```

Knobs (`run_kicked_ising.py`): `--max-weight` (default 8), `--min-abs`
(coefficient threshold, default 1e-4), `--preset` (`gpu`/`hybrid`/`cpu`),
`--indices` (subset of `0..19`).

## Environment / hardware

- Python with the installed `padopauli` package (which brings PyTorch, NumPy, SciPy)
  plus `matplotlib`; recorded with PyTorch 2.11.0+rocm7.2.
- Recorded on an AMD Instinct MI300X GPU via ROCm. With heavy-hex and
  `max_weight=8` the per-field footprint is small; per-field timing, zero-filtered
  term counts and peak VRAM are recorded in `results/kicked_ising_127q_full.json`.

## Reference

Rudolph, Jones, Teng, Angrisani, Holmes, *Pauli Propagation: A Computational
Framework for Simulating Quantum Systems*, arXiv:2505.21606; PauliPropagation.jl
example `3-utility-example.ipynb` (`max_weight=8`, `min_abs_coeff=1e-4`,
`ibmeagletopology`).
