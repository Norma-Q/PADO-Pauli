#!/usr/bin/env python
"""Truncation convergence of the 127-qubit kicked-Ising field sweep.

Overlays the PADO-Pauli <Z_62> curve at max_weight = 3, 4, 5, 8 (all at the same
coefficient threshold build_min_abs = 1e-4, which the legend states) on the IBM
Eagle error-mitigated hardware points, so that the effect of the structural
truncation knob is visible against a fixed external reference.

PADO-Pauli values come from this directory's own run outputs:
    results/kicked_ising_127q_w{3,4,5,6,7}.json  (max_weight = 3..7)
    results/kicked_ising_127q_full.json          (max_weight = 8)
each produced by
    python run_kicked_ising.py --indices all --preset gpu --max-weight W --min-abs 1e-4

The IBM Eagle error-mitigated <Z_62> points are digitized literature values, not
our measurement. Source: Kim et al., "Evidence for the utility of quantum
computing before fault tolerance", Nature 618, 500 (2023); the digitized values
used here are the ones carried in the PauliPropagation.jl utility example
(Rudolph et al., arXiv:2505.21606).

  python make_convergence_figure.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Output location is device-aware: <repo>/results_on_<DEVICE>/{exp}/...
# so the same code can run on an A100, an H100 and an MI300X without one run
# overwriting another. See reproducibility/_outdir.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _outdir import out_dir as _out_dir

RESULTS = _out_dir("kicked_ising_127q", "results")
FIGS = _out_dir("kicked_ising_127q", "figures")
os.makedirs(FIGS, exist_ok=True)

# Every threshold that was swept; PLOT lists the subset drawn in the figure
# (6 and 7 are printed in the summary table below but not drawn).
RUNS = [(3, "kicked_ising_127q_w3.json"), (4, "kicked_ising_127q_w4.json"),
        (5, "kicked_ising_127q_w5.json"), (6, "kicked_ising_127q_w6.json"),
        (7, "kicked_ising_127q_w7.json"), (8, "kicked_ising_127q_full.json")]
PLOT = (3, 4, 5, 8)
COLORS = {3: "#c0392b", 4: "#eda100", 5: "#1baf7a", 8: "#0e7c86"}

IBM_angles = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                       0.6, 0.7, 0.8, 1.0, 1.5707])
IBM_mitigated = np.array([1.01688859, 1.00387483, 0.95615886, 0.95966435,
                          0.83946763, 0.81185907, 0.54640995, 0.45518584,
                          0.19469377, 0.01301832, 0.01016334])

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                              gridspec_kw={"width_ratios": [1.55, 1.0]})
ax.axhline(0.0, color="#6b7280", ls="--", lw=0.9, alpha=0.8, zorder=1)

summary, prev = [], None
for w, fname in RUNS:
    blob = json.load(open(os.path.join(RESULTS, fname)))
    meta = blob["meta"]
    assert int(meta["max_weight"]) == w, (fname, meta["max_weight"])
    rows = sorted(blob["data"], key=lambda r: r["h"])
    h = np.array([r["h"] for r in rows])
    ev = np.array([r["ev"] for r in rows])
    if w in PLOT:
        ax.plot(h, ev, "-o", color=COLORS[w], lw=2.0, ms=4.0, zorder=3,
                label=fr"$w_{{\max}}={w}$")
    resid = np.abs(np.interp(IBM_angles, h, ev) - IBM_mitigated)
    shift = float(np.abs(ev - prev).max()) if prev is not None else float("nan")
    prev = ev
    summary.append((w, float(meta["build_min_abs"]),
                    max(r["zero_filtered_terms"] for r in rows),
                    sum(r["time_s"] for r in rows),
                    max(r["peak_vram_gb"] for r in rows),
                    float(resid.max()), float(np.sqrt(np.mean(resid ** 2))), shift))

ax.scatter(IBM_angles, IBM_mitigated, s=62, marker="o", facecolor="#22303a",
           edgecolor="white", linewidth=1.2, zorder=5, label="IBM Eagle (mitigated)")

# right panel: max / rms deviation from the hardware points vs w_max
W = [s[0] for s in summary]
ax2.plot(W, [s[5] for s in summary], "o-", color="#22303a", lw=1.8, ms=5,
         label="max deviation")
ax2.plot(W, [s[6] for s in summary], "s--", color="#c0392b", lw=1.8, ms=5,
         label="rms deviation")
ax2.set_xlabel(r"max-weight threshold $w_{\max}$")
ax2.set_ylabel(r"deviation from the hardware points")
ax2.set_xticks(W)
ax2.set_yscale("log")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(fontsize=10)

ax.set_xlabel(r"transverse-field angle $h$")
ax.set_ylabel(r"$\langle Z_{62}\rangle$")
ax.set_xticks([0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2])
ax.set_xticklabels(["0", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"])
ax.set_xlim(-0.03, np.pi / 2 + 0.03)
ax.set_ylim(-0.08, 1.10)
ax.grid(True, alpha=0.3)

# The coefficient threshold is held fixed across the sweep; read it back from the
# run metadata rather than hardcoding it, and say so in the legend so the figure
# does not read as if no coefficient truncation were applied.
thresholds = {s[1] for s in summary}
assert len(thresholds) == 1, thresholds
min_abs = thresholds.pop()
exponent = int(round(np.log10(min_abs)))
assert 10.0 ** exponent == min_abs, min_abs
ax.legend(loc="upper right", fontsize=10,
          title=fr"PADO-Pauli, $\mathtt{{build\_min\_abs}}=10^{{{exponent}}}$",
          title_fontsize=10)
fig.tight_layout()
out = os.path.join(FIGS, "fig_kicked_ising_127q_wmax.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"{'w_max':>6} {'min_abs':>9} {'zero-filtered':>14} {'compile s':>10} {'peak GB':>8} "
      f"{'max |dev|':>10} {'rms |dev|':>10} {'shift vs w-1':>13}  plotted")
for w, ma, zf, cs, vram, mx, rms, sh in summary:
    print(f"{w:>6} {ma:>9.0e} {zf:>14} {cs:>10.1f} {vram:>8.2f} "
          f"{mx:>10.4f} {rms:>10.4f} {sh:>13.4f}  {'yes' if w in PLOT else 'no'}")
print("wrote", out)
