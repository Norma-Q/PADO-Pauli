#!/usr/bin/env python
"""Render the <Z_center> vs RX-angle curve from results/kicked_ising_127q_full.json
to figures/fig_kicked_ising_127q.{png,pdf}.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)


def main():
    with open(RESULTS / "kicked_ising_127q_full.json") as f:
        payload = json.load(f)
    meta = payload["meta"]
    data = sorted(payload["data"], key=lambda r: r["i"])
    hs = [r["h"] for r in data]
    evs = [r["ev"] for r in data]
    center = meta.get("center", 60)
    n = meta.get("n_qubits", 127)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(hs, evs, marker="o", color="#1f4e9c", lw=1.8, ms=5,
            label=(fr"$\langle Z_{{{center}}}\rangle$  (PPS: $w_{{\max}}="
                   fr"{meta.get('max_weight')}$, $\delta={meta.get('build_min_abs')}$)"))
    ax.axhline(0.0, color="0.7", lw=0.8, ls="--", zorder=0)

    ax.set_xlabel(r"transverse-field angle $\theta_h$")
    ax.set_ylabel(fr"$\langle Z_{{{center}}}\rangle$")
    ax.set_xlim(-0.02, np.pi / 2 + 0.02)
    ax.set_xticks([0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2])
    ax.set_xticklabels(["0", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"])
    ax.set_title(fr"{n}-qubit kicked Ising "
                 fr"($\mathrm{{RZZ}}(-\pi/2)$, "
                 f"{meta.get('num_trotter_steps', 20)} Trotter steps, "
                 f"IBM Eagle heavy-hex)", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = FIGS / f"fig_kicked_ising_127q.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  -> wrote {out}")

    print(f"\n  field sweep ({len(data)} points), <Z_{center}>:")
    for r in data:
        print(f"    h={r['h']:.5f} ({r['h_over_pi']:.4f}π)  <Z>={r['ev']:+.6f}")


if __name__ == "__main__":
    main()
