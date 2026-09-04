"""Figures for the cross-engine benchmark, drawn from a finished run's summary.json.

    python make_figures.py                     # this device's results
    python make_figures.py --results <dir>     # a specific results/ folder
    python make_figures.py --device A100       # results_on_A100/

Reads only; runs no benchmark and touches no GPU. Output goes next to the data it
came from, in <data root>/engine_benchmarks/figures/, and every figure ships a
CSV of the exact numbers it plots — several series colors sit below 3:1 contrast
on a light surface, and the table is the documented relief for that.

Engines that time out or are skipped are drawn as an explicit "did not finish"
marker rather than dropped: the fact that an engine cannot do a case in the
budget IS a result, and a line that simply stops looks like missing data.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Categorical slots 1-5 of the reference palette, in that order, so the pairs
# that end up adjacent are the ones that were validated as adjacent (worst CVD
# ΔE 9.1 light / 8.4 dark; worst normal-vision ΔE 19.6 / 19.3).
#
# Colour follows the ENGINE, never its rank in a chart: cuPauliProp is orange
# whether it comes first or last, so a filtered chart never repaints survivors.
ENGINES = {
    "pps":             {"label": "PADO",             "color": "#2a78d6", "marker": "o"},
    "cupauliprop":     {"label": "cuPauliProp",      "color": "#eb6834", "marker": "s"},
    "qiskit":          {"label": "Qiskit pauli-prop", "color": "#1baf7a", "marker": "^"},
    "julia":           {"label": "PauliPropagation.jl", "color": "#eda100", "marker": "D"},
    "julia_surrogate": {"label": "PP.jl surrogate",  "color": "#e87ba4", "marker": "v"},
}
ORDER = list(ENGINES)

# Scatter/small-multiple panels can only carry three categorical hues (the full
# five cannot clear the all-pairs floors — no ordering of them can), so the
# frontier figure facets instead: one panel per engine, highlighted against its
# rivals in grey.
HILITE = "#2a78d6"
CONTEXT = "#b8b7b0"

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e5e4df"
SURFACE = "#fcfcfb"

DNF_STATUSES = {"Timeout", "SkippedAfterTimeout", "NoResult", "Error", "BadJSON", "MissingBinary"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_rows(results_dir: Path) -> list[dict]:
    summary = results_dir / "summary.json"
    if not summary.exists():
        raise SystemExit(f"no summary.json in {results_dir} — run benchmark_all_sdks.py first")
    with open(summary) as fh:
        return json.load(fh)["rows"]


def pick(rows, suite, engine, *, status="Success"):
    out = [r for r in rows if r["suite"] == suite and r["engine"] == engine]
    if status:
        out = [r for r in out if r["status"] == status]
    return out


def series(rows, suite, engine, xkey, ykey):
    """(x, y) for the cases this engine finished, sorted by x, y present."""
    pts = [(r[xkey], r[ykey]) for r in pick(rows, suite, engine)
           if r.get(xkey) is not None and r.get(ykey) is not None]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def dnf_x(rows, suite, engine, xkey):
    """x values this engine did not finish — the smallest one is where it gave up.

    A timed-out row can be missing the case parameter itself (older runs did not
    record batch_size when there was no summary to read it from), so fall back to
    what another engine reported for the same case: the case parameter belongs to
    the case, not to whoever finished it. Cases are identified by their label —
    summary rows carry no test id.
    """
    by_case = {r["label"]: r[xkey] for r in rows
               if r["suite"] == suite and r.get(xkey) is not None}
    xs = [r.get(xkey) if r.get(xkey) is not None else by_case.get(r["label"])
          for r in rows
          if r["suite"] == suite and r["engine"] == engine
          and r["status"] in DNF_STATUSES]
    return sorted(x for x in xs if x is not None)


# ---------------------------------------------------------------------------
# Chart furniture
# ---------------------------------------------------------------------------

def new_axes(figsize=(7.2, 4.6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def dress(ax, *, title, xlabel, ylabel, subtitle=None):
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=16 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_SOFT,
                fontsize=9, va="bottom")
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=10)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)


def mark_dnf(ax, xs, y, color):
    """A hollow x at the first case an engine could not finish.

    Anchored to the engine's last successful y, so it reads as "the line stops
    here". An engine with no successful case in the suite has nothing to anchor
    to and is left to the panel's own "no finished case" note.
    """
    if not xs or y is None:
        return
    ax.plot([xs[0]], [y], marker="x", markersize=9, markeredgewidth=2,
            color=color, linestyle="none", zorder=5, clip_on=False)


def save(fig, ax, out_png: Path, table_rows, header):
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    with open(out_png.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(table_rows)
    print(f"  {out_png.name}  (+ {out_png.with_suffix('.csv').name})")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_scaling(rows, figs: Path):
    """Evaluation time against system size, no truncation anywhere."""
    fig, ax = new_axes()
    table = []
    for name in ORDER:
        e = ENGINES[name]
        # PP.jl reports one paired forward+backward sweep, not a separate
        # backward, so its forward-only number is the comparable one here.
        xs, ys = series(rows, "full_scaling", name, "qubits", "t_eval_steady_s")
        if xs:
            ax.plot(xs, ys, color=e["color"], marker=e["marker"], markersize=6,
                    linewidth=2, label=e["label"], zorder=3)
            ax.annotate(e["label"], (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(8, 0), color=INK_SOFT, fontsize=8, va="center")
            table += [[e["label"], x, y] for x, y in zip(xs, ys)]
        mark_dnf(ax, dnf_x(rows, "full_scaling", name, "qubits"), ys[-1] if ys else None,
                 e["color"])
    ax.set_yscale("log")
    dress(ax, title="Evaluation time vs system size",
          subtitle="depth-3 grid-Ising QAOA, no truncation · × marks the first case an engine could not finish",
          xlabel="qubits", ylabel="steady-state evaluation time (s)")
    ax.legend(frameon=False, labelcolor=INK_SOFT, fontsize=9, loc="upper left")
    save(fig, ax, figs / "fig_scaling_eval_time.png", table, ["engine", "qubits", "t_eval_s"])


def fig_batch(rows, figs: Path):
    """Throughput against batch size — the one figure where native batching shows."""
    fig, ax = new_axes(figsize=(9.4, 4.6))
    table = []
    for name in ORDER:
        e = ENGINES[name]
        xs, ys = series(rows, "embedding_batch", name, "batch_size", "batch_throughput_evals_s")
        if xs:
            ax.plot(xs, ys, color=e["color"], marker=e["marker"], markersize=6,
                    linewidth=2, label=e["label"], zorder=3)
            table += [[e["label"], x, y] for x, y in zip(xs, ys)]
        mark_dnf(ax, dnf_x(rows, "embedding_batch", name, "batch_size"),
                 ys[-1] if ys else None, e["color"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    dress(ax, title="Throughput vs batch size",
          subtitle="9-qubit circuit, data-embedding batch · only PADO batches natively; the others loop rows",
          xlabel="batch size", ylabel="evaluations / s")
    # No corner of the axes is free (PADO owns the top, cuPauliProp's flat line
    # crosses the middle, three engines sit on the floor), so the legend lives
    # outside — new_axes above is widened to give it room.
    ax.legend(frameon=False, labelcolor=INK_SOFT, fontsize=9,
              loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    save(fig, ax, figs / "fig_batch_throughput.png", table,
         ["engine", "batch_size", "throughput_evals_s"])


def fig_memory(rows, figs: Path):
    """Peak memory as truncation is loosened. GPU VRAM and CPU RAM are both
    'the memory this engine needed', and no engine reports both, so one axis
    carries both — the engine kind is in the legend."""
    fig, ax = new_axes()
    table = []
    for name in ORDER:
        e = ENGINES[name]
        pts = [(r["max_weight"], r["vram_steady_MB"] if r["vram_steady_MB"] is not None
                else r["ram_steady_MB"])
               for r in pick(rows, "mw_truncation", name) if r.get("max_weight") is not None]
        pts = sorted((x, y) for x, y in pts if y is not None)
        if pts:
            xs = [p[0] for p in pts]
            ys = [max(p[1], 0.1) for p in pts]  # log scale: 0 MB -> below the floor
            kind = "VRAM" if pick(rows, "mw_truncation", name)[0]["engine_kind"] == "gpu" else "RAM"
            ax.plot(xs, ys, color=e["color"], marker=e["marker"], markersize=6,
                    linewidth=2, label=f"{e['label']} ({kind})", zorder=3)
            table += [[e["label"], kind, x, y] for x, y in pts]
        mark_dnf(ax, dnf_x(rows, "mw_truncation", name, "max_weight"),
                 ys[-1] if pts else None, e["color"])
    ax.set_yscale("log")
    dress(ax, title="Peak memory vs max_weight",
          subtitle="16-qubit circuit · values under 0.1 MB are drawn at the axis floor",
          xlabel="max_weight", ylabel="peak memory (MB, log)")
    ax.legend(frameon=False, labelcolor=INK_SOFT, fontsize=9, loc="upper left")
    save(fig, ax, figs / "fig_memory_vs_maxweight.png", table,
         ["engine", "memory_kind", "max_weight", "peak_MB"])


def fig_terms(rows, figs: Path):
    """Two term counts that are NOT the same quantity, drawn as two series.

    Every engine propagates the same set; PADO and the PP.jl surrogate then
    zero-filter it down to what they actually evaluate. Plotting one number per
    engine would compare a filtered count against unfiltered ones.
    """
    fig, ax = new_axes()
    prop_x, prop_y = series(rows, "full_scaling", "cupauliprop", "qubits", "final_n_terms")
    eval_x, eval_y = series(rows, "full_scaling", "pps", "qubits", "final_n_terms")
    ax.plot(prop_x, prop_y, color=ENGINES["cupauliprop"]["color"],
            marker=ENGINES["cupauliprop"]["marker"], markersize=6, linewidth=2,
            label="propagated (every engine)", zorder=3)
    ax.plot(eval_x, eval_y, color=ENGINES["pps"]["color"],
            marker=ENGINES["pps"]["marker"], markersize=6, linewidth=2,
            label="evaluated after zero filter (PADO, PP.jl surrogate)", zorder=3)
    for x, p, e in zip(prop_x, prop_y, eval_y):
        if p and e:
            ax.annotate(f"×{p / e:,.0f}", (x, (p * e) ** 0.5), color=INK_SOFT,
                        fontsize=8, ha="center")
    ax.set_yscale("log")
    dress(ax, title="Pauli terms: propagated vs evaluated",
          subtitle="no truncation · the gap is what the zero filter removes",
          xlabel="qubits", ylabel="terms (log)")
    ax.legend(frameon=False, labelcolor=INK_SOFT, fontsize=9, loc="upper left")
    table = [["propagated", x, y] for x, y in zip(prop_x, prop_y)]
    table += [["evaluated", x, y] for x, y in zip(eval_x, eval_y)]
    save(fig, ax, figs / "fig_term_counts.png", table, ["count_kind", "qubits", "terms"])


def fig_gradient(rows, figs: Path):
    """Gradient error against system size, for the engines that differentiate.

    Three series, so the all-pairs palette cap is satisfied without faceting.
    """
    fig, ax = new_axes()
    table = []
    for name in ("pps", "cupauliprop", "julia"):
        e = ENGINES[name]
        xs, ys = series(rows, "full_scaling", name, "qubits", "grad_abs_err")
        if not xs:
            continue
        ax.plot(xs, ys, color=e["color"], marker=e["marker"], markersize=6,
                linewidth=2, label=e["label"], zorder=3)
        ax.annotate(e["label"], (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, 0), color=INK_SOFT, fontsize=8, va="center")
        table += [[e["label"], x, y] for x, y in zip(xs, ys)]
    ax.set_yscale("log")
    dress(ax, title="Gradient error vs system size",
          subtitle="untruncated, so this is the backward pass itself · "
                   "scored against exact central differences",
          xlabel="qubits", ylabel="max |gradient − exact| (log)")
    ax.legend(frameon=False, labelcolor=INK_SOFT, fontsize=9, loc="upper left")
    save(fig, ax, figs / "fig_gradient_error.png", table, ["engine", "qubits", "grad_abs_err"])


def fig_frontier(rows, figs: Path):
    """Accuracy bought against time spent — the honest form of a speed claim.

    Five engines cannot share one scatter (no ordering of five categorical hues
    clears the all-pairs separation floors), so each gets a panel and the others
    stay as grey context.
    """
    suites = [("mw_truncation", "max_weight — the same knob in every engine"),
              ("coeff_truncation", "min_abs_coeff — NOT the same knob: each engine "
                                   "prunes at a different point")]
    fig, axes = plt.subplots(len(suites), len(ORDER), figsize=(15.5, 7.0),
                             sharex="row", sharey="row", facecolor=SURFACE)
    table = []
    for row_i, (suite, note) in enumerate(suites):
        allpts = {n: series(rows, suite, n, "abs_err", "t_eval_steady_s") for n in ORDER}
        for col_i, name in enumerate(ORDER):
            ax = axes[row_i][col_i]
            ax.set_facecolor(SURFACE)
            for other in ORDER:
                if other == name:
                    continue
                ox, oy = allpts[other]
                if ox:
                    ax.plot(ox, oy, color=CONTEXT, marker="o", markersize=4,
                            linewidth=1.2, zorder=2)
            xs, ys = allpts[name]
            if xs:
                ax.plot(xs, ys, color=HILITE, marker=ENGINES[name]["marker"],
                        markersize=7, linewidth=2.2, zorder=3)
                table += [[suite, ENGINES[name]["label"], x, y] for x, y in zip(xs, ys)]
            else:
                ax.text(0.5, 0.5, "no finished\ncase", transform=ax.transAxes,
                        ha="center", va="center", color=INK_SOFT, fontsize=9)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_SOFT, labelsize=8)
            if row_i == 0:
                ax.set_title(ENGINES[name]["label"], color=INK, fontsize=10)
            if col_i == 0:
                ax.set_ylabel("eval time (s)", color=INK_SOFT, fontsize=9)
            if row_i == len(suites) - 1:
                ax.set_xlabel("|expval − exact|", color=INK_SOFT, fontsize=9)
    # Row notes as figure text, above each band: the panel titles already own
    # y=1.0 in axes coordinates, so anything placed there collides with them.
    for row_i, (_, note) in enumerate(suites):
        y = 0.90 if row_i == 0 else 0.46
        fig.text(0.012, y, note, color=INK_SOFT, fontsize=9, va="bottom")
    fig.suptitle("Accuracy bought vs time spent  (down-left is better)",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.855, bottom=0.10,
                        wspace=0.18, hspace=0.42)
    out = figs / "fig_accuracy_cost_frontier.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["suite", "engine", "abs_err", "t_eval_s"])
        w.writerows(table)
    print(f"  {out.name}  (+ {out.with_suffix('.csv').name})")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path,
                    help="results/ folder of a finished run (default: this device's)")
    ap.add_argument("--device", help="device tag, e.g. A100 — shorthand for its results folder")
    args = ap.parse_args()

    if args.results:
        results = args.results.resolve()
        figs = results.parent / "figures"
    else:
        # Import here so --results works on a machine with no torch at all.
        sys.path.insert(0, str(SCRIPT_DIR))
        import os

        if args.device:
            os.environ["PPS_DEVICE_TAG"] = args.device
        from _outdir import out_dir

        results = Path(out_dir("engine_benchmarks", "results", create=False))
        figs = Path(out_dir("engine_benchmarks", "figures"))
    figs.mkdir(parents=True, exist_ok=True)

    rows = load_rows(results)
    print(f"reading {results}")
    for fn in (fig_scaling, fig_batch, fig_memory, fig_terms, fig_gradient, fig_frontier):
        fn(rows, figs)
    print(f"figures in {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
