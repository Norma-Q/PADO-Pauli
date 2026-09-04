#!/usr/bin/env python3
"""Summarise the repeat-timing runs and write repeat_summary.json.

Reads what run_repeats.sh collected in the device's own data tree
(results_on_<DEVICE>/repeat_timing/runs/repeat*/) and writes the summary beside
it, so an A100 run and an MI300X run never overwrite each other's medians.
PPS_DATA_ROOT / PPS_DEVICE_TAG apply the same way as everywhere else in the
bundle (see ../_outdir.py).

Reports the median as the headline value with the min-max range alongside it;
ratios (diff_mode speedups) are summarised by median too.

  python3 summarize_repeats.py
"""
import json, os, statistics as st, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # reproducibility/
from _outdir import data_root, out_dir  # noqa: E402

RUNS = out_dir("repeat_timing", "runs", create=False)
SC = str(data_root() / "repeat_timing")


def load(job, fname):
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, "repeat*", job))):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            out.append((os.path.basename(os.path.dirname(d)), json.load(open(p))))
    return out


def fmt(vals, unit="", nd=1):
    med, lo, hi = st.median(vals), min(vals), max(vals)
    return f"median {med:.{nd}f}{unit}  (range {lo:.{nd}f}-{hi:.{nd}f}{unit}, n={len(vals)})"


def qubit_summary():
    runs = load("qubit", "qubit_sweep.json")
    if not runs:
        return None
    print(f"\n=== qubit sweep  ({len(runs)} repeats)")
    by_n = {}
    for _, rows in runs:
        for r in rows:
            by_n.setdefault(r["n_qubits"], {"prop": [], "comp": [], "vram": [],
                                            "terms": set(), "zf": set()})
            b = by_n[r["n_qubits"]]
            b["prop"].append(r["propagate_total_s"]); b["comp"].append(r["compile_total_s"])
            b["vram"].append(r["vram_peak_reserved_gb"])
            b["terms"].add(r["propagated_terms"]); b["zf"].add(r["zero_filtered_terms"])
    det = all(len(b["terms"]) == 1 and len(b["zf"]) == 1 for b in by_n.values())
    print(f"  term counts identical across repeats: {det}")
    for n in sorted(by_n):
        b = by_n[n]
        print(f"  n={n:>3}  propagate {fmt(b['prop'],'s',2)}")
        print(f"         compile   {fmt(b['comp'],'s',2)}")
        print(f"         VRAM      {fmt(b['vram'],'GB',3)}")
    med = {n: {k: st.median(v) for k, v in b.items() if isinstance(v, list)}
           for n, b in by_n.items()}
    # cubic VRAM fit on the per-n medians
    ns = sorted(med)
    try:
        import numpy as np
        c = np.polyfit([float(n) for n in ns], [med[n]["vram"] for n in ns], 3)
        pred = np.polyval(c, [float(n) for n in ns])
        res = max(abs(pred[i] - med[n]["vram"]) for i, n in enumerate(ns))
        print(f"\n  cubic VRAM fit ({len(ns)} points, max residual {res:.3f} GB):")
        print(f"    {c[0]:.2e} N^3 + {c[1]:.2e} N^2 + {c[2]:.2e} N + {c[3]:.2f}   (GB)")
        med["_vram_fit"] = [float(x) for x in c]
    except Exception as e:
        print(f"  (fit failed: {e})")
    return med


def diff_mode_summary():
    runs = load("diff_mode", "diff_mode_vjp.json")
    if not runs:
        return None
    print(f"\n=== diff_mode  ({len(runs)} run)")
    print("  each value is the median of REPS evaluations of one compiled program (REPS inside the script).")
    by_w = {}
    for _, d in runs:
        for r in d["results"]:
            b = by_w.setdefault(r["w_max"], {"sp": [], "mf": [], "mp": [], "ap": []})
            b["sp"].append(r["speedup_x"]); b["mf"].append(r["manual_mem_fraction"])
            # result files name the two modes either vjp/autograd or manual_vjp/autograph
            vj = r.get("vjp", r.get("manual_vjp")); ag = r.get("autograd", r.get("autograph"))
            b["mp"].append(vj["peak_alloc_gb"]); b["ap"].append(ag["peak_alloc_gb"])
    for w in sorted(by_w):
        b = by_w[w]
        print(f"  w_max={w}  speedup {fmt(b['sp'],'x',1)}")
        print(f"            manual peak {fmt(b['mp'],'GB',3)}   autograd peak {fmt(b['ap'],'GB',3)}")
        print(f"            mem fraction {fmt(b['mf'],'',4)}")
    allsp = [v for b in by_w.values() for v in b["sp"]]
    allmf = [v for b in by_w.values() for v in b["mf"]]
    meds = [st.median(by_w[w]["sp"]) for w in sorted(by_w)]
    print(f"  speedup medians per w_max: {['%.0fx'%m for m in meds]}, observed range {min(allsp):.1f}x-{max(allsp):.0f}x")
    print(f"  memory saving {(1-max(allmf))*100:.1f}%-{(1-min(allmf))*100:.1f}%")
    wmax = max(by_w)
    print(f"  largest point (w={wmax}): manual {st.median(by_w[wmax]['mp']):.1f} GB vs autograd {st.median(by_w[wmax]['ap']):.1f} GB")
    return by_w


def kicked_summary():
    runs = load("kicked", "kicked_ising_127q_full.json")
    if not runs:
        return None
    print(f"\n=== kicked Ising 127q  ({len(runs)} repeats)")
    tot = [sum(r["time_s"] for r in d["data"]) for _, d in runs]
    vram = [max(r["peak_vram_gb"] for r in d["data"]) for _, d in runs]
    evs = {tuple(round(r["ev"], 9) for r in d["data"]) for _, d in runs}
    print(f"  sweep total time {fmt([t/60 for t in tot],'min',2)}")
    print(f"  peak VRAM        {fmt(vram,'GB',4)}")
    print(f"  curve values identical across repeats: {len(evs)==1}")
    return {"minutes": st.median([t / 60 for t in tot]), "vram": st.median(vram)}


def fwdbwd_summary():
    runs = load("fwdbwd", "random_circuit_fwdbwd.json")
    if not runs:
        return None
    print(f"\n=== fwd+bwd training loop  ({len(runs)} run)")
    print("  each value is the median of 5 training loops on one compiled program (REPS inside the script).")
    out = {}
    for n in sorted({n for _, d in runs for n in d}, key=int):
        pps = [d[n]["pps_loop_s"] for _, d in runs if n in d]
        print(f"  n={n:>3}  PPS loop {fmt(pps,'s',2)}")
        devs = {dv for _, d in runs if n in d for dv in d[n].get("baselines", {})}
        row = {"pps_loop_s": st.median(pps)}
        for dv in sorted(devs):
            sp = [d[n]["baselines"][dv]["speedup"] for _, d in runs
                  if n in d and "speedup" in d[n].get("baselines", {}).get(dv, {})]
            if not sp:
                print(f"        vs {dv:18s} FAILED in every repeat"); continue
            print(f"        vs {dv:18s} {fmt(sp,'x',1)}")
            row[dv] = st.median(sp)
        out[n] = row
    return out


def sweep_summary():
    runs = load("sweep", "random_circuit_sweep.json")
    if not runs:
        return None
    print(f"\n=== random-circuit sweep  ({len(runs)} repeats)")
    out = {}
    for n in sorted({n for _, d in runs for n in d["by_n"]}, key=int):
        acc = [d["by_n"][n]["accuracy"] for _, d in runs if n in d["by_n"]]
        det = len({(a["peak_terms"], a["final_terms"]) for a in acc}) == 1
        print(f"  n={n:>3}  term counts identical across repeats: {det}   max|err| {fmt([a['max_abs'] for a in acc],'',4)}")
        row = {"max_abs": st.median([a["max_abs"] for a in acc]),
               "mean_abs": st.median([a["mean_abs"] for a in acc]), "throughput": {}}
        for B in sorted(runs[0][1]["by_n"][n]["throughput"], key=int):
            per = [d["by_n"][n]["throughput"][B] for _, d in runs if n in d["by_n"]]
            pps = [p["pps_samples_per_s"] for p in per]
            print(f"        batch {B:>3}  PPS {fmt(pps,' ev/s',0)}")
            row["throughput"][B] = {"pps": st.median(pps), "baselines": {}}
            for dv in sorted({k for p in per for k in p.get("baselines", {})}):
                bs = [p["baselines"][dv]["pl_samples_per_s"] for p in per if dv in p.get("baselines", {})]
                sp = [p["baselines"][dv]["speedup"] for p in per if dv in p.get("baselines", {})]
                print(f"              vs {dv:18s} {fmt(bs,' ev/s',1)}   speedup {fmt(sp,'x',0)}")
                row["throughput"][B]["baselines"][dv] = {"pl": st.median(bs), "speedup": st.median(sp)}
        out[n] = row
    return out


def main():
    if not os.path.isdir(RUNS):
        print(f"no runs yet: {RUNS}\nrun run_repeats.sh first.")
        return 1
    q = qubit_summary(); d = diff_mode_summary(); k = kicked_summary()
    f = fwdbwd_summary(); s = sweep_summary()
    summary = {"note": "Medians over the repeat runs; the min-max ranges are printed by summarize_repeats.py.",
               "qubit_sweep_median": q, "kicked_median": k, "fwdbwd_median": f,
               "random_circuit_sweep_median": s,
               "diff_mode_median": {str(w): {kk: st.median(vv) for kk, vv in b.items()}
                                    for w, b in (d or {}).items()}}
    p = os.path.join(SC, "repeat_summary.json")
    json.dump(summary, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
