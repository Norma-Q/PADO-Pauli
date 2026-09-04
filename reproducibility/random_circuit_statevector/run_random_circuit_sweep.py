#!/usr/bin/env python
"""Random quantum circuit (hard for Pauli propagation): PPS vs exact statevector, scale sweep.

Workload: random RX+CZ generative circuit U(gamma) with random angles gamma, followed by
a trainable circuit V(theta) (construction from Hirviniemi, Basheer, Cope, "Random Quantum
Circuits as Seeds for Continuous Generative Models", arXiv:2602.10049; hard for Pauli
propagation). gamma = per-sample embedding input (embedding_idx, hard-detached, not
differentiated); theta = trainable (param_idx). CZ compiled as H.CNOT.H.

Sweep n in {12,16,20} at fixed max_weight=9, 2 observables Z_0 and Z_{n-2}(x)Z_{n-1}.
For each n:
  (A) accuracy        : PPS expvals vs exact statevector, batch 100, fixed theta
  (B) FORWARD throughput (no backward): compile once, then for batch in {128,256,512}
      time the forward pass on PPS (one batched tensor op) vs each PennyLane statevector
      baseline (sample-by-sample) -> samples/s.
Extra: 16q at mw=10 (accuracy only).

Statevector baselines (PL_DEVICES): lightning.qubit (CPU) and the GPU statevector
device for the installed torch build (lightning.gpu on CUDA,
lightning.amdgpu on ROCm).
Part (A) uses only the first entry (all baselines are exact). The CPU baseline is dropped
above PL_CPU_MAX_N=16 qubits. A backend that fails to construct is skipped, not fatal.
Backward/gradient timing is not measured here (see run_fwdbwd.py).

Run: python run_random_circuit_sweep.py
"""
import gc, json, os, sys, time
from pathlib import Path
import numpy as np
import pennylane as qml
import torch

from padopauli import Circuit  # noqa: E402

# Output location is device-aware: <repo>/results_on_<DEVICE>/{exp}/...
# so the same code can run on an A100, an H100 and an MI300X without one run
# overwriting another. See reproducibility/_outdir.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from _outdir import out_dir as _out_dir
from _outdir import lightning_gpu_device as _lightning_gpu_device

OUT = Path(_out_dir("random_circuit_statevector", "results"))
DEV = torch.device("cuda:0")

GEN_LAYERS, V_LAYERS = 3, 2
TAU = float(np.sqrt(0.24)); THS = 0.1
SEED, EVAL = 42, 20260519
ACC_BATCH = 100
TP_BATCHES = [128, 256, 512]
TP_REPS = 5            # PPS forward reps (median); lightning timed once per batch
N_LIST = [12, 16, 20]
MW = 9
# Erdos-Renyi edge probability, held fixed across sizes (default 0.15).
# EDGE_PROB=log uses log(n)/n instead.
EDGE_PROB = os.environ.get("EDGE_PROB", "0.15")
TAG = os.environ.get("TAG", "")
ACC_ONLY = bool(os.environ.get("ACC_ONLY"))   # skip part (B); accuracy is deterministic
PL_DEVICES = [d for d in os.environ.get(                     # see docstring
    "PL_DEVICES", f"lightning.qubit,{_lightning_gpu_device()}").split(",") if d]
# The CPU statevector is only timed up to this qubit count; above it the GPU baseline is
# the one that carries the comparison. Matches run_fwdbwd.py.
PL_CPU_MAX_N = int(os.environ.get("PL_CPU_MAX_N", 16))
CPU_DEVICES = {"lightning.qubit"}


def devices_for(n):
    return [d for d in PL_DEVICES if n <= PL_CPU_MAX_N or d not in CPU_DEVICES]


def _sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()
def _free():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def build(n):
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
    return qc, e, p, graphs, groups, ["Z0", f"Z{n-2}Z{n-1}"]


def make_obs(n, groups):
    return [("Z" * len(gp), list(gp)) for gp in groups]


def make_qnode(n, graphs, groups, dev_name):
    dev = qml.device(dev_name, wires=n)
    def brick(l): s = l % 2; return [(i, i + 1) for i in range(s, n - 1, 2)]
    @qml.qnode(dev, interface=None, diff_method=None)
    def qn(gm, th):
        ei = pi = 0
        for l in range(GEN_LAYERS):
            for q in range(n): qml.RX(float(gm[ei]), wires=q); ei += 1
            for a, b in graphs[l]: qml.Hadamard(b); qml.CNOT([a, b]); qml.Hadamard(b)
        for q in range(n): qml.RX(float(gm[ei]), wires=q); ei += 1
        for q in range(n): qml.RY(float(gm[ei]), wires=q); ei += 1
        for l in range(V_LAYERS):
            for q in range(n):
                qml.RY(float(th[pi]), wires=q); pi += 1
                qml.RZ(float(th[pi]), wires=q); pi += 1
            for a, b in brick(l): qml.Hadamard(b); qml.CNOT([a, b]); qml.Hadamard(b)
        return [qml.expval(qml.prod(*[qml.PauliZ(q) for q in gp]) if len(gp) > 1 else qml.PauliZ(gp[0]))
                for gp in groups]
    return qn


def compile_pps(circ, obs, mw):
    return circ.compile(observables=obs, preset="gpu",
                        dtype="float32", max_weight=mw, chunk_size=25_000_000)


results = {"config": {"gen_layers": GEN_LAYERS, "v_layers": V_LAYERS, "mw": MW,
                      "acc_batch": ACC_BATCH, "tp_batches": TP_BATCHES, "tp_reps": TP_REPS,
                      "n_list": N_LIST, "dtype": "float32", "pl_devices": PL_DEVICES}, "by_n": {}}

for n in N_LIST:
    print(f"\n################## n={n}, mw={MW} ##################")
    circ, ne, npar, graphs, groups, labels = build(n)
    obs = make_obs(n, groups)
    print(f"full={len(circ.gates)}g emb={ne} par={npar} obs={labels}")
    rec = {"n": n, "full_gates": len(circ.gates), "n_embed": ne, "n_params": npar, "obs": labels}

    g = torch.Generator(device="cpu"); g.manual_seed(EVAL)
    theta = (torch.randn((npar,), generator=g, dtype=torch.float32) * THS)
    theta_dev = theta.to(DEV); tnp = theta.numpy()

    qnodes = {}                       # a dead backend must not lose the others
    for dev_name in devices_for(n):
        try:
            qnodes[dev_name] = make_qnode(n, graphs, groups, dev_name)
        except Exception as e:
            print(f"    [skip] {dev_name}: {type(e).__name__}: {str(e)[:120]}")
    if not qnodes:
        raise RuntimeError(f"no usable statevector baseline among {PL_DEVICES}")
    acc_dev = next(iter(qnodes)); qn = qnodes[acc_dev]   # exact => backend-independent

    # ---- (A) accuracy ----
    gamma = (torch.randn((ACC_BATCH, ne), generator=g, dtype=torch.float32) * TAU)
    gnp = gamma.numpy()
    t0 = time.perf_counter()
    ref = np.stack([np.asarray([float(x) for x in qn(gnp[i], tnp)]) for i in range(ACC_BATCH)])
    ref_s = time.perf_counter() - t0

    _free(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter(); prog = compile_pps(circ, obs, MW); cs = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 1024**3
    sm = prog.compile_stats.get("summary", {})
    t1 = time.perf_counter()
    pps = prog.expvals(thetas=theta_dev, embedding=gamma.to(DEV)).detach().cpu().numpy()
    es = time.perf_counter() - t1
    d = np.abs(pps.astype(np.float64) - ref)
    rec["accuracy"] = {
        "compile_s": cs, "vram_gb": vram, "exact_device": acc_dev,
        "peak_terms": sm.get("propagated_terms"), "final_terms": sm.get("final_terms"),
        "max_abs": float(d.max()), "mean_abs": float(d.mean()), "rmse": float(np.sqrt((d ** 2).mean())),
        "per_obs_max": {labels[j]: float(d[:, j].max()) for j in range(len(labels))},
        # scale of the exact values over the same batch: one denominator per cell,
        # so a near-zero individual reference cannot inflate a relative error
        "ref_rms": float(np.sqrt((ref ** 2).mean())),
        "ref_mean_abs": float(np.abs(ref).mean()),
        "ref_rms_per_obs": {labels[j]: float(np.sqrt((ref[:, j] ** 2).mean()))
                            for j in range(len(labels))},
    }
    print(f"[A] compile={cs:.1f}s VRAM={vram:.1f}GB | max|Δ|={d.max():.4f} mean={d.mean():.4f}")

    # ---- (B) forward throughput (no backward) ----
    rec["throughput"] = {}
    if ACC_ONLY:
        results["by_n"][str(n)] = rec
        del prog, circ; _free()
        continue
    print(f"[B] forward throughput (PPS vs {', '.join(qnodes)}):")
    for B in TP_BATCHES:
        gb = (torch.randn((B, ne), generator=g, dtype=torch.float32) * TAU)
        gb_dev = gb.to(DEV); gb_np = gb.numpy()
        # PPS: one batched forward op (median of TP_REPS, after warmup)
        _sync(); _ = prog.expvals(thetas=theta_dev, embedding=gb_dev).detach(); _sync()
        pps_t = []
        for _ in range(TP_REPS):
            _sync(); t0 = time.perf_counter()
            _ = prog.expvals(thetas=theta_dev, embedding=gb_dev).detach()
            _sync(); pps_t.append(time.perf_counter() - t0)
        pps_t_med = float(np.median(pps_t)); pps_sps = B / pps_t_med
        entry = {"pps_time_s": pps_t_med, "pps_samples_per_s": pps_sps, "baselines": {}}
        for dev_name, q in qnodes.items():
            # statevector: sample-by-sample forward (warmup one, then timed pass)
            _ = q(gb_np[0], tnp)
            t0 = time.perf_counter()
            for i in range(B): q(gb_np[i], tnp)
            pl_t = time.perf_counter() - t0; pl_sps = B / pl_t
            entry["baselines"][dev_name] = {"pl_time_s": pl_t, "pl_samples_per_s": pl_sps,
                                            "speedup": pps_sps / pl_sps}
            print(f"    batch {B:4d}: PPS {pps_sps:8.0f} s/s ({pps_t_med:.3f}s)  | "
                  f"{dev_name} {pl_sps:7.1f} s/s ({pl_t:.2f}s)  | x{pps_sps/pl_sps:.0f}")
        rec["throughput"][str(B)] = entry

    results["by_n"][str(n)] = rec
    del prog, circ; _free()  # circ holds the program too

# ---- extra: 16q mw=10 accuracy (raising mw recovers accuracy at steep cost) ----
print("\n################## extra: n=16 mw=10 (accuracy only) ##################")
n = 16
circ, ne, npar, graphs, groups, labels = build(n); obs = make_obs(n, groups)
g = torch.Generator(device="cpu"); g.manual_seed(EVAL)
theta = (torch.randn((npar,), generator=g, dtype=torch.float32) * THS)
gamma = (torch.randn((ACC_BATCH, ne), generator=g, dtype=torch.float32) * TAU)
qn = make_qnode(n, graphs, groups, PL_DEVICES[0])   # accuracy only => backend-independent
ref = np.stack([np.asarray([float(x) for x in qn(gamma.numpy()[i], theta.numpy())]) for i in range(ACC_BATCH)])
_free(); torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter(); prog = compile_pps(circ, obs, 10); cs = time.perf_counter() - t0
vram = torch.cuda.max_memory_allocated() / 1024**3; sm = prog.compile_stats.get("summary", {})
pps = prog.expvals(thetas=theta.to(DEV), embedding=gamma.to(DEV)).detach().cpu().numpy()
d = np.abs(pps.astype(np.float64) - ref)
results["n16_mw10"] = {"compile_s": cs, "vram_gb": vram, "max_abs": float(d.max()),
                       "mean_abs": float(d.mean()),
                       "ref_rms": float(np.sqrt((ref ** 2).mean())),
                       "ref_mean_abs": float(np.abs(ref).mean()),
                       "per_obs_max": {labels[j]: float(d[:, j].max()) for j in range(2)}}
print(f"n=16 mw=10: compile={cs:.1f}s VRAM={vram:.1f}GB max|Δ|={d.max():.4f} mean={d.mean():.4f}")
del prog, circ; _free()  # circ holds the program too

results["edge_prob_setting"] = EDGE_PROB if EDGE_PROB else "log(n)/n"
_out = OUT / f"random_circuit_sweep{TAG}.json"
_out.write_text(json.dumps(results, indent=2))
print(f"\n[saved] {_out}")
