#!/usr/bin/env python
"""Forward+backward training-loop time, PPS vs exact statevector (adjoint).
Same circuit/config as run_random_circuit_sweep.py.

Two exact baselines are timed:
  lightning.qubit   CPU statevector
  lightning.gpu / lightning.amdgpu   GPU statevector, picked by torch build
                    libhsa-runtime64.so.1, see reproducibility/README.md)
Both are exact; only the wall-clock ratios differ. Override with PL_DEVICES
(comma-separated) to time just one.
"""
import gc, json, os, sys, time
from pathlib import Path
import numpy as np
import pennylane as qml
import torch

from padopauli import Circuit

# Erdos-Renyi edge probability, held fixed across sizes (EDGE_PROB=log uses log(n)/n).
EDGE_PROB = os.environ.get("EDGE_PROB", "0.15")


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
OPT_BATCH, OPT_STEPS, OPT_LR, OPT_OBS = 16, 10, 0.05, 0
# Timed training loops per compiled program, plus one discarded warm-up.
REPS = int(os.environ.get("REPS", 5))
N_LIST = [12, 16, 20]; MW = 9
PL_DEVICES = [d for d in os.environ.get(
    "PL_DEVICES", f"lightning.qubit,{_lightning_gpu_device()}").split(",") if d]
# The CPU statevector is only timed up to this qubit count.
PL_CPU_MAX_N = int(os.environ.get("PL_CPU_MAX_N", 16))
CPU_DEVICES = {"lightning.qubit"}


def devices_for(n):
    return [d for d in PL_DEVICES if n <= PL_CPU_MAX_N or d not in CPU_DEVICES]


def _sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def build(n):
    rng = np.random.default_rng(SEED); ep = float(np.log(n) / n) if EDGE_PROB == "log" else float(EDGE_PROB)
    graphs = [[(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < ep] for _ in range(GEN_LAYERS)]
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
    return qc, e, p, graphs, groups


def make_obs(n, groups):
    return [("Z" * len(gp), list(gp)) for gp in groups]


def make_qnode(n, graphs, groups, dev_name):
    dev = qml.device(dev_name, wires=n)
    def brick(l): s = l % 2; return [(i, i + 1) for i in range(s, n - 1, 2)]
    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def qn(gm, th):
        ei = pi = 0
        for l in range(GEN_LAYERS):
            for q in range(n): qml.RX(float(gm[ei]), wires=q); ei += 1
            for a, b in graphs[l]: qml.Hadamard(b); qml.CNOT([a, b]); qml.Hadamard(b)
        for q in range(n): qml.RX(float(gm[ei]), wires=q); ei += 1
        for q in range(n): qml.RY(float(gm[ei]), wires=q); ei += 1
        for l in range(V_LAYERS):
            for q in range(n):
                qml.RY(th[pi], wires=q); pi += 1
                qml.RZ(th[pi], wires=q); pi += 1
            for a, b in brick(l): qml.Hadamard(b); qml.CNOT([a, b]); qml.Hadamard(b)
        return [qml.expval(qml.prod(*[qml.PauliZ(q) for q in gp]) if len(gp) > 1 else qml.PauliZ(gp[0])) for gp in groups]
    return qn


res = {}
for n in N_LIST:
    circ, ne, npar, graphs, groups = build(n); obs = make_obs(n, groups)
    g = torch.Generator(device="cpu"); g.manual_seed(EVAL)
    theta = (torch.randn((npar,), generator=g, dtype=torch.float32) * THS)
    og = torch.Generator(device="cpu"); og.manual_seed(EVAL + 2000)
    gopt = (torch.randn((OPT_BATCH, ne), generator=og, dtype=torch.float32) * TAU)
    gopt_np = gopt.numpy(); theta_init = theta.clone().detach()

    gc.collect(); torch.cuda.empty_cache()
    prog = circ.compile(observables=obs, preset="gpu",
        dtype="float32", max_weight=MW, chunk_size=25_000_000)
    gopt_dev = gopt.to(DEV)
    # One compile, REPS timed loops. The first is a warm-up and is discarded (it absorbs
    # the one-time GPU cost: HIP context, kernel JIT). Every loop restarts from theta_init
    # with a fresh optimizer, so all REPS do exactly the same work.
    pps_t, pps_losses = [], []
    for rep in range(REPS + 1):
        th = theta_init.to(DEV).requires_grad_(True)
        opt = torch.optim.Adam([th], lr=OPT_LR); losses = []
        _sync(); t0 = time.perf_counter()
        for _ in range(OPT_STEPS):
            opt.zero_grad(set_to_none=True)
            out = prog.expvals(thetas=th, embedding=gopt_dev)
            loss = out[:, OPT_OBS].mean(); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        _sync()
        if rep:                       # rep 0 is the warm-up
            pps_t.append(time.perf_counter() - t0); pps_losses = losses
    pps_loop = float(np.median(pps_t))
    del prog, circ; gc.collect(); torch.cuda.empty_cache()  # circ holds the program too

    baselines = {}
    for dev_name in devices_for(n):
        try:
            qn = make_qnode(n, graphs, groups, dev_name)
            pl_t, pl_losses = [], []          # same warm-up + REPS protocol as PPS gets
            for rep in range(REPS + 1):
                thp = theta_init.clone().detach().requires_grad_(True)
                opt = torch.optim.Adam([thp], lr=OPT_LR); losses = []
                t0 = time.perf_counter()
                for _ in range(OPT_STEPS):
                    opt.zero_grad(set_to_none=True)
                    outs = torch.stack([torch.stack(qn(gopt_np[i], thp)) for i in range(OPT_BATCH)])
                    loss = outs[:, OPT_OBS].mean(); loss.backward(); opt.step()
                    losses.append(float(loss.detach().cpu()))
                if rep:
                    pl_t.append(time.perf_counter() - t0); pl_losses = losses
            pl_loop = float(np.median(pl_t))
            baselines[dev_name] = {"loop_s": pl_loop, "ms_step": pl_loop / OPT_STEPS * 1e3,
                                   "speedup": pl_loop / pps_loop, "loss": pl_losses}
            print(f"n={n}: PPS {pps_loop:.3f}s ({pps_loop/OPT_STEPS*1e3:.0f} ms/step) | "
                  f"{dev_name} {pl_loop:.3f}s ({pl_loop/OPT_STEPS*1e3:.0f} ms/step) | "
                  f"PPS x{pl_loop/pps_loop:.1f} | loss PPS {pps_losses[0]:+.3f}->{pps_losses[-1]:+.3f} "
                  f"exact {pl_losses[0]:+.3f}->{pl_losses[-1]:+.3f}")
        except Exception as e:      # one dead backend must not lose the other's timing
            baselines[dev_name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"n={n}: {dev_name} FAILED -- {type(e).__name__}: {str(e)[:120]}")
    res[str(n)] = {"pps_loop_s": pps_loop, "pps_ms_step": pps_loop / OPT_STEPS * 1e3,
                   "pps_loss": pps_losses, "baselines": baselines,
                   "baselines_timed": devices_for(n)}

(OUT / "random_circuit_fwdbwd.json").write_text(json.dumps(res, indent=2))
print("[saved] results/random_circuit_fwdbwd.json")
