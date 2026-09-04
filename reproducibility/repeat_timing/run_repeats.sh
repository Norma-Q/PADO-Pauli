#!/usr/bin/env bash
# Repeat-timing driver for the five timing-sensitive experiments.
#
#   qubit      sweep_qubits_fixed_wmax.py   -> compile/propagate times, VRAM, cubic fit
#   diff_mode  run_diff_mode_benchmark.py   -> vjp vs autograd speed ratio
#   kicked     run_sweep.sh                 -> 127-qubit kicked-Ising sweep wall-clock
#   fwdbwd     run_fwdbwd.py                -> fwd+bwd loop vs exact statevector
#   sweep      run_random_circuit_sweep.py  -> forward throughput and truncation errors
#
# The cuPauliProp/Qiskit/Julia comparison (engine_benchmarks/) is not part of this
# driver: cuPauliProp is CUDA-only.
#
# TWO KINDS OF EXPERIMENT, TWO PLACES TO REPEAT
#   qubit and kicked measure COMPILATION itself, so they are compiled REPEATS times here.
#   diff_mode, fwdbwd and sweep measure EVALUATION against a program compiled once, so
#   they run once here (ONCE_JOBS) and repeat five evaluations inside the script.
#
# Term counts, accuracies and VRAM are deterministic; only wall-clock timings move
# between runs, so only the timing-sensitive experiments are repeated.
#
# PROTOCOL
#   - strictly sequential, one process at a time, no parallelism
#   - each repeat writes into its own results directory, so nothing is
#     overwritten and every repeat stays auditable
#   - a failing repeat is logged and skipped; the run never aborts
#   - the device's results tree (results_on_<DEVICE>/) is restored to its
#     pre-run state at the end, so this driver is non-destructive
#
# USAGE
#   bash run_repeats.sh            # all five jobs
#   REPEATS=3 bash run_repeats.sh  # fewer compile repeats
#   JOBS="qubit kicked" bash run_repeats.sh   # subset
#
# summarize_repeats.py reports the median with the observed min-max range.
set -u

# The code bundle (reproducibility/). Output never lands here -- it goes to
# results_on_<DEVICE>/repeat_timing/, resolved below.
BUNDLE=$(cd "$(dirname "$0")/.." && pwd)
# Uses whatever python is on PATH. Override with PY=/path/to/python if needed.
PY=${PY:-python}
OUT="$(cd "$(dirname "$0")" && pwd)"   # this script's own folder: code only, never output
REPEATS=${REPEATS:-5}
# Jobs that run exactly once no matter what REPEATS says (see above).
ONCE_JOBS=${ONCE_JOBS:-"diff_mode fwdbwd sweep"}
JOBS=${JOBS:-"qubit diff_mode kicked fwdbwd sweep"}
JOB_TIMEOUT=${JOB_TIMEOUT:-21600}

# Repeat outputs are data, so they land in <repo>/results_on_<DEVICE>/repeat_timing/
# rather than next to the script. PPS_DATA_ROOT (exported by run_rerun_all.sh)
# wins; standalone runs resolve it the same way the Python scripts do.
DATA_ROOT=${PPS_DATA_ROOT:-$("$PY" -c 'import sys; sys.path.insert(0, "'"$BUNDLE"'"); from _outdir import data_root; print(data_root())')}
export PPS_DATA_ROOT="$DATA_ROOT"
DATA="$DATA_ROOT/repeat_timing"
mkdir -p "$DATA"
LOGS="$DATA/logs"; RUNS="$DATA/runs"; STATUS="$DATA/status.tsv"
mkdir -p "$LOGS" "$RUNS"
[ -s "$STATUS" ] || printf 'repeat\tjob\tstatus\texit\tseconds\tstarted\n' > "$STATUS"

export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg

# Preflight: the env must carry torch, a working padopauli install, and a GPU.
"$PY" - <<'PYCHECK' || { echo "[FATAL] activate an env with padopauli installed first (see the repository README)"; exit 1; }
import sys, torch
import padopauli.backend  # the compiled engine; fails loudly if the wheel does not match this Python/torch
assert torch.cuda.is_available(), "no GPU visible to torch"
print(f"[env] python={sys.version.split()[0]} torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
PYCHECK

# LD_PRELOAD is a ROCm-only workaround and is not needed on CUDA; honoured if set.
# before running (see reproducibility/README.md); this script does not set a default.
[ -n "${LD_PRELOAD:-}" ] && export LD_PRELOAD

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$DATA/driver.log"; }

# Snapshot the repo results trees so this driver leaves no side effects. A fresh
# snapshot is taken every run; a pre-existing one is moved aside, not deleted.
SNAP="$DATA/_repo_snapshot"
if [ -d "$SNAP" ]; then
  STALE="$DATA/_repo_snapshot.stale.$(date +%Y%m%d_%H%M%S)"
  mv "$SNAP" "$STALE"
  log "moved a pre-existing snapshot aside -> $STALE (restore uses a fresh one)"
fi
mkdir -p "$SNAP"
# A first run on a new device has no results yet; make the trees so the
# snapshot/restore below is a no-op instead of an error.
for d in scaling_accuracy_noise diff_mode_vjp kicked_ising_127q random_circuit_statevector; do
  mkdir -p "$DATA_ROOT/$d/results"
done
cp -a "$DATA_ROOT/scaling_accuracy_noise/results" "$SNAP/qubit_results"
cp -a "$DATA_ROOT/diff_mode_vjp/results"          "$SNAP/diff_mode_results"
cp -a "$DATA_ROOT/kicked_ising_127q/results"      "$SNAP/kicked_results"
cp -a "$DATA_ROOT/random_circuit_statevector/results" "$SNAP/randcirc_results"
log "snapshotted repo results trees into $SNAP"

run_one () {            # run_one <repeat> <job> <workdir> <cmd...>
  local rep="$1"; shift; local job="$1"; shift; local wd="$1"; shift
  local name="r${rep}_${job}"
  local t0 t1 rc state started
  started=$(date +%H:%M:%S)
  log "START  $name"
  t0=$(date +%s)
  ( cd "$wd" && timeout "$JOB_TIMEOUT" "$@" ) > "$LOGS/$name.log" 2>&1
  rc=$?; t1=$(date +%s)
  state=ok; [ $rc -ne 0 ] && state=FAILED; [ $rc -eq 124 ] && state=TIMEOUT
  printf '%d\t%s\t%s\t%d\t%d\t%s\n' "$rep" "$job" "$state" "$rc" "$((t1-t0))" "$started" >> "$STATUS"
  log "$state  $name  (rc=$rc, $((t1-t0))s)"
  return 0
}

collect () {            # collect <repeat> <job> <src-file>...
  local rep="$1"; shift; local job="$1"; shift
  local dst="$RUNS/repeat$rep/$job"; mkdir -p "$dst"
  for f in "$@"; do
    [ -f "$f" ] && cp -a "$f" "$dst/"
    m="${f%.*}.runmeta.json"; [ -f "$m" ] && cp -a "$m" "$dst/"   # provenance sidecar travels with the result
  done
}

log "===== START $(date)  repeats=$REPEATS  jobs=[$JOBS]  once=[$ONCE_JOBS] ====="
$PY -c "import torch;print('torch',torch.__version__,'gpu',torch.cuda.get_device_name(0))" 2>&1 | tee -a "$DATA/driver.log"

for rep in $(seq 1 "$REPEATS"); do
  log "----- repeat $rep / $REPEATS -----"
  for job in $JOBS; do
    case " $ONCE_JOBS " in
      *" $job "*) [ "$rep" -eq 1 ] || { log "skip $job (runs once)"; continue; } ;;
    esac
    case "$job" in
      qubit)
        run_one "$rep" qubit "$BUNDLE/scaling_accuracy_noise" \
                $PY sweep_qubits_fixed_wmax.py
        collect "$rep" qubit "$DATA_ROOT/scaling_accuracy_noise/results/qubit_sweep.json" \
                              "$DATA_ROOT/scaling_accuracy_noise/results/qubit_sweep.csv"
        ;;
      diff_mode)
        run_one "$rep" diff_mode "$BUNDLE/diff_mode_vjp" \
                $PY run_diff_mode_benchmark.py
        collect "$rep" diff_mode "$DATA_ROOT/diff_mode_vjp/results/diff_mode_vjp.json"
        ;;
      kicked)
        run_one "$rep" kicked "$BUNDLE/kicked_ising_127q" \
                bash run_sweep.sh
        collect "$rep" kicked "$DATA_ROOT/kicked_ising_127q/results/kicked_ising_127q_full.json" \
                               "$DATA_ROOT/kicked_ising_127q/results/kicked_ising_127q_full.csv"
        ;;
      fwdbwd)     # PPS vs exact statevector, fwd+bwd training loop
        run_one "$rep" fwdbwd "$BUNDLE/random_circuit_statevector" \
                $PY run_fwdbwd.py
        collect "$rep" fwdbwd "$DATA_ROOT/random_circuit_statevector/results/random_circuit_fwdbwd.json"
        ;;
      sweep)      # forward throughput + truncation accuracy
        run_one "$rep" sweep "$BUNDLE/random_circuit_statevector" \
                $PY run_random_circuit_sweep.py
        collect "$rep" sweep "$DATA_ROOT/random_circuit_statevector/results/random_circuit_sweep.json"
        ;;
      *) log "unknown job '$job' -- skipped" ;;
    esac
  done
done

# restore the repo trees: the run is a measurement, not a state change
cp -a "$SNAP/qubit_results/."     "$DATA_ROOT/scaling_accuracy_noise/results/"
cp -a "$SNAP/diff_mode_results/." "$DATA_ROOT/diff_mode_vjp/results/"
cp -a "$SNAP/kicked_results/."    "$DATA_ROOT/kicked_ising_127q/results/"
cp -a "$SNAP/randcirc_results/."  "$DATA_ROOT/random_circuit_statevector/results/"
log "restored repo results trees from the pre-run snapshot"

log "===== DONE $(date) ====="
column -t "$STATUS" | tee -a "$DATA/driver.log"
echo

# Regenerate repeat_summary.json from the collected runs.
if "$PY" "$OUT/summarize_repeats.py" 2>&1 | tee -a "$DATA/driver.log"; then
  log "repeat_summary.json regenerated from $(ls -d "$RUNS"/repeat* 2>/dev/null | wc -l) repeat(s)"
else
  log "WARN  summarize_repeats.py failed -- repeat_summary.json is NOT current"
fi
