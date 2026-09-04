#!/usr/bin/env bash
# Re-run driver for every experiment in this bundle that runs on an AMD (ROCm) GPU box.
#
# ACTIVATE THE ENV FIRST -- this script does not pick an interpreter. `python`
# on PATH must have padopauli installed (see the repository README):
#
#     bash reproducibility/run_rerun_all.sh
#
# Not included: engine_benchmarks/ (cuPauliProp is CUDA-only; run it separately
# on a CUDA machine).
#
# USAGE
#   bash run_rerun_all.sh                      # every job, in the order below
#   JOBS="grad_quasi noise" bash run_rerun_all.sh   # a subset
#   FORCE=1 bash run_rerun_all.sh              # re-run jobs already marked done
#   DRYRUN=1 bash run_rerun_all.sh             # print the plan and exit
#
# Order is cheapest-and-deterministic first; the long wall-clock jobs (diffmode,
# repeats) go last.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
PY=${PY:-python}
# Logs and the job-status table live beside the data this run produces, in
# <repo>/results_on_<DEVICE>/, so an A100 run and an MI300X run of the same code
# never share a status file. _outdir.py decides the device tag.
DATA_ROOT=$("$PY" -c 'import sys; sys.path.insert(0, "'"$HERE"'"); from _outdir import data_root; print(data_root())') \
    || { echo "[FATAL] could not resolve the data root (see $HERE/_outdir.py)"; exit 1; }
export PPS_DATA_ROOT="$DATA_ROOT"
LOGS="$DATA_ROOT/rerun_logs"; STATUS="$LOGS/status.tsv"
mkdir -p "$LOGS"
[ -s "$STATUS" ] || printf 'job\tstatus\texit\tseconds\tfinished\n' > "$STATUS"

export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
# LD_PRELOAD is a ROCm-only workaround and is not needed on CUDA; honoured if set.
# before running (see reproducibility/README.md); this script does not set a default.
[ -n "${LD_PRELOAD:-}" ] && export LD_PRELOAD

# Extra graph seeds for the qubit sweep and the accuracy-scale sweep. Seed 42 is
# the recorded one and keeps the default output paths; these write
# qubit_sweep_s<seed>.{json,csv} and accuracy_scale_s<seed>.*.
QUBIT_SEEDS=${QUBIT_SEEDS:-"7 123"}   # used by both qubit_seeds and frontier_seeds

JOBS=${JOBS:-"grad_quasi noise minabs frontier frontier_seeds wmax qubit qubit_seeds ablation stress figures kicked diffmode repeats"}

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/driver.log"; }

# --- preflight: fail now, not four hours in --------------------------------
cd "$HERE" || exit 1
"$PY" - <<'PYCHECK' || { echo "[FATAL] activate an env with padopauli installed first (see the repository README)"; exit 1; }
import sys, torch, padopauli
import padopauli.backend  # the compiled engine; fails loudly if the wheel does not match this Python/torch
assert torch.cuda.is_available(), "no GPU visible to torch"
print(f"[env] python={sys.version.split()[0]} torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
print(f"[env] padopauli={padopauli.__version__} from {padopauli.__file__}")
PYCHECK

done_already() { grep -qP "^$1\tok\t" "$STATUS" 2>/dev/null; }

# Record what produced each result file (host, GPU, ROCm/torch/PennyLane versions,
# git commit) in a <stem>.runmeta.json sidecar.
stamp_new_results() {
  local marker="$1"
  local files
  # The data root only: the code tree holds no results any more.
  mapfile -t files < <(find "$DATA_ROOT" -path "$LOGS" -prune -o \
                            -name '*.json' ! -name '*.runmeta.json' -newer "$marker" -print 2>/dev/null)
  [ ${#files[@]} -eq 0 ] && return 0
  ( cd "$HERE" && "$PY" _runmeta.py "${files[@]}" ) >>"$LOGS/runmeta.log" 2>&1 \
    || log "WARN  could not stamp run metadata (see $LOGS/runmeta.log)"
}

run_job() {
  local name="$1"; shift
  if [ "${FORCE:-0}" != "1" ] && done_already "$name"; then
    log "SKIP  $name (already ok; FORCE=1 to redo)"; return 0
  fi
  if [ "${DRYRUN:-0}" = "1" ]; then echo "  would run: $name -> $*"; return 0; fi
  log "START $name"
  local t0 rc marker
  t0=$(date +%s)
  # Anything a job writes is newer than this marker, which is how we find the
  # files to stamp without every script having to know about _runmeta.
  marker="$LOGS/.marker_$name"; : > "$marker"
  ( "$@" ) >"$LOGS/$name.log" 2>&1; rc=$?
  local secs=$(( $(date +%s) - t0 ))
  stamp_new_results "$marker"
  rm -f "$marker"
  printf '%s\t%s\t%d\t%d\t%s\n' "$name" "$([ $rc -eq 0 ] && echo ok || echo FAIL)" "$rc" "$secs" "$(date +%F_%H:%M:%S)" >> "$STATUS"
  if [ $rc -eq 0 ]; then log "DONE  $name (${secs}s)"; else log "FAIL  $name (rc=$rc, ${secs}s) -- see $LOGS/$name.log"; fi
  return 0   # a failed job never aborts the rest of the run
}

# --- job bodies ------------------------------------------------------------
j_grad_quasi() { cd "$HERE/scaling_accuracy_noise" && "$PY" sweep_grad_and_quasi.py; }
j_noise()      { cd "$HERE/scaling_accuracy_noise" && "$PY" sweep_noise.py; }
j_minabs()     { cd "$HERE/scaling_accuracy_noise" && "$PY" sweep_minabs_convergence.py; }
j_frontier()   { cd "$HERE/scaling_accuracy_noise" && "$PY" sweep_accuracy_frontier.py; }
j_wmax()       { cd "$HERE/scaling_accuracy_noise" && "$PY" sweep_wmax_fixed_circuit.py; }
j_qubit()      { cd "$HERE/scaling_accuracy_noise" && "$PY" sweep_qubits_fixed_wmax.py; }
j_figures()    { cd "$HERE/scaling_accuracy_noise" && "$PY" make_figures.py; }
j_ablation()   { cd "$HERE/zero_filter_ablation"   && "$PY" run_zero_filter_ablation.py; }
j_diffmode()   { cd "$HERE/diff_mode_vjp"          && "$PY" run_diff_mode_benchmark.py; }

j_qubit_seeds() {
  cd "$HERE/scaling_accuracy_noise" || return 1
  for s in $QUBIT_SEEDS; do
    echo "=== qubit sweep seed=$s ==="
    "$PY" sweep_qubits_fixed_wmax.py --seed "$s" || return 1
  done
}

# --scale-only runs just the across-n accuracy sweep (n = 20/24/28) and skips
# the fixed-16q frontier.
j_frontier_seeds() {
  cd "$HERE/scaling_accuracy_noise" || return 1
  for s in $QUBIT_SEEDS; do
    echo "=== accuracy scale seed=$s ==="
    "$PY" sweep_accuracy_frontier.py --seed "$s" --scale-only || return 1
  done
}

j_stress() {
  cd "$HERE/random_circuit_statevector" || return 1
  # term counts first: cheap and deterministic.
  "$PY" run_random_circuit_termcounts.py || return 1
  "$PY" run_random_circuit_sweep.py      || return 1
  "$PY" run_fwdbwd.py                    || return 1
}

j_kicked() {
  cd "$HERE/kicked_ising_127q" || return 1
  "$PY" verify_small.py    || return 1      # 9-qubit exact cross-check of the conventions
  PY="$PY" bash run_sweep.sh || return 1    # the 20-field 127q sweep
  "$PY" make_figure.py
}

# Longest job: the five-repeat timing protocol.
j_repeats() { cd "$HERE/repeat_timing" && PY="$PY" bash run_repeats.sh; }

# Notebook re-execution (tutorial 03/04 and examples/). Neither is in the default
# JOBS list: --inplace rewrites the .ipynb files. Opt in explicitly:
#   JOBS=tutorials bash run_rerun_all.sh
#   JOBS=examples  bash run_rerun_all.sh
run_notebooks() {           # run_notebooks <dir> <stem>...
  local dir="$1"; shift
  # These live OUTSIDE this bundle (the repo's shared tutorial/ and examples/)
  # and --inplace rewrites them. Refuse unless explicitly allowed.
  if [ "${ALLOW_SHARED_NOTEBOOKS:-0}" != "1" ]; then
    echo "[SKIP] $dir notebooks rewrite the shared ../$dir/*.ipynb in place."
    echo "       Re-run with ALLOW_SHARED_NOTEBOOKS=1 if that is what you want."
    return 1
  fi
  cd "$HERE/../$dir" || return 1
  # MPLBACKEND=Agg is right for the headless scripts above and wrong here: under
  # Agg the ipython kernel renders nothing inline, so --inplace rewrites every
  # notebook with its figures stripped out. Let ipykernel pick its inline backend.
  unset MPLBACKEND
  for t in "$@"; do
    echo "=== $dir/$t ==="
    "$PY" -m jupyter nbconvert --to notebook --inplace --execute \
        --ExecutePreprocessor.timeout=7200 "$t.ipynb" || return 1
  done
}

TUTORIALS=${TUTORIALS:-"03_training_with_compiled_program 04_embedding_batched_inputs_basics"}
j_tutorials() { run_notebooks tutorial $TUTORIALS; }

# examples/01 is the 127q sweep as a single process; it is the slow one here.
EXAMPLES=${EXAMPLES:-"01_kicked_ising_127q 02_vqe_h2_molecule 03_maxcut_qaoa 04_SAFE_ma-QAOA"}
j_examples() { run_notebooks examples $EXAMPLES; }

# --- go --------------------------------------------------------------------
log "run start | jobs: $JOBS"
for j in $JOBS; do
  if ! declare -F "j_$j" >/dev/null; then log "SKIP  $j (no such job)"; continue; fi
  run_job "$j" "j_$j"
done
log "run end"
echo
echo "=== status ==="; column -t -s$'\t' "$STATUS" 2>/dev/null || cat "$STATUS"
