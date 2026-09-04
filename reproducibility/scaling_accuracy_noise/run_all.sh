#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=${PY:-python}

echo "===== Experiment (1): w_max sweep, fixed 15Q ====="
"$PY" sweep_wmax_fixed_circuit.py

echo
echo "===== Experiment (2): qubit sweep, fixed w_max=3 ====="
"$PY" sweep_qubits_fixed_wmax.py

echo
echo "===== ALL EXPERIMENTS COMPLETE ====="
