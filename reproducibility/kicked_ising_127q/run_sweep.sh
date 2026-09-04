#!/bin/bash
# Run the 20-field kicked-Ising sweep ONE FIELD PER PROCESS. A separate process
# per field means an OOM on a hard field cannot abort the rest. Ends-inward order
# so the curve shape appears early. The structural max_weight cap (the only
# term-count bound PADO-Pauli exposes) keeps memory bounded; build_min_abs is the
# coefficient threshold.
cd "$(dirname "$0")" || exit 1

# Uses `python` on PATH; override with PY=/path/to/python.
PY=${PY:-python}
MAX_WEIGHT=${MAX_WEIGHT:-8}
MIN_ABS=${MIN_ABS:-1e-4}
ORDER="0 19 1 18 2 17 3 16 4 15 5 14 6 13 7 12 8 11 9 10"

echo "=== SWEEP START  max_weight=$MAX_WEIGHT  min_abs=$MIN_ABS  $(date +%H:%M:%S) ==="
for i in $ORDER; do
  echo "=== FIELD i=$i  $(date +%H:%M:%S) ==="
  TQDM_DISABLE=1 "$PY" run_kicked_ising.py \
      --indices "$i" --preset gpu --max-weight "$MAX_WEIGHT" --min-abs "$MIN_ABS" --tag full \
      2>&1 | grep -aE "<Z_|OutOfMemory|MemoryError|Traceback|terms=" \
           | grep -avE "propagate:|it/s"
done
echo "=== SWEEP DONE  $(date +%H:%M:%S) ==="
