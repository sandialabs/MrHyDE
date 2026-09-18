#!/bin/bash

set -euo pipefail
cd "$(dirname "$0")"

MRHYDE_BIN="${MRHYDE_BIN:-$(pwd)/mrhyde}"
NP="${NP:-11}"

if [[ ! -x "$MRHYDE_BIN" ]]; then
  echo "ERROR: MRHYDE_BIN is not executable: $MRHYDE_BIN" >&2
  exit 1
fi

mkdir -p logs

for cfg in "input_base.yaml scan" "input_scale0.yaml scan_scale_0"; do
  set -- $cfg
  input="$1"; tag="$2"
  logfile="logs/${tag}.log"
  echo "=== Running ${input} -> ${logfile} (n=${NP}) ==="
  mpiexec -n "$NP" "$MRHYDE_BIN" "$input" >& "$logfile"
done

echo ""
for tag in scan scan_scale_0; do
  echo "--- logs/${tag}.log ---"
  grep -n -A 20 -F "[MAGNITUDE-SCAN]" "logs/${tag}.log" || echo "NO SCAN OUTPUT FOUND"
  echo ""
done
