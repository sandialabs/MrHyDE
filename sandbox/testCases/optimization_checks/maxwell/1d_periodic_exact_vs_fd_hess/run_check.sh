#!/bin/bash

set -e
cd "$(dirname "$0")"

TAG="r1"
NSTEPS=50
NPROCS=11
MRHYDE="${MRHYDE_BIN:-./mrhyde}"
ROL="rol_decks/rol_check.yaml"

mkdir -p logs

INPUT="input_check.yaml"
LOG="logs/check.log"

sed -e "s|MESH_FILE_PLACEHOLDER|meshes/mesh_${TAG}.yaml|" \
    -e "s|ROL_FILE_PLACEHOLDER|${ROL}|" \
    -e "s|NSTEPS_PLACEHOLDER|${NSTEPS}|" \
    -e "s|OTHER_DECKS_PLACEHOLDER|other_decks_exact|g" \
    input_base.yaml > "$INPUT"

echo "=== check: input=${INPUT} ==="
mpiexec -n "$NPROCS" "$MRHYDE" "$INPUT" >& "$LOG" || {
  echo "  FAILED (exit $?), see ${LOG}"
}

rm -f "$INPUT"

echo ""
echo "=== Done ==="
grep -A 12 "MAGNITUDE-CHECK" "$LOG" | head -30
