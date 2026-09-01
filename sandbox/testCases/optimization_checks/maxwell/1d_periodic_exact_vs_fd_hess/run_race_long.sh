#!/bin/bash
#
# Usage:  ./run_race_long.sh --tag {r1|r2} --label {exact|fd}

set -e
cd "$(dirname "$0")"

TAG=""
LABEL=""
NSTEPS=200
NPROCS=8
MRHYDE="./mrhyde"
ROL="rol_decks/rol_race.yaml"

usage() {
  echo "Usage: $0 --tag {r1|r2} --label {exact|fd}" >&2
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)   TAG="$2";   shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

case "$TAG"   in r1|r2) ;; *) echo "bad --tag: '$TAG'" >&2; usage ;; esac
case "$LABEL" in exact|fd) ;; *) echo "bad --label: '$LABEL'" >&2; usage ;; esac

mkdir -p logs

RUNTAG="${TAG}_${LABEL}"
INPUT="input_${RUNTAG}.yaml"
LOG="logs/mrhyde_${RUNTAG}.log"

sed -e "s|MESH_FILE_PLACEHOLDER|meshes/mesh_${TAG}.yaml|" \
    -e "s|ROL_FILE_PLACEHOLDER|${ROL}|" \
    -e "s|NSTEPS_PLACEHOLDER|${NSTEPS}|" \
    -e "s|OTHER_DECKS_PLACEHOLDER|other_decks_${LABEL}|g" \
    input_base.yaml > "$INPUT"

echo "=== ${RUNTAG}: input=${INPUT}, log=${LOG}, nprocs=${NPROCS} ==="
time mpiexec -n "$NPROCS" "$MRHYDE" "$INPUT" >& "$LOG" || {
  rc=$?
  echo "  FAILED (exit ${rc}), see ${LOG}" >&2
  rm -f "$INPUT"
  exit "$rc"
}

rm -f "$INPUT"
echo "=== ${RUNTAG}: done ==="
