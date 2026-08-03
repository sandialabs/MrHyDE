#!/bin/bash
# Usage:
#   ./run_thunderbird.sh --label {exact|fd} --solver {tr|lbfgs}

source ../../../../../../scripts/load-env.sh
set -e
cd "$(dirname "$0")"

LABEL=""
SOLVER=""
NPROCS=44
MRHYDE="./mrhyde"

usage() {
  echo "Usage: $0 --label {exact|fd} --solver {tr|lbfgs}" >&2
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --label)  LABEL="$2";  shift 2 ;;
    --solver) SOLVER="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

case "$LABEL"  in exact|fd) ;; *) echo "bad --label: '$LABEL'"   >&2; usage ;; esac
case "$SOLVER" in tr|lbfgs) ;; *) echo "bad --solver: '$SOLVER'" >&2; usage ;; esac

mkdir -p logs

RUNTAG="${LABEL}_${SOLVER}"
INPUT="input_${RUNTAG}.yaml"
LOG="logs/mrhyde_${RUNTAG}.log"
ROL="rol_decks/rol_${SOLVER}.yaml"

sed -e "s|OTHER_DECKS_PLACEHOLDER|other_decks_${LABEL}|g" \
    -e "s|ROL_FILE_PLACEHOLDER|${ROL}|" \
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
