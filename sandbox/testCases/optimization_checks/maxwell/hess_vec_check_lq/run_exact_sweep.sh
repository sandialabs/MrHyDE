#!/bin/bash
# Exact HessVec sweep (other_decks_exact has src_gate).

set -e
cd "$(dirname "$0")"

TAGS="r1"
# Two configs on the same input deck: baseline probes at ctrl=0 (degenerate),
# seed probes at a random ctrl (healthy).
MODES="baseline seed"
NSTEPS=10
MRHYDE_BIN="${MRHYDE_BIN:-/Users/abvoron/repos/ACEM/code/mrhyde/mrhyde.exe}"

np_for_tag() { case "$1" in r1) echo 11 ;; *) echo "unknown tag: $1" >&2; exit 1 ;; esac; }

base_for_mode() { echo "input_base.yaml"; }

rol_for_mode() {
  case "$1" in
    baseline) echo "rol_decks/rol_noscale_hess.yaml" ;;
    seed)     echo "rol_decks/rol_seed_hess.yaml" ;;
    *)        echo "unknown mode: $1" >&2; exit 1 ;;
  esac
}

mkdir -p logs
ln -sfn other_decks_exact other_decks

for mode in $MODES; do
  for tag in $TAGS; do
    runtag="${tag}_${mode}"
    logfile="mrhyde_hv_${runtag}_exact.log"
    inputfile="input_ex_${runtag}.yaml"
    rol_src=$(rol_for_mode "$mode")
    base_src=$(base_for_mode "$mode")

    sed -e "s|MESH_FILE_PLACEHOLDER|meshes/mesh_${tag}.yaml|" \
        -e "s|ROL_FILE_PLACEHOLDER|${rol_src}|" \
        -e "s|NSTEPS_PLACEHOLDER|${NSTEPS}|" \
        "${base_src}" > "$inputfile"

    echo "=== EXACT: tag=${tag}, mode=${mode}, base=${base_src} ==="
    np=$(np_for_tag "$tag")
    mpiexec -n "$np" "$MRHYDE_BIN" "$inputfile" >& "$logfile" || {
      echo "  FAILED (exit code $?), see ${logfile}"
      mv "$logfile" logs/ 2>/dev/null || true
      rm -f "$inputfile"
      continue
    }
    mv "$logfile" logs/

    rm -f "$inputfile"
  done
done

rm -f other_decks
echo ""
echo "=== All exact runs complete ==="
ls -la logs/mrhyde_hv_r*_exact.log 2>/dev/null
