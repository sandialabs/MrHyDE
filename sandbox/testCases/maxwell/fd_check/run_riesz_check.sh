#!/bin/bash

set -e
cd "$(dirname "$0")"

TAGS="r1 r2 r3"
MODES="noscale seed w1 seed_w1 noscale_regs0 seed_regs0 seed_w1_regs0 w1_regs1 seed_w1_regs1"
NSTEPS=10
MRHYDE="mrhyde"

np_for_tag() {
  case "$1" in
    r1) echo 11 ;;
    r2) echo 11 ;;
    r3) echo 11 ;;
    *)  echo "unknown tag: $1" >&2; exit 1 ;;
  esac
}

base_for_mode() {
  case "$1" in
    noscale|seed)              echo "input_base.yaml" ;;
    w1|seed_w1)                echo "input_base_w1.yaml" ;;
    noscale_regs0|seed_regs0)  echo "input_base_regs0.yaml" ;;
    seed_w1_regs0)             echo "input_base_w1_regs0.yaml" ;;
    w1_regs1|seed_w1_regs1)    echo "input_base_w1_regs1.yaml" ;;
    *)                         echo "unknown mode: $1" >&2; exit 1 ;;
  esac
}

rol_for_mode() {
  case "$1" in
    noscale|w1|noscale_regs0|w1_regs1)     echo "rol_noscale.yaml" ;;
    seed|seed_w1|seed_regs0|seed_w1_regs0|seed_w1_regs1) echo "rol_seed.yaml" ;;
    *)                                     echo "unknown mode: $1" >&2; exit 1 ;;
  esac
}

for mode in $MODES; do
  for tag in $TAGS; do
    runtag="${tag}_${mode}"
    logfile="mrhyde_${runtag}.log"
    inputfile="input_${runtag}.yaml"
    rol_src=$(rol_for_mode "$mode")
    base_src=$(base_for_mode "$mode")

    sed -e "s|MESH_FILE_PLACEHOLDER|meshes/mesh_${tag}.yaml|" \
        -e "s|ROL_FILE_PLACEHOLDER|rol_decks/${rol_src}|" \
        -e "s|NSTEPS_PLACEHOLDER|${NSTEPS}|" \
        "${base_src}" > "$inputfile"

    echo "=== Running tag=${tag}, mode=${mode}, base=${base_src}, rol=${rol_src}, nsteps=${NSTEPS} ==="
    np=$(np_for_tag "$tag")
    mpiexec -n "$np" "$MRHYDE" "$inputfile" >& "$logfile" || {
      echo "  FAILED (exit code $?), see ${logfile}"
      rm -f "$inputfile"
      continue
    }
    mv "$logfile" logs/

    rm -f "$inputfile"
  done
done

echo ""
echo "=== All runs complete ==="
ls -la logs/mrhyde_r*.log 2>/dev/null
