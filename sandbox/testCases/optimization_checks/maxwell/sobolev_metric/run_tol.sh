#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
BIN=${MRHYDE_BIN:-./mrhyde}
SEED=${SEED:-11}
LIST=${1:-tol.txt}
grep -v '^#' "$LIST" | awk 'NF >= 2 && !seen[$(NF-1) " " $NF]++ {print $(NF-1), $NF}' |
while read -r mesh nt; do
  nsp=$(awk -v m="$mesh" '$1 == m {print $2}' meshes.txt)
  for run in sobolev:exact sobolev_iter:1e-14 sobolev_iter:1e-10 sobolev_iter:1e-8; do
    metric=${run%:*} tag=${run#*:}
    rm -f v.field.*.mm
    python3 mkctrl.py . "$nsp" "$nt" "$SEED" > /dev/null
    sed -e "s#__MESH__#meshes/mesh_${mesh}.yaml#" -e "s#__NSTEPS__#${nt}#" \
      -e "s#tolerance: 1.0e-14#tolerance: ${tag}#" "decks/input_${metric}.yaml" > _run.yaml
    log="log_tol_${mesh}_nt${nt}_${tag}.txt"
    "$BIN" _run.yaml > "$log" 2>&1 || echo "exit $? in $log"
    echo "$tag $mesh nt=$nt: CG $(awk 'NF == 10 && $1 == "1" {print $9; exit}' "$log")"
  done
done
