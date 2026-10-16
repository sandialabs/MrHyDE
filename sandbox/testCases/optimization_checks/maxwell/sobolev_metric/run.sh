#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
BIN=${MRHYDE_BIN:-./mrhyde}
SEED=${SEED:-11}
LIST=${1:-sweeps.txt}
grep -v '^#' "$LIST" | awk 'NF >= 2 && !seen[$(NF-1) " " $NF]++ {print $(NF-1), $NF}' |
while read -r mesh nt; do
  nsp=$(awk -v m="$mesh" '$1 == m {print $2}' meshes.txt)
  for metric in euclid sobolev sobolev_iter; do
    rm -f v.field.*.mm
    python3 mkctrl.py . "$nsp" "$nt" "$SEED" > /dev/null
    sed -e "s#__MESH__#meshes/mesh_${mesh}.yaml#" -e "s#__NSTEPS__#${nt}#" \
      "decks/input_${metric}.yaml" > _run.yaml
    log="log_${metric}_${mesh}_nt${nt}.txt"
    "$BIN" _run.yaml > "$log" 2>&1 || echo "exit $? in $log"
    echo "$metric $mesh nt=$nt: CG $(awk 'NF == 10 && $1 == "1" {print $9; exit}' "$log")"
  done
done
