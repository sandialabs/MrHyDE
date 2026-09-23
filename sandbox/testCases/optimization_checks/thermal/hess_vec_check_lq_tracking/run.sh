#!/bin/bash
# Thermal LQ tracking checks and solves.
# Mode: <hv>-<gamma>[-N<nelem>] with hv in {exact,fd}.
# Env: NP (default 4), MRHYDE_BIN.
# Usage: ./run.sh | ./run.sh check | ./run.sh exact-1e-4-N32

set -e
cd "$(dirname "$0")"
MRHYDE_BIN="${MRHYDE_BIN:-$(pwd)/mrhyde}"
NP="${NP:-4}"
NELEM_DEFAULT=16

DEFAULT_MODES="check \
  exact-1e-4-N4 fd-1e-4-N4 \
  exact-1e-4-N8 fd-1e-4-N8 \
  exact-1e-4-N16 fd-1e-4-N16"
MODES="${*:-$DEFAULT_MODES}"
mkdir -p logs

for mode in $MODES; do
  logfile="logs/mrhyde_${mode}.log"
  case "$mode" in
    check)
      echo "=== LQ tracking check: mode=${mode} ==="
      mpiexec -n "${NP}" "$MRHYDE_BIN" input_base.yaml >& "$logfile" \
        || { echo "  FAILED, see ${logfile}"; continue; }
      ;;
    exact-*|fd-*)
      hv="${mode%%-*}"
      rest="${mode#*-}"
      if [[ "$rest" =~ ^(.+)-N([0-9]+)$ ]]; then
        gamma="${BASH_REMATCH[1]}"
        nelem="${BASH_REMATCH[2]}"
      else
        gamma="$rest"
        nelem="$NELEM_DEFAULT"
      fi
      run_dir="runs/${mode}"
      rm -rf "$run_dir"
      mkdir -p "$run_dir"
      sed -e "s|OTHER_DECKS_DIR|../../other_decks/${hv}|" \
          -e "s|GAMMA|${gamma}|g" \
          -e "s|NELEM|${nelem}|g" \
          input_solve.yaml.template > "$run_dir/input.yaml"
      echo "=== LQ tracking solve: mode=${mode} (hessvec=${hv}, gamma=${gamma}, NX=${nelem}, np=${NP}) ==="
      t0=$(python3 -c 'import time; print(time.time())')
      ( cd "$run_dir" && mpiexec -n "${NP}" "$MRHYDE_BIN" input.yaml ) >& "$logfile" \
        || { echo "  FAILED, see ${logfile}"; continue; }
      t1=$(python3 -c 'import time; print(time.time())')
      python3 -c "print(f'{$t1 - $t0:.2f}')" > "$run_dir/wallclock.sec"
      ;;
    *) echo "unknown mode: $mode" >&2; exit 1 ;;
  esac
done

echo ""
echo "=== Summary ==="

have_check=0; have_solve=0
for m in $MODES; do
  [[ "$m" == "check" ]] && have_check=1 || have_solve=1
done

if [[ $have_check -eq 1 ]]; then
  echo ""
  echo "-- LQ tracking operator checks --"
  printf "  %-16s %-16s %-16s %-16s\n" "GRAD-CHECK" "HESSVEC-CHECK" "SECANT-IDENTITY" "HV-BILINEARITY"
  log="logs/mrhyde_check.log"
  if [[ -f "$log" ]]; then
    grad=$(grep -oE '\[GRAD-CHECK\] best rel_err = [0-9.eE+-]+' "$log" | tail -1 | awk '{print $NF}')
    hv=$(grep -oE '\[HESSVEC-CHECK\] best rel_err = [0-9.eE+-]+' "$log" | tail -1 | awk '{print $NF}')
    sec=$(grep '\[SECANT-IDENTITY\]' "$log" | tail -1 | sed -E 's/.*relative = ([0-9.eE+-]+).*/\1/')
    bil=$(grep '\[HV-BILINEARITY\]' "$log" | tail -1 | sed -E 's/.*relative = ([0-9.eE+-]+).*/\1/')
    printf "  %-16s %-16s %-16s %-16s\n" "${grad:-?}" "${hv:-?}" "${sec:-?}" "${bil:-?}"
  fi
fi

if [[ $have_solve -eq 1 ]]; then
  echo ""
  echo "-- solves: TR-Newton convergence --"
  printf "  %-18s %-13s %-13s %-13s %-8s %-8s %-10s\n" \
    "mode" "iter0 value" "final value" "final gnorm" "n_outer" "sum CG" "wall (s)"
  for mode in $MODES; do
    [[ "$mode" == "check" ]] && continue
    log="logs/mrhyde_${mode}.log"
    [[ -f "$log" ]] || continue
    iter0=$(awk '/^  0 / && NF>=6 {print; exit}' "$log")
    final=$(awk '/^  [1-9][0-9]* / && NF>=6 {last=$0} END{print last}' "$log")
    v0=$(echo "$iter0" | awk '{print $2}')
    vN=$(echo "$final" | awk '{print $2}')
    gN=$(echo "$final" | awk '{print $3}')
    nout=$(echo "$final" | awk '{print $1}')
    sumcg=$(awk '/^  [0-9]+ / && NF>=9 && $9 ~ /^[0-9]+$/ {s+=$9} END{print s+0}' "$log")
    wall=$(cat "runs/${mode}/wallclock.sec" 2>/dev/null || echo "?")
    printf "  %-18s %-13s %-13s %-13s %-8s %-8s %-10s\n" \
      "$mode" "$v0" "$vN" "$gN" "$nout" "$sumcg" "$wall"
  done
fi
