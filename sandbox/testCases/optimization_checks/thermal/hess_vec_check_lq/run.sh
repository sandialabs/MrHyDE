#!/bin/bash
# Thermal linear-quadratic gradient + Hessian-vector diagnostic and full solves.
#
# Solve mode syntax:  <hv>-<gamma>[-N<nelem>]
#   hv    = exact | fd
#   gamma = 1e-3 | 1e-4 | 1e-5 | ...
#   nelem = number of elements per side (default 16); N<nelem> optional
#
# Env: NP (MPI ranks, default 4), MRHYDE_BIN
#
# Usage:
#   ./run.sh                                 # default: check + exact/fd mesh sweep at gamma=1e-4, N=4,8,16
#   ./run.sh check
#   ./run.sh exact-1e-4-N32                  # gamma=1e-4, N=32
#   ./run.sh exact-1e-4-N8 exact-1e-4-N16 exact-1e-4-N32    # custom mesh sweep
#   NP=8 ./run.sh exact-1e-4-N32             # override rank count

set -e
cd "$(dirname "$0")"
MRHYDE_BIN="${MRHYDE_BIN:-$(pwd)/mrhyde}"
NP="${NP:-4}"
NELEM_DEFAULT=16

# Default sweep: exact and fd hessian paths on N = 4, 8, 16 meshes at gamma = 1e-4.
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
      echo "=== LQ check: mode=${mode} ==="
      mpiexec -n "${NP}" "$MRHYDE_BIN" input_base.yaml >& "$logfile" \
        || { echo "  FAILED, see ${logfile}"; continue; }
      ;;
    exact-*|fd-*)
      hv="${mode%%-*}"                     # exact or fd
      rest="${mode#*-}"                    # e.g., 1e-4-N32 or 1e-4
      # Optional -N<nelem> suffix.
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
      echo "=== LQ solve: mode=${mode} (hessvec=${hv}, gamma=${gamma}, NX=${nelem}, np=${NP}) ==="
      t0=$(python3 -c 'import time; print(time.time())')
      ( cd "$run_dir" && mpiexec -n "${NP}" "$MRHYDE_BIN" input.yaml ) >& "$logfile" \
        || { echo "  FAILED, see ${logfile}"; continue; }
      t1=$(python3 -c 'import time; print(time.time())')
      python3 -c "print(f'{$t1 - $t0:.2f}')" > "$run_dir/wallclock.sec"
      ;;
    *) echo "unknown mode: $mode" >&2; exit 1 ;;
  esac
done

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
echo ""
echo "=== Summary ==="

have_check=0; have_solve=0
for m in $MODES; do
  [[ "$m" == "check" ]] && have_check=1 || have_solve=1
done

if [[ $have_check -eq 1 ]]; then
  echo ""
  echo "-- LQ operator checks (all should be at inner-solve floor for exact) --"
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
  echo "-- solves: TR-Newton convergence + native L2 error vs closed-form T* --"
  printf "  %-18s %-13s %-13s %-13s %-13s %-8s %-8s %-10s\n" \
    "mode" "iter0 value" "final value" "final gnorm" "||T-T*||_L2" "n_outer" "sum CG" "wall (s)"
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
    # Native L2 error printed by postproc report() at the post-solve forward.
    l2err=$(grep 'L2 norm of the error for T' "$log" | tail -1 | awk '{print $(NF-3)}')
    l2err="${l2err:-?}"
    wall=$(cat "runs/${mode}/wallclock.sec" 2>/dev/null || echo "?")
    printf "  %-18s %-13s %-13s %-13s %-13s %-8s %-8s %-10s\n" \
      "$mode" "$v0" "$vN" "$gN" "$l2err" "$nout" "$sumcg" "$wall"
  done
fi
