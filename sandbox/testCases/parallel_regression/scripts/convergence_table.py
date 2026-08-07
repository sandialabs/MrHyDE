#!/usr/bin/env python3
"""Print convergence rows from runs/<partition>/output.log.

Reports:
  * Newton iteration count (max observed iteration index)
  * Final scaled nonlinear residual
  * EMPTY/NO_NEWTON status for incomplete logs
"""

import glob
import os
import re
import sys

ITER_RE = re.compile(r"^\*+ Iteration:\s+(\d+)")
SCALED_RE = re.compile(r"^\*+ Scaled Norm of nonlinear residual:\s+(\S+)")


def parse_one(log_path):
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return {"status": "EMPTY", "newton": None, "scaled": None}
    max_iter = -1
    last_scaled = None
    with open(log_path) as f:
        for line in f:
            m = ITER_RE.match(line)
            if m:
                max_iter = max(max_iter, int(m.group(1)))
                continue
            m = SCALED_RE.match(line)
            if m:
                last_scaled = m.group(1)
    if max_iter < 0:
        return {"status": "NO_NEWTON", "newton": None, "scaled": None}
    return {"status": "OK", "newton": max_iter, "scaled": last_scaled}


def main():
    runs_root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "runs")
    runs_root = os.path.abspath(runs_root)
    part_dirs = sorted(d for d in glob.glob(os.path.join(runs_root, "np*_*"))
                       if os.path.isdir(d))
    if not part_dirs:
        print(f"no run directories under {runs_root}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for pd in part_dirs:
        name = os.path.basename(pd)
        r = parse_one(os.path.join(pd, "output.log"))
        rows.append((name, r))

    hdr = f"{'partition':22s} | {'status':9s} | {'newton_iters':>12} | {'scaled_nonlin_res':>18}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in rows:
        newton = "-" if r["newton"] is None else f"{r['newton']:d}"
        scaled = "-" if r["scaled"] is None else r["scaled"]
        print(f"{name:22s} | {r['status']:9s} | {newton:>12} | {scaled:>18}")


if __name__ == "__main__":
    main()
