#!/usr/bin/env python3
"""Wide summary for porous_mixed parallel-regression runs.

Layout:  <root>/<mesh>/<partition>/output.log      (no order subdir)
Columns: partition | Newton iters | p L2 | u L2

Usage:
  ./l2_summary.py                # defaults to ./runs
  ./l2_summary.py <runs_dir>
"""

import os
import re
import sys

if len(sys.argv) > 1:
    ROOT = os.path.abspath(sys.argv[1])
else:
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, "runs"))

MESH_TYPES = ("tet", "hex")
L2_FIELD_RE = re.compile(r"L2 norm of the error for\s+(\S+)\s*=\s*([-+0-9.eE]+)")
ITER_RE = re.compile(r"^\*+ Iteration:\s+(\d+)")


def parse(log_path):
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return None, None, None
    fields = {}
    max_iter = -1
    with open(log_path) as f:
        for line in f:
            m = ITER_RE.match(line)
            if m:
                v = int(m.group(1))
                if v > max_iter:
                    max_iter = v
                continue
            m = L2_FIELD_RE.search(line)
            if m and m.group(1) not in fields:
                try:
                    fields[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    iters = None if max_iter < 0 else max_iter
    return fields.get("p"), fields.get("u"), iters


def partitions(root, mesh):
    d = os.path.join(root, mesh)
    if not os.path.isdir(d):
        return []
    return sorted(x for x in os.listdir(d)
                  if re.match(r"^np\d+_", x) and os.path.isdir(os.path.join(d, x)))


print(f"# tree: {ROOT}")

for mesh in MESH_TYPES:
    parts = partitions(ROOT, mesh)
    if not parts:
        continue
    hdr = f"{'partition':22s} | {'iters':>8} | {'p L2':>13}  | {'u L2':>13} "
    print()
    print("=" * len(hdr))
    print(f"# {mesh}")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for part in parts:
        p_l2, u_l2, it = parse(os.path.join(ROOT, mesh, part, "output.log"))
        iters_str = "-" if it is None else f"{it:d}"
        p_str = "-" if p_l2 is None else f"{p_l2:.6e}"
        u_str = "-" if u_l2 is None else f"{u_l2:.6e}"
        print(f"{part:22s} | {iters_str:>8} | {p_str:>13}  | {u_str:>13} ")
