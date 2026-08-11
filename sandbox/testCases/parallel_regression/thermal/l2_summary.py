#!/usr/bin/env python3
"""Wide summary for thermal parallel-regression runs.

Layout:  <root>/<mesh>/T<order>/<partition>/output.log
Columns: partition | T<n> iters | T<n> L2

Usage:
  ./l2_summary.py                # defaults to ./runs
  ./l2_summary.py <runs_dir>
"""

import glob
import os
import re
import sys

if len(sys.argv) > 1:
    ROOT = os.path.abspath(sys.argv[1])
else:
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, "runs"))

MESH_TYPES = ("tet", "hex")
L2_T_RE = re.compile(r"L2 norm of the error for T\s*=\s*([-+0-9.eE]+)")
ITER_RE = re.compile(r"^\*+ Iteration:\s+(\d+)")


def parse(log_path):
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return None, None
    l2 = None
    max_iter = -1
    with open(log_path) as f:
        for line in f:
            m = ITER_RE.match(line)
            if m:
                v = int(m.group(1))
                if v > max_iter:
                    max_iter = v
                continue
            if l2 is None:
                m = L2_T_RE.search(line)
                if m:
                    try:
                        l2 = float(m.group(1))
                    except ValueError:
                        pass
    return l2, (None if max_iter < 0 else max_iter)


def orders(root, mesh):
    d = os.path.join(root, mesh)
    if not os.path.isdir(d):
        return []
    xs = [x for x in os.listdir(d)
          if re.match(r"^T\d+$", x) and os.path.isdir(os.path.join(d, x))]
    xs.sort(key=lambda t: int(t[1:]))
    return xs


def partitions(root, mesh, orders):
    parts = set()
    for o in orders:
        for p in glob.glob(os.path.join(root, mesh, o, "np*_*")):
            parts.add(os.path.basename(p))
    return sorted(parts)


print(f"# tree: {ROOT}")

for mesh in MESH_TYPES:
    T_orders = orders(ROOT, mesh)
    if not T_orders:
        continue
    parts = partitions(ROOT, mesh, T_orders)
    if not parts:
        continue
    iter_cols = " | ".join(f"{o+' iters':>8}" for o in T_orders)
    l2_cols = " | ".join(f"{o+' L2':>13} " for o in T_orders)
    hdr = f"{'partition':22s} | {iter_cols} | {l2_cols}"
    print()
    print("=" * len(hdr))
    print(f"# {mesh}")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for part in parts:
        iters_vals, l2_vals = [], []
        for o in T_orders:
            l2, it = parse(os.path.join(ROOT, mesh, o, part, "output.log"))
            iters_vals.append("-" if it is None else f"{it:d}")
            l2_vals.append("-" if l2 is None else f"{l2:.6e}")
        iter_str = " | ".join(f"{v:>8}" for v in iters_vals)
        l2_str = " | ".join(f"{v:>13} " for v in l2_vals)
        print(f"{part:22s} | {iter_str} | {l2_str}")
