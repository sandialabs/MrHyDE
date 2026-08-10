#!/usr/bin/env python3
"""Wide summary: one table per mesh type; columns are partition,
iters T1..Tn, L2 T1..Tn."""

import glob
import os
import re
import sys

if len(sys.argv) > 1:
    ROOT = sys.argv[1]
else:
    # default: sibling ../thermal/runs_orig relative to this script
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, "..", "thermal", "runs"))
MESH_TYPES = ("tet", "hex")

L2_RE = re.compile(r"L2 norm of the error for T\s*=\s*([-+0-9.eE]+)")
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
                m = L2_RE.search(line)
                if m:
                    try:
                        l2 = float(m.group(1))
                    except ValueError:
                        pass
    iters = None if max_iter < 0 else max_iter
    return l2, iters


def orders_present(root, mesh):
    d = os.path.join(root, mesh)
    if not os.path.isdir(d):
        return []
    orders = [x for x in os.listdir(d)
              if re.match(r"^T\d+$", x) and os.path.isdir(os.path.join(d, x))]
    orders.sort(key=lambda t: int(t[1:]))
    return orders


def all_partitions(root, mesh, orders):
    parts = set()
    for o in orders:
        for p in glob.glob(os.path.join(root, mesh, o, "np*_*")):
            parts.add(os.path.basename(p))
    return sorted(parts)


print(f"# tree: {ROOT}")

for mesh in MESH_TYPES:
    orders = orders_present(ROOT, mesh)
    if not orders:
        continue
    parts = all_partitions(ROOT, mesh, orders)
    if not parts:
        continue

    iter_cols = " | ".join(f"{o+' iters':>8}" for o in orders)
    l2_cols = " | ".join(f"{o+' L2':>13} " for o in orders)
    hdr = f"{'partition':22s} | {iter_cols} | {l2_cols}"

    print()
    print("=" * len(hdr))
    print(f"# {mesh}")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for part in parts:
        iters_vals = []
        l2_vals = []
        for o in orders:
            l2, it = parse(os.path.join(ROOT, mesh, o, part, "output.log"))
            iters_vals.append("-" if it is None else f"{it:d}")
            l2_vals.append("-" if l2 is None else f"{l2:.6e}")
        iter_str = " | ".join(f"{v:>8}" for v in iters_vals)
        l2_str = " | ".join(f"{v:>13} " for v in l2_vals)
        print(f"{part:22s} | {iter_str} | {l2_str}")
