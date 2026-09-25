#!/usr/bin/env python3
"""Wide summary for maxwell_hcurl parallel-regression runs.

Layout:  <root>/<mesh>/E<order>/<partition>/output.log
Columns: partition | E L2 E<n> | B L2 E<n>   (at final observed time step)
Plus a 'spread' row per order = max-min across partitions.

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
L2_FIELD_TIME_RE = re.compile(
    r"L2 norm of the error for\s+(\S+)\s*=\s*([-+0-9.eE]+)\s*\(time\s*=\s*([-+0-9.eE]+)\)")


def parse(log_path):
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return None, None
    E_by_t, B_by_t = {}, {}
    with open(log_path) as f:
        for line in f:
            m = L2_FIELD_TIME_RE.search(line)
            if not m:
                continue
            try:
                var, val, t = m.group(1), float(m.group(2)), float(m.group(3))
            except ValueError:
                continue
            if var == "E":
                E_by_t[t] = val
            elif var == "B":
                B_by_t[t] = val
    return (E_by_t[max(E_by_t)] if E_by_t else None,
            B_by_t[max(B_by_t)] if B_by_t else None)


def orders(root, mesh):
    d = os.path.join(root, mesh)
    if not os.path.isdir(d):
        return []
    xs = [x for x in os.listdir(d)
          if re.match(r"^E\d+$", x) and os.path.isdir(os.path.join(d, x))]
    xs.sort(key=lambda t: int(t[1:]))
    return xs


def partitions(root, mesh, orders):
    parts = set()
    for o in orders:
        for p in glob.glob(os.path.join(root, mesh, o, "np*_*")):
            parts.add(os.path.basename(p))
    return sorted(parts)


def spread(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f"{max(vals) - min(vals):.3e}"


print(f"# tree: {ROOT}")

for mesh in MESH_TYPES:
    E_orders = orders(ROOT, mesh)
    if not E_orders:
        continue
    parts = partitions(ROOT, mesh, E_orders)
    if not parts:
        continue
    E_cols = " | ".join(f"{'E L2 ' + o:>15}" for o in E_orders)
    B_cols = " | ".join(f"{'B L2 ' + o:>15}" for o in E_orders)
    hdr = f"{'partition':22s} | {E_cols} | {B_cols}"
    print()
    print("=" * len(hdr))
    print(f"# {mesh}")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for part in parts:
        E_vals, B_vals = [], []
        for o in E_orders:
            E_l2, B_l2 = parse(os.path.join(ROOT, mesh, o, part, "output.log"))
            E_vals.append("-" if E_l2 is None else f"{E_l2:.6e}")
            B_vals.append("-" if B_l2 is None else f"{B_l2:.6e}")
        E_str = " | ".join(f"{v:>15}" for v in E_vals)
        B_str = " | ".join(f"{v:>15}" for v in B_vals)
        print(f"{part:22s} | {E_str} | {B_str}")

    print()
    for o in E_orders:
        E_all = [parse(os.path.join(ROOT, mesh, o, p, "output.log"))[0] for p in parts]
        B_all = [parse(os.path.join(ROOT, mesh, o, p, "output.log"))[1] for p in parts]
        print(f"spread {o:6s}   E L2: {spread(E_all):>10}   B L2: {spread(B_all):>10}")
