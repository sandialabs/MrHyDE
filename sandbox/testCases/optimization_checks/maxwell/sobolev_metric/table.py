#!/usr/bin/env python3
"""Prints one iteration-count table per refinement family from run.sh's logs.
A trailing * marks a run whose inner CG hit the cap instead of converging.
"""

import os
import re

here = os.path.dirname(os.path.abspath(__file__))

meta = {}
for line in open(os.path.join(here, "meshes.txt")):
    if line.strip() and not line.startswith("#"):
        m, nsp, nu, nx, nz = line.split()[:5]
        meta[m] = dict(nu=int(nu), nx=int(nx), nz=int(nz))

fam = {}
for line in open(os.path.join(here, "sweeps.txt")):
    if line.strip() and not line.startswith("#"):
        f, m, nt = line.split()[:3]
        fam.setdefault(f, []).append((m, int(nt)))


def itb(metric, mesh, nt):
    p = os.path.join(here, "log_%s_%s_nt%d.txt" % (metric, mesh, nt))
    if not os.path.exists(p):
        return "--"
    for line in open(p, errors="replace"):
        g = line.split()
        if len(g) == 10 and g[0] == "1" and re.fullmatch(r"\d+", g[8]):
            return g[8] if g[9] == "0" else g[8] + "*"
    return "--"


TITLE = {"dt": "Refine time  (2, 2, N_t)", "z": "Refine along z  (2, N_Z, 8)",
         "tx": "Refine transversely  (N_X, 2, 4)",
         "dtz": "Refine time on the z-refined mesh  (2, 12, N_t)"}
LABEL = {"dt": ("N_t", lambda m, nt: nt), "z": ("N_Z", lambda m, nt: meta[m]["nz"]),
         "tx": ("N_X", lambda m, nt: meta[m]["nx"]), "dtz": ("N_t", lambda m, nt: nt)}

for f in ("dt", "z", "tx", "dtz"):
    if f not in fam:
        continue
    name, lvl = LABEL[f]
    print("\n### %s\n" % TITLE[f])
    print("| %s | control DoFs/slab | Euclid | Sobolev (exact) | Sobolev (RefMaxwell) |" % name)
    print("|---|---|---|---|---|")
    for m, nt in fam[f]:
        print("| %d | %d | %s | %s | %s |" % (
            lvl(m, nt), meta[m]["nu"], itb("euclid", m, nt),
            itb("sobolev", m, nt), itb("sobolev_iter", m, nt)))

# RefMaxwell inner-tolerance table, if run_tol.sh was run.
tolpath = os.path.join(here, "tol.txt")
if os.path.exists(tolpath):
    tags = ["exact", "1e-14", "1e-10", "1e-8"]

    def tolcell(mesh, nt, tag):
        p = os.path.join(here, "log_tol_%s_nt%d_%s.txt" % (mesh, nt, tag))
        if not os.path.exists(p):
            return "--"
        for line in open(p, errors="replace"):
            g = line.split()
            if len(g) == 10 and g[0] == "1" and re.fullmatch(r"\d+", g[8]):
                return g[8]
        return "--"

    pts = [(c.split()[0], int(c.split()[1])) for c in open(tolpath)
           if c.strip() and not c.startswith("#")]
    if any(os.path.exists(os.path.join(here, "log_tol_%s_nt%d_exact.txt" % (m, nt))) for m, nt in pts):
        print("\n### RefMaxwell inner tolerance  (outer CG count)\n")
        print("| case (N_X, N_Z, N_t) | " + " | ".join(tags) + " |")
        print("|---" + "|---" * len(tags) + "|")
        for m, nt in pts:
            case = "(%d, %d, %d)" % (meta[m]["nx"], meta[m]["nz"], nt)
            print("| %s | %s |" % (case, " | ".join(tolcell(m, nt, t) for t in tags)))
