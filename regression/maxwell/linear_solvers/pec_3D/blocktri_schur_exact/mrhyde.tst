#!/usr/bin/env python3

import re
import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check, Results

its = mrhyde_test_support('''Exact Schur complement through the discrete curl on hexes and tetrahedra, the absolute-row-sum Schur weight on tetrahedra, and the RefMaxwell addon beta under exact-curl.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k regression,maxwell,HCURL,HDIV,blocktriangular,schur,schur_hcurl,refmaxwell,iters,tet,addon

CASES = [
    ("exact_curl_hex",       10,  4.1,  5),
    ("exact_curl_tet",       10, 11.5, 13),
    ("abs_pivot_diag_tet",   10, 16.5, 17),
    ("exact_curl_addon_hex", 10,  4.1,  5),
]

res = Results()
status = enable_trilinos_debug()
logs = {}
for name, solves, mean, imax in CASES:
    log = "mrhyde_%s.log" % name
    status += its.call("mpiexec -n 4 ../../../../mrhyde input_%s.yaml >& %s" % (name, log))
    logs[name] = open(log, errors="replace").read()
    check(solves=solves, mean=mean, imax=imax, log=log, res=res, label="%s iterations" % name)

for name in ("exact_curl_hex", "exact_curl_tet"):
    rels = [float(r) for r in re.findall(r"\[BlockTri\] exact-curl: s = \S+ rel = (\S+)", logs[name])]
    res.add(bool(rels) and max(rels) < 1e-12, "%s curl identity" % name,
            "rel %s" % (rels or "missing; no [BlockTri] exact-curl line at verbosity >= 5"))

used = [int(n) for n in re.findall(r"used_abs=(\d+)", logs["abs_pivot_diag_tet"])]
res.add(bool(used) and min(used) > 0, "absolute row sums used",
        "used_abs %s" % (used or "missing; no Schur weight line at verbosity >= 5"))

seen = [float(b) for b in re.findall(r"\[ADDON\] beta = ([-0-9.eE+]+)", logs["exact_curl_addon_hex"])]
res.add(bool(seen) and all(abs(b - 0.1) < 1.0e-10 for b in seen), "exact_curl_addon_hex beta",
        "%s, expected 0.1" % (seen or "missing; no [ADDON] line at verbosity >= 5"))

sys.exit(status + res.write())
