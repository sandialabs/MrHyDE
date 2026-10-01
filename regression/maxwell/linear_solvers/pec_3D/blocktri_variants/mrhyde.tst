#!/usr/bin/env python3

import re
import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check, Results

its = mrhyde_test_support('''Schur variants and per-split routing on one 2x2 Maxwell system.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k regression,maxwell,HCURL,HDIV,blocktriangular,schur,schur_hcurl,schur_hdiv,onelevel,refmaxwell,iters,routing

# name, then the solves/mean/max each variant was recorded at.
CASES = [
    ("schur_base",          10, 38.5, 43),   # S = J11
    ("schur_diag",          10, 14.5, 15),   # S = J11 - J10 lumpdiag(J00)^-1 J01
    ("triangle_lower",      10, 14.9, 16),   # same as schur_diag, lower triangle
    ("refmaxwell_on_pivot", 10, 39.0, 45),   # RefMaxwell on split 0, HDIV target
]

res = Results()
status = enable_trilinos_debug()
logs = {}
for name, solves, mean, imax in CASES:
    log = "mrhyde_%s.log" % name
    status += its.call("mpiexec -n 4 ../../../../mrhyde input_%s.yaml >& %s" % (name, log))
    logs[name] = open(log, errors="replace").read()
    check(solves=solves, mean=mean, imax=imax, log=log, res=res, label="%s iterations" % name)

picked = re.findall(r"\[BlockTri\] triangle = (\w+)", logs["triangle_lower"])
res.add(bool(picked) and all(p == "lower" for p in picked), "lower triangle selected",
        "picked %s" % (picked or "no [BlockTri] triangle line at verbosity >= 5"))

# RefMaxwell has to attach to the split the deck names, not to the Schur target.
built = re.findall(r"\[RefMaxwell\] Built new preconditioner hierarchy \(split (\d+)\)",
                   logs["refmaxwell_on_pivot"])
res.add(bool(built) and all(r == "0" for r in built), "RefMaxwell built on the pivot split",
        "splits built: %s" % (built or "no hierarchy reported"))

sys.exit(status + res.write())
