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

CASES = [
    ("schur_base",          10, 38.5, 43),
    ("schur_diag",          10, 14.5, 15),
    ("triangle_lower",      10, 14.9, 16),
    ("refmaxwell_on_pivot", 10, 39.0, 45),
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

built = re.findall(r"\[RefMaxwell\] Built new preconditioner hierarchy \(split (\d+)\)",
                   logs["refmaxwell_on_pivot"])
res.add(bool(built) and all(r == "0" for r in built), "RefMaxwell built on the pivot split",
        "splits built: %s" % (built or "no hierarchy reported"))

sys.exit(status + res.write())
