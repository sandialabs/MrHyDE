#!/usr/bin/env python3

import re
import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check, Results

its = mrhyde_test_support('''Order-2 E and B on hexes: RefMaxwell with an order-2 nodal space, and p-multigrid with an order-1 RefMaxwell coarse solve.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k regression,maxwell,HCURL,HDIV,blocktriangular,schur,refmaxwell,pmultigrid,order2,iters

CASES = [
    ("refmaxwell_hgp_p2", 10, 16.8, 18),
    ("pmg_p2",            10, 11.7, 14),
]

res = Results()
status = enable_trilinos_debug()
logs = {}
for name, solves, mean, imax in CASES:
    log = "mrhyde_%s.log" % name
    status += its.call("mpiexec -n 4 ../../../../mrhyde input_%s.yaml >& %s" % (name, log))
    logs[name] = open(log, errors="replace").read()
    check(solves=solves, mean=mean, imax=imax, log=log, res=res, label="%s iterations" % name)

rels = [float(r) for r in re.findall(r"\[AUX\] p-multigrid P check: .* = (\S+)", logs["pmg_p2"])]
res.add(bool(rels) and max(rels) < 1e-12, "pmg_p2 prolongator",
        "rel %s" % (rels or "missing; no [AUX] P check line at verbosity >= 5"))

sys.exit(status + res.write())
