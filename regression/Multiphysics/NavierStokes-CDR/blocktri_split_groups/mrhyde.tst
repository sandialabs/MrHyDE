#!/usr/bin/env python3

import sys
sys.path.append("../../../scripts")
sys.path.append("../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import iterations, Results

its = mrhyde_test_support('''Block diagonal against pivot/target, per group of field splits.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 1
#TESTING -k regression,Navier-Stokes,CDR,coupled,blocktriangular,nblock,splitgroups

CAP = 2000

res = Results()
status = enable_trilinos_debug()
counts = {}
for name in ("chain", "pivot_target"):
    log = "mrhyde_%s.log" % name
    status += its.call("../../../mrhyde input_%s.yaml >& %s" % (name, log))
    counts[name] = iterations(open(log, errors="replace").read().splitlines())


def solves(name):
    return counts[name][1:] or counts[name]


chain, pivot = max(solves("chain")), max(solves("pivot_target"))
res.add(pivot <= chain and pivot < CAP, "a block-diagonal scalar costs nothing",
        "chain max %d, pivot/target max %d, cap %d" % (chain, pivot, CAP))

sys.exit(status + res.write())
