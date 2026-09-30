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

# Four fields in one set (ux, uy, pr, c), one split each, every split solved exactly by KLU.
# The iteration count therefore measures only how the splits are composed:
#
#   chain           no grouping, one elimination chain through all four splits
#   pivot_target    flow {ux, uy, pr} pivot/target, scalar {c} block diagonal
CAP = 2000   # 'max linear iters' in input_solver_common.yaml

res = Results()
status = enable_trilinos_debug()
counts = {}
for name in ("chain", "pivot_target"):
    log = "mrhyde_%s.log" % name
    status += its.call("../../../mrhyde input_%s.yaml >& %s" % (name, log))
    counts[name] = iterations(open(log, errors="replace").read().splitlines())


def solves(name):
    """The Newton solves, dropping the zero-residual projection a steady-state run leads
    with. iterations() only skips that one when there is a time loop to anchor on."""
    return counts[name][1:] or counts[name]


# Taking the scalar out of the elimination and leaving the saddle point as a pivot/target
# chain costs nothing. The continuity equation never sees 'c', so the correction the full
# chain adds for it is zero, and the shorter elimination leaves a slightly better sweep.
#
# Belos returns Unconverged here even when the recursive residual meets the tolerance, since
# its explicit residual check is stricter at this preconditioner quality. The gate is the
# iteration count, the same statistic every other iteration test uses.
chain, pivot = max(solves("chain")), max(solves("pivot_target"))
res.add(pivot <= chain and pivot < CAP, "a block-diagonal scalar costs nothing",
        "chain max %d, pivot/target max %d, cap %d" % (chain, pivot, CAP))

sys.exit(status + res.write())
