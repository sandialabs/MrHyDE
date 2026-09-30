#!/usr/bin/env python3

import sys
sys.path.append("../../scripts")
sys.path.append("../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import iterations, Results

its = mrhyde_test_support('''Split grouping in parallel: one group has to reproduce the flat sweep.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k Stokes,blocktriangular,nblock,splitgroups,regression

# Taylor-Hood with usePSPG: false leaves the pressure diagonal exactly zero, so 'pr' has to
# share a group with the velocity to pick up its Schur correction. One group is therefore the
# only grouping this problem admits, which is what makes it a clean identity gate; the
# Multiphysics/NavierStokes-CDR deck is where the groupings are actually compared.
res = Results()
status = enable_trilinos_debug()


def run(deck):
    log = "mrhyde_%s.log" % deck
    st = its.call("mpiexec -n 4 ../../mrhyde input_%s.yaml >& %s" % (deck, log))
    return st, iterations(open(log, errors="replace").read().splitlines())


status_flat, flat = run("flat")
status_one, one = run("onegroup")
status += status_flat + status_one

res.add(bool(flat) and flat == one, "one group reproduces the flat sweep on 4 ranks",
        "flat %s against grouped %s" % (flat, one))

its.call("mpiexec -n 4 ../../mrhyde input_missing.yaml >& mrhyde_missing.log",
         ignore_status=True)
want = "every split must appear exactly once"
hit = want in open("mrhyde_missing.log", errors="replace").read()
res.add(hit, "a split left out of the groups is rejected",
        "" if hit else "did not report: " + want)

sys.exit(status + res.write())
