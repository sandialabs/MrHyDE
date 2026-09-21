#!/usr/bin/env python3

import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from parse_log import iterations, stats, Results

its = mrhyde_test_support('''Block-triangular, KLU on both blocks, at 1/2/4 ranks.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,direct,parallel,regression

RANKS = [1, 2, 4]
ITER_TOL = 1

res = Results()
status = 0
runs = {}
for n in RANKS:
    log = "mrhyde_np%d.log" % n
    status += its.call("mpiexec -n %d ../../../../mrhyde input.yaml >& %s" % (n, log))
    counts, s = iterations(open(log).read().splitlines()), stats(log)
    if not counts or s is None:
        res.add(False, "np=%d ran" % n, "no Belos solves in " + log)
        continue
    res.add(not s["unconv"], "np=%d converged" % n,
            "%d solves, %d unconverged" % (s["solves"], s["unconv"]))
    runs[n] = counts

ref = RANKS[0]
for n in RANKS[1:]:
    if n not in runs or ref not in runs:
        continue
    if len(runs[n]) != len(runs[ref]):
        res.add(False, "np=%d vs np=%d" % (n, ref),
                "%d solves against %d" % (len(runs[n]), len(runs[ref])))
        continue
    off = [i for i, (a, b) in enumerate(zip(runs[ref], runs[n])) if abs(a - b) > ITER_TOL]
    res.add(not off, "np=%d vs np=%d" % (n, ref),
            "solves %s differ by more than %d" % (off, ITER_TOL) if off
            else "%d solves agree within %d" % (len(runs[n]), ITER_TOL))

sys.exit(status + res.write())
