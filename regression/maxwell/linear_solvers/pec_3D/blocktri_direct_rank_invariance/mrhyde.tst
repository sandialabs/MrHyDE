#!/usr/bin/env python3

import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from parse_log import iterations, stats

its = mrhyde_test_support('''Block-triangular, KLU on both blocks, at 1/2/4 ranks.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,direct,parallel,regression

RANKS = [1, 2, 4]
ITER_TOL = 1

status = 0
runs = {}
for n in RANKS:
    log = "mrhyde_np%d.log" % n
    status += its.call("mpiexec -n %d ../../../../mrhyde input.yaml >& %s" % (n, log))
    counts = iterations(open(log).read().splitlines())
    s = stats(log)
    if not counts or s is None:
        print("Failure: %s has no Belos solves. Is verbosity 10 set?" % log)
        status += 1
        continue
    if s["unconv"]:
        print("Failure: np=%d had %d of %d solves hit the iteration limit"
              % (n, s["unconv"], s["solves"]))
        status += 1
    runs[n] = counts
    print("np=%d  %s" % (n, counts))

if len(runs) == len(RANKS):
    ref_n = RANKS[0]
    ref = runs[ref_n]
    for n in RANKS[1:]:
        cur = runs[n]
        if len(cur) != len(ref):
            print("Failure: np=%d ran %d solves, np=%d ran %d"
                  % (n, len(cur), ref_n, len(ref)))
            status += 1
            continue
        for i, (a, b) in enumerate(zip(ref, cur)):
            if abs(a - b) > ITER_TOL:
                print("Failure: solve %d took %d iters at np=%d and %d at np=%d (tol %d)"
                      % (i, b, n, a, ref_n, ITER_TOL))
                status += 1

sys.exit(status)
