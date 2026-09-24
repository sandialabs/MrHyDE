#!/usr/bin/env python3

import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import stats, Results

its = mrhyde_test_support('''RefMaxwell iteration counts stay near flat under h refinement.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,refmaxwell,parallel,regression

# 3x the elements per step. An h-independent preconditioner holds max iterations
# nearly flat here; SA-AMG on the same Schur block roughly doubles over the same
# sequence, so the gate separates the two.
MESHES = ["N8x8x4", "N16x8x4", "N24x12x6"]
GROWTH = 1.6

res = Results()
status = enable_trilinos_debug()
imax = {}
for mesh in MESHES:
    log = "mrhyde_%s.log" % mesh
    status += its.call("mpiexec -n 4 ../../../../mrhyde input_%s.yaml >& %s" % (mesh, log))
    s = stats(log)
    if s is None:
        res.add(False, "%s ran" % mesh, "no Belos solves in " + log)
        continue
    res.add(s["solves"] == 10 and not s["unconv"], "%s converged" % mesh,
            "%d solves, %d unconverged" % (s["solves"], s["unconv"]))
    imax[mesh] = s["max"]

if len(imax) == len(MESHES):
    seq = [imax[m] for m in MESHES]
    grew = seq[-1] / float(seq[0])
    res.add(grew <= GROWTH, "h independence",
            "max iters %s, coarsest to finest x%.2f (gate x%.1f)" % (seq, grew, GROWTH))
    rising = [i for i in range(1, len(seq)) if seq[i] < seq[i-1] - 1]
    res.add(not rising, "monotone in h",
            "max iters drop at %s, which means the sequence is not resolving" % rising
            if rising else "counts do not fall as the mesh refines")

sys.exit(status + res.write())
