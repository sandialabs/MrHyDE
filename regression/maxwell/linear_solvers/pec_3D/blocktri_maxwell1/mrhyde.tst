#!/usr/bin/env python3

import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check, Results

its = mrhyde_test_support('''Maxwell1 Schur block: Kn from the edge mass matrix and Kn from SM.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k regression,maxwell,HCURL,HDIV,blocktriangular,schur,schur_hcurl,maxwell1,iters,experimental

CASES = [
    ("kn_from_m1", 10, 7.1, 8),
    ("kn_from_sm", 10, 7.1, 8),
]

res = Results()
# TPETRA_DEBUG off: the KLU coarse solve hits Amesos2 reindex_impl, which builds an
# overlapping column map with a non-overlapping global size, and MueLu::Maxwell1::compute
# aborts on an edge-sized map in Tpetra_Map_def.hpp. Both are upstream issues.
status = enable_trilinos_debug(tpetra=False)
for name, solves, mean, imax in CASES:
    log = "mrhyde_%s.log" % name
    status += its.call("mpiexec -n 4 ../../../../mrhyde input_%s.yaml >& %s" % (name, log))
    check(solves=solves, mean=mean, imax=imax, log=log, res=res, label="%s iterations" % name)

sys.exit(status + res.write())
