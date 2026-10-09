#!/usr/bin/env python3

import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check, Results

its = mrhyde_test_support('''Level-0 Hiptmair on the diag Schur complement, with and without a pivot XML.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k regression,maxwell,HCURL,HDIV,blocktriangular,schur,schur_hcurl,hiptmair,iters,routing

CASES = [
    ("hiptmair",   10,  7.2,  8),
    ("bare_pivot", 10, 19.6, 22),
]

res = Results()
status = enable_trilinos_debug()
for name, solves, mean, imax in CASES:
    log = "mrhyde_%s.log" % name
    status += its.call("mpiexec -n 4 ../../../../mrhyde input_%s.yaml >& %s" % (name, log))
    check(solves=solves, mean=mean, imax=imax, log=log, res=res, label="%s iterations" % name)

sys.exit(status + res.write())
