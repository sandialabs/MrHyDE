#!/usr/bin/env python3

import re
import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check, Results

its = mrhyde_test_support('''Block-triangular with the lower triangle.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,schur,regression

res = Results()
status = enable_trilinos_debug()
status += its.call('mpiexec -n 4 ../../../../mrhyde >& mrhyde.log')

picked = re.findall(r"\[BlockTri\] triangle = (\w+)", open("mrhyde.log").read())
res.add(picked and all(p == "lower" for p in picked), "lower triangle selected",
        "picked %s" % (picked or "no [BlockTri] triangle line at verbosity >= 5"))
check(solves=10, mean=14.9, imax=16, res=res)

sys.exit(status + res.write())
