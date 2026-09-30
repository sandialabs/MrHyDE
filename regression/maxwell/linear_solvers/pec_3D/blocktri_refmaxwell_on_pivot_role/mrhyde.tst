#!/usr/bin/env python3

import re
import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check, Results

its = mrhyde_test_support('''Per-role settings routing: RefMaxwell attaches to the pivot role, not the Schur role.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k regression,maxwell,HCURL,HDIV,blocktriangular,schur,schur_hdiv,refmaxwell,routing

res = Results()
status = enable_trilinos_debug()
status += its.call('mpiexec -n 4 ../../../../mrhyde >& mrhyde.log')

text = open("mrhyde.log").read()
built = re.findall(r"\[RefMaxwell\] Built new preconditioner hierarchy( \(Schur\))?", text)
res.add(built and not any(built), "RefMaxwell built on the pivot",
        "%d hierarchies, Schur-side: %d" % (len(built), sum(1 for b in built if b)))
check(solves=10, mean=39.0, imax=45, res=res)

sys.exit(status + res.write())
