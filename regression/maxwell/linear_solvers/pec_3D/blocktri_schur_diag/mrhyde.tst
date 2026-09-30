#!/usr/bin/env python3

import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check

its = mrhyde_test_support('''Block-triangular: diagonal pivot on B, Chebyshev on the diag Schur complement.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,schur,regression

status = enable_trilinos_debug()
status += its.call('mpiexec -n 4 ../../../../mrhyde >& mrhyde.log')
status += check(solves=10, mean=14.5, imax=15)

sys.exit(status)
