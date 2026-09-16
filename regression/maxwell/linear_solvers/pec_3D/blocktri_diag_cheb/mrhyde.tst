#!/usr/bin/env python3

import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from parse_log import check

its = mrhyde_test_support('''Block-triangular: diagonal pivot on B, Chebyshev on the diag Schur complement.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,schur,regression

status = its.call('mpiexec -n 4 ../../../../mrhyde >& mrhyde.log')
status += check(solves=12, mean=12.5, imax=15)

sys.exit(status)
