#!/usr/bin/env python3

import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check

# 44 is expected: S lands on H(div) and one-level Chebyshev cannot resolve
# its near-kernel. The test pins the path, not a good count.

its = mrhyde_test_support('''Block-triangular with pivot block 0: Schur forms on the HDIV block.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HDIV,blocktriangular,schur,pivot0,regression

status = enable_trilinos_debug()
status += its.call('mpiexec -n 4 ../../../../mrhyde >& mrhyde.log')
status += check(solves=10, mean=44.3, imax=57)

sys.exit(status)
