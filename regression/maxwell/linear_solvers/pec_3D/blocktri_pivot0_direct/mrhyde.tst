#!/usr/bin/env python3

import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from parse_log import check

its = mrhyde_test_support('''Block-triangular pivoting on block 0, KLU on both blocks.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,direct,regression

status = its.call('mpiexec -n 4 ../../../../mrhyde >& mrhyde.log')
status += check(solves=12, mean=6.33, imax=8)

sys.exit(status)
