#!/usr/bin/env python3

import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from parse_log import check

its = mrhyde_test_support('''Block-diagonal with AMG on the mass matrices.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blockdiagonal,mass,regression

status = its.call('mpiexec -n 4 ../../../../mrhyde >& mrhyde.log')
status += check(solves=12, mean=24.67, imax=33)

sys.exit(status)
