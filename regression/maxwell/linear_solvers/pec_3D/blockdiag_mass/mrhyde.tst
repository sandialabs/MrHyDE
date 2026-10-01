#!/usr/bin/env python3

import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check

its = mrhyde_test_support('''Block diagonal 'use mass matrix': AMG on M1 and M2, not the Jacobian blocks.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k regression,maxwell,HCURL,HDIV,blockdiagonal,amg,mass,iters

status = enable_trilinos_debug()
status += its.call('mpiexec -n 4 ../../../../mrhyde >& mrhyde.log')
status += check(solves=10, mean=27.7, imax=31)

sys.exit(status)
