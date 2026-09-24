#!/usr/bin/env python3

import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check

its = mrhyde_test_support('''Block-triangular pivoting on block 0, KLU on both blocks.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,direct,regression

# TPETRA_DEBUG off: Amesos2 reindex_impl builds an overlapping column map
# with a non-overlapping global size, which trips Tpetra's own check.
status = enable_trilinos_debug(tpetra=False)
status += its.call('mpiexec -n 4 ../../../../mrhyde >& mrhyde.log')
status += check(solves=10, mean=7.4, imax=8)

sys.exit(status)
