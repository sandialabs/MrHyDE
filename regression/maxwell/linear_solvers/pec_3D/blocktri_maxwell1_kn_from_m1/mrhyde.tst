#!/usr/bin/env python3

import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check

its = mrhyde_test_support('''Maxwell1 Schur block with 'use Kn from M1: true': Kn from the edge mass matrix.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k regression,maxwell,HCURL,HDIV,blocktriangular,schur,schur_hcurl,maxwell1,iters

# TPETRA_DEBUG off: the KLU coarse solve hits Amesos2 reindex_impl, which
# builds an overlapping column map with a non-overlapping global size.
status = enable_trilinos_debug(tpetra=False)
status += its.call('mpiexec -n 4 ../../../../mrhyde >& mrhyde.log')
status += check(solves=10, mean=7.1, imax=8)

sys.exit(status)
