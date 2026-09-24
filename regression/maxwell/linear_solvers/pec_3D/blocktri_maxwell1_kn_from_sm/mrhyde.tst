#!/usr/bin/env python3

import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from parse_log import check

# Release only: under Tpetra debug MueLu::Maxwell1::compute aborts on an
# edge-sized map, Tpetra_Map_def.hpp:518. Upstream issue.

its = mrhyde_test_support('''Block-triangular Maxwell1 Schur with Kn built from SM, the default path.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,schur,regression

status = its.call('mpiexec -n 4 ../../../../mrhyde >& mrhyde.log')
status += check(solves=10, mean=7.1, imax=8)

sys.exit(status)
