#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os
sys.path.append("../../scripts")
from mrhyde_test_support import *

# ==============================================================================
# Parsing input

desc = ''' Magnitude scan probe on a Maxwell E-B control problem. Reports each
objective and regularization term in unweighted and weighted forms; exits
after the scan (Iteration Limit: 0). Diffs mrhyde.log against mrhyde.gold. '''

its = mrhyde_test_support(desc)

its.opts.verbose = True

root = 'mrhyde'

#TESTING active
#TESTING -n 11
#TESTING -k maxwell,optimization,scale,scan

# ==============================================================================
status = 0

if its.opts.preprocess:
  status += its.call('echo "  No preprocessing, yet."')

status += its.call('mpiexec -n 11 ../../mrhyde >& mrhyde.log')
status += its.clean_log()
status += its.call('diff -y mrhyde.log mrhyde.gold')

if its.opts.clean and not status:
  status += its.call('rm -f mrhyde.log')

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
