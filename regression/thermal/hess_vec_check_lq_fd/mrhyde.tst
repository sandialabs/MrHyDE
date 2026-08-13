#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os
sys.path.append("../../scripts")
from mrhyde_test_support import *

# ==============================================================================
# Parsing input

desc = ''' LQ Hessian-vector identity check on a distributed-source thermal
control problem. FD-of-gradients HessVec path (no src_gate). Diffs the
[HESSVEC-CHECK], [HV-*], [SECANT-IDENTITY] tag lines against mrhyde.gold. '''

its = mrhyde_test_support(desc)

its.opts.verbose = True

root = 'mrhyde'

#TESTING active
#TESTING -n 4
#TESTING -k thermal,optimization,hessvec,lq

# ==============================================================================
status = 0

if its.opts.preprocess:
  status += its.call('echo "  No preprocessing, yet."')

status += its.call('mpiexec -n 4 ../../mrhyde >& mrhyde.log')
status += its.clean_log()

status += its.call('diff -y mrhyde.log mrhyde.gold')

if its.opts.clean and not status:
  status += its.call('rm -f mrhyde.log')

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
