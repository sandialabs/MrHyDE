#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os
sys.path.append("../../scripts")
from mrhyde_test_support import *

# ==============================================================================
# Parsing input

desc = '''3D PEC Cavity TM_110 Mode Test
       Verifies 3D PEC (n x E = 0) on all 6 faces using TM_110 mode.
       Ez = sin(pi*x)*sin(pi*y), Ex = Ey = 0, all B components = 0
       '''

its = mrhyde_test_support(desc)

print('Because of the diff test on the log file, this test needs ')
print('to run with "-v".  There is a buffering issue.')
print('Setting the verbosity to True.')
its.opts.verbose = True

#-------------------------------------------------------------------------------
# Problem Parameters

root = 'mrhyde'

# These comments are for testing with the runtest.py utility.
#TESTING active
#TESTING -n 2
#TESTING -k maxwell,HCURL,PEC,3D,cavity,TM110

# ==============================================================================
status = 0

status += its.call('mpiexec -n 2 ../../mrhyde >& mrhyde.log')
status += its.clean_log()
status += its.call('diff -y %s.log %s.gold' % (root, root))

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
