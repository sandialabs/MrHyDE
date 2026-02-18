#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, os
sys.path.append("../../scripts")
from mrhyde_test_support import *

# ==============================================================================
# Parsing input

desc = '''HCURL 3D Component-based Dirichlet BC verification
       E = (-y, x, 0) with curl(E) = (0, 0, 2)
       L2 error expected at machine precision
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
#TESTING -k maxwell,HCURL,Dirichlet,3D,component-BC

# ==============================================================================
status = 0

status += its.call('mpiexec -n 2 ../../mrhyde >& mrhyde.log')
status += its.clean_log()
status += its.call('diff -y %s.log %s.gold' % (root, root))

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
