#!/usr/bin/env python2.7
#-------------------------------------------------------------------------------

import sys, os
import subprocess as sp
import string
import shutil
sys.path.append("../../scripts")
from mrhyde_test_support import *

# ==============================================================================
# Parsing input

# No reason to format the description as it will be reformatted by optparse.
desc = ''' gradient check for non-mortar thermal problem
       '''

its = mrhyde_test_support(desc)

print('Because of the diff test on the log file, this test needs ')
print('to run with "-v".  There is a buffering issue.')
print('Setting the verbosity to True.')
its.opts.verbose = True

#-------------------------------------------------------------------------------
# Problem Parameters

root = 'mrhyde'   # root filename for test
aeps = 5.0e-15     # absolute error tolerance
reps = 1.0e-12     # relative error tolerance
fdtol= 5.0e-10     # finite difference gradient tolerance

# These comments are for testing with the runtest.py utility.
#TESTING active
#TESTING -n 4
#TESTING -k thermal,optimization,transient,scalar-parameters

# ==============================================================================
status = 0

# ------------------------------
if its.opts.preprocess:
  if its.opts.verbose != 'none': print('---> Preprocessing %s' % (root))
  status += its.call('echo "  No preprocessing, yet."')

status += its.call('mpiexec -n 4 ../../mrhyde >& mrhyde.log')
status += its.clean_log()
status += its.call('rm final_params.dat param_stash.dat ROL_out.txt')


flog = '%s.log' % (root)
reflog = '%s.gold' % (root)

testdat = [0.0, 0.0, 0.0]
prog = 0
for line in open(flog):
  if prog < 3 :
    if "1.00000000" in  line:
      w = line.split()
      if len(w)>3 :
        testdat[prog] = float(w[3])
        prog = prog+1

refdat = [0.0, 0.0, 0.0]
prog = 0
for line in open(reflog):
  if prog < 3:
    if "1.00000000" in  line: 
      w = line.split()
      if len(w)>3 :
        refdat[prog] = float(w[3])
        prog = prog+1

for chk in range(3):
  if abs(refdat[chk]-testdat[chk]) > aeps :
    print('  Failure: gradient error is too large.')

# ------------------------------



# ------------------------------
if its.opts.graphics and not status:
  if its.opts.verbose != 'none': print('---> Graphics %s' % (root))
  status += its.call('echo "  No graphics, yet."')

# ------------------------------
if its.opts.clean and not status:
  if its.opts.verbose != 'none': print('---> Clean %s' % (root))
  os.chdir('obj-org')
  status += its.call('ichos_clean')
  status += its.call('rm -rf shot.*')
  os.chdir('..')
  status += its.call('ichos_clean')

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
