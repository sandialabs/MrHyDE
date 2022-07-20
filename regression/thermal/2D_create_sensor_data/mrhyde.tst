#!/usr/bin/env python3
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
#TESTING -n 1
#TESTING -k thermal,sensors

# ==============================================================================
status = 0

# ------------------------------
if its.opts.preprocess:
  if its.opts.verbose != 'none': print('---> Preprocessing %s' % (root))
  status += its.call('echo "  No preprocessing, yet."')

status += its.clean_log()
status += its.call('mpiexec -n 1 ../../mrhyde >& mrhyde.log')


err = 0.0

# read the list of files to compare
filenames = ['sensor.objval', 'sensor.objgrad']
for filename in filenames:
  # this creates a list filled with strings from each line of the file
  outfile = open(filename+'.out','r')
  outfilevalues = outfile.readlines()
  outfile.close()

  # this creates a list filled with strings from each line of the file
  reffile = open(filename+'.gold','r')
  reffilevalues = reffile.readlines()
  reffile.close()

  # convert the strings to floats and compute the error
  for i in range(0, len(outfilevalues)):
    err += abs(float(outfilevalues[i]) - float(reffilevalues[i]))

if err > aeps:
  status += 1

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
