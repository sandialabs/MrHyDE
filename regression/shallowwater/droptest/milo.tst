#!/usr/bin/env python2.7
#-------------------------------------------------------------------------------

import sys, os
import subprocess as sp
import string
import shutil
from milo_test_support import *
#from numpy import isnan, isinf
#from math import isnan, isinf

# ==============================================================================
# Parsing input

# No reason to format the description as it will be reformatted by optparse.
desc = '''stokes verification
       '''

its = milo_test_support(desc)

print('Because of the diff test on the log file, this test needs ')
print('to run with "-v".  There is a buffering issue.')
print('Setting the verbosity to True.')
its.opts.verbose = True

#-------------------------------------------------------------------------------
# Problem Parameters

root = 'milo'   # root filename for test
aeps = 1.0e-14     # absolute error tolerance
reps = 1.0e-12     # relative error tolerance
fdtol= 5.0e-10     # finite difference gradient tolerance

# These comments are for testing with the runtest.py utility.
#TESTING active
#TESTING -n 4
#TESTING -k shallowwater,transient,nonlinear

# ==============================================================================
status = 0

# ------------------------------
if its.opts.preprocess:
  if its.opts.verbose != 'none': print('---> Preprocessing %s' % (root))
  status += its.call('echo "  No preprocessing, yet."')

status += its.call('./run.sh')
# ------------------------------
#if its.opts.execute:
#  if its.opts.verbose != 'none': print '---> Execute %s' % (root)
#  os.chdir('obj-org')
#  #status += its.ichos(root)
#  status += its.call('./run.sh')
#  os.chdir('..')
#  #status += its.call('ichos_clean')
#  #status += its.ichos_opt(root)
#  #status += its.call('./run.sh')

# ------------------------------
flog = '%s.log' % (root)
reflog = 'ref/%s.ocs' % (root)

for line in open(flog):
  if "L2 norm of the error for H" in  line: Hline = line
w = Hline.split()
Herr = float(w[9])

for line in open(reflog):
  if "L2 norm of the error for H" in  line: refHline = line
w = refHline.split()
refHerr = float(w[9])

if abs(Herr-refHerr) > aeps :
  status += 1
  print('  Failure: L2 error for H too large.')

for line in open(flog):
  if "L2 norm of the error for Hu" in  line: Huline = line
w = Huline.split()
Huerr = float(w[9])

for line in open(reflog):
  if "L2 norm of the error for Hu" in  line: refHuline = line
w = refHuline.split()
refHuerr = float(w[9])

if abs(Huerr-refHuerr) > aeps :
  status += 1
  print('  Failure: L2 error for uy too large.')

for line in open(flog):
  if "L2 norm of the error for Hv" in  line: Hvline = line
w = Hvline.split()
Hverr = float(w[9])

for line in open(reflog):
  if "L2 norm of the error for Hv" in  line: refHvline = line
w = refHvline.split()
refHverr = float(w[9])

if abs(Hverr-refHverr) > aeps :
  status += 1
  print('  Failure: L2 error for Hv too large.')

# ------------------------------
# ------------------------------
# ------------------------------
if its.opts.baseline and not status:
  if its.opts.verbose != 'none': print('---> Baseline %s' % (root))
  try :
    shutil.copy2('%s.ocs' %(root), 'ref/%s.ocs' %(root))
  except (IOError, os.error) as why:
    print(why)
    status += 1

  try :
    shutil.copy2('%s.rst' %(root), 'ref/%s.rst' %(root))
  except (IOError, os.error) as why:
    print(why)
    status += 1

  try :
    shutil.copy2('%s.adj.rst' %(root), 'ref/%s.adj.rst' %(root))
  except (IOError, os.error) as why:
    print(why)
    status += 1

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
