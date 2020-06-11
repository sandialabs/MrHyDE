#!/usr/bin/env python2.7
#-------------------------------------------------------------------------------

import sys, os
import subprocess as sp
import string
import shutil
from milo_test_support import *
from numpy import isnan, isinf
#from math import isnan, isinf

# ==============================================================================
# Parsing input

# No reason to format the description as it will be reformatted by optparse.
desc = '''thermal verification
       '''

its = milo_test_support(desc)

print 'Because of the diff test on the log file, this test needs '
print 'to run with "-v".  There is a buffering issue.'
print 'Setting the verbosity to True.'
its.opts.verbose = True

#-------------------------------------------------------------------------------
# Problem Parameters

root = 'milo'   # root filename for test
aeps = 5.0e-15     # absolute error tolerance
reps = 1.0e-12     # relative error tolerance
fdtol= 5.0e-10     # finite difference gradient tolerance

# These comments are for testing with the runtest.py utility.
#TESTING active
#TESTING -n 1
#TESTING -k medium

# ==============================================================================
status = 0

# ------------------------------
if its.opts.preprocess:
  if its.opts.verbose != 'none': print '---> Preprocessing %s' % (root)
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
#if its.opts.diff:
#  if its.opts.verbose != 'none': print '---> Diff %s' % (root)
#  # Test 1
#  fline = ''
#  if its.opts.nprocs > 1:
#    flog = '%s.%i.log' % (root, its.opts.nprocs)
#  else:
#    flog = '%s.log' % (root)
#  for line in open(flog):
#    #if "err w.r.t. fourth order fd" in line: fline = line
#    if "Value of Objective Function" in  line: fline = line
#  w = fline.split()
#  fderr = float(w[6])
#  if its.opts.verbose != 'none':
#    print '\n-> Is 4th order FD error, %g, > %g?' % (abs(fderr), fdtol)
#  if abs(fderr) > fdtol or isnan(fderr) or isinf(fderr):
#    status += 1
#    print '  Failure 4th order FD error too large.'

  # Test 2
  #
status += its.call('diff -y %s.log ./ref/%s.ocs' % (root, root))
  #status += its.call("awk 'NR==1 {print substr($0,0,38)} NR>1 {print substr($0,0,41);}' < %s.ocs | diff - ref/%s.ocs" % (root, root))

  # Test 3
#  cmd = 'ichos_diff.exe -aeps %g -reps %g -r1 ref/%s.rst -r2 %s.rst %s' \
#        %(aeps, reps, root, root, root)
#  status += its.call(cmd)

  # Test 4
#  cmd = 'ichos_diff.exe -aeps %g -reps %g -r1 ref/%s.adj.rst -r2 %s.adj.rst %s'\
#        %(aeps, reps, root, root, root)
#  status += its.call(cmd)

# ------------------------------
if its.opts.baseline and not status:
  if its.opts.verbose != 'none': print '---> Baseline %s' % (root)
  try :
    shutil.copy2('%s.ocs' %(root), 'ref/%s.ocs' %(root))
  except (IOError, os.error), why:
    print why
    status += 1

  try :
    shutil.copy2('%s.rst' %(root), 'ref/%s.rst' %(root))
  except (IOError, os.error), why:
    print why
    status += 1

  try :
    shutil.copy2('%s.adj.rst' %(root), 'ref/%s.adj.rst' %(root))
  except (IOError, os.error), why:
    print why
    status += 1

# ------------------------------
if its.opts.graphics and not status:
  if its.opts.verbose != 'none': print '---> Graphics %s' % (root)
  status += its.call('echo "  No graphics, yet."')

# ------------------------------
if its.opts.clean and not status:
  if its.opts.verbose != 'none': print '---> Clean %s' % (root)
  os.chdir('obj-org')
  status += its.call('ichos_clean')
  status += its.call('rm -rf shot.*')
  os.chdir('..')
  status += its.call('ichos_clean')

# ==============================================================================
if status == 0: print 'Success.'
else:           print 'Failure.'
sys.exit(status)
