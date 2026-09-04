#!/usr/bin/env python3

import sys
import os

sys.path.append("../../../scripts")
from mrhyde_test_support import *

desc = '''3D PEC cavity with block-triangular Schur preconditioner (S_tilde = C).'''
its = mrhyde_test_support(desc)

print('Because of the diff test on the log file, this test needs ')
print('to run with "-v".  There is a buffering issue.')
print('Setting the verbosity to True.')
its.opts.verbose = True

root = 'mrhyde'

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,schur,regression

status = 0

if its.opts.preprocess:
  if its.opts.verbose != 'none':
    print('---> Preprocessing %s' % (root))
  status += its.call('echo "  No preprocessing, yet."')

status += its.call('mpiexec -n 4 ../../../mrhyde >& mrhyde.log')
status += its.clean_log()

status += its.call('diff -y %s.log %s.gold' % (root, root))

if its.opts.graphics and not status:
  if its.opts.verbose != 'none':
    print('---> Graphics %s' % (root))
  status += its.call('echo "  No graphics, yet."')

if its.opts.clean and not status:
  if its.opts.verbose != 'none':
    print('---> Clean %s' % (root))
  os.chdir('obj-org')
  status += its.call('ichos_clean')
  status += its.call('rm -rf shot.*')
  os.chdir('..')
  status += its.call('ichos_clean')

if status == 0:
  print('Success.')
else:
  print('Failure.')
sys.exit(status)
