#!/usr/bin/env python3

import os
import re
import subprocess
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_mrhyde_root = os.path.normpath(os.path.join(_script_dir, "..", "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_mrhyde_root, "regression", "scripts"))
from mrhyde_test_support import *

desc = '''3D PEC cavity with block-triangular Schur preconditioner (S_tilde = C).'''
its = mrhyde_test_support(desc)
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,schur,regression

_mrhyde_exe = os.path.join(_mrhyde_root, "regression", "mrhyde")
cmd = "mpiexec -n 4 " + _mrhyde_exe + " >& mrhyde.log"
if its.opts.verbose:
  print('---> ' + cmd)
result = subprocess.run(cmd, shell=True)
status = result.returncode
its.clean_log()

iters = []
last_iter = None
iter_re = re.compile(r'^Iter\\s+([0-9]+),')
with open('mrhyde.log', 'r') as f:
  for line in f:
    m = iter_re.match(line.strip())
    if m:
      last_iter = int(m.group(1))
    if 'Norm of solution:' in line and last_iter is not None:
      iters.append(last_iter)
      last_iter = None

if not iters:
  print('Warning: no nonlinear linear-solve iteration counts found in log.')
else:
  max_iter = max(iters)
  print('Linear iterations per step:', iters)
  print('Maximum linear iterations:', max_iter)
  if max_iter > 120:
    print('Warning: iteration ceiling exceeded (max_iter > 120).')

if status == 0:
  print('Success.')
else:
  print('Failure.')
sys.exit(status)
