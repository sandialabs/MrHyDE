#!/usr/bin/env python3

import subprocess
import sys

sys.path.append("../../../scripts")
from mrhyde_test_support import *

desc = '''Expect failure when RefMaxwell key appears in AMG Settings.'''
its = mrhyde_test_support(desc)

#TESTING active
#TESTING -n 2
#TESTING -k maxwell,HCURL,blocktriangular,negative,parsing

status = 0

if its.opts.execute:
  cmd = 'mpiexec -n 2 ../../../mrhyde >& mrhyde.log'
  if its.opts.verbose:
    print('---> ' + cmd)
  rc = subprocess.call(cmd, shell=True, executable='/bin/bash')
  if rc == 0:
    print('Failure: expected mrhyde to fail for misplaced key.')
    status += 1
  checks = [
    "grep -q \"Unknown key 'use lumped M0inv' in section 'Pivot Block Settings.AMG Settings'.\" mrhyde.log",
    "grep -q \"Allowed parameters:\" mrhyde.log",
    "grep -q \"  - smoother: type\" mrhyde.log",
    "grep -q \"Allowed sublists:\" mrhyde.log",
    "grep -q \"  - smoother: params\" mrhyde.log"
  ]
  missing_msg = False
  for check in checks:
    if subprocess.call(check, shell=True, executable='/bin/bash') != 0:
      missing_msg = True
      status += 1
  if missing_msg:
    print('Failure: expected misplaced-key error message missing from log.')

if status == 0:
  print('Success.')
else:
  print('Failure.')
sys.exit(status)
