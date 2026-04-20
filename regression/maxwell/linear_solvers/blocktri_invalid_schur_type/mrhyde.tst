#!/usr/bin/env python3

import subprocess
import sys

sys.path.append("../../../scripts")
from mrhyde_test_support import *

desc = '''Expect failure on unsupported Schur approximation type.'''
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
    print('Failure: expected mrhyde to fail for invalid Schur type.')
    status += 1
  grep_cmd = "grep -q \"Unsupported Schur approximation type 'exact'\" mrhyde.log"
  grep_rc = subprocess.call(grep_cmd, shell=True, executable='/bin/bash')
  if grep_rc != 0:
    print('Failure: expected Schur-type error message missing from log.')
    status += 1

if status == 0:
  print('Success.')
else:
  print('Failure.')
sys.exit(status)
