#!/usr/bin/env python3
import sys, os
sys.path.append("../../../scripts")
from mrhyde_test_support import *

desc = '''Maxwell monolith vs block-diag Jacobi: compare outputs'''
its = mrhyde_test_support(desc)
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,solver

status = 0
try:
    status += its.call('cp input_monolith_jac.yaml input.yaml')
    status += its.call('mpiexec -n 4 ../../../mrhyde >& mrhyde_monolith.log')
    status += its.clean_log('mrhyde_monolith.log')
    status += its.call('cp input_blockdiag_jacobi.yaml input.yaml')
    status += its.call('mpiexec -n 4 ../../../mrhyde >& mrhyde_blockdiag.log')
    status += its.clean_log('mrhyde_blockdiag.log')
    status += its.call('diff mrhyde_monolith.log mrhyde_blockdiag.log')
finally:
    if os.path.exists('input.yaml'):
        os.remove('input.yaml')

if status == 0:
    print('Success.')
else:
    print('Failure.')
sys.exit(status)
