#!/usr/bin/env python3

import os
import re
import sys
sys.path.append("../../../../scripts")
from mrhyde_test_support import *

its = mrhyde_test_support('''Monolith Jacobi vs block-diagonal Jacobi: outputs must match.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,blockdiagonal,equivalence,regression

KEEP = re.compile(r'(Iteration:|Norm of nonlinear residual|Scaled Norm of nonlinear residual|'
                  r'Belos Iterative Solver|^Iter\s+\d+,|Beginning Time Step|Current time is|'
                  r'Integrated quantities|electric_energy|magnetic_energy|total_energy)')


def filter_log(src, dst):
    with open(src) as f, open(dst, 'w') as g:
        g.writelines(line for line in f if KEEP.search(line))


status = 0
try:
    for deck, log in (('input_monolith_jac.yaml', 'mrhyde_monolith'),
                      ('input_blockdiag_jacobi.yaml', 'mrhyde_blockdiag')):
        status += its.call('cp %s input.yaml' % deck)
        status += its.call('mpiexec -n 4 ../../../../mrhyde >& %s.log' % log)
        filter_log('%s.log' % log, '%s.filtered' % log)
    status += its.call('diff mrhyde_monolith.filtered mrhyde_blockdiag.filtered')
finally:
    if os.path.exists('input.yaml'):
        os.remove('input.yaml')

print('Success.' if status == 0 else 'Failure.')
sys.exit(status)
