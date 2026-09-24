#!/usr/bin/env python3

import os
import re
import sys
sys.path.append("../../scripts")
sys.path.append("../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import Results

its = mrhyde_test_support('''Three-field Stokes: block Jacobi must equal monolithic Jacobi.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 1
#TESTING -k Stokes,blockdiagonal,equivalence,regression

KEEP = re.compile(r'(Iteration:|Norm of nonlinear residual|Belos Iterative Solver|'
                  r'^Iter\s+\d+,|L2 norm of the error)')


def filter_log(src, dst):
    with open(src) as f, open(dst, 'w') as g:
        g.writelines(line for line in f if KEEP.search(line))


res = Results()
status = enable_trilinos_debug()
try:
    for deck, log in (('input_monolith_jac.yaml', 'mrhyde_monolith'),
                      ('input_blockdiag_jacobi.yaml', 'mrhyde_blockdiag')):
        status += its.call('cp %s input.yaml' % deck)
        status += its.call('mpiexec -n 1 ../../mrhyde >& %s.log' % log)
        filter_log('%s.log' % log, '%s.filtered' % log)
finally:
    if os.path.exists('input.yaml'):
        os.remove('input.yaml')

# Guards the N-block Teko path: a 2-block fallback would still match the monolith.
nblocks = [l.split()[1] for l in open('mrhyde_blockdiag.log') if '[BlockDiag]' in l]
res.add(nblocks[:1] == ['3'], "three variable blocks",
        "built %s blocks" % nblocks[0] if nblocks else "block-diagonal path never ran")

mono = open('mrhyde_monolith.filtered').read()
bd = open('mrhyde_blockdiag.filtered').read()
res.add(mono == bd, "block Jacobi equals monolithic Jacobi",
        "%d filtered lines identical" % len(bd.splitlines()) if mono == bd
        else "filtered output differs")

sys.exit(status + res.write())
