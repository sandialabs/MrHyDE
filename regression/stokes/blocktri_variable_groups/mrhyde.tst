#!/usr/bin/env python3

import os
import re
import sys
sys.path.append("../../scripts")
sys.path.append("../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import iterations, stats, Results

its = mrhyde_test_support('''Block-triangular 'variable groups': ux+uy fuse into one pivot split, pressure-mass Schur.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k Stokes,blocktriangular,nblock,parallel,regression

RANKS = [1, 2, 4]
ITER_CAP = 41
EXACT_TOL = 1.0e-10

res = Results()
status = enable_trilinos_debug()


def run(deck, log, n=1):
    status = its.call('cp %s input.yaml' % deck)
    status += its.call('mpiexec -n %d ../../mrhyde >& %s' % (n, log))
    return status


def l2(log):
    out = {}
    for line in open(log):
        m = re.search(r'L2 norm of the error for (\w+) = ([\d.eE+-]+)', line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


runs = {}
for n in RANKS:
    log = 'mrhyde_blocktri_np%d.log' % n
    status += run('input_blocktri_mass.yaml', log, n)
    counts, s = iterations(open(log).read().splitlines()), stats(log)
    if counts and s is not None and not s["unconv"]:
        runs[n] = counts

status += run('input_blocktri_exact_pivot.yaml', 'mrhyde_blocktri_exact.log')
if os.path.exists('input.yaml'):
    os.remove('input.yaml')

res.add(sorted(runs) == RANKS, "converged at np=1,2,4",
        "" if sorted(runs) == RANKS else "converged only at %s" % sorted(runs))

splits = [l.strip() for l in open('mrhyde_blocktri_np1.log')
         if '[BlockTri]' in l and 'variable blocks' in l]
want = "3 variable blocks, 2 splits (from variable groups); schur target 'pressure'"
res.add(bool(splits) and want in splits[0], "ux+uy fuse into one pivot split",
        "" if splits and want in splits[0] else (splits[0] if splits else "N-block path never ran"))

tri, tri4 = l2('mrhyde_blocktri_np1.log'), l2('mrhyde_blocktri_np4.log')
worst = max(tri.values()) if tri else 1.0
res.add(worst < EXACT_TOL, "solution exact to 1e-10", "worst L2 %.1e" % worst)

agree = bool(tri) and set(tri) == set(tri4) and all(abs(tri[k] - tri4[k]) < 1.0e-8 for k in tri)
res.add(agree, "L2 error is rank-invariant", "" if agree else "%s against %s" % (tri, tri4))

exact = iterations(open('mrhyde_blocktri_exact.log').read().splitlines())
res.add(bool(exact) and max(exact) == 1, "exact pivot solves in 1 iteration",
        "" if exact and max(exact) == 1 else "max %s" % (max(exact) if exact else "no solves"))

worst_it = {n: max(c) for n, c in runs.items()}
under = bool(worst_it) and all(v <= ITER_CAP for v in worst_it.values())
res.add(under, "iterations under %d" % ITER_CAP, "max %s" % sorted(worst_it.items()))

status += its.call('../../mrhyde input_blocktri_seq3.yaml >& mrhyde_seq3.log')
seq_splits = [l for l in open('mrhyde_seq3.log') if '[BlockTri]' in l and '3 splits' in l]
res.add(bool(seq_splits), "3-split sequential path runs",
        "" if seq_splits else "no [BlockTri] line reporting 3 splits")

sys.exit(status + res.write())
