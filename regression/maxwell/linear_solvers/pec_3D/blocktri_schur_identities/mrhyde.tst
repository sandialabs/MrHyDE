#!/usr/bin/env python3

import re
import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check, Results

its = mrhyde_test_support('''Block-triangular: AMG pivot, RefMaxwell Schur block.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,refmaxwell,parallel,regression

IDENTITY_TOL = 1.0e-12
IDENTITIES = ["round-trip", "J10*D0", "schur", "D0-scale"]

res = Results()
status = enable_trilinos_debug()
status += its.call('mpiexec -n 4 ../../../../mrhyde >& mrhyde.log')

found = {}
for line in open("mrhyde.log"):
    m = re.search(r"\[BLOCK-VERIFY\] (\S+) rel = ([-0-9.eE+]+)", line)
    if m:
        found.setdefault(m.group(1), []).append(float(m.group(2)))

for name in IDENTITIES:
    vals = found.get(name)
    if not vals:
        res.add(False, name, "no [BLOCK-VERIFY] line; is verbosity 5 or higher set?")
        continue
    res.add(max(vals) <= IDENTITY_TOL, name,
            "%d checks, worst %.3e, limit %.1e" % (len(vals), max(vals), IDENTITY_TOL))

check(solves=10, mean=8.5, imax=9, res=res)

sys.exit(status + res.write())
