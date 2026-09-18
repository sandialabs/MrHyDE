#!/usr/bin/env python3

import re
import sys
sys.path.append("../../../../scripts")
from mrhyde_test_support import *

its = mrhyde_test_support('''Setup-time block identities: round-trip, J10*D0, assembled vs matrix-free Schur.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,refmaxwell,parallel,regression

TOL = 1.0e-12
EXPECTED = ["round-trip", "J10*D0", "schur"]

status = its.call('mpiexec -n 4 ../../../../mrhyde input.yaml >& mrhyde.log')

found = {}
for line in open("mrhyde.log"):
    m = re.search(r"\[BLOCK-VERIFY\] (\S+) rel = ([-0-9.eE+]+)", line)
    if m:
        found.setdefault(m.group(1), []).append(float(m.group(2)))

for name in EXPECTED:
    if name not in found:
        print("Failure: no [BLOCK-VERIFY] %s line. Is verbosity 5 or higher set?" % name)
        status += 1
        continue
    worst = max(found[name])
    print("%s: %d checks, worst %.3e" % (name, len(found[name]), worst))
    if worst > TOL:
        print("Failure: %s rel %.3e exceeds %.1e" % (name, worst, TOL))
        status += 1

sys.exit(status)
