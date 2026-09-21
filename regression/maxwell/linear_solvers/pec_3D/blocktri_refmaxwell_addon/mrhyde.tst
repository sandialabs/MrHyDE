#!/usr/bin/env python3

import re
import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from parse_log import check

its = mrhyde_test_support('''RefMaxwell with the addon enabled: checks the recovered beta and the iteration count.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,refmaxwell,addon,regression

# beta = alpha_u^2 * gamma / (alpha_t * mu) = 0.25 * 1 / (2.5 * 1)
BETA = 0.1
BETA_TOL = 1.0e-10

status = its.call('mpiexec -n 4 ../../../../mrhyde input.yaml >& mrhyde.log')

seen = [float(m) for m in
        re.findall(r"\[ADDON\] beta = ([-0-9.eE+]+)", open("mrhyde.log").read())]
if not seen:
    print("Failure: no [ADDON] beta line. Is the addon enabled and verbosity >= 5?")
    status += 1
else:
    print("beta: %d build(s), values %s" % (len(seen), seen))
    for b in seen:
        if abs(b - BETA) > BETA_TOL:
            print("Failure: beta %.12g, expected %.12g" % (b, BETA))
            status += 1

if "refmaxwell: disable addon : bool = 0" not in open("mrhyde.log").read():
    print("Failure: MueLu reports the addon disabled.")
    status += 1

status += check(solves=12, mean=7.42, imax=9)

sys.exit(status)
