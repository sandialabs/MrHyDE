#!/usr/bin/env python3

import re
import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from parse_log import check, Results

its = mrhyde_test_support('''RefMaxwell with the addon enabled: recovered beta and iteration count.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,refmaxwell,addon,regression

# beta = alpha_u^2 * gamma / (alpha_t * mu) = 0.25 / 2.5
BETA, BETA_TOL = 0.1, 1.0e-10

res = Results()
status = its.call('mpiexec -n 4 ../../../../mrhyde input.yaml >& mrhyde.log')
text = open("mrhyde.log").read()

seen = [float(m) for m in re.findall(r"\[ADDON\] beta = ([-0-9.eE+]+)", text)]
res.add(seen and all(abs(b - BETA) < BETA_TOL for b in seen), "recovered beta",
        "%s, expected %.12g" % (seen or "no [ADDON] line at verbosity >= 5", BETA))

# The first hierarchy has no addon: beta comes off the Schur correction, which
# does not exist yet. Only the last build matters.
flags = re.findall(r"refmaxwell: disable addon : bool = ([01])", text)
res.add(flags and flags[-1] == "0", "addon on for the last build",
        "flag per build: %s" % ("".join(flags) or "MueLu never echoed it"))

# The +/- 2 band cannot separate this from the no-addon deck's 7.42.
check(solves=12, mean=7.67, imax=9, res=res)

sys.exit(status + res.write())
