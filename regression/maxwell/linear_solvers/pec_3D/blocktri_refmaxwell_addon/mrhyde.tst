#!/usr/bin/env python3

import re
import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check, Results

its = mrhyde_test_support('''RefMaxwell with the addon enabled: recovered beta and iteration count.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k regression,maxwell,HCURL,HDIV,blocktriangular,schur,schur_hcurl,refmaxwell,algebra

BETA, BETA_TOL = 0.1, 1.0e-10

res = Results()
status = enable_trilinos_debug()
status += its.call('mpiexec -n 4 ../../../../mrhyde input.yaml >& mrhyde.log')
text = open("mrhyde.log").read()

seen = [float(m) for m in re.findall(r"\[ADDON\] beta = ([-0-9.eE+]+)", text)]
res.add(seen and all(abs(b - BETA) < BETA_TOL for b in seen), "recovered beta",
        "%s, expected %.12g" % (seen or "no [ADDON] line at verbosity >= 5", BETA))

flags = re.findall(r"refmaxwell: disable addon : bool = ([01])", text)
res.add(flags and all(f == "0" for f in flags), "addon on for every build",
        "flag per build: %s" % ("".join(flags) or "MueLu never echoed it"))

check(solves=10, mean=8.8, imax=9, res=res)

sys.exit(status + res.write())
