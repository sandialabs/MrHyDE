#!/usr/bin/env python3

import re
import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import stats, Results

its = mrhyde_test_support('''A Hiptmair block must rebuild; its neighbour must still reuse.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blockdiagonal,amg,parallel,regression

LOG = "mrhyde.log"

def count(text, block, action):
    return len(re.findall(r"\[InverseLibrary\] BlockDiag block %d MueLu: %s" % (block, action), text))

res = Results()
status = enable_trilinos_debug()
status += its.call("mpiexec -n 4 ../../../../mrhyde >& %s" % LOG)
text = open(LOG).read()
s = stats(LOG)

if s is None:
    res.add(False, "ran", "no Belos solves in " + LOG)
else:
    res.add(not s["unconv"] and s["solves"] > 0, "converged",
            "%d solves, %d unconverged" % (s["solves"], s["unconv"]))
    hip_new, hip_reuse = count(text, 0, "built new"), count(text, 0, "rebuilt in place")
    sa_new, sa_reuse = count(text, 1, "built new"), count(text, 1, "rebuilt in place")
    res.add(hip_reuse == 0 and hip_new == s["solves"], "hiptmair block rebuilds",
            "%d built, %d reused; any reuse means a stale D0^T A D0" % (hip_new, hip_reuse))
    res.add(sa_reuse > 0, "plain block still reuses",
            "%d built, %d reused; 0 reuse means the guard is too broad" % (sa_new, sa_reuse))

sys.exit(status + res.write())
