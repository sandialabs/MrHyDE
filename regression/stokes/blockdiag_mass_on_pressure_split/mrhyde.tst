#!/usr/bin/env python3

import re
import sys
sys.path.append("../../scripts")
sys.path.append("../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check, stats, Results

its = mrhyde_test_support('''Three-split block diagonal with 'use mass matrix' on the zero-diagonal pressure split.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k Stokes,blockdiagonal,nblock,mass,regression

EXACT_TOL = 1.0e-10

res = Results()
status = enable_trilinos_debug()
status += its.call('mpiexec -n 4 ../../mrhyde >& mrhyde.log')

text = open("mrhyde.log", errors="replace").read()

three = all(("  %s -> " % v) in text for v in ("ux", "pr", "uy"))
res.add(three, "per-split sublists applied",
        "" if three else "log does not echo a ux, pr and uy split sublist")

mass = "[BlockDiag] Block 1: substituting mass matrix" in text
res.add(mass, "pressure block uses its mass matrix",
        "" if mass else "no mass substitution on block 1")

err = {m.group(1): float(m.group(2)) for m in
       re.finditer(r'L2 norm of the error for (\w+) = ([\d.eE+-]+)', text)}
worst = max(err.values()) if err else 1.0
res.add(worst < EXACT_TOL, "solution exact to 1e-10", "worst L2 %.1e" % worst)

s = stats("mrhyde.log")
conv = s is not None and not s["unconv"]
res.add(conv, "all solves converged",
        "" if conv else "%s of %s solves hit the iteration limit"
        % (s["unconv"], s["solves"]) if s else "no solves")

check(solves=3, mean=83.3, imax=129, res=res)

sys.exit(status + res.write())
