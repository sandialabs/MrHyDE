#!/usr/bin/env python3

import re
import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import iterations, stats, Results

its = mrhyde_test_support('''Reuse type must not change the answer: none vs update vs full.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,refmaxwell,parallel,regression

# J is identical at every solve, so this catches reuse not happening and reset
# disagreeing with rebuild, not a reset carrying a stale copy of that matrix.
FREEZE_TOL = 1
BUILD = re.compile(r"\[RefMaxwell\] Built new preconditioner hierarchy")
RESET = re.compile(r"\[RefMaxwell\] Reusing existing hierarchy")
MODES = ["none", "update", "full"]

res = Results()
status = enable_trilinos_debug()
runs, builds, resets = {}, {}, {}
for mode in MODES:
    log = "mrhyde_%s.log" % mode
    status += its.call("mpiexec -n 4 ../../../../mrhyde input_%s.yaml >& %s" % (mode, log))
    text = open(log).read()
    counts, s = iterations(text.splitlines()), stats(log)
    if not counts or s is None:
        res.add(False, "%s ran" % mode, "no Belos solves in " + log)
        continue
    res.add(not s["unconv"], "%s converged" % mode,
            "%d solves, %d unconverged" % (s["solves"], s["unconv"]))
    runs[mode] = counts
    builds[mode], resets[mode] = len(BUILD.findall(text)), len(RESET.findall(text))

def compare(a, b, tol):
    if len(runs[a]) != len(runs[b]):
        return False, "%s ran %d solves, %s ran %d" % (a, len(runs[a]), b, len(runs[b]))
    off = [i for i, (x, y) in enumerate(zip(runs[a], runs[b])) if abs(x - y) > tol]
    if off:
        return False, "solves %s differ by more than %d" % (off, tol)
    return True, "%d solves agree within %d" % (len(runs[a]), tol)

if len(runs) == len(MODES):
    res.add(resets["update"] > 0, "update reuses",
            "%d resetMatrix calls; 0 means the short-circuit is dead" % resets["update"])
    res.add(builds["none"] > builds["update"], "none rebuilds",
            "none %d builds, update %d" % (builds["none"], builds["update"]))
    for a, b, tol in (("none", "update", 0), ("none", "full", FREEZE_TOL)):
        ok, detail = compare(a, b, tol)
        res.add(ok, "%s vs %s" % (a, b), detail)

sys.exit(status + res.write())
