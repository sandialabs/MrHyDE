#!/usr/bin/env python3

import re
import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import band, iterations, stats, Results

its = mrhyde_test_support('''MueLu sub-blocks reuse their hierarchy under 'update'.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,amg,parallel,regression

LABELS = ["BlockTri pivot MueLu", "BlockTri Schur MueLu"]
MODES = ["none", "update"]
ENERGY = re.compile(r"electric_energy = ([-0-9.e+]+)\s+magnetic_energy = ([-0-9.e+]+)")

def count(text, label, action):
    return len(re.findall(r"\[InverseLibrary\] %s: %s" % (re.escape(label), action), text))

res = Results()
status = enable_trilinos_debug()
runs, rebuilt, built, energy = {}, {}, {}, {}
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
    rebuilt[mode] = {L: count(text, L, "rebuilt in place") for L in LABELS}
    built[mode] = {L: count(text, L, "built new") for L in LABELS}
    hits = ENERGY.findall(text)
    energy[mode] = [float(v) for v in hits[-1]] if hits else None

if len(runs) == len(MODES):
    for L in LABELS:
        res.add(rebuilt["update"][L] > 0, "update reuses %s" % L,
                "%d in-place rebuilds; 0 means every solve paid a full setup"
                % rebuilt["update"][L])
        res.add(rebuilt["none"][L] == 0 and built["none"][L] > 1, "none rebuilds %s" % L,
                "none: %d built, %d in place" % (built["none"][L], rebuilt["none"][L]))

    # The linear solves run to 1e-8, so reuse must not move the answer.
    if energy["none"] is None or energy["update"] is None:
        res.add(False, "energies reported", "no integrated quantities in one of the logs")
    else:
        off = max(abs(a - b) / max(abs(a), 1.0e-30)
                  for a, b in zip(energy["none"], energy["update"]))
        res.add(off < 1.0e-5, "none vs update energy",
                "worst relative difference %.2e" % off)

    # 'reuse: type: RP' keeps R and P, so the hierarchy is not bit-identical.
    if len(runs["none"]) != len(runs["update"]):
        res.add(False, "none vs update iters", "none ran %d solves, update ran %d"
                % (len(runs["none"]), len(runs["update"])))
    else:
        tol = band(max(runs["none"]))
        off = [i for i, (x, y) in enumerate(zip(runs["none"], runs["update"]))
               if abs(x - y) > tol]
        res.add(not off, "none vs update iters",
                "solves %s differ by more than %d" % (off, tol) if off
                else "%d solves agree within %d" % (len(runs["none"]), tol))

sys.exit(status + res.write())
