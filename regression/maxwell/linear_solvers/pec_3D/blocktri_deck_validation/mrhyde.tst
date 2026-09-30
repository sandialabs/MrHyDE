#!/usr/bin/env python3

import re
import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import check, Results

its = mrhyde_test_support('''Schur operator identities, plus the two deck errors that must abort.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k regression,maxwell,HCURL,HDIV,blocktriangular,schur,schur_hcurl,refmaxwell,algebra,routing

IDENTITY_TOL = 1.0e-12
IDENTITIES = {"round-trip": "blocked op round-trips",
              "J10*D0":     "J10 annihilates D0",
              "schur":      "S matches J11 - J10 Dinv J01",
              "D0-scale":   "D0 scaling is consistent"}

res = Results()
status = enable_trilinos_debug()
status += its.call('mpiexec -n 4 ../../../../mrhyde input_identities.yaml >& mrhyde_identities.log')

found = {}
for line in open("mrhyde_identities.log"):
    m = re.search(r"\[BLOCK-VERIFY\] (\S+) rel = ([-0-9.eE+]+)", line)
    if m:
        found.setdefault(m.group(1), []).append(float(m.group(2)))

for tag, label in IDENTITIES.items():
    vals = found.get(tag)
    if not vals:
        res.add(False, label, "no [BLOCK-VERIFY] line; is verbosity 5 or higher set?")
        continue
    res.add(max(vals) <= IDENTITY_TOL, label,
            "%d checks, worst %.3e, limit %.1e" % (len(vals), max(vals), IDENTITY_TOL))

check(solves=10, mean=8.5, imax=9, log="mrhyde_identities.log", res=res)


def aborts_with(deck, log, want, label):
    """Both decks are expected to abort, hence ignore_status."""
    its.call('mpiexec -n 4 ../../../../mrhyde %s >& %s' % (deck, log), ignore_status=True)
    hit = want in open(log, errors="replace").read()
    res.add(hit, label, "" if hit else "did not report: " + want)


aborts_with('input_badname.yaml', 'mrhyde_badname.log',
            "names no variable in this set", "misspelled variable name is rejected")
aborts_with('input_orphan_split.yaml', 'mrhyde_orphan.log',
            "sublist for the split of that name",
            "split with no settings sublist is rejected")

# maxwell.cpp pushes E then B unconditionally; split indices depend on that order.
order = [l for l in open('mrhyde_identities.log', errors="replace")
         if '[BlockTri] variables:' in l]
want = '0=E, 1=B'
res.add(bool(order) and want in order[0], "variables declared in physics order",
        "" if order and want in order[0] else (order[0].strip() if order else "no variables line"))

sys.exit(status + res.write())
