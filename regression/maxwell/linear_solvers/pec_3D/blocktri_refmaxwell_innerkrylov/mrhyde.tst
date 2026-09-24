#!/usr/bin/env python3

import sys
sys.path.append("../../../../scripts")
sys.path.append("../../../../../scripts/data_processing")
from mrhyde_test_support import *
from trilinos_env import enable_trilinos_debug
from parse_log import iterations, stats, check, Results

its = mrhyde_test_support('''AMG pivot and RefMaxwell Schur blocks wrapped in inner Belos solves.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 4
#TESTING -k maxwell,HCURL,blocktriangular,refmaxwell,innerkrylov,regression

WRAP = "wrapping block preconditioner in inner Belos"

res = Results()
status = enable_trilinos_debug()
status += its.call('mpiexec -n 4 ../../../../mrhyde input.yaml >& mrhyde.log')
status += its.call('mpiexec -n 4 ../../../../mrhyde input_nowrap.yaml >& mrhyde_nowrap.log')

built = [l.split("[", 1)[-1].split("]", 1)[0] for l in open("mrhyde.log") if WRAP in l]
res.add(len(built) == 2, "both blocks wrapped",
        "wrapped " + ", ".join(built) if built else "wrapper never ran")
plain_built = [l for l in open("mrhyde_nowrap.log") if WRAP in l]
res.add(not plain_built, "control is unwrapped",
        "control built one too" if plain_built else "no inner solve, as intended")

wrapped = iterations(open("mrhyde.log").read().splitlines())
plain = iterations(open("mrhyde_nowrap.log").read().splitlines())
if wrapped and plain:
    mw, mp = sum(wrapped) / len(wrapped), sum(plain) / len(plain)
    res.add(mw < 0.8 * mp, "wrapping cuts outer iterations by >20%",
            "wrapped %.2f vs unwrapped %.2f (%.0f%% of control)" % (mw, mp, 100.0 * mw / mp))
else:
    res.add(False, "both runs solved", "no Belos solves in one of the logs")

GUARD = "'inner krylov solver' requires Block GMRES"
its.call('mpiexec -n 4 ../../../../mrhyde input_noflex.yaml >& mrhyde_noflex.log',
         ignore_status=True)
lines = open("mrhyde_noflex.log").read().splitlines()
fired = [i for i, l in enumerate(lines) if GUARD in l]
res.add(bool(fired), "non-flexible outer refused",
        "guard fired, run aborted as intended" if fired
        else "ran anyway without 'Flexible Gmres: true'")
if fired:
    open("mrhyde_noflex.log", "w").write("\n".join(lines[:fired[0] + 1]) + "\n")

check(solves=10, mean=7.6, imax=8, res=res)

sys.exit(status + res.write())
