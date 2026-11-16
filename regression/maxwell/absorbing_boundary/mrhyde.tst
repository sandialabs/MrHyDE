#!/usr/bin/env python3

import os
import re
import sys
sys.path.append("../../scripts")
sys.path.append("../../../scripts/data_processing")
from mrhyde_test_support import *
from parse_log import Results, error_norms

its = mrhyde_test_support('''Maxwell absorbing boundary: with refractive index 2, a plane pulse
must leave through the ABC, and a box without it must keep its energy.''')
its.opts.verbose = True

#TESTING active
#TESTING -n 1
#TESTING -k regression,Maxwells,HCURL,HDIV,absorbing-boundary

R2, C02 = 4.0, 1.0   # r^2 and c_0^2 in the decks


def energy(deck, log):
    """W(T)/W(0) with W = r^2 ||E||^2 + c_0^2 ||B||^2, or None if the run stopped early."""
    tend = float(re.search(r"final time:\s*(\S+)", open(deck).read()).group(1))
    err = error_norms(open(log).read().splitlines())
    if "B" not in err or abs(err["B"][-1][0] - tend) > 1e-6 * tend:
        return None
    w = [R2 * e**2 + C02 * b**2 for (_, e), (_, b) in zip(err["E"], err["B"])]
    return w[-1] / w[0]


res = Results()
status = 0
try:
    for deck, log in (('input_abc.yaml', 'mrhyde_abc.log'), ('input_pmc.yaml', 'mrhyde_pmc.log')):
        status += its.call('cp %s input.yaml' % deck)
        status += its.call('mpiexec -n 1 ../../mrhyde >& %s' % log)
finally:
    if os.path.exists('input.yaml'):
        os.remove('input.yaml')

abc = energy('input_abc.yaml', 'mrhyde_abc.log')
res.add(abc is not None and abc < 1e-3, "ABC absorbs the pulse",
        "run did not finish" if abc is None else "energy left %.2e of initial, limit 1e-3" % abc)
pmc = energy('input_pmc.yaml', 'mrhyde_pmc.log')
res.add(pmc is not None and abs(pmc - 1.0) < 1e-3, "PMC box keeps the energy",
        "run did not finish" if pmc is None else "energy %.6f of initial, limit 1 +/- 1e-3" % pmc)

sys.exit(status + res.write())
