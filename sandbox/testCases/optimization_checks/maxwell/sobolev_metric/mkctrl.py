#!/usr/bin/env python3
"""Random starting control, one MatrixMarket file per time slab.

Usage: mkctrl.py <dir> <n_spatial> <n_slabs> <seed>
"""

import os
import random
import sys

d, nsp, nslabs, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
os.makedirs(d, exist_ok=True)
rng = random.Random(seed)
for s in range(nslabs):
    with open(os.path.join(d, "v.field.%d.mm" % s), "w") as f:
        f.write("%%MatrixMarket matrix array real general\n")
        f.write("%d 1\n" % nsp)
        for _ in range(nsp):
            f.write("%.17e\n" % (1.0e-3 * rng.gauss(0.0, 1.0)))
print("wrote %d control files of %d entries into %s" % (nslabs, nsp, d))
