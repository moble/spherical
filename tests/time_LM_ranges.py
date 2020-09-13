#! /usr/bin/env ipython

# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

try:
    from IPython import get_ipython
    ipython = get_ipython()
except AttributeError:
    print("This script must be run with `ipython`")
    raise

import sys
import numpy as np
import random
import quaternionic
import numba as nb
import spherical as sf

ru = lambda: random.uniform(-1, 1)

ells = range(16 + 1) + [24, 32]
evals = np.empty_like(ells, dtype=int)
picoseconds = np.empty_like(ells, dtype=float)
ell_min = 0
for i, ell_max in enumerate(ells):
    evals[i] = (ell_max * (11 + ell_max * (12 + 4 * ell_max)) + ell_min * (1 - 4 * ell_min ** 2) + 3) // 3
    LMpM = np.empty((evals[i], 3), dtype=int)
    result = ipython.magic("timeit -o global LMpM; LMpM = sf.LMpM_range(ell_min, ell_max)")
    picoseconds[i] = 1e12 * result.best / evals[i]
    print("With ell_max={0}, and {1} evaluations, each LMpM averages {2:.0f} ps".format(ell_max, evals[i],
                                                                                        picoseconds[i]))
    result = ipython.magic(
        "timeit -o global LMpM; LMpM = np.array([[ell,mp,m] for ell in range(ell_min,ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])")
    picoseconds[i] = 1e12 * result.best / evals[i]
    print("\tWith ell_max={0}, and {1} evaluations, each LMpM averages {2:.0f} ps".format(ell_max, evals[i],
                                                                                          picoseconds[i]))
    sys.stdout.flush()
