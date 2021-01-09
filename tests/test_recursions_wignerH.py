#!/usr/bin/env python

# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import numpy as np
from spherical.recursions.wignerH import *


def test_sign():
    for x in np.arange(-20, 21):
        if x == 0:
            assert sign(x) == 1
        else:
            assert sign(x) == np.sign(x)


def test_nm_index():
    n_max = 17
    indices = [[n, m] for n in range(n_max+1) for m in range(-n, n+1)]
    for n in range(n_max+1):
        for m in range(-n, n+1):
            assert [n, m] == indices[nm_index(n, m)]


def test_nabsm_index():
    n_max = 17
    indices = [[n, m] for n in range(n_max+1) for m in range(n+1)]
    for n in range(n_max+1):
        for m in range(n+1):
            assert [n, m] == indices[nabsm_index(n, m)], (n, m, nabsm_index(n, m), indices)


def test_nmpm_index():
    n_max = 17
    indices = [[n, mp, m] for n in range(n_max+1) for mp in range(-n, n+1) for m in range(-n, n+1)]
    for n in range(n_max+1):
        for mp in range(-n, n+1):
            for m in range(-n, n+1):
                assert [n, mp, m] == indices[nmpm_index(n, mp, m)]
