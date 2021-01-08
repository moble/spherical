#!/usr/bin/env python

# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import math
import cmath
import numpy as np
import quaternionic
import spherical as sf
import pytest

slow = pytest.mark.slow
precision_SWSH = 2.e-15


@slow
def test_SWSH_grid(special_angles, ell_max_slow):
    ell_max = ell_max_slow
    LM = sf.LM_range(0, ell_max)

    # Test flat array arrangement
    R_grid = np.array([quaternionic.array.from_euler_angles(alpha, beta, gamma).normalized
                       for alpha in special_angles
                       for beta in special_angles
                       for gamma in special_angles])
    for s in range(-ell_max + 1, ell_max):
        # values_explicit = np.array([sf.SWSH(R, s, LM) for R in R_grid])
        values_explicit = sf.SWSH(R_grid, s, LM)
        values_grid = sf.SWSH_grid(R_grid, s, ell_max)
        assert np.array_equal(values_explicit, values_grid)

    # Test nested array arrangement
    R_grid = np.array([[[quaternionic.array.from_euler_angles(alpha, beta, gamma)
                         for alpha in special_angles]
                        for beta in special_angles]
                       for gamma in special_angles])
    for s in range(-ell_max + 1, ell_max):
        # values_explicit = np.array([[[sf.SWSH(R, s, LM) for R in R1] for R1 in R2] for R2 in R_grid])
        values_explicit = sf.SWSH(R_grid, s, LM)
        values_grid = sf.SWSH_grid(R_grid, s, ell_max)
        assert np.array_equal(values_explicit, values_grid)


def test_SWSH_signatures(Rs):
    """There are two ways to call the SWSH function: with an array of Rs, or with an array of (ell,m) values.  This
    test ensures that the results are the same in both cases."""
    s_max = 5
    ss = np.arange(-s_max, s_max+1)
    ell_max = 6
    ell_ms = sf.LM_range(0, ell_max)
    SWSHs1 = np.zeros((Rs.size//4, ss.size, ell_ms.size//2), dtype=np.complex)
    SWSHs2 = np.zeros_like(SWSHs1)
    for i_s, s in enumerate(ss):
        for i_ellm, (ell, m) in enumerate(ell_ms):
            if ell >= abs(s):
                value = sf.SWSH(Rs, s, [ell, m])
            else:
                value = 0.0j
            SWSHs1[:, i_s, i_ellm] = value
    for i_s, s in enumerate(ss):
        for i_R, R in enumerate(Rs):
            SWSHs2[i_R, i_s, :] = sf.SWSH(R, s, ell_ms)
    assert np.array_equal(SWSHs1, SWSHs2)
