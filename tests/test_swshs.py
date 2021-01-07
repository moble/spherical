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


def test_SWSH_WignerD_expression(special_angles, ell_max):
    for iota in special_angles:
        for phi in special_angles:
            for ell in range(ell_max + 1):
                for s in range(-ell, ell + 1):
                    R = quaternionic.array.from_euler_angles(phi, iota, 0)
                    LM = np.array([[ell, m] for m in range(-ell, ell + 1)])
                    Y = sf.SWSH(R, s, LM)
                    LMS = np.array([[ell, m, -s] for m in range(-ell, ell + 1)])
                    Rs = R.two_spinor
                    D = (-1.) ** (s) * math.sqrt((2 * ell + 1) / (4 * np.pi)) * sf.Wigner_D_element(Rs.a, Rs.b, LMS)
                    assert np.allclose(Y, D, atol=ell ** 6 * precision_SWSH, rtol=ell ** 6 * precision_SWSH)


@slow
def test_SWSH_spin_behavior(Rs, special_angles, ell_max):
    # We expect that the SWSHs behave according to
    # sYlm( R * exp(gamma*z/2) ) = sYlm(R) * exp(-1j*s*gamma)
    # See http://moble.github.io/spherical/SWSHs.html#fn:2
    # for a more detailed explanation
    # print("")
    for i, R in enumerate(Rs):
        # print("\t{0} of {1}: R = {2}".format(i, len(Rs), R))
        for gamma in special_angles:
            for ell in range(ell_max + 1):
                for s in range(-ell, ell + 1):
                    LM = np.array([[ell, m] for m in range(-ell, ell + 1)])
                    Rgamma = R * quaternionic.array(math.cos(gamma / 2.), 0, 0, math.sin(gamma / 2.))
                    sYlm1 = sf.SWSH(Rgamma, s, LM)
                    sYlm2 = sf.SWSH(R, s, LM) * cmath.exp(-1j * s * gamma)
                    # print(R, gamma, ell, s, np.max(np.abs(sYlm1-sYlm2)))
                    assert np.allclose(sYlm1, sYlm2, atol=ell ** 6 * precision_SWSH, rtol=ell ** 6 * precision_SWSH)


@slow
def test_SWSH_conjugation(special_angles, ell_max):
    # {s}Y{l,m}.conjugate() = (-1.)**(s+m) {-s}Y{l,-m}
    indices1 = sf.LM_range(0, ell_max)
    indices2 = np.array([[ell, -m] for ell, m in indices1])
    neg1_to_m = np.array([(-1.)**m for ell, m in indices1])
    for iota in special_angles:
        for phi in special_angles:
            R = quaternionic.array.from_spherical_coordinates(iota, phi)
            for s in range(1-ell_max, ell_max):
                assert np.allclose(sf.SWSH(R, s, indices1),
                                   (-1.)**s * neg1_to_m * np.conjugate(sf.SWSH(R, -s, indices2)),
                                   atol=1e-15, rtol=1e-15)


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
    SWSHs1 = np.empty((Rs.size//4, ss.size, ell_ms.size//2), dtype=np.complex)
    SWSHs2 = np.empty_like(SWSHs1)
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
