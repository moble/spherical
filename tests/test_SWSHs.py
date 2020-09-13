#!/usr/bin/env python

# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

# Try to keep imports to a minimum; from the standard library as much
# as possible.  We have to conda install all dependencies, and it's
# not right to make Travis do too much work.
import math
import cmath
import numpy as np
import quaternionic
import spherical as sf
import pytest

slow = pytest.mark.slow

precision_SWSH = 2.e-15


def slow_Wignerd(iota, ell, m, s):
    # Eq. II.8 of Ajith et al. (2007) 'Data formats...'
    k_min = max(0, m - s)
    k_max = min(ell + m, ell - s)
    return sum([((-1.) ** (k) * math.sqrt(
        math.factorial(ell + m) * math.factorial(ell - m) * math.factorial(ell + s) * math.factorial(ell - s))
                 * math.cos(iota / 2.) ** (2 * ell + m - s - 2 * k) * math.sin(iota / 2.) ** (2 * k + s - m)
                 / float(
        math.factorial(ell + m - k) * math.factorial(ell - s - k) * math.factorial(k) * math.factorial(k + s - m)))
                for k in range(k_min, k_max + 1)])


def slow_sYlm(s, ell, m, iota, phi):
    # Eq. II.7 of Ajith et al. (2007) 'Data formats...'
    # Note the weird definition w.r.t. `-s`
    return (-1.) ** (-s) * math.sqrt((2 * ell + 1) / (4 * np.pi)) * slow_Wignerd(iota, ell, m, -s) * cmath.exp(
        1j * m * phi)


## This is just to test my implementation of the equations give in the paper
def test_SWSH_NINJA_values(special_angles, ell_max):
    def m2Y22(iota, phi):
        return math.sqrt(5 / (64 * np.pi)) * (1 + math.cos(iota)) ** 2 * cmath.exp(2j * phi)

    def m2Y21(iota, phi):
        return math.sqrt(5 / (16 * np.pi)) * math.sin(iota) * (1 + math.cos(iota)) * cmath.exp(1j * phi)

    def m2Y20(iota, phi):
        return math.sqrt(15 / (32 * np.pi)) * math.sin(iota) ** 2

    def m2Y2m1(iota, phi):
        return math.sqrt(5 / (16 * np.pi)) * math.sin(iota) * (1 - math.cos(iota)) * cmath.exp(-1j * phi)

    def m2Y2m2(iota, phi):
        return math.sqrt(5 / (64 * np.pi)) * (1 - math.cos(iota)) ** 2 * cmath.exp(-2j * phi)

    for iota in special_angles:
        for phi in special_angles:
            assert abs(slow_sYlm(-2, 2, 2, iota, phi) - m2Y22(iota, phi)) < ell_max * precision_SWSH
            assert abs(slow_sYlm(-2, 2, 1, iota, phi) - m2Y21(iota, phi)) < ell_max * precision_SWSH
            assert abs(slow_sYlm(-2, 2, 0, iota, phi) - m2Y20(iota, phi)) < ell_max * precision_SWSH
            assert abs(slow_sYlm(-2, 2, -1, iota, phi) - m2Y2m1(iota, phi)) < ell_max * precision_SWSH
            assert abs(slow_sYlm(-2, 2, -2, iota, phi) - m2Y2m2(iota, phi)) < ell_max * precision_SWSH


@slow
def test_SWSH_values(special_angles, ell_max):
    for iota in special_angles:
        for phi in special_angles:
            for ell in range(ell_max + 1):
                for s in range(-ell, ell + 1):
                    for m in range(-ell, ell + 1):
                        R = quaternionic.array.from_euler_angles(phi, iota, 0)
                        assert abs(sf.SWSH(R, s, np.array([[ell, m]]))
                                   - slow_sYlm(s, ell, m, iota, phi)) < ell_max ** 6 * precision_SWSH


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


def test_SWSH_grid(special_angles, ell_max):
    LM = sf.LM_range(0, ell_max)

    # Test flat array arrangement
    R_grid = np.array([quaternionic.array.from_euler_angles(alpha, beta, gamma).normalized
                       for alpha in special_angles
                       for beta in special_angles
                       for gamma in special_angles])
    for s in range(-ell_max + 1, ell_max):
        values_explicit = np.array([sf.SWSH(R, s, LM) for R in R_grid])
        values_grid = sf.SWSH_grid(R_grid, s, ell_max)
        assert np.array_equal(values_explicit, values_grid)

    # Test nested array arrangement
    R_grid = np.array([[[quaternionic.array.from_euler_angles(alpha, beta, gamma)
                         for alpha in special_angles]
                        for beta in special_angles]
                       for gamma in special_angles])
    for s in range(-ell_max + 1, ell_max):
        values_explicit = np.array([[[sf.SWSH(R, s, LM) for R in R1] for R1 in R2] for R2 in R_grid])
        values_grid = sf.SWSH_grid(R_grid, s, ell_max)
        assert np.array_equal(values_explicit, values_grid)


def test_SWSH_signatures(Rs):
    """There are two ways to call the SWSH function: with an array of Rs, or with an array of (ell,m) values.  This
    test ensures that the results are the same in both cases."""
    s_max = 5
    ss = np.arange(-s_max, s_max+1)
    ell_max = 5
    ell_ms = sf.LM_range(0, ell_max)
    SWSHs1 = np.empty((Rs.size//4, ss.size, ell_ms.size//2), dtype=np.complex)
    SWSHs2 = np.empty_like(SWSHs1)
    for i_s, s in enumerate(ss):
        for i_ellm, (ell, m) in enumerate(ell_ms):
            SWSHs1[:, i_s, i_ellm] = sf.SWSH(Rs, s, [ell, m])
    for i_s, s in enumerate(ss):
        for i_R, R in enumerate(Rs):
            SWSHs2[i_R, i_s, :] = sf.SWSH(R, s, ell_ms)
    assert np.array_equal(SWSHs1, SWSHs2)
