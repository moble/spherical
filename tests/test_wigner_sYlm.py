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


def slow_Wignerd(iota, ell, m, s):
    # Eq. II.8 of Ajith et al. (2007) 'Data formats...'
    k_min = max(0, m - s)
    k_max = min(ell + m, ell - s)
    return sum([
        ((-1.) ** (k)
         * math.cos(iota / 2.) ** (2 * ell + m - s - 2 * k)
         * math.sin(iota / 2.) ** (2 * k + s - m)
         * math.sqrt(
             math.factorial(ell + m)
             * math.factorial(ell - m)
             * math.factorial(ell + s)
             * math.factorial(ell - s)
         )
         / float(
             math.factorial(ell + m - k)
             * math.factorial(ell - s - k)
             * math.factorial(k)
             * math.factorial(k + s - m))
         )
        for k in range(k_min, k_max + 1)
    ])


def slow_sYlm(s, ell, m, iota, phi):
    # Eq. II.7 of Ajith et al. (2007) 'Data formats...'
    # Note the weird definition w.r.t. `-s`
    if abs(s) > ell or abs(m) > ell:
        return 0j
    return (
        (-1.) ** (-s)
        * math.sqrt((2 * ell + 1) / (4 * np.pi))
        * slow_Wignerd(iota, ell, m, -s)
        * cmath.exp(1j * m * phi)
    )


## This is just to test my implementation of the equations give in the paper.
## Note that this is a test of the testing code itself, not of the main code.
def test_NINJA_consistency(special_angles, ell_max):
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
def test_sYlm_NINJA_expressions(special_angles, ell_max_slow, eps):
    ϵ = 5 * ell_max_slow**6 * eps  # This is mostly due to the expressions above being inaccurate
    wigner = sf.Wigner(ell_max_slow)
    for iota in special_angles:
        for phi in special_angles:
            R = quaternionic.array.from_euler_angles(phi, iota, 0)
            Y1 = np.array([
                [
                    slow_sYlm(s, ell, m, iota, phi)
                    for ell in range(ell_max_slow + 1)
                    for m in range(-ell, ell + 1)
                ]
                for s in range(-ell_max_slow, ell_max_slow + 1)
            ])
            Y2 = np.array([wigner.sYlm(s, R) for s in range(-ell_max_slow, ell_max_slow + 1)])
            assert np.allclose(Y1, Y2, rtol=ϵ, atol=ϵ)


@slow
def test_sYlm_WignerD_expression(special_angles, ell_max_slow, eps):
    # ₛYₗₘ(R) = (-1)ˢ √((2ℓ+1)/(4π)) 𝔇ˡₘ₋ₛ(R)
    #        = (-1)ˢ √((2ℓ+1)/(4π)) 𝔇̄ˡ₋ₛₘ(R̄)

    ϵ = 2 * ell_max_slow * eps
    wigner = sf.Wigner(ell_max_slow)
    for iota in special_angles:
        for phi in special_angles:
            R = quaternionic.array.from_euler_angles(phi, iota, 0)
            D_R̄ = wigner.D(R.conjugate())
            for s in range(-ell_max_slow, ell_max_slow + 1):
                Y = wigner.sYlm(s, R)
                for ell in range(abs(s), ell_max_slow + 1):
                    Y_ℓ = Y[sf.Yindex(ell, -ell):sf.Yindex(ell, ell)+1]
                    Y_D = (
                        (-1.) ** (s) * math.sqrt((2 * ell + 1) / (4 * np.pi))
                        * D_R̄[sf.WignerDindex(ell, -s, -ell):sf.WignerDindex(ell, -s, ell)+1].conjugate()
                    )
                    assert np.allclose(Y_ℓ, Y_D, atol=ϵ, rtol=ϵ)