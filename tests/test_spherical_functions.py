#!/usr/bin/env python

# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

# Try to keep imports to a minimum; from the standard library as much
# as possible.  We have to conda install all dependencies, and it's
# not right to make Travis do too much work.
import math
import numpy as np
import quaternionic
import spherical as sf

import numba  # This is to check to make sure we're actually using numba


def test_constant_as_ell_0_mode(special_angles):
    indices = np.array([[0, 0]])
    np.random.seed(123)
    for imaginary_part in [0.0, 1.0j]:  # Test both real and imaginary constants
        for rep in range(1000):
            constant = np.random.uniform(-1, 1) + imaginary_part * np.random.uniform(-1, 1)
            const_ell_m = sf.constant_as_ell_0_mode(constant)
            assert abs(constant - sf.constant_from_ell_0_mode(const_ell_m)) < 1e-15
            for theta in special_angles:
                for phi in special_angles:
                    dot = np.dot(const_ell_m, sf.SWSH(quaternionic.array.from_spherical_coordinates(theta, phi), 0, indices))
                    assert abs(constant - dot) < 1e-15


def test_vector_as_ell_1_modes(special_angles):
    indices = np.array([[1, -1], [1, 0], [1, 1]])

    def nhat(theta, phi):
        return np.array([math.sin(theta) * math.cos(phi),
                         math.sin(theta) * math.sin(phi),
                         math.cos(theta)])

    np.random.seed(123)
    for rep in range(1000):
        vector = np.random.uniform(-1, 1, size=(3,))
        vec_ell_m = sf.vector_as_ell_1_modes(vector)
        assert np.allclose(vector, sf.vector_from_ell_1_modes(vec_ell_m), atol=1.0e-16, rtol=1.0e-15)
        for theta in special_angles:
            for phi in special_angles:
                dot1 = np.dot(vector, nhat(theta, phi))
                dot2 = np.dot(vec_ell_m, sf.SWSH(quaternionic.array.from_spherical_coordinates(theta, phi), 0, indices)).real
                assert abs(dot1 - dot2) < 1e-15


def test_finite_constant_arrays():
    assert np.all(np.isfinite(sf.factorials))
    assert np.all(np.isfinite(sf._binomial_coefficients))
    assert np.all(np.isfinite(sf._ladder_operator_coefficients))
    assert np.all(np.isfinite(sf._Wigner_coefficients))


def nCk(n, k):
    """Simple binomial function, so we don't have to import anything"""
    from operator import mul  # or mul=lambda x,y:x*y
    from fractions import Fraction
    from functools import reduce

    return int(reduce(mul, (Fraction(n - i, i + 1) for i in range(k)), 1))


def test_factorials():
    for i in range(len(sf.factorials)):
        assert sf.factorial(i) == sf.factorials[i]
        assert float(math.factorial(i)) == sf.factorial(i)


def test_binomial_coefficients():
    for n in range(2 * sf.ell_max + 1):
        for k in range(n + 1):
            a = nCk(n, k)
            b = sf.binomial_coefficient(n, k)
            assert abs(a - b) / (abs(a) + abs(b)) < 3.e-14


def test_ladder_operator_coefficient():
    # for ell in range(sf.ell_max + 1):
    #     for m in range(-ell, ell + 1):
    #         a = math.sqrt(ell * (ell + 1) - m * (m + 1))
    #         b = sf.ladder_operator_coefficient(ell, m)
    #         if (m == ell):
    #             assert b == 0.0
    #         else:
    #             assert abs(a - b) / (abs(a) + abs(b)) < 3e-16
    for twoell in range(2*sf.ell_max + 1):
        for twom in range(-twoell, twoell + 1, 2):
            a = math.sqrt(twoell * (twoell + 2) - twom * (twom + 2))/2
            b = sf._ladder_operator_coefficient(twoell, twom)
            c = sf.ladder_operator_coefficient(twoell/2, twom/2)
            if (twom == twoell):
                assert b == 0.0 and c == 0.0
            else:
                assert abs(a - b) / (abs(a) + abs(b)) < 3e-16 and abs(a - c) / (abs(a) + abs(c)) < 3e-16

def test_Wigner_coefficient():
    import mpmath
    mpmath.mp.dps = 4 * sf.ell_max
    i = 0
    for twoell in range(2*sf.ell_max + 1):
        for twomp in range(-twoell, twoell + 1, 2):
            for twom in range(-twoell, twoell + 1, 2):
                tworho_min = max(0, twomp - twom)
                a = sf._Wigner_coefficient(twoell, twomp, twom)
                b = float(mpmath.sqrt(mpmath.fac((twoell + twom)//2) * mpmath.fac((twoell - twom)//2)
                                      / (mpmath.fac((twoell + twomp)//2) * mpmath.fac((twoell - twomp)//2)))
                          * mpmath.binomial((twoell + twomp)//2, tworho_min//2)
                          * mpmath.binomial((twoell - twomp)//2, (twoell - twom - tworho_min)//2))
                assert np.allclose(a, b), (twoell, twomp, twom, i, sf._Wigner_index(twoell, twomp, twom))
                i += 1



def test_LM_range(ell_max):
    for l_max in range(ell_max + 1):
        for l_min in range(l_max + 1):
            assert np.array_equal(sf.LM_range(l_min, l_max),
                                  np.array([[ell, m] for ell in range(l_min, l_max + 1) for m in range(-ell, ell + 1)]))


def test_LM_index(ell_max):
    for ell_min in range(ell_max + 1):
        LM = sf.LM_range(ell_min, ell_max)
        for ell in range(ell_min, ell_max + 1):
            for m in range(-ell, ell + 1):
                assert np.array_equal(np.array([ell, m]), LM[sf.LM_index(ell, m, ell_min)])


def test_LM_total_size(ell_max):
    for l_min in range(ell_max + 1):
        for l_max in range(l_min, ell_max + 1):
            assert sf.LM_index(l_max + 1, -(l_max + 1), l_min) == sf.LM_total_size(l_min, l_max)


def test_LMpM_range(ell_max):
    for l_max in range(ell_max + 1):
        assert np.array_equal(sf.LMpM_range(l_max, l_max),
                              np.array([[l_max, mp, m]
                                        for mp in range(-l_max, l_max + 1)
                                        for m in range(-l_max, l_max + 1)]))
        for l_min in range(l_max + 1):
            assert np.array_equal(sf.LMpM_range(l_min, l_max),
                                  np.array([[ell, mp, m]
                                            for ell in range(l_min, l_max + 1)
                                            for mp in range(-ell, ell + 1)
                                            for m in range(-ell, ell + 1)]))


def test_LMpM_range_half_integer(ell_max):
    for twoell_max in range(2*ell_max + 1):
        assert np.array_equal(sf.LMpM_range_half_integer(twoell_max/2, twoell_max/2),
                              np.array([[twoell_max/2, twomp/2, twom/2]
                                        for twomp in range(-twoell_max, twoell_max + 1, 2)
                                        for twom in range(-twoell_max, twoell_max + 1, 2)]))
        # for twoell_min in range(twoell_max):
        for twoell_min in range(min(10, twoell_max)):
            a = sf.LMpM_range_half_integer(twoell_min/2, twoell_max/2)
            b = np.array([[twoell/2, twomp/2, twom/2]
                          for twoell in range(twoell_min, twoell_max + 1)
                          for twomp in range(-twoell, twoell + 1, 2)
                          for twom in range(-twoell, twoell + 1, 2)])
            assert np.array_equal(a, b)


def test_LMpM_index(ell_max):
    for ell_min in range(ell_max + 1):
        LMpM = sf.LMpM_range(ell_min, ell_max)
        for ell in range(ell_min, ell_max + 1):
            for mp in range(-ell, ell + 1):
                for m in range(-ell, ell + 1):
                    assert np.array_equal(np.array([ell, mp, m]), LMpM[sf.LMpM_index(ell, mp, m, ell_min)])


def test_LMpM_total_size(ell_max):
    for l_min in range(ell_max + 1):
        for l_max in range(l_min, ell_max + 1):
            assert sf.LMpM_index(l_max + 1, -(l_max + 1), -(l_max + 1), l_min) == sf.LMpM_total_size(l_min, l_max)

