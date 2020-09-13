#!/usr/bin/env python

# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

# Try to keep imports to a minimum; from the standard library as much
# as possible.  We have to conda install all dependencies, and it's
# not right to make Travis do too much work.
import spherical as sf
import pytest

precision_Wigner3j = 1.e-15

try:
    import sympy
    sympy_not_present = False
except ImportError:
    sympy_not_present = True

requires_sympy = pytest.mark.skipif(sympy_not_present, reason="Requires SymPy to be importable")


def test_Wigner3j_properties():
    assert abs(sf.Wigner3j(2, 6, 4, 0, 0, 0) - 0.1869893980016914) < precision_Wigner3j
    ## The following test various symmetries and other properties of
    ## the Wigner 3-j symbols
    j_max = 8
    for j1 in range(j_max+1):
        for j2 in range(j_max+1):
            for j3 in range(j_max+1):
                # Selection rule
                if ((j1 + j2 + j3) % 2 != 0):
                    assert abs(sf.Wigner3j(j1, j2, j3, 0, 0, 0)) < precision_Wigner3j
                for m1 in range(-j1, j1 + 1):
                    for m2 in range(-j2, j2 + 1):
                        # Selection rule
                        if abs(j1 - j2) > j3 or j1 + j2 < j3:
                            assert abs(sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)) < precision_Wigner3j
                        # Test even permutations
                        assert abs(sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                                   - sf.Wigner3j(j2, j3, j1, m2, -m1 - m2, m1)) < precision_Wigner3j
                        assert abs(sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                                   - sf.Wigner3j(j2, j3, j1, m2, -m1 - m2, m1)) < precision_Wigner3j
                        assert abs(sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                                   - sf.Wigner3j(j3, j1, j2, -m1 - m2, m1, m2)) < precision_Wigner3j
                        # Test odd permutations
                        assert abs(
                            sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                            - (-1.) ** (j1 + j2 + j3) * sf.Wigner3j(j2, j1, j3, m2, m1, -m1 - m2)) < precision_Wigner3j
                        assert abs(
                            sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                            - (-1.) ** (j1 + j2 + j3) * sf.Wigner3j(j1, j3, j2, m1, -m1 - m2, m2)) < precision_Wigner3j
                        # Test sign change
                        assert abs(
                            sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                            - (-1.) ** (j1 + j2 + j3) * sf.Wigner3j(j1, j2, j3, -m1, -m2, m1 + m2)) < precision_Wigner3j
                        # Regge symmetries (skip for non-integer values)
                        if ((j2 + j3 - m1) % 2 == 0):
                            assert abs(sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                                       - sf.Wigner3j(j1, (j2 + j3 - m1) // 2, (j2 + j3 + m1) // 2, j3 - j2,
                                                     (j2 - j3 - m1) // 2 + m1 + m2,
                                                     (j2 - j3 + m1) // 2 - m1 - m2)) < precision_Wigner3j
                        if ((j2 + j3 - m1) % 2 == 0):
                            assert abs(sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                                       - sf.Wigner3j(j1, (j2 + j3 - m1) // 2, (j2 + j3 + m1) // 2, j3 - j2,
                                                     (j2 - j3 - m1) // 2 + m1 + m2,
                                                     (j2 - j3 + m1) // 2 - m1 - m2)) < precision_Wigner3j
                        if ((j2 + j3 + m1) % 2 == 0 and (j1 + j3 + m2) % 2 == 0 and (j1 + j2 - m1 - m2) % 2 == 0):
                            assert (abs(sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                                        - (-1.) ** (j1 + j2 + j3) * sf.Wigner3j((j2 + j3 + m1) // 2,
                                                                               (j1 + j3 + m2) // 2,
                                                                               (j1 + j2 - m1 - m2) // 2,
                                                                               j1 - (j2 + j3 - m1) // 2,
                                                                               j2 - (j1 + j3 - m2) // 2,
                                                                               j3 - (j1 + j2 + m1 + m2) // 2))
                                    < precision_Wigner3j)


@requires_sympy
def test_Wigner3j_values():
    from sympy import N
    from sympy.physics.wigner import wigner_3j
    from spherical import Wigner3j
    j_max = 8
    for j1 in range(j_max+1):
        for j2 in range(j1, j_max+1):
            for j3 in range(j2, j_max+1):
                for m1 in range(-j1, j1 + 1):
                    for m2 in range(-j2, j2 + 1):
                        m3 = -m1-m2
                        if j3 >= abs(m3):
                            sf_3j = Wigner3j(j1, j2, j3, m1, m2, m3)
                            sy_3j = N(wigner_3j(j1, j2, j3, m1, m2, m3))
                            assert abs(sf_3j - sy_3j) < precision_Wigner3j
