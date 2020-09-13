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

precision_Wigner_D_element = 4.e-14


def test_Wigner_D_linear_indices(ell_max):
    for l_min in range(ell_max):
        for l_max in range(l_min + 1, ell_max + 1):
            LMpM = sf.LMpM_range(l_min, l_max)

            assert len(LMpM) == sf._total_size_D_matrices(l_min, l_max)

            for ell in range(l_min, l_max + 1):
                assert list(LMpM[sf._linear_matrix_offset(ell, l_min)]) == [ell, -ell, -ell]

            for ell in range(l_min, l_max + 1):
                for mp in range(-ell, ell + 1):
                    i = sf._linear_matrix_diagonal_index(ell, mp)
                    o = sf._linear_matrix_offset(ell, l_min)
                    assert list(LMpM[i + o]) == [ell, mp, mp]
                    for m in range(-ell, ell + 1):
                        i = sf._linear_matrix_index(ell, mp, m)
                        o = sf._linear_matrix_offset(ell, l_min)
                        assert list(LMpM[i + o]) == [ell, mp, m]


def test_Wigner_D_matrices_negative_argument(Rs, ell_max):
    # For integer ell, D(R)=D(-R)
    LMpM = sf.LMpM_range(0, ell_max)
    a = np.empty((LMpM.shape[0],), dtype=complex)
    b = np.empty((LMpM.shape[0],), dtype=complex)
    for R in Rs:
        R2s = R.two_spinor
        sf._Wigner_D_matrices(R2s.a, R2s.b, 0, ell_max, a)
        sf._Wigner_D_matrices(-R2s.a, -R2s.b, 0, ell_max, b)
        assert np.allclose(a, b, rtol=ell_max * precision_Wigner_D_element)


@slow
def test_Wigner_D_elements_representation_property(Rs, ell_max):
    # Test the representation property for special and random angles
    # Try half-integers, too
    ell_max = min(8, ell_max)
    twoLMpM = np.round(2*sf.LMpM_range_half_integer(0, ell_max)).astype(int)
    print("")
    D1 = np.empty((twoLMpM.shape[0],), dtype=complex)
    D2 = np.empty((twoLMpM.shape[0],), dtype=complex)
    D12 = np.empty((twoLMpM.shape[0],), dtype=complex)
    for i, R1 in enumerate(Rs):
        print("\t{0} of {1}: R1 = {2}".format(i+1, len(Rs), R1))
        for j, R2 in enumerate(Rs):
            # print("\t\t{0} of {1}: R2 = {2}".format(j+1, len(Rs), R2))
            R12 = R1 * R2
            R1s = R1.two_spinor
            R2s = R2.two_spinor
            R12s = R12.two_spinor
            sf._Wigner_D_element(R1s.a, R1s.b, twoLMpM, D1)
            sf._Wigner_D_element(R2s.a, R2s.b, twoLMpM, D2)
            sf._Wigner_D_element(R12s.a, R12s.b, twoLMpM, D12)
            M12 = np.array([np.sum([D1[sf._Wigner_index(twoell, twomp, twompp)]
                                    * D2[sf._Wigner_index(twoell, twompp, twom)]
                                    for twompp in range(-twoell, twoell + 1, 2)])
                            for twoell in range(2*ell_max + 1)
                            for twomp in range(-twoell, twoell + 1, 2)
                            for twom in range(-twoell, twoell + 1, 2)])
            # if not np.allclose(M12, D12, atol=ell_max * precision_Wigner_D_element):
            #     for k in range(min(100, M12.size)):
            #         print(twoLMpM[k], "\t", abs(D12[k]-M12[k]), "\t\t", D12[k], "\t", M12[k])
            #     print(D12.shape, M12.shape)
            assert np.allclose(M12, D12, atol=ell_max * precision_Wigner_D_element)


@slow
def test_Wigner_D_matrices_representation_property(Rs, ell_max):
    # Test the representation property for special and random angles
    # Can't try half-integers because Wigner_D_matrices doesn't accept them
    ell_max = min(8, ell_max)
    LMpM = sf.LMpM_range(0, ell_max)
    print("")
    D1 = np.empty((LMpM.shape[0],), dtype=complex)
    D2 = np.empty((LMpM.shape[0],), dtype=complex)
    D12 = np.empty((LMpM.shape[0],), dtype=complex)
    for i, R1 in enumerate(Rs):
        print("\t{0} of {1}: R1 = {2}".format(i+1, len(Rs), R1))
        for j, R2 in enumerate(Rs[i:]):
            # print("\t\t{0} of {1}: R2 = {2}".format(j+1, len(Rs), R2))
            R12 = R1 * R2
            R1s = R1.two_spinor
            R2s = R2.two_spinor
            R12s = R12.two_spinor
            sf._Wigner_D_matrices(R1s.a, R1s.b, 0, ell_max, D1)
            sf._Wigner_D_matrices(R2s.a, R2s.b, 0, ell_max, D2)
            sf._Wigner_D_matrices(R12s.a, R12s.b, 0, ell_max, D12)
            M12 = np.array([np.sum([D1[sf.LMpM_index(ell, mp, mpp, 0)] * D2[sf.LMpM_index(ell, mpp, m, 0)]
                                    for mpp in range(-ell, ell + 1)])
                            for ell in range(ell_max + 1)
                            for mp in range(-ell, ell + 1)
                            for m in range(-ell, ell + 1)])
            assert np.allclose(M12, D12, atol=ell_max * precision_Wigner_D_element)


@slow
def test_Wigner_D_matrix_inverse(Rs, ell_max):
    # Ensure that the matrix of the inverse rotation is the inverse of
    # the matrix of the rotation
    print()
    for i, R in enumerate(Rs):
        print("\t{0} of {1}: {2}".format(i+1, len(Rs), R))
        for twoell in range(2*ell_max + 1):
            LMpM = np.array([[twoell/2, twomp/2, twom/2]
                             for twomp in range(-twoell, twoell+1, 2)
                             for twom in range(-twoell, twoell+1, 2)])
            Rspinor = R.two_spinor
            D1 = sf.Wigner_D_element(Rspinor.a, Rspinor.b, LMpM)
            D2 = sf.Wigner_D_element(Rspinor.a.conjugate(), -Rspinor.b, LMpM)
            D1 = D1.reshape((twoell + 1, twoell + 1))
            D2 = D2.reshape((twoell + 1, twoell + 1))
            assert np.allclose(D1.dot(D2), np.identity(twoell + 1),
                               atol=ell_max ** 4 * precision_Wigner_D_element,
                               rtol=ell_max ** 4 * precision_Wigner_D_element)


@slow
def test_Wigner_D_element_symmetries(Rs, ell_max):
    LMpM = sf.LMpM_range(0, ell_max)
    # D_{mp,m}(R) = (-1)^{mp+m} \bar{D}_{-mp,-m}(R)
    MpPM = np.array([mp + m for ell in range(ell_max + 1) for mp in range(-ell, ell + 1) for m in range(-ell, ell + 1)])
    LmMpmM = np.array(
        [[ell, -mp, -m] for ell in range(ell_max + 1) for mp in range(-ell, ell + 1) for m in range(-ell, ell + 1)])
    print()
    for R in Rs:
        print("\t", R)
        assert np.allclose(sf.Wigner_D_element(R, LMpM), (-1.) ** MpPM * np.conjugate(sf.Wigner_D_element(R, LmMpmM)),
                           atol=ell_max ** 2 * precision_Wigner_D_element,
                           rtol=ell_max ** 2 * precision_Wigner_D_element)
    # D is a unitary matrix, so its conjugate transpose is its
    # inverse.  D(R) should equal the matrix inverse of D(R^{-1}).
    # So: D_{mp,m}(R) = \bar{D}_{m,mp}(\bar{R})
    LMMp = np.array(
        [[ell, m, mp] for ell in range(ell_max + 1) for mp in range(-ell, ell + 1) for m in range(-ell, ell + 1)])
    for R in Rs:
        print("\t", R)
        assert np.allclose(sf.Wigner_D_element(R, LMpM), np.conjugate(sf.Wigner_D_element(R.inverse, LMMp)),
                           atol=ell_max ** 4 * precision_Wigner_D_element,
                           rtol=ell_max ** 4 * precision_Wigner_D_element)


def test_Wigner_D_element_roundoff(Rs, ell_max):
    LMpM = sf.LMpM_range(0, ell_max)
    # Test rotations with |Ra|<1e-15
    expected = [((-1.) ** ell if mp == -m else 0.0) for ell in range(ell_max + 1) for mp in range(-ell, ell + 1) for m in
                range(-ell, ell + 1)]
    assert np.allclose(sf.Wigner_D_element(quaternionic.x, LMpM), expected,
                       atol=ell_max * precision_Wigner_D_element, rtol=ell_max * precision_Wigner_D_element)
    expected = [((-1.) ** (ell + m) if mp == -m else 0.0) for ell in range(ell_max + 1) for mp in range(-ell, ell + 1)
                for m in range(-ell, ell + 1)]
    assert np.allclose(sf.Wigner_D_element(quaternionic.y, LMpM), expected,
                       atol=ell_max * precision_Wigner_D_element, rtol=ell_max * precision_Wigner_D_element)
    for theta in np.linspace(0, 2 * np.pi):
        expected = [((-1.) ** (ell + m) * (np.cos(theta) + 1j * np.sin(theta)) ** (2 * m) if mp == -m else 0.0)
                    for ell in range(ell_max + 1) for mp in range(-ell, ell + 1) for m in range(-ell, ell + 1)]
        assert np.allclose(sf.Wigner_D_element(np.cos(theta) * quaternionic.y + np.sin(theta) * quaternionic.x, LMpM),
                           expected,
                           atol=ell_max * precision_Wigner_D_element, rtol=ell_max * precision_Wigner_D_element)
    # Test rotations with |Rb|<1e-15
    expected = [(1.0 if mp == m else 0.0) for ell in range(ell_max + 1) for mp in range(-ell, ell + 1) for m in
                range(-ell, ell + 1)]
    assert np.allclose(sf.Wigner_D_element(quaternionic.one, LMpM), expected,
                       atol=ell_max * precision_Wigner_D_element, rtol=ell_max * precision_Wigner_D_element)
    expected = [((-1.) ** m if mp == m else 0.0) for ell in range(ell_max + 1) for mp in range(-ell, ell + 1) for m in
                range(-ell, ell + 1)]
    assert np.allclose(sf.Wigner_D_element(quaternionic.z, LMpM), expected,
                       atol=ell_max * precision_Wigner_D_element, rtol=ell_max * precision_Wigner_D_element)
    for theta in np.linspace(0, 2 * np.pi):
        expected = [((np.cos(theta) + 1j * np.sin(theta)) ** (2 * m) if mp == m else 0.0)
                    for ell in range(ell_max + 1) for mp in range(-ell, ell + 1) for m in range(-ell, ell + 1)]
        assert np.allclose(sf.Wigner_D_element(np.cos(theta) * quaternionic.one + np.sin(theta) * quaternionic.z, LMpM),
                           expected,
                           atol=ell_max * precision_Wigner_D_element, rtol=ell_max * precision_Wigner_D_element)


def test_Wigner_D_element_underflow(Rs, ell_max):
    # NOTE: This is a delicate test, which depends on the result underflowing exactly when expected.
    # In particular, it should underflow to 0.0 when |mp+m|>32, but should never undeflow to 0.0
    # when |mp+m|<32.  So it's not the end of the world if this test fails, but it does mean that
    # the underflow properties have changed, so it might be worth a look.
    assert sf.ell_max >= 15  # Test can't work if this has been set lower
    eps = 1.e-10
    ell_max = max(sf.ell_max, ell_max)
    # Test |Ra|=1e-10
    LMpM = np.array(
        [[ell, mp, m] for ell in range(ell_max + 1) for mp in range(-ell, ell + 1) for m in range(-ell, ell + 1) if
         abs(mp + m) > 32])
    R = quaternionic.array(eps, 1, 0, 0).normalized
    assert np.all(sf.Wigner_D_element(R, LMpM) == 0j)
    LMpM = np.array(
        [[ell, mp, m] for ell in range(ell_max + 1) for mp in range(-ell, ell + 1) for m in range(-ell, ell + 1) if
         abs(mp + m) < 32])
    R = quaternionic.array(eps, 1, 0, 0).normalized
    assert np.all(sf.Wigner_D_element(R, LMpM) != 0j)
    # Test |Rb|=1e-10
    LMpM = np.array(
        [[ell, mp, m] for ell in range(ell_max + 1) for mp in range(-ell, ell + 1) for m in range(-ell, ell + 1) if
         abs(m - mp) > 32])
    R = quaternionic.array(1, eps, 0, 0).normalized
    assert np.all(sf.Wigner_D_element(R, LMpM) == 0j)
    LMpM = np.array(
        [[ell, mp, m] for ell in range(ell_max + 1) for mp in range(-ell, ell + 1) for m in range(-ell, ell + 1) if
         abs(m - mp) < 32])
    R = quaternionic.array(1, eps, 0, 0).normalized
    assert np.all(sf.Wigner_D_element(R, LMpM) != 0j)


def test_Wigner_D_element_overflow(Rs, ell_max):
    assert sf.ell_max >= 15  # Test can't work if this has been set lower
    ell_max = max(sf.ell_max, ell_max)
    LMpM = sf.LMpM_range(0, ell_max)
    # Test |Ra|=1e-10
    R = quaternionic.array(1.e-10, 1, 0, 0).normalized
    assert np.all(np.isfinite(sf.Wigner_D_element(R, LMpM)))
    # Test |Rb|=1e-10
    R = quaternionic.array(1, 1.e-10, 0, 0).normalized
    assert np.all(np.isfinite(sf.Wigner_D_element(R, LMpM)))


def slow_Wignerd(beta, ell, mp, m):
    # https://en.wikipedia.org/wiki/Wigner_D-matrix#Wigner_.28small.29_d-matrix
    Prefactor = math.sqrt(
        math.factorial(ell + mp) * math.factorial(ell - mp) * math.factorial(ell + m) * math.factorial(ell - m))
    s_min = int(round(max(0, round(m - mp))))
    s_max = int(round(min(round(ell + m), round(ell - mp))))
    assert isinstance(s_max, int), type(s_max)
    assert isinstance(s_min, int), type(s_min)
    return Prefactor * sum([((-1.) ** (mp - m + s)
                             * math.cos(beta / 2.) ** (2 * ell + m - mp - 2 * s)
                             * math.sin(beta / 2.) ** (mp - m + 2 * s)
                             / float(math.factorial(ell + m - s) * math.factorial(s) * math.factorial(mp - m + s)
                                     * math.factorial(ell - mp - s)))
                            for s in range(s_min, s_max + 1)])


def slow_Wigner_D_element(alpha, beta, gamma, ell, mp, m):
    # https://en.wikipedia.org/wiki/Wigner_D-matrix#Definition_of_the_Wigner_D-matrix
    return cmath.exp(-1j * mp * alpha) * slow_Wignerd(beta, ell, mp, m) * cmath.exp(-1j * m * gamma)


@slow
def test_Wigner_D_element_values(special_angles, ell_max):
    LMpM = sf.LMpM_range_half_integer(0, ell_max // 2)
    # Compare with more explicit forms given in Euler angles
    print("")
    for alpha in special_angles:
        print("\talpha={0}".format(alpha))  # Need to show some progress to Travis
        for beta in special_angles:
            print("\t\tbeta={0}".format(beta))
            for gamma in special_angles:
                a = np.conjugate(np.array([slow_Wigner_D_element(alpha, beta, gamma, ell, mp, m) for ell,mp,m in LMpM]))
                b = sf.Wigner_D_element(quaternionic.array.from_euler_angles(alpha, beta, gamma), LMpM)
                # if not np.allclose(a, b,
                #     atol=ell_max ** 6 * precision_Wigner_D_element,
                #     rtol=ell_max ** 6 * precision_Wigner_D_element):
                #     for i in range(min(a.shape[0], 100)):
                #         print(LMpM[i], "\t", abs(a[i]-b[i]), "\t\t", a[i], "\t", b[i])
                assert np.allclose(a, b,
                    atol=ell_max ** 6 * precision_Wigner_D_element,
                    rtol=ell_max ** 6 * precision_Wigner_D_element)


@slow
def test_Wigner_D_matrix(Rs, ell_max):
    for l_min in [0, 1, 2, ell_max // 2, ell_max - 1]:
        print("")
        for l_max in range(l_min + 1, ell_max + 1):
            print("\tWorking on (l_min,l_max)=({0},{1})".format(l_min, l_max))
            LMpM = sf.LMpM_range(l_min, l_max)
            for R in Rs:
                elements = sf.Wigner_D_element(R, LMpM)
                matrix = np.empty(LMpM.shape[0], dtype=complex)
                Rspinor = R.two_spinor
                sf._Wigner_D_matrices(Rspinor.a, Rspinor.b, l_min, l_max, matrix)
                assert np.allclose(elements, matrix,
                                   atol=1e3 * l_max * ell_max * precision_Wigner_D_element,
                                   rtol=1e3 * l_max * ell_max * precision_Wigner_D_element)


@slow
def test_Wigner_D_input_types(Rs, special_angles, ell_max):
    LMpM = sf.LMpM_range(0, ell_max // 2)
    # Compare with more explicit forms given in Euler angles
    print("")
    for alpha in special_angles:
        print("\talpha={0}".format(alpha))  # Need to show some progress to Travis
        for beta in special_angles:
            for gamma in special_angles:
                a = sf.Wigner_D_element(alpha, beta, gamma, LMpM)
                b = sf.Wigner_D_element(quaternionic.array.from_euler_angles(alpha, beta, gamma), LMpM)
                assert np.allclose(a, b,
                                   atol=ell_max ** 6 * precision_Wigner_D_element,
                                   rtol=ell_max ** 6 * precision_Wigner_D_element)
    for R in Rs:
        a = sf.Wigner_D_element(R, LMpM)
        Rspinor = R.two_spinor
        b = sf.Wigner_D_element(Rspinor.a, Rspinor.b, LMpM)
        assert np.allclose(a, b,
                           atol=ell_max ** 6 * precision_Wigner_D_element,
                           rtol=ell_max ** 6 * precision_Wigner_D_element)


def test_Wigner_D_signatures(Rs):
    """There are two ways to call the WignerD function: with an array of Rs, or with an array of (ell,mp,m) values.
    This test ensures that the results are the same in both cases."""
    # from spherical.WignerD import _Wigner_D_elements
    ell_max = 6
    ell_mp_m = sf.LMpM_range(0, ell_max)
    Ds1 = np.zeros((Rs.size//4, ell_mp_m.shape[0]), dtype=np.complex)
    Ds2 = np.zeros_like(Ds1)
    for i, R in enumerate(Rs):
        Ds1[i, :] = sf.Wigner_D_element(R, ell_mp_m)
    for i, (ell, mp, m) in enumerate(ell_mp_m):
        Ds2[:, i] = sf.Wigner_D_element(Rs, ell, mp, m)
    assert np.allclose(Ds1, Ds2, rtol=3e-15, atol=3e-15)
