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

precision_Wigner_D_element = 4.e-14


def test_Wigner_D_negative_argument(Rs, ell_max, eps):
    # For integer ell, D(R)=D(-R)
    #
    # This test passes (specifically, using these tolerances) for at least
    # ell_max=100, but takes a few minutes at that level.
    a = np.empty(sf.WignerDsize(0, ell_max), dtype=complex)
    b = np.empty(sf.WignerDsize(0, ell_max), dtype=complex)
    wigner = sf.Wigner(ell_max)
    for R in Rs:
        wigner.D(R, out=a)
        wigner.D(-R, out=b)
        # sf.wigner_D(R, 0, ell_max, out=a)
        # sf.wigner_D(-R, 0, ell_max, out=b)
        assert np.allclose(a, b, rtol=ell_max*eps, atol=2*ell_max*eps)


@slow
def test_Wigner_D_representation_property(Rs, ell_max, eps):
    # Test the representation property for special and random angles
    # For each l, 𝔇ˡₘₚ,ₘ(R1 * R2) = Σₘₚₚ 𝔇ˡₘₚ,ₘₚₚ(R1) * 𝔇ˡₘₚₚ,ₘ(R2)
    print("")
    D1 = np.empty(sf.WignerDsize(0, ell_max), dtype=complex)
    D2 = np.empty(sf.WignerDsize(0, ell_max), dtype=complex)
    D12 = np.empty(sf.WignerDsize(0, ell_max), dtype=complex)
    wigner = sf.Wigner(ell_max)
    for i, R1 in enumerate(Rs):
        print(f"\t{i+1} of {len(Rs)}: R1 = {R1}")
        for j, R2 in enumerate(Rs):
            # print(f"\t\t{j+1} of {len(Rs)}: R2 = {R2}")
            R12 = R1 * R2
            wigner.D(R1, out=D1)
            wigner.D(R2, out=D2)
            wigner.D(R12, out=D12)
            for ell in range(ell_max+1):
                ϵ = (2*ell+1)**2 * eps
                i1 = sf.WignerDindex(ell, -ell, -ell)
                i2 = sf.WignerDindex(ell, ell, ell)
                shape = (2*ell+1, 2*ell+1)
                Dˡ1 = D1[i1:i2+1].reshape(shape)
                Dˡ2 = D2[i1:i2+1].reshape(shape)
                Dˡ12 = D12[i1:i2+1].reshape(shape)
                assert np.allclose(Dˡ1 @ Dˡ2, Dˡ12, rtol=ϵ, atol=ϵ), ell
                # assert np.allclose(Dˡ1 @ Dˡ2, Dˡ12, atol=ell_max * precision_Wigner_D_element), ell


def test_Wigner_D_inverse_property(Rs, ell_max, eps):
    # Test the inverse property for special and random angles
    # For each l, 𝔇ˡₘₚ,ₘ(R⁻¹) should be the inverse matrix of 𝔇ˡₘₚ,ₘ(R)
    D1 = np.empty(sf.WignerDsize(0, ell_max), dtype=complex)
    D2 = np.empty(sf.WignerDsize(0, ell_max), dtype=complex)
    wigner = sf.Wigner(ell_max)
    for i, R in enumerate(Rs):
        # print(f"\t{i+1} of {len(Rs)}: R = {R}")
        wigner.D(R, out=D1)
        wigner.D(R.inverse, out=D2)
        for ell in range(ell_max+1):
            ϵ = (2*ell+1)**2 * eps
            i1 = sf.WignerDindex(ell, -ell, -ell)
            i2 = sf.WignerDindex(ell, ell, ell)
            shape = (2*ell+1, 2*ell+1)
            Dˡ1 = D1[i1:i2+1].reshape(shape)
            Dˡ2 = D2[i1:i2+1].reshape(shape)
            assert np.allclose(Dˡ1 @ Dˡ2, np.identity(2*ell+1), rtol=ϵ, atol=ϵ), ell


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
    eps = 1.e-10
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
