#!/usr/bin/env python

# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import math
import cmath
import numpy as np
import quaternionic
import spherical as sf
import pytest

from .conftest import requires_sympy

slow = pytest.mark.slow

precision_Wigner_D_element = 4.e-14


def test_Wigner_D_negative_argument(Rs, ell_max, eps):
    # For integer ell, D(R)=D(-R)
    #
    # This test passes (specifically, using these tolerances) for at least
    # ell_max=100, but takes a few minutes at that level.
    a = np.zeros(sf.WignerDsize(0, ell_max), dtype=complex)
    b = np.zeros(sf.WignerDsize(0, ell_max), dtype=complex)
    wigner = sf.Wigner(ell_max)
    for R in Rs:
        wigner.D(R, out=a)
        wigner.D(-R, out=b)
        # sf.wigner_D(R, 0, ell_max, out=a)
        # sf.wigner_D(-R, 0, ell_max, out=b)
        assert np.allclose(a, b, rtol=ell_max*eps, atol=2*ell_max*eps)


@slow
def test_Wigner_D_representation_property(Rs, ell_max_slow, eps):
    # Test the representation property for special and random angles
    # For each l, 𝔇ˡₘₚ,ₘ(R1 * R2) = Σₘₚₚ 𝔇ˡₘₚ,ₘₚₚ(R1) * 𝔇ˡₘₚₚ,ₘ(R2)
    import time
    print("")
    t1 = time.perf_counter()
    D1 = np.zeros(sf.WignerDsize(0, ell_max_slow), dtype=complex)
    D2 = np.zeros(sf.WignerDsize(0, ell_max_slow), dtype=complex)
    D12 = np.zeros(sf.WignerDsize(0, ell_max_slow), dtype=complex)
    wigner = sf.Wigner(ell_max_slow)
    for i, R1 in enumerate(Rs):
        print(f"\t{i+1} of {len(Rs)}: R1 = {R1}")
        for j, R2 in enumerate(Rs):
            # print(f"\t\t{j+1} of {len(Rs)}: R2 = {R2}")
            R12 = R1 * R2
            wigner.D(R1, out=D1)
            wigner.D(R2, out=D2)
            wigner.D(R12, out=D12)
            for ell in range(ell_max_slow+1):
                ϵ = (2*ell+1)**2 * eps
                i1 = sf.WignerDindex(ell, -ell, -ell)
                i2 = sf.WignerDindex(ell, ell, ell)
                shape = (2*ell+1, 2*ell+1)
                Dˡ1 = D1[i1:i2+1].reshape(shape)
                Dˡ2 = D2[i1:i2+1].reshape(shape)
                Dˡ12 = D12[i1:i2+1].reshape(shape)
                assert np.allclose(Dˡ1 @ Dˡ2, Dˡ12, rtol=ϵ, atol=ϵ), ell
                # assert np.allclose(Dˡ1 @ Dˡ2, Dˡ12, atol=ell_max_slow * precision_Wigner_D_element), ell
    t2 = time.perf_counter()
    print(f"\tFinished in {t2-t1:.4f} seconds.")


def test_Wigner_D_inverse_property(Rs, ell_max, eps):
    # Test the inverse property for special and random angles
    # For each l, 𝔇ˡₘₚ,ₘ(R⁻¹) should be the inverse matrix of 𝔇ˡₘₚ,ₘ(R)
    D1 = np.zeros(sf.WignerDsize(0, ell_max), dtype=complex)
    D2 = np.zeros(sf.WignerDsize(0, ell_max), dtype=complex)
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
def test_Wigner_D_symmetries(Rs, ell_max_slow, eps):
    # We have two obvious symmetries to test.  First,
    #
    #   D_{mp,m}(R) = (-1)^{mp+m} \bar{D}_{-mp,-m}(R)
    #
    # Second, since D is a unitary matrix, its conjugate transpose is its
    # inverse; because of the representation property, D(R) should equal the
    # matrix inverse of D(R⁻¹).  Thus,
    #
    #   D_{mp,m}(R) = \bar{D}_{m,mp}(\bar{R})

    ϵ = 5 * ell_max_slow * eps
    D1 = np.zeros(sf.WignerDsize(0, ell_max_slow), dtype=complex)
    D2 = np.zeros(sf.WignerDsize(0, ell_max_slow), dtype=complex)
    wigner = sf.Wigner(ell_max_slow)
    ell_mp_m = sf.WignerDrange(0, ell_max_slow)

    flipped_indices = np.array([
        sf.WignerDindex(ell, -mp, -m)
        for ell, mp, m in ell_mp_m
    ])
    swapped_indices = np.array([
        sf.WignerDindex(ell, m, mp)
        for ell, mp, m in ell_mp_m
    ])
    signs = (-1) ** np.abs(np.sum(ell_mp_m[:, 1:], axis=1))
    for R in Rs:
        wigner.D(R, out=D1)
        wigner.D(R.inverse, out=D2)
        # D_{mp,m}(R) = (-1)^{mp+m} \bar{D}_{-mp,-m}(R)
        a = D1
        b = signs * D1[flipped_indices].conjugate()
        assert np.allclose(a, b, rtol=ϵ, atol=ϵ)
        # D_{mp,m}(R) = \bar{D}_{m,mp}(\bar{R})
        b = D2[swapped_indices].conjugate()
        assert np.allclose(a, b, rtol=ϵ, atol=ϵ)


def test_Wigner_D_roundoff(Rs, ell_max, eps):
    # Testing rotations in special regions with simple expressions for 𝔇

    ϵ = 5 * ell_max * eps
    D = np.zeros(sf.WignerDsize(0, ell_max), dtype=complex)
    wigner = sf.Wigner(ell_max)

    # Test rotations with |Ra|<1e-15
    wigner.D(quaternionic.x, out=D)
    actual = D
    expected = np.array([
        ((-1.) ** ell if mp == -m else 0.0)
        for ell in range(ell_max + 1)
        for mp in range(-ell, ell + 1)
        for m in  range(-ell, ell + 1)
    ])
    assert np.allclose(actual, expected, rtol=ϵ, atol=ϵ)

    wigner.D(quaternionic.y, out=D)
    actual = D
    expected = np.array([
        ((-1.) ** (ell + m) if mp == -m else 0.0)
        for ell in range(ell_max + 1)
        for mp in range(-ell, ell + 1)
        for m in range(-ell, ell + 1)
    ])
    assert np.allclose(actual, expected, rtol=ϵ, atol=ϵ)

    for theta in np.linspace(0, 2 * np.pi):
        wigner.D(np.cos(theta) * quaternionic.y + np.sin(theta) * quaternionic.x, out=D)
        actual = D
        expected = np.array([
            ((-1.) ** (ell + m) * (np.cos(theta) + 1j * np.sin(theta)) ** (2 * m) if mp == -m else 0.0)
            for ell in range(ell_max + 1)
            for mp in range(-ell, ell + 1)
            for m in range(-ell, ell + 1)
        ])
        assert np.allclose(actual, expected, rtol=ϵ, atol=ϵ)

    # Test rotations with |Rb|<1e-15
    wigner.D(quaternionic.one, out=D)
    actual = D
    expected = np.array([
        (1.0 if mp == m else 0.0)
        for ell in range(ell_max + 1)
        for mp in range(-ell, ell + 1)
        for m in range(-ell, ell + 1)
    ])
    assert np.allclose(actual, expected, rtol=ϵ, atol=ϵ)

    wigner.D(quaternionic.z, out=D)
    actual = D
    expected = np.array([
        ((-1.) ** m if mp == m else 0.0)
        for ell in range(ell_max + 1)
        for mp in range(-ell, ell + 1)
        for m in range(-ell, ell + 1)
    ])
    assert np.allclose(actual, expected, rtol=ϵ, atol=ϵ)

    for theta in np.linspace(0, 2 * np.pi):
        wigner.D(np.cos(theta) * quaternionic.one + np.sin(theta) * quaternionic.z, out=D)
        actual = D
        expected = np.array([
            ((np.cos(theta) + 1j * np.sin(theta)) ** (2 * m) if mp == m else 0.0)
            for ell in range(ell_max + 1)
            for mp in range(-ell, ell + 1)
            for m in range(-ell, ell + 1)
        ])
        assert np.allclose(actual, expected, rtol=ϵ, atol=ϵ)


@pytest.mark.xfail
def test_Wigner_D_underflow(Rs, ell_max, eps):
    # NOTE: This is a delicate test, which depends on the result underflowing exactly when expected.
    # In particular, it should underflow to 0.0 when |mp+m|>32, but should never underflow to 0.0
    # when |mp+m|<32.  So it's not the end of the world if this test fails, but it does mean that
    # the underflow properties have changed, so it might be worth a look.
    epsilon = 1.e-10

    ϵ = 5 * ell_max * eps
    D = np.zeros(sf.WignerDsize(0, ell_max), dtype=complex)
    wigner = sf.Wigner(ell_max)
    ell_mp_m = sf.WignerDrange(0, ell_max)

    # Test |Ra|=1e-10
    R = quaternionic.array(epsilon, 1, 0, 0).normalized
    wigner.D(R, out=D)
    # print(R.to_euler_angles.tolist())
    # print(D.tolist())
    non_underflowing_indices = np.abs(ell_mp_m[:, 1] + ell_mp_m[:, 2]) < 32
    assert np.all(D[non_underflowing_indices] != 0j)
    underflowing_indices = np.abs(ell_mp_m[:, 1] + ell_mp_m[:, 2]) > 32
    assert np.all(D[underflowing_indices] == 0j)

    # Test |Rb|=1e-10
    R = quaternionic.array(1, epsilon, 0, 0).normalized
    wigner.D(R, out=D)
    non_underflowing_indices = np.abs(ell_mp_m[:, 1] - ell_mp_m[:, 2]) < 32
    assert np.all(D[non_underflowing_indices] != 0j)
    underflowing_indices = np.abs(ell_mp_m[:, 1] - ell_mp_m[:, 2]) > 32
    assert np.all(D[underflowing_indices] == 0j)


def test_Wigner_D_non_overflow(ell_max):
    D = np.zeros(sf.WignerDsize(0, ell_max), dtype=complex)
    wigner = sf.Wigner(ell_max)

    # Test |Ra|=1e-10
    R = quaternionic.array(1.e-10, 1, 0, 0).normalized
    assert np.all(np.isfinite(wigner.D(R, out=D)))

    # Test |Rb|=1e-10
    R = quaternionic.array(1, 1.e-10, 0, 0).normalized
    assert np.all(np.isfinite(wigner.D(R, out=D)))


def Wigner_d_Wikipedia(beta, ell, mp, m):
    # https://en.wikipedia.org/wiki/Wigner_D-matrix#Wigner_.28small.29_d-matrix
    Prefactor = math.sqrt(
        math.factorial(ell + mp)
        * math.factorial(ell - mp)
        * math.factorial(ell + m)
        * math.factorial(ell - m)
    )
    s_min = int(round(max(0, round(m - mp))))
    s_max = int(round(min(round(ell + m), round(ell - mp))))
    assert isinstance(s_max, int), type(s_max)
    assert isinstance(s_min, int), type(s_min)
    return Prefactor * sum([
        (
            (-1.) ** (mp - m + s)
            * math.cos(beta / 2.) ** (2 * ell + m - mp - 2 * s)
            * math.sin(beta / 2.) ** (mp - m + 2 * s)
            / float(
                math.factorial(ell + m - s)
                * math.factorial(s)
                * math.factorial(mp - m + s)
                * math.factorial(ell - mp - s)
            )
        )
        for s in range(s_min, s_max + 1)
    ])


def Wigner_D_Wikipedia(alpha, beta, gamma, ell, mp, m):
    # https://en.wikipedia.org/wiki/Wigner_D-matrix#Definition_of_the_Wigner_D-matrix
    return cmath.exp(-1j * mp * alpha) * Wigner_d_Wikipedia(beta, ell, mp, m) * cmath.exp(-1j * m * gamma)


@slow
def test_Wigner_D_vs_Wikipedia(special_angles, ell_max_slow, eps):
    ell_max = ell_max_slow
    ϵ = 5 * ell_max**6 * eps

    D = np.zeros(sf.WignerDsize(0, ell_max), dtype=complex)
    wigner = sf.Wigner(ell_max)
    ell_mp_m = sf.WignerDrange(0, ell_max)

    print("")
    for alpha in special_angles:
        print("\talpha={0}".format(alpha))  # Need to show some progress for CI
        for beta in special_angles:
            print("\t\tbeta={0}".format(beta))
            for gamma in special_angles:
                a = np.conjugate(np.array([Wigner_D_Wikipedia(alpha, beta, gamma, ell, mp, m) for ell,mp,m in ell_mp_m]))
                b = wigner.D(quaternionic.array.from_euler_angles(alpha, beta, gamma), out=D)
                assert np.allclose(a, b, rtol=ϵ, atol=ϵ)


@slow
@requires_sympy
def test_Wigner_D_vs_sympy(special_angles, ell_max_slow, eps):
    from sympy import S, N
    from sympy.physics.quantum.spin import WignerD as Wigner_D_sympy

    # Note that this does not fully respect ell_max_slow because
    # this test is extraordinarily slow
    ell_max = min(4, ell_max_slow)
    ϵ = 2 * ell_max * eps

    wigner = sf.Wigner(ell_max)
    max_error = 0.0

    j = 0
    k = 0
    print()
    a = special_angles[::4]
    b = special_angles[len(special_angles)//2::4]
    c = special_angles[::2]
    for α in a:
        for β in b:
            for γ in c:
                R = quaternionic.array.from_euler_angles(α, β, γ)
                𝔇 = wigner.D(R)

                k += 1
                print(f"\tAngle iteration {k} of {a.size*b.size*c.size}")
                for ell in range(wigner.ell_max+1):
                    for mp in range(-ell, ell+1):
                        for m in range(-ell, ell+1):
                            sympyD = N(Wigner_D_sympy(ell, mp, m, α, β, γ).doit(), n=24).conjugate()
                            sphericalD = 𝔇[wigner.Dindex(ell, mp, m)]
                            error = float(abs(sympyD-sphericalD))
                            assert error < ϵ, (
                                f"Testing Wigner d recursion: ell={ell}, m'={mp}, m={m}, "
                                f"sympy:{sympyD}, spherical:{sphericalD}, error={error}"
                            )
                            max_error = max(error, max_error)
    print(f"\tmax_error={max_error} after checking {(len(special_angles)**3)*wigner.Dsize} values")
