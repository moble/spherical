#!/usr/bin/env python

# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import math
import cmath
import pickle
import copy
import numpy as np
import spherical as sf
import pytest


def test_modes_creation():
    for s in range(-2, 2 + 1):
        ell_min = abs(s)
        ell_max = 8

        # Test successful creation with real data of the right shape
        a = np.random.rand(3, 7, sf.LM_total_size(ell_min, ell_max)*2)
        m = sf.Modes(a, spin_weight=s, ell_min=ell_min, ell_max=ell_max)
        assert m.s == s
        assert m.ell_min == 0  # NOTE: This is hard coded!!!
        assert m.ell_max == ell_max
        assert np.array_equal(a.view(complex), m[..., sf.LM_total_size(0, ell_min-1):])
        assert np.all(m[..., :sf.LM_total_size(0, abs(s)-1)] == 0.0)
        m = sf.Modes(a, spin_weight=s, ell_min=ell_min)  # ell_max is deduced!
        assert m.s == s
        assert m.ell_min == 0  # NOTE: This is hard coded!!!
        assert m.ell_max == ell_max
        assert np.array_equal(a.view(complex), m[..., sf.LM_total_size(0, ell_min-1):])
        assert np.all(m[..., :sf.LM_total_size(0, abs(s)-1)] == 0.0)

        # Test successful creation with complex data of the right shape
        a = a.view(complex)
        m = sf.Modes(a, spin_weight=s, ell_min=ell_min, ell_max=ell_max)
        assert m.s == s
        assert m.ell_min == 0  # NOTE: This is hard coded!!!
        assert m.ell_max == ell_max
        assert np.array_equal(a, m[..., sf.LM_total_size(0, ell_min-1):])
        assert np.all(m[..., :sf.LM_total_size(0, abs(s)-1)] == 0.0)
        m = sf.Modes(a, spin_weight=s, ell_min=ell_min)  # ell_max is deduced!
        assert m.s == s
        assert m.ell_min == 0  # NOTE: This is hard coded!!!
        assert m.ell_max == ell_max
        assert np.array_equal(a, m[..., sf.LM_total_size(0, ell_min-1):])
        assert np.all(m[..., :sf.LM_total_size(0, abs(s)-1)] == 0.0)

        # Test failed creation with complex data of inconsistent shape
        if ell_min != 0:
            with pytest.raises(ValueError):
                m = sf.Modes(a, spin_weight=s)
        with pytest.raises(ValueError):
            m = sf.Modes(a, spin_weight=s, ell_min=ell_min-1, ell_max=ell_max)
        with pytest.raises(ValueError):
            m = sf.Modes(a, spin_weight=s, ell_min=ell_min+1, ell_max=ell_max)
        with pytest.raises(ValueError):
            m = sf.Modes(a, spin_weight=s, ell_min=ell_min, ell_max=ell_max-1)
        with pytest.raises(ValueError):
            m = sf.Modes(a, spin_weight=s, ell_min=ell_min, ell_max=ell_max+1)

        # Test failed creation with complex data of impossible shape
        with pytest.raises(ValueError):
            m = sf.Modes(a[..., 1:], spin_weight=s, ell_min=ell_min)

        # Test successful creation with complex data containing extraneous data at ell<abs(s)
        a = np.random.rand(3, 7, sf.LM_total_size(0, ell_max)*2)
        a = a.view(complex)
        m = sf.Modes(a, spin_weight=s)
        assert m.s == s
        assert m.ell_min == 0  # NOTE: This is hard coded!!!
        assert m.ell_max == ell_max
        assert np.all(m[..., :sf.LM_total_size(0, abs(s)-1)] == 0.0)


def np_copy(m):
    return np.copy(m)

def np_array_copy(m):
    return np.array(m, copy=True)

def np_array_copy_subok(m):
    return np.array(m, copy=True, subok=True)

def ndarray_copy(m):
    return m.copy()

def pickle_roundtrip(m):
    return pickle.loads(pickle.dumps(m))

def copy_copy(m):
    return copy.copy(m)

def copy_deepcopy(m):
    return copy.deepcopy(m)

# Note that np.copy and np.array(..., copy=True) return ndarray's, and thus lose information
copy_xfail = lambda f: pytest.param(f, marks=pytest.mark.xfail(reason="Unexpected numpy defaults"))

@pytest.mark.parametrize("copier", [
    copy_xfail(np_copy), copy_xfail(np_array_copy), np_array_copy_subok,
    ndarray_copy, pickle_roundtrip, copy_copy, copy_deepcopy
])
def test_modes_copying_and_pickling(copier):
    for s in range(-2, 2 + 1):
        ell_min = abs(s)
        ell_max = 8
        a = np.random.rand(3, 7, sf.LM_total_size(ell_min, ell_max)*2).view(complex)
        m = sf.Modes(a, spin_weight=s, ell_min=ell_min, ell_max=ell_max)
        c = copier(m)
        assert m is not c
        assert np.array_equal(c, m)
        assert isinstance(c, type(m))
        assert c.s == m.s
        assert c.ell_min == m.ell_min
        assert c.ell_max == m.ell_max


@pytest.mark.xfail
def test_modes_grid():
    for s in range(-2, 2 + 1):
        ell_min = abs(s)
        ell_max = 8
        a = np.random.rand(3, 7, sf.LM_total_size(ell_min, ell_max)*2).view(complex)
        m = sf.Modes(a, spin_weight=s, ell_min=ell_min, ell_max=ell_max)
        n = 2*ell_max+1
        for n_theta, n_phi in [[None, None], [n, None], [None, n], [n, n], [n+1, n], [n, n+1], [n+1, n+1]]:
            g = m.grid(n_theta=n_theta, n_phi=n_phi)
            assert g.dtype == np.complex
            assert g.shape[:-2] == a.shape[:-1]
            if n_theta is None:
                n_theta = n
            if n_phi is None:
                n_phi = n
            assert g.shape[-2:] == (n_theta, n_phi)


@pytest.mark.xfail
def test_modes_addition():
    tolerance = 1e-14
    np.random.seed(1234)
    for s1 in range(-2, 2 + 1):
        ell_min1 = abs(s1)
        ell_max1 = 8
        a1 = np.random.rand(3, 7, sf.LM_total_size(ell_min1, ell_max1)*2)
        a2 = np.random.rand(*a1.shape)
        a1 = a1.view(complex)
        a2 = a2.view(complex)
        m1 = sf.Modes(a1, spin_weight=s1, ell_min=ell_min1, ell_max=ell_max1)
        m2 = sf.Modes(a2, spin_weight=s1, ell_min=ell_min1, ell_max=ell_max1)
        m1m2 = m1+m2
        assert m1m2.s == s1
        assert m1m2.ell_max == m1.ell_max
        assert np.array_equal(m1m2, m1.add(m2))
        assert np.array_equal(m1m2, m1.view(np.ndarray)+m2.view(np.ndarray))
        for s2 in range(-s1, s1 + 1):
            ell_min2 = ell_min1+1
            ell_max2 = ell_max1-1
            a2 = np.random.rand(3, 7, sf.LM_total_size(ell_min2, ell_max2)*2).view(complex)
            m2 = sf.Modes(a2, spin_weight=s2, ell_min=ell_min2, ell_max=ell_max2)
            if s1 != s2:
                # Don't allow addition of non-zero data
                with pytest.raises(ValueError):
                    m1m2 = m1.add(m2)
                # Do allow addition with various forms of 0, for convenience
                for m3 in [
                        m1.add(0), m1.add(np.zeros(1)), m1.add(np.zeros((1,))), m1.add(np.zeros((7,))), m1.add(np.zeros((3,7))),
                        m1+0, m1+np.zeros(1), m1+np.zeros((1,)), m1+np.zeros((7,)), m1+np.zeros((3,7)),
                        0+m1, np.zeros(1)+m1, np.zeros((1,))+m1, np.zeros((7,))+m1, np.zeros((3,7))+m1,
                ]:
                    assert m3.s == s1
                    assert m3.ell_min == m1.ell_min
                    assert m3.ell_max == m1.ell_max
                    assert np.array_equal(m1, m3)
            else:
                for m1m2 in [m1.add(m2), m2.add(m1)]:
                    assert m1m2.s == s1
                    assert m1m2.ell_max == m1.ell_max
                    i1 = sf.LM_total_size(0, min(ell_min1, ell_min2)-1)
                    i2 = sf.LM_total_size(0, max(ell_min1, ell_min2)-1)
                    i3 = sf.LM_total_size(0, min(ell_max1, ell_max2))
                    i4 = sf.LM_total_size(0, max(ell_max1, ell_max2))
                    assert np.array_equiv(m1m2[..., :i1], 0.0)
                    assert np.array_equal(m1m2[..., i1:i2], a1[..., :i2-i1])
                    assert np.array_equal(m1m2[..., i1:i2], m1.view(np.ndarray)[..., i1:i2])
                    assert np.array_equal(m1m2[..., i2:i3], m1.view(np.ndarray)[..., i2:i3]+m2.view(np.ndarray)[..., i2:i3])
                    assert np.array_equal(m1m2[..., i3:i4], m1.view(np.ndarray)[..., i3:i4])
                    g12 = m1m2.grid()
                    n_theta, n_phi = g12.shape[-2:]
                    g1 = m1.grid(n_theta, n_phi)
                    g2 = m2.grid(n_theta, n_phi)
                    assert np.allclose(g1+g2, g12, rtol=tolerance, atol=tolerance)


@pytest.mark.xfail
def test_modes_subtraction():
    tolerance = 1e-14
    np.random.seed(1234)
    for s1 in range(-2, 2 + 1):
        ell_min1 = abs(s1)
        ell_max1 = 8
        a1 = np.random.rand(3, 7, sf.LM_total_size(ell_min1, ell_max1)*2)
        a2 = np.random.rand(*a1.shape)
        a1 = a1.view(complex)
        a2 = a2.view(complex)
        m1 = sf.Modes(a1, spin_weight=s1, ell_min=ell_min1, ell_max=ell_max1)
        m2 = sf.Modes(a2, spin_weight=s1, ell_min=ell_min1, ell_max=ell_max1)
        m1m2 = m1-m2
        assert m1m2.s == s1
        assert m1m2.ell_max == m1.ell_max
        assert np.array_equal(m1m2, m1.subtract(m2))
        assert np.array_equal(m1m2, m1.view(np.ndarray)-m2.view(np.ndarray))
        for s2 in range(-s1, s1 + 1):
            ell_min2 = ell_min1+1
            ell_max2 = ell_max1-1
            a2 = np.random.rand(3, 7, sf.LM_total_size(ell_min2, ell_max2)*2).view(complex)
            m2 = sf.Modes(a2, spin_weight=s2, ell_min=ell_min2, ell_max=ell_max2)
            if s1 != s2:
                # Don't allow addition of non-zero data
                with pytest.raises(ValueError):
                    m1m2 = m1.subtract(m2)
                # Do allow subtraction with various forms of 0, for convenience
                for m3 in [
                        m1.subtract(0), m1.subtract(np.zeros(1)), m1.subtract(np.zeros((1,))),
                        m1.subtract(np.zeros((7,))), m1.subtract(np.zeros((3,7))),
                        m1-0, m1-np.zeros(1), m1-np.zeros((1,)), m1-np.zeros((7,)), m1-np.zeros((3,7)),
                ]:
                    assert m3.s == s1
                    assert m3.ell_min == m1.ell_min
                    assert m3.ell_max == m1.ell_max
                    assert np.array_equal(m1, m3)
                for m3 in [
                        0-m1, np.zeros(1)-m1, np.zeros((1,))-m1, np.zeros((7,))-m1, np.zeros((3,7))-m1,
                ]:
                    assert m3.s == s1
                    assert m3.ell_min == m1.ell_min
                    assert m3.ell_max == m1.ell_max
                    assert np.array_equal(-m1, m3)
            else:
                m1m2 = m1.subtract(m2)
                assert m1m2.s == s1
                assert m1m2.ell_max == m1.ell_max
                i1 = sf.LM_total_size(0, min(ell_min1, ell_min2)-1)
                i2 = sf.LM_total_size(0, max(ell_min1, ell_min2)-1)
                i3 = sf.LM_total_size(0, min(ell_max1, ell_max2))
                i4 = sf.LM_total_size(0, max(ell_max1, ell_max2))
                assert np.array_equiv(m1m2[..., :i1], 0.0)
                assert np.array_equal(m1m2[..., i1:i2], a1[..., :i2-i1])
                assert np.array_equal(m1m2[..., i1:i2], m1.view(np.ndarray)[..., i1:i2])
                assert np.array_equal(m1m2[..., i2:i3], m1.view(np.ndarray)[..., i2:i3]-m2.view(np.ndarray)[..., i2:i3])
                assert np.array_equal(m1m2[..., i3:i4], m1.view(np.ndarray)[..., i3:i4])
                g12 = m1m2.grid()
                n_theta, n_phi = g12.shape[-2:]
                g1 = m1.grid(n_theta, n_phi)
                g2 = m2.grid(n_theta, n_phi)
                assert np.allclose(g1-g2, g12, rtol=tolerance, atol=tolerance)


@pytest.mark.xfail
def test_modes_multiplication():
    tolerance = 1e-13
    np.random.seed(1234)
    # Test without truncation
    for i_mul, mul in enumerate([np.multiply, lambda a, b: a.multiply(b), lambda a, b: a.multiply(b, truncator=max)]):
        for s1 in range(-2, 2 + 1):
            ell_min1 = abs(s1)
            ell_max1 = 8
            a1 = np.random.rand(3, 7, sf.LM_total_size(ell_min1, ell_max1)*2).view(complex)
            m1 = sf.Modes(a1, spin_weight=s1, ell_min=ell_min1, ell_max=ell_max1)
            # Check scalar multiplications
            s = np.random.rand()
            m1s = mul(m1, s)
            assert m1.s == s1
            assert m1s.ell_max == m1.ell_max
            g1s = m1s.grid()
            n_theta, n_phi = g1s.shape[-2:]
            g1 = m1.grid(n_theta, n_phi)
            assert np.allclose(g1*s, g1s, rtol=tolerance, atol=tolerance)
            if mul is np.multiply:
                sm1 = mul(s, m1)
                assert sm1.s == s1
                assert sm1.ell_max == m1.ell_max
                sg1 = sm1.grid()
                n_theta, n_phi = sg1.shape[-2:]
                g1 = m1.grid(n_theta, n_phi)
                assert np.allclose(s*g1, sg1, rtol=tolerance, atol=tolerance)
            # Check scalar-array multiplications
            s = np.random.rand(3, 7)
            m1s = mul(m1, s)
            assert m1.s == s1
            assert m1s.ell_max == m1.ell_max
            g1s = m1s.grid()
            n_theta, n_phi = g1s.shape[-2:]
            g1 = m1.grid(n_theta, n_phi)
            assert np.allclose(g1*s, g1s, rtol=tolerance, atol=tolerance)
            if mul is np.multiply:
                sm1 = mul(s, m1)
                assert sm1.s == s1
                assert sm1.ell_max == m1.ell_max
                sg1 = sm1.grid()
                n_theta, n_phi = sg1.shape[-2:]
                g1 = m1.grid(n_theta, n_phi)
                assert np.allclose(s*g1, sg1, rtol=tolerance, atol=tolerance)
            # Check spin-weighted multiplications
            for s2 in range(-s1, s1 + 1):
                ell_min2 = ell_min1+1
                ell_max2 = ell_max1-1
                a2 = np.random.rand(3, 7, sf.LM_total_size(ell_min2, ell_max2)*2).view(complex)
                m2 = sf.Modes(a2, spin_weight=s2, ell_min=ell_min2, ell_max=ell_max2)
                m1m2 = mul(m1, m2)
                assert m1m2.s == s1 + s2
                if i_mul == 2:
                    assert m1m2.ell_max == max(m1.ell_max, m2.ell_max)
                else:
                    assert m1m2.ell_max == m1.ell_max + m2.ell_max
                    g12 = m1m2.grid()
                    n_theta, n_phi = g12.shape[-2:]
                    g1 = m1.grid(n_theta, n_phi)
                    g2 = m2.grid(n_theta, n_phi)
                    assert np.allclose(g1*g2, g12, rtol=tolerance, atol=tolerance)


@pytest.mark.xfail
def test_modes_conjugate():
    tolerance = 1e-15
    np.random.seed(1234)
    for inplace in [False, True]:
        for s in range(-2, 2 + 1):
            ell_min = abs(s)
            ell_max = 8
            a = np.random.rand(3, 7, sf.LM_total_size(ell_min, ell_max)*2).view(complex)
            m = sf.Modes(a, spin_weight=s, ell_min=ell_min, ell_max=ell_max)
            g = m.grid()
            s = m.s
            ell_min = m.ell_min
            ell_max = m.ell_max
            shape = m.shape
            mbar = m.conjugate(inplace)
            gbar = mbar.grid()
            assert s == -mbar.s
            assert ell_min == mbar.ell_min
            assert ell_max == mbar.ell_max
            assert shape == mbar.shape
            assert np.allclose(g, np.conjugate(gbar), rtol=tolerance, atol=tolerance)


@pytest.mark.xfail
def test_modes_real():
    tolerance = 1e-14
    np.random.seed(1234)
    for inplace in [False, True]:
        s = 0
        ell_min = abs(s)
        ell_max = 8
        a = np.random.rand(3, 7, sf.LM_total_size(ell_min, ell_max)*2).view(complex)
        # Test success with spin_weight==0
        m = sf.Modes(a, spin_weight=s, ell_min=ell_min, ell_max=ell_max)
        g = m.grid()
        s = m.s
        ell_min = m.ell_min
        ell_max = m.ell_max
        shape = m.shape
        mreal = m._real_func(inplace)
        greal = mreal.grid()
        assert s == mreal.s
        assert ell_min == mreal.ell_min
        assert ell_max == mreal.ell_max
        assert shape == mreal.shape
        assert np.allclose(greal, np.real(greal)+0.0j, rtol=tolerance, atol=tolerance)
        assert np.allclose(np.real(g), np.real(greal), rtol=tolerance, atol=tolerance)
        assert np.allclose(np.zeros_like(g, dtype=float), np.imag(greal), rtol=tolerance, atol=tolerance)
        # Test failure with s!=0
        for s in [-3, -2, -1, 1, 2, 3]:
            m = sf.Modes(a, spin_weight=s, ell_min=ell_min, ell_max=ell_max)
            with pytest.raises(ValueError):
                mreal = m._real_func(inplace)


@pytest.mark.xfail
def test_modes_imag():
    tolerance = 1e-14
    np.random.seed(1234)
    for inplace in [False, True]:
        s = 0
        ell_min = abs(s)
        ell_max = 8
        a = np.random.rand(3, 7, sf.LM_total_size(ell_min, ell_max)*2).view(complex)
        # Test success with spin_weight==0
        m = sf.Modes(a, spin_weight=s, ell_min=ell_min, ell_max=ell_max)
        g = m.grid()
        s = m.s
        ell_min = m.ell_min
        ell_max = m.ell_max
        shape = m.shape
        mimag = m._imag_func(inplace)
        gimag = mimag.grid()
        assert s == mimag.s
        assert ell_min == mimag.ell_min
        assert ell_max == mimag.ell_max
        assert shape == mimag.shape
        assert np.allclose(gimag, np.real(gimag), rtol=tolerance, atol=tolerance)  # gimag is purely real
        assert np.allclose(
            np.array(np.imag(g.ndarray), dtype=complex),
            gimag.ndarray,
            rtol=tolerance, atol=tolerance
        )  # imag(g) == gimag
        assert np.allclose(
            np.imag(gimag.ndarray),
            np.zeros_like(g.ndarray, dtype=float),
            rtol=tolerance, atol=tolerance
        ) # imag(gimag) == 0
        # Test failure with s!=0
        for s in [-3, -2, -1, 1, 2, 3]:
            m = sf.Modes(a, spin_weight=s, ell_min=ell_min, ell_max=ell_max)
            with pytest.raises(ValueError):
                mimag = m._imag_func(inplace)


def test_modes_squared_angular_momenta():
    tolerance = 1e-13
    np.random.seed(1234)
    L2 = sf.Modes.Lsquared
    Lz = sf.Modes.Lz
    Lp = sf.Modes.Lplus
    Lm = sf.Modes.Lminus
    R2 = sf.Modes.Rsquared
    Rz = sf.Modes.Rz
    Rp = sf.Modes.Rplus
    Rm = sf.Modes.Rminus
    for s in range(-2, 2+1):
        ell_min = abs(s)
        ell_max = 8
        a = np.random.rand(3, 7, sf.LM_total_size(ell_min, ell_max)*2).view(complex)
        m = sf.Modes(a, spin_weight=s, ell_min=ell_min, ell_max=ell_max)

        # Test L^2 = 0.5(L+L- + L-L+) + LzLz
        m1 = L2(m)
        m2 = 0.5 * (Lp(Lm(m)) + Lm(Lp(m))) + Lz(Lz(m))
        assert np.allclose(m1, m2, rtol=tolerance, atol=tolerance)

        # Test R^2 = 0.5(R+R- + R-R+) + RzRz
        m1 = R2(m)
        m2 = 0.5 * (Rp(Rm(m)) + Rm(Rp(m))) + Rz(Rz(m))
        assert np.allclose(m1, m2, rtol=tolerance, atol=tolerance)

        # Test L^2 = R^2
        m1 = L2(m)
        m2 = R2(m)
        assert np.allclose(m1, m2, rtol=tolerance, atol=tolerance)


def test_modes_derivative_commutators():
    tolerance = 1e-13
    np.random.seed(1234)
    # Note that post-fix operators are in the opposite order compared
    # to prefixed commutators, so we pull the post-fix operators out
    # as functions to make things look right.
    np.random.seed(1234)
    L2 = sf.Modes.Lsquared
    Lz = sf.Modes.Lz
    Lp = sf.Modes.Lplus
    Lm = sf.Modes.Lminus
    R2 = sf.Modes.Rsquared
    Rz = sf.Modes.Rz
    Rp = sf.Modes.Rplus
    Rm = sf.Modes.Rminus
    eth = lambda modes: modes.eth
    ethbar = lambda modes: modes.ethbar
    for s in range(-2, 2+1):
        ell_min = abs(s)
        ell_max = 8
        a = np.random.rand(3, 7, sf.LM_total_size(ell_min, ell_max)*2).view(complex)
        m = sf.Modes(a, spin_weight=s, ell_min=ell_min, ell_max=ell_max)
        # Test [Ri, Lj] = 0
        for R in [Rz, Rp, Rm]:
            for L in [Lz, Lp, Lm]:
                assert np.max(np.abs(L(R(m)) - R(L(m)))) < tolerance
        # Test [L2, Lj] = 0
        for L in [Lz, Lp, Lm]:
            assert np.max(np.abs(L2(L(m)) - L(L2(m)))) < 5*tolerance
        # Test [R2, Rj] = 0
        for R in [Rz, Rp, Rm]:
            assert np.max(np.abs(R2(R(m)) - R(R2(m)))) < 5*tolerance
        # Test [Lz, Lp] = Lp
        assert np.allclose(Lz(Lp(m)) - Lp(Lz(m)), Lp(m), rtol=tolerance, atol=tolerance)
        # Test [Lz, Lm] = -Lm
        assert np.allclose(Lz(Lm(m)) - Lm(Lz(m)), -Lm(m), rtol=tolerance, atol=tolerance)
        # Test [Lp, Lm] = 2Lz
        assert np.allclose(Lp(Lm(m)) - Lm(Lp(m)), 2 * Lz(m), rtol=tolerance, atol=tolerance)
        # Test [Rz, Rp] = Rp
        assert np.allclose(Rz(Rp(m)) - Rp(Rz(m)), Rp(m), rtol=tolerance, atol=tolerance)
        # Test [Rz, Rm] = -Rm
        assert np.allclose(Rz(Rm(m)) - Rm(Rz(m)), -Rm(m), rtol=tolerance, atol=tolerance)
        # Test [Rp, Rm] = 2Rz
        assert np.allclose(Rp(Rm(m)) - Rm(Rp(m)), 2 * Rz(m), rtol=tolerance, atol=tolerance)
        # Test [ethbar, eth] = 2s
        assert np.allclose(ethbar(eth(m)) - eth(ethbar(m)), 2 * m.s * m, rtol=tolerance, atol=tolerance)


@pytest.mark.xfail
def test_modes_derivatives_on_grids():
    # Test various SWSH-derivative expressions on grids
    tolerance = 2e-14
    np.random.seed(1234)
    for s in range(-2, 2+1):
        ell_min = 0
        ell_max = abs(s)+5
        zeros = lambda: np.zeros(sf.LM_total_size(ell_min, ell_max), dtype=complex)
        for ell in range(abs(s), ell_max+1):
            for m in range(-ell, ell+1):
                sYlm = sf.Modes(zeros(), spin_weight=s, ell_min=ell_min, ell_max=ell_max)
                sYlm[sYlm.index(ell, m)] = 1.0
                g_sYlm = sYlm.grid()
                n_theta, n_phi = g_sYlm.shape[-2:]

                # Test Lsquared {s}Y{l,m} = l * (l+1) * {s}Y{l,m}
                L2_sYlm = sYlm.Lsquared()
                g_L2_sYlm = L2_sYlm.grid(n_theta, n_phi)
                factor = ell * (ell+1)
                assert np.allclose(g_L2_sYlm, factor*g_sYlm, rtol=tolerance, atol=tolerance)

                # Test Lz {s}Y{l,m} = m * {s}Y{l,m}
                Lz_sYlm = sYlm.Lz()
                g_Lz_sYlm = Lz_sYlm.grid(n_theta, n_phi)
                factor = m
                assert np.allclose(g_Lz_sYlm, factor*g_sYlm, rtol=tolerance, atol=tolerance)

                # Test Lplus {s}Y{l,m} = sqrt((l-m)*(l+m+1)) {s}Y{l,m+1}
                invalid = abs(m+1) > ell
                sYlmp1 = sf.Modes(zeros(), spin_weight=s, ell_min=ell_min, ell_max=ell_max)
                if invalid:
                    with pytest.raises(ValueError):
                        sYlmp1.index(ell, m+1)
                else:
                    sYlmp1[sYlmp1.index(ell, m+1)] = 1.0
                g_sYlmp1 = sYlmp1.grid(n_theta, n_phi)
                Lp_sYlm = sYlm.Lplus()
                g_Lp_sYlm = Lp_sYlm.grid(n_theta, n_phi)
                factor = 0.0 if invalid else math.sqrt((ell-m)*(ell+m+1))
                assert np.allclose(g_Lp_sYlm, factor*g_sYlmp1, rtol=tolerance, atol=tolerance)

                # Test Lminus {s}Y{l,m} = sqrt((l+m)*(l-m+1)) * {s}Y{l,m-1}
                invalid = abs(m-1) > ell
                sYlmm1 = sf.Modes(zeros(), spin_weight=s, ell_min=ell_min, ell_max=ell_max)
                if invalid:
                    with pytest.raises(ValueError):
                        sYlmm1.index(ell, m-1)
                else:
                    sYlmm1[sYlmm1.index(ell, m-1)] = 1.0
                g_sYlmm1 = sYlmm1.grid(n_theta, n_phi)
                Lm_sYlm = sYlm.Lminus()
                g_Lm_sYlm = Lm_sYlm.grid(n_theta, n_phi)
                factor = 0.0 if invalid else math.sqrt((ell+m)*(ell-m+1))
                assert np.allclose(g_Lm_sYlm, factor*g_sYlmm1, rtol=tolerance, atol=tolerance)

                # Test Rsquared {s}Y{l,m} = l * (l+1) * {s}Y{l,m}
                R2_sYlm = sYlm.Rsquared()
                g_R2_sYlm = R2_sYlm.grid(n_theta, n_phi)
                factor = ell * (ell+1)
                assert np.allclose(g_R2_sYlm, factor*g_sYlm, rtol=tolerance, atol=tolerance)

                # Test Rz {s}Y{l,m} = -s * {s}Y{l,m}
                Rz_sYlm = sYlm.Rz()
                g_Rz_sYlm = Rz_sYlm.grid(n_theta, n_phi)
                factor = -s
                assert np.allclose(g_Rz_sYlm, factor*g_sYlm, rtol=tolerance, atol=tolerance)

                # Test Rplus {s}Y{l,m} = sqrt((l+s)(l-s+1)) {s-1}Y{l,m}
                invalid = abs(s-1) > ell
                sm1Ylm = sf.Modes(zeros(), spin_weight=s-1, ell_min=ell_min, ell_max=ell_max)
                if invalid:
                    with pytest.raises(ValueError):
                        sm1Ylm.index(ell, m)
                else:
                    sm1Ylm[sm1Ylm.index(ell, m)] = 1.0
                g_sm1Ylm = sm1Ylm.grid(n_theta, n_phi)
                Rp_sYlm = sYlm.Rplus()
                g_Rp_sYlm = Rp_sYlm.grid(n_theta, n_phi)
                factor = 0.0 if invalid else math.sqrt((ell+s)*(ell-s+1))
                assert np.allclose(g_Rp_sYlm, factor*g_sm1Ylm, rtol=tolerance, atol=tolerance)

                # Test Rminus {s}Y{l,m} = sqrt((l-s)(l+s+1)) {s+1}Y{l,m}
                invalid = abs(s+1) > ell
                sp1Ylm = sf.Modes(zeros(), spin_weight=s+1, ell_min=ell_min, ell_max=ell_max)
                if invalid:
                    with pytest.raises(ValueError):
                        sp1Ylm.index(ell, m)
                else:
                    sp1Ylm[sp1Ylm.index(ell, m)] = 1.0
                Rm_sYlm = sYlm.Rminus()
                g_sp1Ylm = sp1Ylm.grid(n_theta, n_phi)
                g_Rm_sYlm = Rm_sYlm.grid(n_theta, n_phi)
                factor = 0.0 if invalid else math.sqrt((ell-s)*(ell+s+1))
                assert np.allclose(g_Rm_sYlm, factor*g_sp1Ylm, rtol=tolerance, atol=tolerance)

                # Test eth {s}Y{l,m} = sqrt((l-s)(l+s+1)) {s+1}Y{l,m}
                invalid = abs(s+1) > ell
                sp1Ylm = sf.Modes(zeros(), spin_weight=s+1, ell_min=ell_min, ell_max=ell_max)
                if invalid:
                    with pytest.raises(ValueError):
                        sp1Ylm.index(ell, m)
                else:
                    sp1Ylm[sp1Ylm.index(ell, m)] = 1.0
                eth_sYlm = sYlm.eth
                g_sp1Ylm = sp1Ylm.grid(n_theta, n_phi)
                g_eth_sYlm = eth_sYlm.grid(n_theta, n_phi)
                factor = 0.0 if invalid else math.sqrt((ell-s)*(ell+s+1))
                assert np.allclose(g_eth_sYlm, factor*g_sp1Ylm, rtol=tolerance, atol=tolerance)

                # Test ethbar {s}Y{l,m} = -sqrt((l+s)(l-s+1)) {s-1}Y{l,m}
                invalid = abs(s-1) > ell
                sm1Ylm = sf.Modes(zeros(), spin_weight=s-1, ell_min=ell_min, ell_max=ell_max)
                if invalid:
                    with pytest.raises(ValueError):
                        sm1Ylm.index(ell, m)
                else:
                    sm1Ylm[sm1Ylm.index(ell, m)] = 1.0
                g_sm1Ylm = sm1Ylm.grid(n_theta, n_phi)
                ethbar_sYlm = sYlm.ethbar
                g_ethbar_sYlm = ethbar_sYlm.grid(n_theta, n_phi)
                factor = 0.0 if invalid else -math.sqrt((ell+s)*(ell-s+1))
                assert np.allclose(g_ethbar_sYlm, factor*g_sm1Ylm, rtol=tolerance, atol=tolerance)

                # Test ethbar eth sYlm = -(l-s)(l+s+1) sYlm
                ethbar_eth_sYlm = sYlm.eth.ethbar
                g_ethbar_eth_sYlm = ethbar_eth_sYlm.grid(n_theta, n_phi)
                factor = 0.0 if (abs(s+1) > ell or abs(s) > ell) else -(ell-s)*(ell+s+1)
                assert np.allclose(g_ethbar_eth_sYlm, factor*g_sYlm, rtol=tolerance, atol=tolerance)


def test_modes_norm():
    tolerance = 1e-15
    np.random.seed(1234)
    for s in range(-2, 2 + 1):
        ell_min = abs(s)
        ell_max = 8
        a = np.random.rand(3, 7, sf.LM_total_size(ell_min, ell_max)*2).view(complex)
        m = sf.Modes(a, spin_weight=s, ell_min=ell_min, ell_max=ell_max)
        mmbar = m.multiply(m.conjugate())
        norm = np.sqrt(2*math.sqrt(np.pi) * mmbar[..., 0].view(np.ndarray).real)
        assert np.allclose(norm, m.norm(), rtol=tolerance, atol=tolerance)


def test_modes_ufuncs():
    for s1 in range(-2, 2 + 1):
        ell_min1 = abs(s1)
        ell_max1 = 8
        a1 = np.random.rand(11, sf.LM_total_size(ell_min1, ell_max1)*2).view(complex)
        m1 = sf.Modes(a1, spin_weight=s1, ell_min=ell_min1, ell_max=ell_max1)
        positivem1 = +m1
        assert np.array_equal(m1.view(np.ndarray), positivem1.view(np.ndarray))
        negativem1 = -m1
        assert np.array_equal(-(m1.view(np.ndarray)), negativem1.view(np.ndarray))
        # for s2 in range(-2, 2 + 1):
        #     ell_min2 = abs(s2)
        #     ell_max2 = 8
        #     a2 = np.random.rand(3, 7, sf.LM_total_size(ell_min2, ell_max2)*2).view(complex)
        #     m2 = sf.Modes(a2, spin_weight=s2, ell_min=ell_min2, ell_max=ell_max2)
