#!/usr/bin/env python

# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import numpy as np

def test_complex_powers(eps):
    from spherical import complex_powers
    M = 10_000
    np.random.seed(12345)
    # Test 1000 random values roughly less than 2π, and a bunch of fractions of exactly π
    θs = np.concatenate((np.random.rand(1000)*6.3,
                         np.pi * np.array([0, 1/8, 1/4, 1/3, 1/2, 2/3, 3/4, 1, 5/4, 3/2, 7/4, 2, 9/4])))
    zs = np.exp(1j * θs)
    zpowers1 = complex_powers(zs, M)
    assert zpowers1.shape == zs.shape+(M+1,)
    mbroadcaster = np.arange(M+1).reshape((1,)*zs.ndim + (M+1,))
    zbroadcaster = zs.reshape(zs.shape+(1,))
    zpowers2 = zbroadcaster**mbroadcaster
    assert np.allclose(zpowers1, zpowers2, atol=3 * eps * M, rtol=0)
    assert np.all(np.abs(zpowers1 - zpowers2)[:, 1:] < 4 * eps * np.arange(M+1)[np.newaxis, 1:])


