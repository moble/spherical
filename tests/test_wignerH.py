#!/usr/bin/env python

# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import math
import cmath
import numpy as np
import spherical as sf
import pytest

slow = pytest.mark.slow
eps = np.finfo(float).eps


from spherical.recursions.wignerH import *


def test_wedge_size():
    for ℓₘₐₓ in range(17):
        for mpₘₐₓ in range(17):
            indices = [
                (n, mp, m)
                for n in range(ℓₘₐₓ+1)
                for mp in range(-min(n, mpₘₐₓ), min(n, mpₘₐₓ)+1)
                for m in range(abs(mp), n+1)
            ]
            #print(indices)
            #print(f"{ℓₘₐₓ:_}, {mpₘₐₓ:_}, {wedge_size(ℓₘₐₓ, mpₘₐₓ):_}, {len(indices):_}")
            assert wedge_size(ℓₘₐₓ, mpₘₐₓ) == len(indices)


def test_wedge_index():
    # total_comparisons = 0
    for ℓₘₐₓ in range(17):
        for mpₘₐₓ in range(ℓₘₐₓ+1):
            indices = [
                (n, mp, m)
                for n in range(ℓₘₐₓ+1)
                for mp in range(-min(n, mpₘₐₓ), min(n, mpₘₐₓ)+1)
                for m in range(abs(mp), n+1)
            ]
            for i, (n, mp, m) in enumerate(indices):
                if mpₘₐₓ == ℓₘₐₓ:
                    j = wedge_index(n, mp, m, None)
                else:
                    j = wedge_index(n, mp, m, mpₘₐₓ)
                assert i == j, (i, j, (n, mp, m, mpₘₐₓ), indices)
                # total_comparisons += 1
            for ℓ in range(ℓₘₐₓ+1):
                #for mp in range(-mpₘₐₓ, mpₘₐₓ+1):
                for mp in range(-ℓ, ℓ+1):
                    m_limit = mpₘₐₓ if abs(mp) > mpₘₐₓ else ℓ
                    #for m in range(-ℓ, ℓ+1):
                    for m in range(-m_limit, m_limit+1):
                        j = wedge_index(ℓ, mp, m, mpₘₐₓ)
                        if m < -mp:
                            if m < mp:
                                i = indices.index((ℓ, -mp, -m))
                            else:
                                i = indices.index((ℓ, -m, -mp))
                        else:
                            if m < mp:
                                i = indices.index((ℓ, m, mp))
                            else:
                                i = indices.index((ℓ, mp, m))
                        assert i == j, (i, j, (ℓ, mp, m, mpₘₐₓ), indices)
                        # total_comparisons += 1
    # print(f"total comparisons: {total_comparisons}")


def test_sign():
    for x in np.arange(-20, 21):
        if x == 0:
            assert sign(x) == 1
        else:
            assert sign(x) == np.sign(x)


def test_nm_index():
    n_max = 17
    indices = [[n, m] for n in range(n_max+1) for m in range(-n, n+1)]
    for n in range(n_max+1):
        for m in range(-n, n+1):
            assert [n, m] == indices[nm_index(n, m)]


def test_nabsm_index():
    n_max = 17
    indices = [[n, m] for n in range(n_max+1) for m in range(n+1)]
    for n in range(n_max+1):
        for m in range(n+1):
            assert [n, m] == indices[nabsm_index(n, m)], (n, m, nabsm_index(n, m), indices)


def test_nmpm_index():
    n_max = 17
    indices = [[n, mp, m] for n in range(n_max+1) for mp in range(-n, n+1) for m in range(-n, n+1)]
    for n in range(n_max+1):
        for mp in range(-n, n+1):
            for m in range(-n, n+1):
                assert [n, mp, m] == indices[nmpm_index(n, mp, m)]


def test_wigner_d():
    import quaternionic
    ell_max = 17
    β = np.linspace(0, np.pi)
    R = quaternionic.array.from_spherical_coordinates(β, 0)
    d0 = np.array([sf.Wigner_D_matrices(Ri, 0, ell_max) for Ri in R]).T
    d1 = HCalculator(ell_max).wigner_d(np.cos(β))
    assert np.max(np.abs(d1-d0)) < 4e3 * eps


def test_wigner_d_mp_max():
    import quaternionic
    ell_max = 17
    for mp_max in range(ell_max-1, ell_max+1):
        β = np.linspace(0, np.pi)
        R = quaternionic.array.from_spherical_coordinates(β, 0)
        d0 = np.array([sf.Wigner_D_matrices(Ri, 0, ell_max) for Ri in R]).T
        d1 = HCalculator(ell_max, mp_max).wigner_d(np.cos(β))
        d0 = np.array([
            d0[i]
            for ell in range(0, ell_max+1)
            for mp in range(-mp_max, mp_max+1)
            for m in range(-ell, ell+1)
            for i in [sf.LMpM_index(ell, mp, m, 0)]
        ])
        d1 = np.array([
            d1[i]
            for ell in range(0, ell_max+1)
            for mp in range(-mp_max, mp_max+1)
            for m in range(-ell, ell+1)
            for i in [sf.LMpM_index(ell, mp, m, 0)]
        ])
        assert np.max(np.abs(d1-d0)) < 4e3 * eps
        raise NotImplementedError("wigner_d didn't get the memo about mp_max (because LMpM_index and total size didn't)")


def test_wigner_D():
    import quaternionic
    ell_max = 17
    H = HCalculator(ell_max)
    for i in range(50):
        R = quaternionic.array(np.random.rand(4)).normalized
        d0 = sf.Wigner_D_matrices(R, 0, ell_max).T
        d1 = H.wigner_D(R)
        if np.max(np.abs(d1-d0)) >= 1e4 * eps:
            print()
            print(i)
            print("d0 =", d0.tolist())
            print("d1 =", d1.tolist())
        assert np.max(np.abs(d1-d0)) < 1e4 * eps