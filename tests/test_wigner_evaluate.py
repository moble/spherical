#!/usr/bin/env python

# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import numpy as np
import quaternionic
import spherical as sf
import pytest

from .conftest import requires_spinsfast

slow = pytest.mark.slow


@slow
def test_wigner_evaluate(ell_max_slow, eps):
    # import time

    ell_max = max(3, ell_max_slow)
    np.random.seed(1234)
    ϵ = 10 * (2 * ell_max + 1) * eps
    n_theta = n_phi = 2 * ell_max + 1
    max_s = 2
    wigner = sf.Wigner(ell_max, mp_max=max_s)

    for rotors in [
            quaternionic.array.from_spherical_coordinates(sf.theta_phi(n_theta, n_phi)),
            quaternionic.array(np.random.rand(n_theta, n_phi, 4)).normalized
    ]:

        for s in range(-max_s, max_s + 1):
            ell_min = abs(s)

            a1 = np.random.rand(7, sf.Ysize(ell_min, ell_max)*2).view(complex)
            a1[:, sf.Yindex(ell_min, -ell_min, ell_min):sf.Yindex(abs(s), -abs(s), ell_min)] = 0.0
            m1 = sf.Modes(a1, spin_weight=s, ell_min=ell_min, ell_max=ell_max)

            # tic = time.perf_counter()
            f1 = wigner.evaluate(m1, rotors)
            # toc = time.perf_counter()
            assert f1.shape == m1.shape[:-1] + rotors.shape[:-1]
            # print(f"Evaluation for s={s} took {toc-tic:.4f} seconds")
            # print(f1.shape)

            sYlm = np.zeros((sf.Ysize(0, ell_max),) + rotors.shape[:-1], dtype=complex)
            for i, Rs in enumerate(rotors):
                for j, R in enumerate(Rs):
                    wigner.sYlm(s, R, out=sYlm[:, i, j])
            f2 = np.tensordot(m1.view(np.ndarray), sYlm, axes=([-1], [0]))
            assert f2.shape == m1.shape[:-1] + rotors.shape[:-1]

            assert np.allclose(f1, f2, rtol=ϵ, atol=ϵ), f"max|f1-f2|={np.max(np.abs(f1-f2))} > ϵ={ϵ}"


@requires_spinsfast
@slow
def test_wigner_spinsfast(ell_max_slow, eps):
    # import time

    ell_max = max(3, ell_max_slow)
    np.random.seed(1234)
    ϵ = 10 * (2 * ell_max + 1) * eps
    n_theta = n_phi = 2 * ell_max + 1
    max_s = 2
    wigner = sf.Wigner(ell_max, mp_max=max_s)

    rotors = quaternionic.array.from_spherical_coordinates(sf.theta_phi(n_theta, n_phi))

    for s in range(-max_s, max_s + 1):
        ell_min = abs(s)

        a1 = np.random.rand(7, sf.Ysize(ell_min, ell_max)*2).view(complex)
        m1 = sf.Modes(a1, spin_weight=s, ell_min=ell_min, ell_max=ell_max)

        f1 = wigner.evaluate(m1, rotors)
        assert f1.shape == m1.shape[:-1] + rotors.shape[:-1]

        f2 = m1.grid(n_theta, n_phi, use_spinsfast=True)
        assert f2.shape == m1.shape[:-1] + rotors.shape[:-1]

        assert np.allclose(f1, f2.ndarray, rtol=ϵ, atol=ϵ), f"max|f1-f2|={np.max(np.abs(f1-f2.ndarray))} > ϵ={ϵ}"
