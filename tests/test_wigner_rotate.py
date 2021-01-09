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
def test_wigner_rotate_composition(Rs, ell_max_slow, eps):
    ell_min = 0
    ell_max = max(3, ell_max_slow)
    np.random.seed(1234)
    ϵ = (10 * (2 * ell_max + 1))**2 * eps
    wigner = sf.Wigner(ell_max)
    skipping = 5

    for i, R1 in enumerate(Rs[::skipping]):
        for j, R2 in enumerate(Rs[::skipping]):
            for spin_weight in range(-2, 2+1):
                a1 = np.random.rand(7, sf.Ysize(ell_min, ell_max)*2).view(complex)
                a1[:, sf.Yindex(ell_min, -ell_min, ell_min):sf.Yindex(abs(spin_weight), -abs(spin_weight), ell_min)] = 0.0
                m1 = sf.Modes(a1, spin_weight=spin_weight, ell_min=ell_min, ell_max=ell_max)

                fA = wigner.rotate(wigner.rotate(m1, R1), R2)
                fB = wigner.rotate(m1, R1*R2)

                assert np.allclose(fA, fB, rtol=ϵ, atol=ϵ), f"{np.max(np.abs(fA-fB))} > {ϵ} for R1={R1} R2={R2}"
