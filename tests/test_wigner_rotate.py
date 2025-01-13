#!/usr/bin/env python

# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import math
import numpy as np
import quaternionic
import spherical as sf
import pytest

slow = pytest.mark.slow


# @slow  ### Note that the following marks only the `horner=True` case as slow
@pytest.mark.parametrize("horner", [pytest.param(True, marks=pytest.mark.slow), False])
def test_wigner_rotate_composition(horner, Rs, ell_max_slow, eps):
    import time
    ell_min = 0
    ell_max = max(3, ell_max_slow)
    np.random.seed(1234)
    ϵ = (10 * (2 * ell_max + 1))**2 * eps
    wigner = sf.Wigner(ell_max)
    skipping = 5

    print()
    max_error = 0.0
    total_time = 0.0
    Rs = Rs[::skipping]
    for i, R1 in enumerate(Rs):
        # print(f"\tR1[{i+1}] of {len(Rs)}")
        for j, R2 in enumerate(Rs):
            for spin_weight in range(-2, 2+1):
                a1 = np.random.rand(7, sf.Ysize(ell_min, ell_max)*2).view(complex)
                a1[:, sf.Yindex(ell_min, -ell_min, ell_min):sf.Yindex(abs(spin_weight), -abs(spin_weight), ell_min)] = 0.0
                m1 = sf.Modes(a1, spin_weight=spin_weight, ell_min=ell_min, ell_max=ell_max)

                t1 = time.perf_counter()
                fA = wigner.rotate(wigner.rotate(m1, R1, horner=horner), R2, horner=horner)
                fB = wigner.rotate(m1, R1*R2, horner=horner)
                t2 = time.perf_counter()

                max_error = max(np.max(np.abs(fA-fB)), max_error)
                total_time += t2 - t1

                # import warnings
                # warnings.warn("Eliminating assert for debugging")
                assert np.allclose(fA.ndarray, fB.ndarray, rtol=ϵ, atol=ϵ), f"{np.max(np.abs((fA-fB).ndarray))} > {ϵ} for R1={R1} R2={R2}"

    print(f"\tmax_error[{horner}] = {max_error}")
    print(f"\ttotal_time[{horner}] = {total_time}")


@pytest.mark.parametrize("horner", [True, False])
def test_wigner_rotate_vector(horner, special_angles, Rs, eps):
    """Rotating a vector == rotating the mode-representation of that vector

    Note that the wigner.rotate function rotates the *basis* in which the modes are
    represented, so we rotate the modes by the inverse of the rotation we apply to
    the vector.

    """
    ell_min = 1
    ell_max = 1
    wigner = sf.Wigner(ell_max, ell_min=ell_min)

    def nhat(theta, phi):
        return quaternionic.array.from_vector_part([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta)
        ])

    for theta in special_angles[special_angles >= 0]:
        for phi in special_angles:
            v = nhat(theta, phi)
            vₗₘ = sf.Modes(sf.vector_as_ell_1_modes(v.vector), ell_min=ell_min, ell_max=ell_max, spin_weight=0)
            for R in Rs:
                vprm1 = (R * v * R.conjugate()).vector
                vₗₙ = wigner.rotate(vₗₘ, R.conjugate(), horner=horner).ndarray[1:]  # See note above
                vprm2 = sf.vector_from_ell_1_modes(vₗₙ).real
                assert np.allclose(vprm1, vprm2, atol=5*eps, rtol=0), (
                    f"\ntheta: {theta}\n"
                    f"phi: {phi}\n"
                    f"R: {R}\n"
                    f"v: {v}\n"
                    f"vprm1: {vprm1}\n"
                    f"vprm2: {vprm2}\n"
                    f"vprm1-vprm2: {vprm1-vprm2}\n"
                )
