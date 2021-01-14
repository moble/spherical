#!/usr/bin/env python

# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import math
import numpy as np
import quaternionic
import spherical as sf


def test_constant_as_ell_0_mode(special_angles):
    ell_max = 1
    wigner = sf.Wigner(ell_max)
    np.random.seed(123)
    for imaginary_part in [0.0, 1.0j]:  # Test both real and imaginary constants
        for _ in range(1000):
            constant = np.random.uniform(-1, 1) + imaginary_part * np.random.uniform(-1, 1)
            const_ell_m = sf.constant_as_ell_0_mode(constant)
            assert abs(constant - sf.constant_from_ell_0_mode(const_ell_m)) < 1e-15
            for theta in special_angles:
                for phi in special_angles:
                    Y = wigner.sYlm(0, quaternionic.array.from_spherical_coordinates(theta, phi))
                    dot = np.dot(const_ell_m, Y[0])
                    assert abs(constant - dot) < 1e-15, imaginary_part


def test_vector_as_ell_1_modes(special_angles):
    ell_min = 1
    ell_max = 1
    wigner = sf.Wigner(ell_max, ell_min=ell_min)

    def nhat(theta, phi):
        return np.array([math.sin(theta) * math.cos(phi),
                         math.sin(theta) * math.sin(phi),
                         math.cos(theta)])

    np.random.seed(123)
    for _ in range(1000):
        vector = np.random.uniform(-1, 1, size=(3,))
        vec_ell_m = sf.vector_as_ell_1_modes(vector)
        assert np.allclose(vector, sf.vector_from_ell_1_modes(vec_ell_m), atol=1.0e-16, rtol=1.0e-15)
        for theta in special_angles:
            for phi in special_angles:
                dot1 = np.dot(vector, nhat(theta, phi))
                dot2 = np.dot(vec_ell_m, wigner.sYlm(0, quaternionic.array.from_spherical_coordinates(theta, phi))).real
                assert abs(dot1 - dot2) < 1e-15
