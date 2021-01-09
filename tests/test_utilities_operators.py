#!/usr/bin/env python

# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import platform
import numpy as np
import pytest
import quaternionic
import spherical as sf

from .conftest import requires_spinsfast


@pytest.mark.parametrize("eth, spin_weight_of_eth", [(sf.eth_NP, 1), (sf.eth_GHP, 1),
                                                     (sf.ethbar_NP, -1), (sf.ethbar_GHP, -1)])
@requires_spinsfast
def test_eth_derivation(eth, spin_weight_of_eth):
    """Ensure that the various `eth` operators are derivations -- i.e., they obey the Leibniz product law

    Given two spin-weighted functions `f` and `g`, we need to test that

        eth(f * g) = eth(f) * g + f * eth(g)

    This test generates a set of random modes with equal power for `f` and `g`
    (though more realistic functions can be expected to have exponentially decaying
    mode amplitudes).  Because of the large power in high-ell modes, we need to
    double the number of modes in the representation of their product, which is why
    we use

        n_theta = n_phi = 4 * ell_max + 1

    These `f` and `g` functions must be transformed to the physical-space
    representation, multiplied there, the product transformed back to spectral
    space, the eth operator evaluated, and then transformed back again to physical
    space for comparison.

    We test both the Newman-Penrose and Geroch-Held-Penrose versions of eth, as
    well as their conjugates.

    """
    import spinsfast
    ell_max = 16
    n_modes = sf.LM_total_size(0, ell_max)
    n_theta = n_phi = 4 * ell_max + 1
    for s1 in range(-2, 2 + 1):
        for s2 in range(-s1, s1 + 1):
            np.random.seed(1234)
            ell_min1 = abs(s1)
            ell_min2 = abs(s2)
            f = np.random.rand(n_modes) + 1j * np.random.rand(n_modes)
            f[:sf.LM_total_size(0, ell_min1-1)] = 0j
            f_j_k = spinsfast.salm2map(f, s1, ell_max, n_theta, n_phi)
            g = np.random.rand(n_modes) + 1j * np.random.rand(n_modes)
            g[:sf.LM_total_size(0, ell_min2-1)] = 0j
            g_j_k = spinsfast.salm2map(g, s2, ell_max, n_theta, n_phi)
            fg_j_k = f_j_k * g_j_k
            fg = spinsfast.map2salm(fg_j_k, s1+s2, 2*ell_max)
            ethf = eth(f, s1, ell_min=0)
            ethg = eth(g, s2, ell_min=0)
            ethfg = eth(fg, s1+s2, ell_min=0)
            ethf_j_k = spinsfast.salm2map(ethf, s1+spin_weight_of_eth, ell_max, n_theta, n_phi)
            ethg_j_k = spinsfast.salm2map(ethg, s2+spin_weight_of_eth, ell_max, n_theta, n_phi)
            ethfg_j_k = spinsfast.salm2map(ethfg, s1+s2+spin_weight_of_eth, 2*ell_max, n_theta, n_phi)
            assert np.allclose(ethfg_j_k, ethf_j_k * g_j_k + f_j_k * ethg_j_k, rtol=1e-10, atol=1e-10)


def test_ethbar_inverse_NP():
    for s in range(-5, 5+1):
        f = np.random.random(81) + 1j * np.random.random(81)
        f[:sf.LM_index(abs(s), -abs(s), 0)] = 0.0

        # Test ethbar_inverse_NP(ethbar_NP(x)) == x
        ethbarf = sf.ethbar_NP(f, s, 0)
        ethbarf_prime = sf.ethbar_inverse_NP(ethbarf, s-1, 0)
        assert np.allclose(f[sf.LM_index(abs(s-1), -abs(s-1), 0):],
                           ethbarf_prime[sf.LM_index(abs(s-1), -abs(s-1), 0):],
                           atol=0, rtol=1e-15)
        if sf.LM_index(abs(s), -abs(s), 0) < sf.LM_index(abs(s-1), -abs(s-1), 0):
            assert abs(ethbarf_prime[sf.LM_index(abs(s), -abs(s), 0):sf.LM_index(abs(s-1), -abs(s-1), 0)]).max() == 0.0

        # Test ethbar_NP(ethbar_inverse_NP(x)) == x
        fprime = sf.ethbar_inverse_NP(f, s, 0)
        ethbar_fprime = sf.ethbar_NP(fprime, s+1, 0)
        assert np.allclose(f[sf.LM_index(abs(s+1), -abs(s+1), 0):],
                           ethbar_fprime[sf.LM_index(abs(s+1), -abs(s+1), 0):],
                           atol=0, rtol=1e-15)
        if sf.LM_index(abs(s), -abs(s), 0) < sf.LM_index(abs(s+1), -abs(s+1), 0):
            assert abs(ethbar_fprime[sf.LM_index(abs(s), -abs(s), 0):sf.LM_index(abs(s+1), -abs(s+1), 0)]).max() == 0.0
