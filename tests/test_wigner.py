#!/usr/bin/env python

# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import sympy
import numpy as np
import spherical as sf


def test_H(eps):
    from sympy.physics.quantum.spin import WignerD as sympyWignerD

    """Eq. (29) of arxiv:1403.7698: d^{m',m}_{n}(β) = ϵ(m') ϵ(-m) H^{m',m}_{n}(β)"""

    def ϵ(m):
        m = np.asarray(m)
        eps = np.ones_like(m)
        eps[m >= 0] = (-1)**m[m >= 0]
        return eps

    ell_max = 4
    alpha, beta, gamma = 0.0, 0.1, 0.0
    w = sf.Wigner(ell_max)
    Hnmpm = w.H(np.exp(1j * beta))
    max_error = 0.0

    for n in range(w.ell_max+1):
        for mp in range(-n, n+1):
            for m in range(-n, n+1):
                sympyd = sympy.re(sympy.N(sympyWignerD(n, mp, m, alpha, beta, gamma).doit()))
                myd = ϵ(mp) * ϵ(-m) * Hnmpm[sf.WignerHindex(n, mp, m)]
                # sympyd = sympy.re(sympy.N(sympyWignerD(n, mp, m, alpha, -beta, gamma).doit()))
                # myd = ϵ(-mp) * ϵ(m) * Hnmpm[sf.WignerHindex(n, mp, m)]
                error = float(abs(sympyd-myd))
                assert error < 3*eps, f"Testing Wigner d recursion: n={n}, m'={mp}, m={m}, sympy:{sympyd}, spherical:{myd}, error={error}"
                max_error = max(error, max_error)

    # print(f"\nTesting H (Wigner d recursion): max error = {max_error}")
