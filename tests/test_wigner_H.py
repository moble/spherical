#!/usr/bin/env python

# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import sympy
import numpy as np
import spherical as sf
import pytest

from .conftest import requires_sympy

slow = pytest.mark.slow


@requires_sympy
@slow
def test_H_vs_sympy(ell_max_slow, eps):
    from sympy.physics.quantum.spin import WignerD as Wigner_D_sympy

    """Eq. (29) of arxiv:1403.7698: d^{m',m}_{n}(β) = ϵ(m') ϵ(-m) H^{m',m}_{n}(β)"""

    def ϵ(m):
        m = np.asarray(m)
        eps = np.ones_like(m)
        eps[m >= 0] = (-1)**m[m >= 0]
        return eps

    ell_max = max(4, ell_max_slow // 2)
    alpha, beta, gamma = 0.0, 0.1, 0.0
    max_error = 0.0

    for mp_max in range(ell_max):
        w = sf.Wigner(ell_max, mp_max=mp_max)
        workspace = w.new_workspace()
        Hwedge, Hv, Hextra, _, _, _ = w._split_workspace(workspace)
        Hnmpm = w.H(np.exp(1j * beta), Hwedge, Hv, Hextra)
        for n in range(w.ell_max+1):
            for mp in range(-min(n, mp_max), min(n, mp_max)+1):
                for m in range(-n, n+1):
                    sympyd = sympy.re(sympy.N(Wigner_D_sympy(n, mp, m, alpha, beta, gamma).doit()))
                    sphericald = ϵ(mp) * ϵ(-m) * Hnmpm[sf.WignerHindex(n, mp, m, mp_max)]
                    error = float(abs(sympyd-sphericald))
                    assert error < 4.1 * eps, (
                        f"Testing Wigner d recursion with n={n}, m'={mp}, m={m}, mp_max={mp_max}, "
                        f"sympyd={sympyd}, sphericald={sphericald}, error={error}"
                    )
