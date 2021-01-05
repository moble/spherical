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
    # errors = np.empty(hcalc.Hsize)

    # print("Hnmpm:")
    # print(repr(Hnmpm))

    # print()
    for n in range(w.ell_max+1):
        # print(f'Testing n={n} compared to sympy')

        # print("sympy:")
        # print([[sympy.re(sympy.N(sympyWignerD(n, mp, m, alpha, -beta, gamma).doit()))
        #         for mp in range(-n, n+1)]
        #        for m in range(-n, n+1)
        #        ])
        # print()
        # print("spherical:")
        # print([[ϵ(mp) * ϵ(-m) * Hnmpm[sf.WignerHindex(n, mp, m)]
        #         for mp in range(-n, n+1)]
        #        for m in range(-n, n+1)
        #        ])

        for mp in range(-n, n+1):
            for m in range(-n, n+1):
                sympyd = sympy.re(sympy.N(sympyWignerD(n, mp, m, alpha, -beta, gamma).doit()))
                # myd = ϵ(mp) * ϵ(-m) * Hnmpm[sf.WignerHindex(n, mp, m)]
                myd = ϵ(-mp) * ϵ(m) * Hnmpm[sf.WignerHindex(n, mp, m)]
                error = float(abs(sympyd-myd))
                # error = float(min(abs(sympyd+myd), abs(sympyd-myd)))
                if error >= 3*eps:
                    print(
                        f"Hnmpm[sf.WignerHindex({n}, {mp}, {m})]: "
                        f"{Hnmpm.tolist()}[{sf.WignerHindex(n, mp, m)}] = "
                        f"{Hnmpm[sf.WignerHindex(n, mp, m)]}"
                    )
                assert error < 3*eps, f"Testing Wigner d recursion: n={n}, m'={mp}, m={m}, sympy:{sympyd}, spherical:{myd}, error={error}"
                max_error = max(error, max_error)
                # errors[i] = float(min(abs(sympyd+myd), abs(sympyd-myd)))
                # print("{:>5} {:>5} {:>5} {:24} {:24} {:24}".format(n, mp, m, float(sympyd), myd, errors[i]))

    # print(f"\nTesting H (Wigner d recursion): max error = {max_error}")


# def test_WignerDRecursion_timing():
#     import timeit
#     import textwrap
#     print()
#     hcalc = HCalculator(8)
#     for ell_max in [8, 100]:
#         cosβ = 2*np.random.rand(2*ell_max+1, 2*ell_max+1) - 1
#         workspace = hcalc.workspace(cosβ)
#         size = hcalc(cosβ, workspace=workspace).size  # Run once to ensure everything is compiled
#         number = 1000 // ell_max
#         time = timeit.timeit('hcalc(cosβ, workspace=workspace)', number=number, globals={'hcalc': hcalc, 'cosβ': cosβ, 'workspace': workspace})
#         print('Time for ell_max={} grid points was {}ms per call; {}ns per element'.format(100, 1_000*time/number, 1_000_000_000*time/(number*size)))


# @pytest.mark.skipif(platform.system() == "Windows", reason="line_profiler is missing")
# def test_WignerDRecursion_lineprofiling():
#     from line_profiler import LineProfiler
#     ell_max = 8
#     hcalc = HCalculator(ell_max)
#     cosβ = 2*np.random.rand(100, 100) - 1
#     workspace = hcalc.workspace(cosβ)
#     hcalc(cosβ, workspace=workspace)  # Run once to ensure everything is compiled
#     profiler = LineProfiler(hcalc.__call__)#, _step_2, _step_3, _step_4, _step_5, _step_6)
#     profiler.runctx('hcalc(cosβ, workspace=workspace)', {'hcalc': hcalc, 'cosβ': cosβ, 'workspace': workspace}, {})
#     print()
#     profiler.print_stats()
