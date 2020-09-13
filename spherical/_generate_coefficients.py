from spherical import ell_max

# The coefficients were originally produced with the following code,
# though obviously this doesn't need to be run each time.

import numpy as np
import mpmath

mpmath.mp.dps = 4 * ell_max

_binomial_coefficients = np.empty((((2 * ell_max + 1) * (2 * ell_max + 2)) // 2,), dtype=float)
i = 0
for n in range(2 * ell_max + 1):
    for k in range(n + 1):
        _binomial_coefficients[i] = float(mpmath.binomial(n, k))
        i += 1
print(i, _binomial_coefficients.shape)
np.save('binomial_coefficients', _binomial_coefficients)

_ladder_operator_coefficients = np.empty((((2*ell_max + 3) * ell_max + 1),), dtype=float)
i = 0
for twoell in range(2*ell_max + 1):
    for twom in range(-twoell, twoell + 1, 2):
        _ladder_operator_coefficients[i] = mpmath.sqrt((twoell * (twoell + 2) - twom * (twom + 2))/4)
        i += 1
print(i, _ladder_operator_coefficients.shape)
np.save('ladder_operator_coefficients', _ladder_operator_coefficients)

# _Wigner_coefficients = np.empty(((4 * ell_max ** 3 + 12 * ell_max ** 2 + 11 * ell_max + 3) // 3,), dtype=float)
# i = 0
# for ell in range(ell_max + 1):
#     for mp in range(-ell, ell + 1):
#         for m in range(-ell, ell + 1):
#             rho_min = max(0, mp - m)
#             _Wigner_coefficients[i] = float(mpmath.sqrt(mpmath.fac(ell + m) * mpmath.fac(ell - m)
#                                                         / (mpmath.fac(ell + mp) * mpmath.fac(ell - mp)))
#                                             * mpmath.binomial(ell + mp, rho_min)
#                                             * mpmath.binomial(ell - mp, ell - m - rho_min))
#             i += 1

_Wigner_coefficients = np.empty(((8 * ell_max ** 3 + 18 * ell_max ** 2 + 13 * ell_max + 3) // 3,), dtype=float)
i = 0
for twoell in range(0, 2*ell_max + 1):
    for twomp in range(-twoell, twoell + 1, 2):
        for twom in range(-twoell, twoell + 1, 2):
            tworho_min = max(0, twomp - twom)
            _Wigner_coefficients[i] = float(mpmath.sqrt(mpmath.fac((twoell + twom)//2) * mpmath.fac((twoell - twom)//2)
                                                        / (mpmath.fac((twoell + twomp)//2) * mpmath.fac((twoell - twomp)//2)))
                                            * mpmath.binomial((twoell + twomp)//2, tworho_min//2)
                                            * mpmath.binomial((twoell - twomp)//2, (twoell - twom - tworho_min)//2))
            i += 1
print(i, _Wigner_coefficients.shape)
np.save('Wigner_coefficients', _Wigner_coefficients)
