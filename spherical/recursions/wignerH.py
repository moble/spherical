"""Algorithm for computing H, as given by arxiv:1403.7698

H is related to Wigner's (small) d via

    dₗⁿᵐ = ϵₙ ϵ₋ₘ Hₗⁿᵐ,

where

         ⎧ 1 for k≤0
    ϵₖ = ⎨
         ⎩ (-1)ᵏ for k>0

H has various advantages over d, including the fact that it can be efficiently
and robustly valculated via recurrence relations, and the following symmetry
relations:

    H^{m', m}_n(β) = H^{m, m'}_n(β)
    H^{m', m}_n(β) = H^{-m', -m}_n(β)
    H^{m', m}_n(β) = (-1)^{n+m+m'} H^{-m', m}_n(π - β)
    H^{m', m}_n(β) = (-1)^{m+m'} H^{m', m}_n(-β)

Because of these symmetries, we only need to evaluate at most 1/4 of all the
elements.

"""

import numpy as np
import quaternionic

from .. import jit, complex_powers, WignerHindex

sqrt3 = np.sqrt(3)
inverse_sqrt2 = 1.0 / np.sqrt(2)


@jit
def ϵ(m):
    if m <= 0:
        return 1
    elif m%2:
        return -1
    else:
        return 1


@jit
def sign(m):
    """Return sign of input, with sign(0)=1"""
    if m >= 0:
        return 1
    else:
        return -1


@jit
def nm_index(n, m):
    """Return flat index into arrray of [n, m] pairs.

    Assumes array is ordered as

        [
            [n, m]
            for n in range(n_max+1)
            for m in range(-n, n+1)
        ]

    """
    return m + n * (n + 1)


@jit
def nabsm_index(n, absm):
    """Return flat index into arrray of [n, abs(m)] pairs

    Assumes array is ordered as

        [
            [n, m]
            for n in range(n_max+1)
            for m in range(n+1)
        ]

    """
    return absm + (n * (n + 1)) // 2


@jit
def nmpm_index(n, mp, m):
    """Return flat index into arrray of [n, mp, m]

    Assumes array is ordered as

        [
            [n, mp, m]
            for n in range(n_max+1)
            for mp in range(-n, n+1)
            for m in range(-n, n+1)
        ]

    """
    return (((4 * n + 6) * n + 6 * mp + 5) * n + 3 * (m + mp)) // 3


@jit
def _step_1(Hwedge):
    """If n=0 set H_{0}^{0,0}=1."""
    Hwedge[0] = 1.0


@jit
def _step_2(g, h, n_max, mp_max, Hwedge, Hextra, Hv, expiβ):
    """Compute values H^{0,m}_{n}(β)for m=0,...,n and H^{0,m}_{n+1}(β) for m=0,...,n+1 using Eq. (32):

        H^{0,m}_{n}(β) = (-1)^m √((n-|m|)! / (n+|m|)!) P^{|m|}_{n}(cos β)
                       = (-1)^m (sin β)^m P̂^{|m|}_{n}(cos β) / √(k (2n+1))

    This function computes the associated Legendre functions directly by recursion
    as explained by Holmes and Featherstone (2002), doi:10.1007/s00190-002-0216-2.
    Note that I had to adjust certain steps for consistency with the notation
    assumed by arxiv:1403.7698 -- mostly involving factors of (-1)**m.

    NOTE: Though not specified in arxiv:1403.7698, there is not enough information
    for step 4 unless we also use symmetry to set H^{1,0}_{n} here.  Similarly,
    step 5 needs additional information, which depends on setting H^{0, -1}_{n}
    from its symmetric equivalent H^{0, 1}_{n} in this step.

    """
    cosβ = expiβ.real
    sinβ = expiβ.imag
    if n_max > 0:
        # n = 1
        n0n_index = WignerHindex(1, 0, 1, mp_max)
        nn_index = nm_index(1, 1)
        Hwedge[n0n_index] = sqrt3  # Un-normalized
        Hwedge[n0n_index-1] = (g[nn_index-1] * cosβ) * inverse_sqrt2  # Normalized
        # n = 2, ..., n_max+1
        for n in range(2, n_max+2):
            if n <= n_max:
                n0n_index = WignerHindex(n, 0, n, mp_max)
                H = Hwedge
            else:
                n0n_index = n
                H = Hextra
            nm10nm1_index = WignerHindex(n-1, 0, n-1, mp_max)
            nn_index = nm_index(n, n)
            const = np.sqrt(1.0 + 0.5/n)
            gi = g[nn_index-1]
            # m = n
            H[n0n_index] = const * Hwedge[nm10nm1_index]
            # m = n-1
            H[n0n_index-1] = gi * cosβ * H[n0n_index]
            # m = n-2, ..., 1
            for i in range(2, n):
                gi = g[nn_index-i]
                hi = h[nn_index-i]
                H[n0n_index-i] = gi * cosβ * H[n0n_index-i+1] - hi * sinβ**2 * H[n0n_index-i+2]
            # m = 0, with normalization
            const = 1.0 / np.sqrt(4*n+2)
            gi = g[nn_index-n]
            hi = h[nn_index-n]
            H[n0n_index-n] = (gi * cosβ * H[n0n_index-n+1] - hi * sinβ**2 * H[n0n_index-n+2]) * const
            # Now, loop back through, correcting the normalization for this row, except for n=n element
            prefactor = const
            for i in range(1, n):
                prefactor *= sinβ
                H[n0n_index-n+i] *= prefactor
            # Supply extra edge cases as noted in docstring
            if n <= n_max:
                Hv[nm_index(n, 1)] = Hwedge[WignerHindex(n, 0, 1, mp_max)]
                Hv[nm_index(n, 0)] = Hwedge[WignerHindex(n, 0, 1, mp_max)]
        # Correct normalization of m=n elements
        prefactor = 1.0
        for n in range(1, n_max+1):
            prefactor *= sinβ
            Hwedge[WignerHindex(n, 0, n, mp_max)] *= prefactor / np.sqrt(4*n+2)
        for n in [n_max+1]:
            prefactor *= sinβ
            Hextra[n] *= prefactor / np.sqrt(4*n+2)
        # Supply extra edge cases as noted in docstring
        Hv[nm_index(1, 1)] = Hwedge[WignerHindex(1, 0, 1, mp_max)]
        Hv[nm_index(1, 0)] = Hwedge[WignerHindex(1, 0, 1, mp_max)]


@jit
def _step_3(a, b, n_max, mp_max, Hwedge, Hextra, expiβ):
    """Use relation (41) to compute H^{1,m}_{n}(β) for m=1,...,n.  Using symmetry and shift
    of the indices this relation can be written as

        b^{0}_{n+1} H^{1, m}_{n} =   (b^{−m−1}_{n+1} (1−cosβ))/2 H^{0, m+1}_{n+1}
                                   − (b^{ m−1}_{n+1} (1+cosβ))/2 H^{0, m−1}_{n+1}
                                   − a^{m}_{n} sinβ H^{0, m}_{n+1}

    """
    cosβ = expiβ.real
    sinβ = expiβ.imag
    if n_max > 0 and mp_max > 0:
        for n in range(1, n_max+1):
            # m = 1, ..., n
            i1 = WignerHindex(n, 1, 1, mp_max)
            if n+1 <= n_max:
                i2 = WignerHindex(n+1, 0, 0, mp_max)
                H2 = Hwedge
            else:
                i2 = 0
                H2 = Hextra
            i3 = nm_index(n+1, 0)
            i4 = nabsm_index(n, 1)
            inverse_b5 = 1.0 / b[i3]
            for i in range(n):
                b6 = b[-i+i3-2]
                b7 = b[i+i3]
                a8 = a[i+i4]
                Hwedge[i+i1] = inverse_b5 * (
                    0.5 * (
                          b6 * (1-cosβ) * H2[i+i2+2]
                        - b7 * (1+cosβ) * H2[i+i2]
                    )
                    - a8 * sinβ * H2[i+i2+1]
                )


@jit
def _step_4(d, n_max, mp_max, Hwedge, Hv):
    """Recursively compute H^{m'+1, m}_{n}(β) for m'=1,...,n−1, m=m',...,n using relation (50) resolved
    with respect to H^{m'+1, m}_{n}:

      d^{m'}_{n} H^{m'+1, m}_{n} =   d^{m'−1}_{n} H^{m'−1, m}_{n}
                                   − d^{m−1}_{n} H^{m', m−1}_{n}
                                   + d^{m}_{n} H^{m', m+1}_{n}

    (where the last term drops out for m=n).

    """
    if n_max > 0 and mp_max > 0:
        for n in range(2, n_max+1):
            for mp in range(1, min(n, mp_max)):
                # m = m', ..., n-1
                # i1 = WignerHindex(n, mp+1, mp, mp_max)
                i1 = WignerHindex(n, mp+1, mp+1, mp_max) - 1
                i2 = WignerHindex(n, mp-1, mp, mp_max)
                # i3 = WignerHindex(n, mp, mp-1, mp_max)
                i3 = WignerHindex(n, mp, mp, mp_max) - 1
                i4 = WignerHindex(n, mp, mp+1, mp_max)
                i5 = nm_index(n, mp)
                i6 = nm_index(n, mp-1)
                inverse_d5 = 1.0 / d[i5]
                d6 = d[i6]
                for i in [0]:
                    d7 = d[i+i6]
                    d8 = d[i+i5]
                    Hv[i+nm_index(n, mp+1)] = inverse_d5 * (
                          d6 * Hwedge[i+i2]
                        - d7 * Hv[i+nm_index(n, mp)]
                        + d8 * Hwedge[i+i4]
                    )
                for i in range(1, n-mp):
                    d7 = d[i+i6]
                    d8 = d[i+i5]
                    Hwedge[i+i1] = inverse_d5 * (
                          d6 * Hwedge[i+i2]
                        - d7 * Hwedge[i+i3]
                        + d8 * Hwedge[i+i4]
                    )
                # m = n
                for i in [n-mp]:
                    Hwedge[i+i1] = inverse_d5 * (
                          d6 * Hwedge[i+i2]
                        - d[i+i6] * Hwedge[i+i3]
                    )


@jit
def _step_5(d, n_max, mp_max, Hwedge, Hv):
    """Recursively compute H^{m'−1, m}_{n}(β) for m'=−1,...,−n+1, m=−m',...,n using relation (50)
    resolved with respect to H^{m'−1, m}_{n}:

      d^{m'−1}_{n} H^{m'−1, m}_{n} = d^{m'}_{n} H^{m'+1, m}_{n}
                                     + d^{m−1}_{n} H^{m', m−1}_{n}
                                     − d^{m}_{n} H^{m', m+1}_{n}

    (where the last term drops out for m=n).

    NOTE: Although arxiv:1403.7698 specifies the loop over mp to start at -1, I
    find it necessary to start at 0, or there will be missing information.  This
    also requires setting the (m',m)=(0,-1) components before beginning this loop.

    """
    if n_max > 0 and mp_max > 0:
        for n in range(0, n_max+1):
            for mp in range(0, -min(n, mp_max), -1):
                # m = -m', ..., n-1
                # i1 = WignerHindex(n, mp-1, -mp, mp_max)
                i1 = WignerHindex(n, mp-1, -mp+1, mp_max) - 1
                # i2 = WignerHindex(n, mp+1, -mp, mp_max)
                i2 = WignerHindex(n, mp+1, -mp+1, mp_max) - 1
                # i3 = WignerHindex(n, mp, -mp-1, mp_max)
                i3 = WignerHindex(n, mp, -mp, mp_max) - 1
                i4 = WignerHindex(n, mp, -mp+1, mp_max)
                i5 = nm_index(n, mp-1)
                i6 = nm_index(n, mp)
                i7 = nm_index(n, -mp-1)
                i8 = nm_index(n, -mp)
                inverse_d5 = 1.0 / d[i5]
                d6 = d[i6]
                for i in [0]:
                    d7 = d[i+i7]
                    d8 = d[i+i8]
                    if mp == 0:
                        Hv[i+nm_index(n, mp-1)] = inverse_d5 * (
                              d6 * Hv[i+nm_index(n, mp+1)]
                            + d7 * Hv[i+nm_index(n, mp)]
                            - d8 * Hwedge[i+i4]
                        )
                    else:
                        Hv[i+nm_index(n, mp-1)] = inverse_d5 * (
                              d6 * Hwedge[i+i2]
                            + d7 * Hv[i+nm_index(n, mp)]
                            - d8 * Hwedge[i+i4]
                        )
                for i in range(1, n+mp):
                    d7 = d[i+i7]
                    d8 = d[i+i8]
                    Hwedge[i+i1] = inverse_d5 * (
                          d6 * Hwedge[i+i2]
                        + d7 * Hwedge[i+i3]
                        - d8 * Hwedge[i+i4]
                    )
                # m = n
                i = n+mp
                Hwedge[i+i1] = inverse_d5 * (
                      d6 * Hwedge[i+i2]
                    + d[i+i7] * Hwedge[i+i3]
                )
