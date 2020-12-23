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
from functools import lru_cache
from scipy.special import factorial
import sympy

from .. import jit, LMpM_total_size, LMpM_index

sqrt3 = np.sqrt(3)
sqrt2 = np.sqrt(2)
inverse_sqrt2 = 1.0 / sqrt2


@jit
def ϵ(m):
    if m <= 0:
        return 1
    elif m%2:
        return -1
    else:
        return 1


@jit
def wedge_size(ℓₘₐₓ, mpₘₐₓ=None):
    """Total size of wedges of width mpₘₐₓ up to ℓₘₐₓ

    Parameters
    ----------
    ℓₘₐₓ : int
    mpₘₐₓ : int, optional
        If None, it is assumed to be at least ℓ

    See Also
    --------
    wedge_index : Index inside these wedges

    Notes
    -----
    Here, it is assumed that only data with m≥|m'| are stored, and only
    corresponding values are passed.  We also assume |m|≤ℓ and |m'|≤ℓ.  Neither of
    these are checked.  The wedge array that this function indexes is ordered as

        [
            (ℓ, mp, m) for ℓ in range(ℓₘₐₓ+1)
            for mp in range(-mpₘₐₓ, mpₘₐₓ+1)
            for m in range(abs(mp), n+1)
        ]

    See the docstring of `wedge_index` for

    """
    if ℓₘₐₓ < 0:
        return 0
    if mpₘₐₓ is None or mpₘₐₓ > ℓₘₐₓ:
        return (ℓₘₐₓ+1) * (ℓₘₐₓ+2) * (2*ℓₘₐₓ+3) // 6
    else:
        return ((ℓₘₐₓ+1)*(ℓₘₐₓ+2)*(2*ℓₘₐₓ+3) - 2*(ℓₘₐₓ-mpₘₐₓ)*(ℓₘₐₓ-mpₘₐₓ+1)*(ℓₘₐₓ-mpₘₐₓ+2)) // 6


@jit
def _wedge_index(ℓ, mp, m, mpₘₐₓ):
    """Helper function for `wedge_index`"""
    i = wedge_size(ℓ-1, mpₘₐₓ)  # total size of everything with smaller ℓ
    if mp<1:
        i += (mpₘₐₓ + mp) * (2*ℓ - mpₘₐₓ + mp + 1) // 2  # size of wedge to the left of m'
    else:
        i += (mpₘₐₓ + 1) * (2*ℓ - mpₘₐₓ + 2) // 2  # size of entire left half of wedge
        i += (mp - 1) * (2*ℓ - mp + 2) // 2  # size of right half of wedge to the left of m'
    i += m - abs(mp)  # size of column in wedge between m and |m'|
    return i


@jit
def wedge_index(ℓ, mp, m, mpₘₐₓ=None):
    """Index to "wedge" arrays

    Parameters
    ----------
    ℓ : int
    mp : int
    m : int
    mpₘₐₓ : int, optional
        If None, it is assumed to be at least ℓ

    See Also
    --------
    wedge_size : Total size of wedge array

    Notes
    -----
    Here, it is assumed that only data with m≥|m'| are stored, and only
    corresponding values are passed.  We also assume |m|≤ℓ and |m'|≤ℓ.  Neither of
    these are checked.  The wedge array that this function indexes is ordered as

        [
            (ℓ, mp, m) for ℓ in range(ℓₘₐₓ+1)
            for mp in range(-mpₘₐₓ, mpₘₐₓ+1)
            for m in range(abs(mp), n+1)
        ]

    """
    mp_max = ℓ
    if mpₘₐₓ is not None:
        mp_max = min(mpₘₐₓ, mp_max)
    if m < -mp:
        if m < mp:
            return _wedge_index(ℓ, -mp, -m, mp_max)
        else:
            return _wedge_index(ℓ, -m, -mp, mp_max)
    else:
        if m < mp:
            return _wedge_index(ℓ, m, mp, mp_max)
        else:
            return _wedge_index(ℓ, mp, m, mp_max)


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
    Hwedge[0, :] = 1.0


@jit
def _step_2(g, h, n_max, mp_max, Hwedge, Hextra, Hv, cosβ, sinβ):
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
    prefactor = np.empty_like(sinβ)
    # n = 1
    n0n_index = wedge_index(1, 0, 1, mp_max)
    nn_index = nm_index(1, 1)
    Hwedge[n0n_index, :] = sqrt3  # Un-normalized
    Hwedge[n0n_index-1, :] = (g[nn_index-1] * cosβ) * inverse_sqrt2  # Normalized
    # n = 2, ..., n_max+1
    for n in range(2, n_max+2):
        if n <= n_max:
            n0n_index = wedge_index(n, 0, n, mp_max)
            H = Hwedge
        else:
            n0n_index = n
            H = Hextra
        nm10nm1_index = wedge_index(n-1, 0, n-1, mp_max)
        nn_index = nm_index(n, n)
        const = np.sqrt(1.0 + 0.5/n)
        gi = g[nn_index-1]
        for j in range(H.shape[1]):
            # m = n
            H[n0n_index, j] = const * Hwedge[nm10nm1_index, j]
            # m = n-1
            H[n0n_index-1, j] = gi * cosβ[j] * H[n0n_index, j]
        # m = n-2, ..., 1
        for i in range(2, n):
            gi = g[nn_index-i]
            hi = h[nn_index-i]
            for j in range(H.shape[1]):
                H[n0n_index-i, j] = gi * cosβ[j] * H[n0n_index-i+1, j] - hi * sinβ[j]**2 * H[n0n_index-i+2, j]
        # if n==2:
        #     print("args2 =", (g, h, n_max, Hwedge, Hextra, Hv, cosβ, sinβ))
        # m = 0, with normalization
        const = 1.0 / np.sqrt(4*n+2)
        gi = g[nn_index-n]
        hi = h[nn_index-n]
        for j in range(H.shape[1]):
            H[n0n_index-n, j] = (gi * cosβ[j] * H[n0n_index-n+1, j] - hi * sinβ[j]**2 * H[n0n_index-n+2, j]) * const
        # Now, loop back through, correcting the normalization for this row, except for n=n element
        prefactor[:] = const
        for i in range(1, n):
            prefactor *= sinβ
            H[n0n_index-n+i, :] *= prefactor
        # Supply extra edge cases as noted in docstring
        if n <= n_max:
            Hv[nm_index(n, 1), :] = Hwedge[wedge_index(n, 0, 1, mp_max)]
            Hv[nm_index(n, 0), :] = Hwedge[wedge_index(n, 0, 1, mp_max)]
    # Correct normalization of m=n elements
    prefactor[:] = 1.0
    for n in range(1, n_max+1):
        prefactor *= sinβ
        Hwedge[wedge_index(n, 0, n, mp_max), :] *= prefactor / np.sqrt(4*n+2)
    for n in [n_max+1]:
        prefactor *= sinβ
        Hextra[n, :] *= prefactor / np.sqrt(4*n+2)
    # Supply extra edge cases as noted in docstring
    Hv[nm_index(1, 1), :] = Hwedge[wedge_index(1, 0, 1, mp_max)]
    Hv[nm_index(1, 0), :] = Hwedge[wedge_index(1, 0, 1, mp_max)]


@jit
def _step_3(a, b, n_max, mp_max, Hwedge, Hextra, cosβ, sinβ):
    """Use relation (41) to compute H^{1,m}_{n}(β) for m=1,...,n.  Using symmetry and shift
    of the indices this relation can be written as

        b^{0}_{n+1} H^{1, m}_{n} =   (b^{−m−1}_{n+1} (1−cosβ))/2 H^{0, m+1}_{n+1}
                                   − (b^{ m−1}_{n+1} (1+cosβ))/2 H^{0, m−1}_{n+1}
                                   − a^{m}_{n} sinβ H^{0, m}_{n+1}

    """
    for n in range(1, n_max+1):
        # m = 1, ..., n
        i1 = wedge_index(n, 1, 1, mp_max)
        if n+1 <= n_max:
            i2 = wedge_index(n+1, 0, 0, mp_max)
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
            for j in range(Hwedge.shape[1]):
                Hwedge[i+i1, j] = inverse_b5 * (
                    0.5 * (
                          b6 * (1-cosβ[j]) * H2[i+i2+2, j]
                        - b7 * (1+cosβ[j]) * H2[i+i2, j]
                    )
                    - a8 * sinβ[j] * H2[i+i2+1, j]
                )


#@jit
def _step_4(d, n_max, mp_max, Hwedge, Hv):
    """Recursively compute H^{m'+1, m}_{n}(β) for m'=1,...,n−1, m=m',...,n using relation (50) resolved
    with respect to H^{m'+1, m}_{n}:

      d^{m'}_{n} H^{m'+1, m}_{n} =   d^{m'−1}_{n} H^{m'−1, m}_{n}
                                   − d^{m−1}_{n} H^{m', m−1}_{n}
                                   + d^{m}_{n} H^{m', m+1}_{n}

    (where the last term drops out for m=n).

    """
    for n in range(2, n_max+1):
        for mp in range(1, n):
            # m = m', ..., n-1
            # i1 = wedge_index(n, mp+1, mp, mp_max)
            i1 = wedge_index(n, mp+1, mp+1, mp_max) - 1
            i2 = wedge_index(n, mp-1, mp, mp_max)
            # i3 = wedge_index(n, mp, mp-1, mp_max)
            i3 = wedge_index(n, mp, mp, mp_max) - 1
            i4 = wedge_index(n, mp, mp+1, mp_max)
            i5 = nm_index(n, mp)
            i6 = nm_index(n, mp-1)
            inverse_d5 = 1.0 / d[i5]
            d6 = d[i6]
            for i in [0]:
                d7 = d[i+i6]
                d8 = d[i+i5]
                for j in range(Hwedge.shape[1]):
                    Hv[i+nm_index(n, mp+1), j] = inverse_d5 * (
                          d6 * Hwedge[i+i2, j]
                        - d7 * Hv[i+nm_index(n, mp), j]
                        + d8 * Hwedge[i+i4, j]
                    )
            for i in range(1, n-mp):
                d7 = d[i+i6]
                d8 = d[i+i5]
                for j in range(Hwedge.shape[1]):
                    Hwedge[i+i1, j] = inverse_d5 * (
                          d6 * Hwedge[i+i2, j]
                        - d7 * Hwedge[i+i3, j]
                        + d8 * Hwedge[i+i4, j]
                    )
            # if n == 2 and mp == 1:
            #     print("args2 =", (d, n_max, Hwedge, Hv, i1, i2, i3, i4))
            # m = n
            for i in [n-mp]:
                for j in range(Hwedge.shape[1]):
                    Hwedge[i+i1, j] = inverse_d5 * (
                          d6 * Hwedge[i+i2, j]
                        - d[i+i6] * Hwedge[i+i3, j]
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
    for n in range(0, n_max+1):
        for mp in range(0, -n, -1):
            # m = -m', ..., n-1
            # i1 = wedge_index(n, mp-1, -mp, mp_max)
            i1 = wedge_index(n, mp-1, -mp+1, mp_max) - 1
            # i2 = wedge_index(n, mp+1, -mp, mp_max)
            i2 = wedge_index(n, mp+1, -mp+1, mp_max) - 1
            # i3 = wedge_index(n, mp, -mp-1, mp_max)
            i3 = wedge_index(n, mp, -mp, mp_max) - 1
            i4 = wedge_index(n, mp, -mp+1, mp_max)
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
                    for j in range(Hwedge.shape[1]):
                        Hv[i+nm_index(n, mp-1), j] = inverse_d5 * (
                              d6 * Hv[i+nm_index(n, mp+1), j]
                            + d7 * Hv[i+nm_index(n, mp), j]
                            - d8 * Hwedge[i+i4, j]
                        )
                else:
                    for j in range(Hwedge.shape[1]):
                        Hv[i+nm_index(n, mp-1), j] = inverse_d5 * (
                              d6 * Hwedge[i+i2, j]
                            + d7 * Hv[i+nm_index(n, mp), j]
                            - d8 * Hwedge[i+i4, j]
                        )
            for i in range(1, n+mp):
                d7 = d[i+i7]
                d8 = d[i+i8]
                for j in range(Hwedge.shape[1]):
                    Hwedge[i+i1, j] = inverse_d5 * (
                          d6 * Hwedge[i+i2, j]
                        + d7 * Hwedge[i+i3, j]
                        - d8 * Hwedge[i+i4, j]
                    )
            # m = n
            i = n+mp
            for j in range(Hwedge.shape[1]):
                Hwedge[i+i1, j] = inverse_d5 * (
                      d6 * Hwedge[i+i2, j]
                    + d[i+i7] * Hwedge[i+i3, j]
                )


class HCalculator(object):
    def __init__(self, n_max, mp_max=None):
        """Object to repeatedly calculate Wigner H values

        In particular the simplicity of the first two of these relations implies that
        we only need to compute one fourth of the total number of elements.  There is
        also a very accurate and efficient recursion method to compute these values.

        Create this object using the largest value of `n` (also commonly denoted `j` or
        `ℓ`) you expect to need, optionally create the `workspace` for a given shape of
        cosβ using that method, and then call this object for a given value or array of
        cosβ values.

        The returned object is a series of "wedges" of the matrix, for the various
        values of `n`, comprising just a quarter of the elements of the full matrix;
        all remaining elements are determined by the first two symmetries above.  This
        wedge has an initial dimension representing a multi-index for (ℓ, mp, m)
        values, while following dimensions are just the same as cosβ.  Any required
        value may be obtained with `wedge_index(ℓ, mp, m, mp_max)`.  The inner call
        translates a general index into the equivalent index lying inside the wedge,
        while the outer call translates that corrected (ℓ, mp, m) tuple into a linear
        index into the array.

        Example Usage
        -------------
        hcalc = HCalculator(n_max)
        workspace = hcalc.workspace(cosβ)  # Note that cosβ can be an array of many values
        # Possibly loop over many values of cosβ here
        wedge = hcalc(cosβ, sinβ, workspace)

        """
        self.n_max = int(n_max)
        if mp_max is not None:
            self.mp_max = abs(int(mp_max))
        else:
            self.mp_max = None
        if self.n_max < 0:
            raise ValueError('Nonsensical value for n_max = {0}'.format(self.n_max))
        self.wedge_size = wedge_size(self.n_max, self.mp_max)
        n = np.array([n for n in range(self.n_max+2) for m in range(-n, n+1)])
        m = np.array([m for n in range(self.n_max+2) for m in range(-n, n+1)])
        absn = np.array([n for n in range(self.n_max+2) for m in range(n+1)])
        absm = np.array([m for n in range(self.n_max+2) for m in range(n+1)])
        self.a = np.sqrt((absn+1+absm) * (absn+1-absm) / ((2*absn+1)*(2*absn+3)))
        self.b = np.sqrt((n-m-1) * (n-m) / ((2*n-1)*(2*n+1)))
        self.b[m<0] *= -1
        self.d = 0.5 * np.sqrt((n-m) * (n+m+1))
        self.d[m<0] *= -1
        with np.errstate(divide='ignore', invalid='ignore'):
            self.g = 2*(m+1) / np.sqrt((n-m)*(n+m+1))
            self.h = np.sqrt((n+m+2)*(n-m-1) / ((n-m)*(n+m+1)))
        if not (
            np.all(np.isfinite(self.a)) and
            np.all(np.isfinite(self.b)) and
            np.all(np.isfinite(self.d))
        ):
            raise ValueError("Found a non-finite value inside this object")

    def workspace(self, cosβ=[1.0]):
        """Return a new workspace sized for cosβ."""
        cosβ = np.asarray(cosβ, dtype=float)
        return np.zeros((self.wedge_size+(self.n_max+1)**2+self.n_max+2,) + cosβ.shape, dtype=float)

    def __call__(self, cosβ, sinβ=None, workspace=None):
        cosβ = np.asarray(cosβ, dtype=float)
        if np.max(cosβ) > 1.0 or np.min(cosβ) < -1.0:
            raise ValueError('Nonsensical value for range of cosβ: [{0}, {1}]'.format(np.min(cosβ), np.max(cosβ)))
        cosβshape = cosβ.shape
        cosβ = cosβ.ravel(order='K')
        if sinβ is None:
            sinβ = np.sqrt(1 - cosβ**2)
        else:
            if sinβ.shape != cosβshape:
                raise ValueError(
                    f"Input cosβ and sinβ must be the same shape; their shapes are {cosβshape} and {sinβ.shape}."
                )
            sinβ = sinβ.ravel(order='K')
        workspace = workspace if workspace is not None else self.workspace(cosβ)
        Hwedge = workspace[:self.wedge_size]
        Hv = workspace[self.wedge_size:self.wedge_size+(self.n_max+1)**2]
        Hextra = workspace[self.wedge_size+(self.n_max+1)**2:self.wedge_size+(self.n_max+1)**2+self.n_max+2]
        _step_1(Hwedge)
        _step_2(self.g, self.h, self.n_max, self.mp_max, Hwedge, Hextra, Hv, cosβ, sinβ)
        _step_3(self.a, self.b, self.n_max, self.mp_max, Hwedge, Hextra, cosβ, sinβ)
        _step_4(self.d, self.n_max, self.mp_max, Hwedge, Hv)
        _step_5(self.d, self.n_max, self.mp_max, Hwedge, Hv)
        Hwedge.reshape((-1,)+cosβshape)
        return Hwedge

    def wigner_d(self, cosβ, sinβ=None, workspace=None):
        """Return Wigner's d matrix"""
        ell_min = 0
        ell_max = self.n_max
        Hwedge = self(cosβ, sinβ, workspace)
        d = np.empty((LMpM_total_size(ell_min, ell_max),) + cosβ.shape)
        for ell in range(ell_min, ell_max+1):
            for mp in range(-ell, ell+1):
                for m in range(-ell, ell+1):
                    i_d = LMpM_index(ell, mp, m, ell_min)
                    i_H = wedge_index(ell, mp, m, self.mp_max)
                    d[i_d] = ϵ(mp) * ϵ(-m) * Hwedge[i_H]
        return d
