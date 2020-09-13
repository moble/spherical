import numpy as np
from functools import lru_cache
from scipy.special import factorial
import spherical as sf
from .. import jit


sqrt3 = np.sqrt(3)
sqrt2 = np.sqrt(2)


"""Algorithm for computing H, as given by arxiv:1403.7698

dₗⁿᵐ = ϵₙ ϵ₋ₘ Hₗⁿᵐ, where

       ⎧ 1 for k≤0
  ϵₖ = ⎨
       ⎩ (-1)ᵏ for k>0

Symmetry relations:
H^{m', m}_n(\beta) = H^{m, m'}_n(\beta)
H^{m', m}_n(\beta) = H^{-m', -m}_n(\beta)
H^{m', m}_n(\beta) = (-1)^{n+m+m'} H^{-m', m}_n(\pi - \beta)
H^{m', m}_n(\beta) = (-1)^{m+m'} H^{m', m}_n(-\beta)

1: If n=0 set H_{0}^{0,0}=1.

2: Compute values H^{0,m}_{n}(β)for m=0,...,n and H^{0,m}_{n+1}(β) for m=0,...,n+1 using Eq. (32):
     H^{0,m}_{n}(β) = (-1)^m \sqrt{(n-|m|)!/(n+|m|)!} P^{|m|}_{n}(cos β)

3: Use relation (41) to compute H^{1,m}_{n}(β) for m=1,...,n.  Using symmetry and shift of the
   indices this relation can be written as
     b^{0}_{n+1} H^{1,m}_{n} = \frac{b^{−m−1}_{n+1} (1−cosβ)}{2} H^{0,m+1}_{n+1}
                             − \frac{b^{m−1}_{n+1}  (1+cosβ)}{2} H^{0,m−1}_{n+1}
                             − a^{m}_{n} sinβ H^{0,m}_{n+1}

4: Recursively compute H^{m′+1,m}_{n}(β) for m′=1,...,n−1, m=m′,...,n using relation (50) resolved
   with respect to H^{m′+1,m}_{n}:
     d^{m′}_{n} H^{m′+1,m}_{n} = d^{m′−1}_{n} H^{m′−1,m}_{n}
                               − d^{m−1}_{n} H^{m′,m−1}_{n}
                               + d^{m}_{n} H^{m′,m+1}_{n}
   (where the last term drops out for m=n).

5. Recursively compute H^{m′−1,m}_{n}(β) for m′=−1,...,−n+1, m=−m′,...,n using relation (50)
   resolved with respect to H^{m′−1,m}_{n}:
     d^{m′−1}_{n} H^{m′−1,m}_{n} = d^{m′}_{n} H^{m′+1,m}_{n}
                                 + d^{m−1}_{n} H^{m′,m−1}_{n}
                                 − d^{m}_{n} H^{m′,m+1}_{n}
   (where the last term drops out for m=n).

6: Apply the first and the second symmetry relations above to obtain all other values H^{m′,m}_{n}
   outside the computational triangle m=0,...,n, m′=−m,...,m.

"""


@jit
def wedge_size(ℓₘₐₓ):
    return (ℓₘₐₓ+1) * (ℓₘₐₓ+2) * (2*ℓₘₐₓ+3) // 6

@jit
def wedge_index(ℓ, mp, m):
    """Index to "wedge" arrays

    Here, it is assumed that only data with m≥|m'| are stored, and only corresponding values are
    passed.  We also assume |m|≤ℓ and |m'|≤ℓ.  Neither of these are checked.  The wedge array that
    this function indexes is ordered as

        [(n, mp, m) for n in range(n_max+1) for mp in range(-n, n+1) for m in range(abs(mp), n+1)]

    """
    # if m < -mp:
    #     if m < mp:
    #         return wedge_index(ℓ, -mp, -m)
    #     else:
    #         return wedge_index(ℓ, -m, -mp)
    # else:
    #     if m < mp:
    #         return wedge_index(ℓ, m, mp)
    if mp>=0:
        return (2*(ℓ)*(ℓ+1)*(ℓ+2) + 3*mp*(2*ℓ-mp+1) + 6*m) // 6
    else:
        return (2*(ℓ)*(ℓ+1)*(ℓ+2) + 3*mp*(2*ℓ+mp+3) + 6*m) // 6


@jit
def wedgeify_index(ℓ, mp, m):
    """Convert general (ℓ, m′, m) index into index appropriate for H wedge

    The wedge is stored as

        [(n, mp, m) for n in range(n_max+1) for mp in range(-n, n+1) for m in range(abs(mp), n+1)]

    This function returns an (ℓ, m′, m) taking advantage of the H symmetries to be the corresponding
    index lying inside the wedge.

    Obtaining values of the H wedge from other indices:
    ================================
    m\m'  -3  -2  -1   0   1   2   3
     3     v   v   v   v   v   v   v
     2    nt   v   v   v   v   v   t
     1    nt  nt   v   v   v   t   t
     0    nt  nt  nt   v   t   t   t
    -1    nt  nt   n   n   t   t   t
    -2    nt   n   n   n   n   t   t
    -3     n   n   n   n   n   n   t
    ================================
    v = value
    t = transpose, (m',m) -> (m,m')
    n = negative, (m',m) -> (-m',-m)
    nt = negative transpose, (m',m) -> (-m,-m')

    Note that the boundaries between the lower wedges are overlap.  For example, the common border
    of the `n` and `nt` wedges could use either negative or negative transpose and obtain the same
    element of the main wedge.

    """
    if m < -mp:
        if m < mp:
            return [ℓ, -mp, -m]
        else:
            return [ℓ, -m, -mp]
    else:
        if m < mp:
            return [ℓ, m, mp]
        else:
            return [ℓ, mp, m]


@jit
def sign(m):
    if m >= 0:
        return 1
    else:
        return -1


@jit
def nm_index(n, m):
    """Return flat index into arrray of [n, m] pairs.
    
    Assumes array is ordered as [[n, m] for n in range(n_max+1) for m in range(-n, n+1)]

    """
    return m + n * (n + 1)


@jit
def nabsm_index(n, absm):
    """Return flat index into arrray of [n, abs(m)] pairs
    
    Assumes array is ordered as [[n, m] for n in range(n_max+1) for m in range(n+1)]

    """
    return absm + (n * (n + 1)) // 2


@jit
def nmpm_index(n, mp, m):
    """Return flat index into arrray of [n, mp, m]
    
    Assumes array is ordered as

        [[n, mp, m] for n in range(n_max+1) for mp in range(-n, n+1) for m in range(-n, n+1)]

    """
    return (((4 * n + 6) * n + 6 * mp + 5) * n + 3 * (m + mp)) // 3


@jit
def _step_1(Hwedge):
    """If n=0 set H_{0}^{0,0}=1."""
    Hwedge[0, :] = 1.0


@jit
def _step_2(g, h, n_max, Hwedge, Hextra, Hv, cosβ, sinβ):
    """Compute values H^{0,m}_{n}(β)for m=0,...,n and H^{0,m}_{n+1}(β) for m=0,...,n+1 using Eq. (32):
        H^{0,m}_{n}(β) = (-1)^m \sqrt{(n-|m|)! / (n+|m|)!} P^{|m|}_{n}(cos β)
                       = (-1)^m (sin β)^m \hat{P}^{|m|}_{n}(cos β) / \sqrt{k (2n+1)}

    This function computes the associated Legendre functions directly by recursion as explained by
    Holmes and Featherstone (2002), doi:10.1007/s00190-002-0216-2.  Note that I had to adjust
    certain steps for consistency with the notation assumed by arxiv:1403.7698 -- mostly involving
    factors of (-1)**m.

    NOTE: Though not specified in arxiv:1403.7698, there is not enough information for step 4 unless we
    also use symmetry to set H^{1,0}_{n} here.  Similarly, step 5 needs additional information, which
    depends on setting H^{0, -1}_{n} from its symmetric equivalent H^{0, 1}_{n} in this step.

    """
    prefactor = np.empty_like(sinβ)
    # n = 1
    n0n_index = wedge_index(1, 0, 1)
    nn_index = nm_index(1, 1)
    Hwedge[n0n_index, :] = sqrt3  # Un-normalized
    Hwedge[n0n_index-1, :] = (g[nn_index-1] * cosβ) / sqrt2  # Normalized
    # n = 2, ..., n_max+1
    for n in range(2, n_max+2):
        if n <= n_max:
            n0n_index = wedge_index(n, 0, n)
            H = Hwedge
        else:
            n0n_index = n
            H = Hextra
        nm10nm1_index = wedge_index(n-1, 0, n-1)
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
        # m = 0, with normalization
        const = np.sqrt(4*n+2)
        gi = g[nn_index-n]
        hi = h[nn_index-n]
        for j in range(H.shape[1]):
            H[n0n_index-n, j] = (gi * cosβ[j] * H[n0n_index-n+1, j] - hi * sinβ[j]**2 * H[n0n_index-n+2, j]) / const
        # Now, loop back through, correcting the normalization for this row, except for n=n element
        prefactor[:] = 1/const
        for i in range(1, n):
            prefactor *= sinβ
            H[n0n_index-n+i, :] *= prefactor
        # Supply extra edge cases as noted in docstring
        if n <= n_max:
            Hv[nm_index(n, 1), :] = Hwedge[wedge_index(n, 0, 1)]
            Hv[nm_index(n, 0), :] = Hwedge[wedge_index(n, 0, 1)]
    # Correct normalization of m=n elements
    prefactor[:] = 1.0
    for n in range(1, n_max+1):
        prefactor *= sinβ
        Hwedge[wedge_index(n, 0, n), :] *= prefactor / np.sqrt(4*n+2)
    for n in [n_max+1]:
        prefactor *= sinβ
        Hextra[n, :] *= prefactor / np.sqrt(4*n+2)
    # Supply extra edge cases as noted in docstring
    Hv[nm_index(1, 1), :] = Hwedge[wedge_index(1, 0, 1)]
    Hv[nm_index(1, 0), :] = Hwedge[wedge_index(1, 0, 1)]


@jit
def _step_3(a, b, n_max, Hwedge, Hextra, cosβ, sinβ):
    """Use relation (41) to compute H^{1,m}_{n}(β) for m=1,...,n.  Using symmetry and shift of the
    indices this relation can be written as
        b^{0}_{n+1} H^{1, m}_{n} =   \frac{b^{−m−1}_{n+1} (1−cosβ)}{2} H^{0, m+1}_{n+1}
                                   − \frac{b^{ m−1}_{n+1} (1+cosβ)}{2} H^{0, m−1}_{n+1}
                                   − a^{m}_{n} sinβ H^{0, m}_{n+1}
    """
    for n in range(1, n_max+1):
        # m = 1, ..., n
        i1 = wedge_index(n, 1, 1)
        if n+1 <= n_max:
            i2 = wedge_index(n+1, 0, 0)
            H2 = Hwedge
        else:
            i2 = 0
            H2 = Hextra
        i3 = nm_index(n+1, 0)
        i4 = nabsm_index(n, 1)
        b5 = b[i3]
        for i in range(n):
            b6 = b[-i+i3-2]
            b7 = b[i+i3]
            a8 = a[i+i4]
            for j in range(Hwedge.shape[1]):
                Hwedge[i+i1, j] = (1 / b5) * (
                    0.5 * (
                          b6 * (1-cosβ[j]) * H2[i+i2+2, j]
                        - b7 * (1+cosβ[j]) * H2[i+i2, j]
                    )
                    - a8 * sinβ[j] * H2[i+i2+1, j]
                )


@jit
def _step_4(d, n_max, Hwedge, Hv):
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
            i1 = wedge_index(n, mp+1, mp)
            i2 = wedge_index(n, mp-1, mp)
            i3 = wedge_index(n, mp, mp-1)
            i4 = wedge_index(n, mp, mp+1)
            i5 = nm_index(n, mp)
            i6 = nm_index(n, mp-1)
            d5 = d[i5]
            d6 = d[i6]
            for i in [0]:
                d7 = d[i+i6]
                d8 = d[i+i5]
                for j in range(Hwedge.shape[1]):
                    Hv[i+nm_index(n, mp+1), j] = (1 / d5) * (
                          d6 * Hwedge[i+i2, j]
                        - d7 * Hv[i+nm_index(n, mp), j]
                        + d8 * Hwedge[i+i4, j]
                    )
            for i in range(1, n-mp):
                d7 = d[i+i6]
                d8 = d[i+i5]
                for j in range(Hwedge.shape[1]):
                    Hwedge[i+i1, j] = (1 / d5) * (
                          d6 * Hwedge[i+i2, j]
                        - d7 * Hwedge[i+i3, j]
                        + d8 * Hwedge[i+i4, j]
                    )
            # m = n
            for i in [n-mp]:
                for j in range(Hwedge.shape[1]):
                    Hwedge[i+i1, j] = (1 / d5) * (
                          d6 * Hwedge[i+i2, j]
                        - d[i+i6] * Hwedge[i+i3, j]
                    )


@jit
def _step_5(d, n_max, Hwedge, Hv):
    """Recursively compute H^{m'−1, m}_{n}(β) for m'=−1,...,−n+1, m=−m',...,n using relation (50)
    resolved with respect to H^{m'−1, m}_{n}:
      d^{m'−1}_{n} H^{m'−1, m}_{n} = d^{m'}_{n} H^{m'+1, m}_{n}
                                     + d^{m−1}_{n} H^{m', m−1}_{n}
                                     − d^{m}_{n} H^{m', m+1}_{n}
    (where the last term drops out for m=n).

    NOTE: Although arxiv:1403.7698 specifies the loop over mp to start at -1, I find it necessary to
    start at 0, or there will be missing information.  This also requires setting the (m',m)=(0,-1)
    components before beginning this loop.

    """
    for n in range(0, n_max+1):
        for mp in range(0, -n, -1):
            # m = -m', ..., n-1
            i1 = wedge_index(n, mp-1, -mp)
            i2 = wedge_index(n, mp+1, -mp)
            i3 = wedge_index(n, mp, -mp-1)
            i4 = wedge_index(n, mp, -mp+1)
            i5 = nm_index(n, mp-1)
            i6 = nm_index(n, mp)
            i7 = nm_index(n, -mp-1)
            i8 = nm_index(n, -mp)
            d5 = d[i5]
            d6 = d[i6]
            for i in [0]:
                d7 = d[i+i7]
                d8 = d[i+i8]
                if mp == 0:
                    for j in range(Hwedge.shape[1]):
                        Hv[i+nm_index(n, mp-1), j] = (1 / d5) * (
                              d6 * Hv[i+nm_index(n, mp+1), j]
                            + d7 * Hv[i+nm_index(n, mp), j]
                            - d8 * Hwedge[i+i4, j]
                        )
                else:
                    for j in range(Hwedge.shape[1]):
                        Hv[i+nm_index(n, mp-1), j] = (1 / d5) * (
                              d6 * Hwedge[i+i2, j]
                            + d7 * Hv[i+nm_index(n, mp), j]
                            - d8 * Hwedge[i+i4, j]
                        )
            for i in range(1, n+mp):
                d7 = d[i+i7]
                d8 = d[i+i8]
                for j in range(Hwedge.shape[1]):
                    Hwedge[i+i1, j] = (1 / d5) * (
                          d6 * Hwedge[i+i2, j]
                        + d7 * Hwedge[i+i3, j]
                        - d8 * Hwedge[i+i4, j]
                    )
            # m = n
            i = n+mp
            for j in range(Hwedge.shape[1]):
                Hwedge[i+i1, j] = (1 / d5) * (
                      d6 * Hwedge[i+i2, j]
                    + d[i+i7] * Hwedge[i+i3, j]
                )


class HCalculator(object):
    def __init__(self, n_max):
        """Object to repeatedly calculate Wigner H values

        The H matrix is related to Wigner's d matrix as given by arxiv:1403.7698 according to

        dₗⁿᵐ = ϵₙ ϵ₋ₘ Hₗⁿᵐ, where

               ⎧ 1 for k≤0
          ϵₖ = ⎨
               ⎩ (-1)ᵏ for k>0

        This object has various advantageous features compared to the d matrix.  It obeys these
        symmetry relations:

            H^{m', m}_n(\beta) = H^{m, m'}_n(\beta)
            H^{m', m}_n(\beta) = H^{-m', -m}_n(\beta)
            H^{m', m}_n(\beta) = (-1)^{n+m+m'} H^{-m', m}_n(\pi - \beta)
            H^{m', m}_n(\beta) = (-1)^{m+m'} H^{m', m}_n(-\beta)

        In particular the simplicity of the first two of these relations implies that we only need
        to compute one fourth of the total number of elements.  There is also a very accurate and
        efficient recursion method to compute these values.

        Create this object using the largest value of `n` (also commonly denoted `j` or `ℓ`) you
        expect to need, optionally create the `workspace` for a given shape of cosβ using that method,
        and then call this object for a given value or array of cosβ values.

        The returned object is a series of "wedges" of the matrix, for the various values of `n`,
        comprising just a quarter of the elements of the full matrix; all remaining elements are
        determined by the first two symmetries above.  This wedge has an initial dimension
        representing a multi-index for (ℓ, mp, m) values, while following dimensions are just the
        same as cosβ.  Any required value may be obtained with `wedge_index(*wedgeify_index(ℓ, mp,
        m))`.  The inner call translates a general index into the equivalent index lying inside the
        wedge, while the outer call translates that corrected (ℓ, mp, m) tuple into a linear index
        into the array.

        Example Usage
        =============

        hcalc = HCalculator(n_max)
        workspace = hcalc.workspace(cosβ)  # Note that cosβ can be an array of many values
        # Possibly loop over many values of cosβ here
        wedge = hcalc(cosβ, sinβ, workspace)

        """
        self.n_max = int(n_max)
        if self.n_max < 0:
            raise ValueError('Nonsensical value for n_max = {0}'.format(self.n_max))
        self.wedge_size = wedge_size(self.n_max)
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

    def workspace(self, cosβ=1.0):
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
        _step_2(self.g, self.h, self.n_max, Hwedge, Hextra, Hv, cosβ, sinβ)
        _step_3(self.a, self.b, self.n_max, Hwedge, Hextra, cosβ, sinβ)
        _step_4(self.d, self.n_max, Hwedge, Hv)
        _step_5(self.d, self.n_max, Hwedge, Hv)
        Hwedge.reshape((-1,)+cosβshape)
        return Hwedge
