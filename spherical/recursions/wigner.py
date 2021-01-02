
import numpy as np
import quaternionic

from . import complex_powers
from .. import jit, LMpM_total_size, LMpM_index

sqrt3 = np.sqrt(3)
sqrt2 = np.sqrt(2)
inverse_sqrt2 = 1.0 / sqrt2
inverse_4pi = 1.0 / (4 * np.pi)


@jit
def WignerHsize(mp_max, ell_max):
    """Total size of array of wedges of width mp_max up to ell_max

    Parameters
    ----------
    ell_max : int
    mp_max : int, optional
        If None, it is assumed to be at least ell

    See Also
    --------
    WignerHindex : Index inside these wedges

    Notes
    -----
    Here, it is assumed that only data with m≥|m'| are stored, and only
    corresponding values are passed.  We also assume |m|≤ell and |m'|≤ell.  Neither of
    these are checked.  The wedge array that this function indexes is ordered as

        [
            H(ell, mp, m) for ell in range(ell_max+1)
            for mp in range(-min(ell, mp_max), min(ell, mp_max)+1)
            for m in range(abs(mp), ell+1)
        ]

    """
    if ell_max < 0:
        return 0
    if mp_max is None or mp_max > ell_max:
        return (ell_max+1) * (ell_max+2) * (2*ell_max+3) // 6
    else:
        return ((ell_max+1)*(ell_max+2)*(2*ell_max+3) - 2*(ell_max-mp_max)*(ell_max-mp_max+1)*(ell_max-mp_max+2)) // 6

@jit
def _WignerHindex(ell, mp, m, mp_max):
    """Helper function for `WignerHindex`"""
    i = WignerHsize(ell-1, mp_max)  # total size of everything with smaller ell
    if mp<1:
        i += (mp_max + mp) * (2*ell - mp_max + mp + 1) // 2  # size of wedge to the left of m'
    else:
        i += (mp_max + 1) * (2*ell - mp_max + 2) // 2  # size of entire left half of wedge
        i += (mp - 1) * (2*ell - mp + 2) // 2  # size of right half of wedge to the left of m'
    i += m - abs(mp)  # size of column in wedge between m and |m'|
    return i


@jit
def WignerHindex(ell, mp, m, mp_max=None):
    """Index to "wedge" arrays

    Parameters
    ----------
    ell : int
    mp : int
    m : int
    mp_max : int, optional
        If None, it is assumed to be at least ell

    See Also
    --------
    WignerHsize : Total size of wedge array

    Notes
    -----
    Here, it is assumed that only data with m≥|m'| are stored, and only corresponding
    values are passed.  We also assume |m|≤ell and |m'|≤ell.  Neither of these are
    checked.  The wedge array that this function indexes is ordered as

        [
            H(ell, mp, m) for ell in range(ell_max+1)
            for mp in range(-min(ell, mp_max), min(ell, mp_max)+1)
            for m in range(abs(mp), ell+1)
        ]

    """
    mpmax = ell
    if mp_max is not None:
        mpmax = min(mp_max, mpmax)
    if m < -mp:
        if m < mp:
            return _WignerHindex(ell, -mp, -m, mpmax)
        else:
            return _WignerHindex(ell, -m, -mp, mpmax)
    else:
        if m < mp:
            return _WignerHindex(ell, m, mp, mpmax)
        else:
            return _WignerHindex(ell, mp, m, mpmax)


@jit
def WignerDsize(ell_min, mp_max, ell_max):
    """Compute total size of Wigner 𝔇 matrix

    Parameters
    ----------
    ell_min : int
        Integer satisfying 0 <= ell_min <= ell_max
    mp_max : int
        Integer satisfying 0 <= mp_max
    ell_max : int
        Integer satisfying 0 <= ell_min <= ell_max

    Returns
    -------
    i : int
        Total size of Wigner 𝔇 matrix arranged as described below

    See Also
    --------
    WignerDindex : Index of a particular element of the 𝔇 matrix

    Notes
    -----
    This assumes that the Wigner 𝔇 matrix is arranged as

        [
            𝔇(ℓ, mp, m)
            for ℓ in range(ell_min, ell_max+1)
            for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
            for m in range(-ℓ, ℓ+1)
        ]

    """
    # from sympy import symbols, summation, horner
    # from sympy.printing.pycode import pycode
    # ell,mp,m,ell_min,ell_max,mp_max = symbols('ell,mp,m,ell_min,ell_max,mp_max', integer=True)
    # 
    # def nice(expr):
    #     return horner(expr.expand().simplify(), (mp_max, ell_min, ell_max))
    #
    # # Assuming ell_min <= ell_max <= mp_max:
    # WignerDsize_ellmin_ellmax_mpmax = horner(
    #     (
    #         # sum over all mp=0 elements
    #         summation(summation(summation(1, (m, -ell, ell)), (mp, 0, 0)), (ell, ell_min, ell_max))
    #         # sum over all ell_min <= ell <= ell_max elements
    #         + summation(summation(summation(2, (m, -ell, ell)), (mp, 1, ell)), (ell, ell_min, ell_max))
    #     ).expand().simplify(),
    #     (ell_min, ell_max)
    # )
    # print(f"({pycode(nice(3*WignerDsize_ellmin_ellmax_mpmax.subs(ell_max, ell-1)))}) // 3")
    # 
    # # Assuming ell_min <= mp_max <= ell_max:
    # WignerDsize_ellmin_mpmax_ellmax = horner(
    #     (
    #         # sum over all mp=0 elements
    #         summation(summation(summation(1, (m, -ell, ell)), (mp, 0, 0)), (ell, ell_min, ell_max))
    #         # sum over all ell <= mp_max elements
    #         + summation(summation(summation(2, (m, -ell, ell)), (mp, 1, ell)), (ell, ell_min, mp_max))
    #         # sum over all ell >= mp_max elements
    #         + summation(summation(summation(2, (m, -ell, ell)), (mp, 1, mp_max)), (ell, mp_max+1, ell_max))
    #     ).expand().simplify(),
    #     (mp_max, ell_min, ell_max)
    # )
    # print(f"({pycode(nice(3*WignerDsize_ellmin_mpmax_ellmax.subs(ell_max, ell-1)))}) // 3")
    #
    # # Assuming mp_max <= ell_min <= ell_max:
    # WignerDsize_mpmax_ellmin_ellmax = horner(
    #     (
    #         # sum over all mp=0 elements
    #         summation(summation(summation(1, (m, -ell, ell)), (mp, 0, 0)), (ell, ell_min, ell_max))
    #         # sum over all remaining |mp| <= mp_max elements
    #         + summation(summation(summation(2, (m, -ell, ell)), (mp, 1, mp_max)), (ell, ell_min, ell_max))
    #     ).expand().simplify(),
    #     (mp_max, ell_min, ell_max)
    # )
    # print(f"{pycode(nice(WignerDsize_mpmax_ellmin_ellmax.subs(ell_max, ell-1)).factor())}")
    if mp_max >= ell_max:
        return (
            ell_max * (ell_max * (4 * ell_max + 12) + 11)
            + ell_min * (1 - 4 * ell_min**2)
            + 3
        ) // 3
    if mp_max > ell_min:
        return (
            3 * ell_max * (ell_max + 2)
            + ell_min * (1 - 4 * ell_min**2)
            + mp_max * (
                3 * ell_max * (2 * ell_max + 4)
                + mp_max * (-2 * mp_max - 3) + 5
            )
            + 3
        ) // 3
    else:
        return (ell_max * (ell_max + 2) - ell_min**2) * (1 + 2 * mp_max) + 2 * mp_max + 1


@jit
def WignerDindex(ell_min, mp_max, ell_max, ell, mp, m):
    """Compute index into Wigner 𝔇 matrix

    Parameters
    ----------
    ell_min : int
        Integer satisfying 0 <= ell_min <= ell_max
    mp_max : int
        Integer satisfying 0 <= mp_max
    ell_max : int
        Integer satisfying 0 <= ell_min <= ell_max
    ell : int
        Integer satisfying ell_min <= ell <= ell_max
    mp : int
        Integer satisfying -min(ell_max, mp_max) <= mp <= min(ell_max, mp_max)
    m : int
        Integer satisfying -ell <= m <= ell

    Returns
    -------
    i : int
        Index into Wigner 𝔇 matrix arranged as described below

    See Also
    --------
    WignerDsize : Total size of the 𝔇 matrix

    Notes
    -----
    This assumes that the Wigner 𝔇 matrix is arranged as

        [
            𝔇(ℓ, mp, m)
            for ℓ in range(ell_min, ell_max+1)
            for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
            for m in range(-ℓ, ℓ+1)
        ]

    """
    i = (mp + min(mp_max, ell)) * (2 * ell + 1) + m + ell
    if ell > ell_min:
        i += WignerDsize(ell_min, mp_max, ell-1)
    return i


@jit
def Ysize(ell_min, ell_max):
    """Compute total size of array of mode weights

    Parameters
    ----------
    ell_min : int
        Integer satisfying 0 <= ell_min <= ell_max
    ell_max : int
        Integer satisfying 0 <= ell_min <= ell_max

    Returns
    -------
    i : int
        Total size of array of mode weights arranged as described below

    See Also
    --------
    Yindex : Index of a particular element of the mode weight array

    Notes
    -----
    This assumes that the modes are arranged (with fixed s value) as

        [
            Y(s, ℓ, m)
            for ℓ in range(ell_min, ell_max+1)
            for m in range(-ℓ, ℓ+1)
        ]

    """
    # from sympy import symbols, summation, horner
    # from sympy.printing.pycode import pycode
    # ell,m,ell_min,ell_max = symbols('ell,m,ell_min,ell_max', integer=True)
    # 
    # def nice(expr):
    #     return horner(expr.expand().simplify(), (mp_max, ell_min, ell_max))
    #
    # Ysize = horner(
    #     summation(summation(1, (m, -ell, ell)), (ell, ell_min, ell_max)).expand().simplify(),
    #     (ell_min, ell_max)
    # )
    return ell_max * (ell_max + 2) - ell_min**2 + 1


@jit
def Yindex(ell_min, ell, m):
    """Compute index into array of mode weights

    Parameters
    ----------
    ell_min : int
        Integer satisfying 0 <= ell_min
    ell : int
        Integer satisfying ell_min <= ell <= ell_max
    m : int
        Integer satisfying -ell <= m <= ell

    Returns
    -------
    i : int
        Index of a particular element of the mode-weight array as described below

    See Also
    --------
    Ysize : Total size of array of mode weights

    Notes
    -----
    This assumes that the modes are arranged (with fixed s value) as

        [
            Y(s, ℓ, m)
            for ℓ in range(ell_min, ell_max+1)
            for m in range(-ℓ, ℓ+1)
        ]

    """
    # from sympy import symbols, summation, horner
    # from sympy.printing.pycode import pycode
    # ell,m,mp,ell_min, = symbols('ell,m,mp,ell_min', integer=True)
    # 
    # def nice(expr):
    #     return horner(expr.expand().simplify(), (ell_min, ell, m))
    #
    # Yindex = horner(
    #     (Ysize.subs(ell_max, ell-1) + summation(1, (mp, -ell, m)) - 1).expand().simplify(),
    #     (ell_max, ell, m)
    # )
    if ell > ell_min:
        return ell*(ell + 1) - ell_min**2 + m
    else:
        return m + ell


class Wigner:
    def __init__(self, ell_max, ell_min=0, mp_max=np.iinfo(np.int64).max):
        self.ell_min = int(ell_min)
        self.ell_max = int(ell_max)
        self.mp_max = min(abs(int(mp_max)), self.ell_min)

        if ell_min < 0 or ell_min > ell_max:
            raise ValueError(f"ell_min={ell_min} must be non-negative and no greater than ell_max={ell_max}")
        if ell_max < 0:
            raise ValueError(f"ell_max={ell_max} must be non-negative")

        if mp_max >= ell_max:
            self.index = self._index
        else:
            self.index = self._index_mp_max

        self._Hsize = WignerHsize(self.mp_max, self.ell_max)
        self._dsize = WignerDsize(self.ell_min, self.mp_max, self.ell_max)
        self._Dsize = WignerDsize(self.ell_min, self.mp_max, self.ell_max)
        self._Ysize = Ysize(self.ell_min, self.ell_max)

        self.Hwedge = np.empty(self.wedge_size], dtype=float)
        self.Hv = np.empty((self.n_max+1)**2, dtype=float)
        self.Hextra = np.empty(self.n_max+2, dtype=float)

    @property
    def Hsize(self):
        """Total size of Wigner H array

        The H array represents just 1/4 of the total possible indices of the H matrix,
        which are the same as for the Wigner d and 𝔇 matrices.

        This incorporates the mp_max, and ell_max information associated with this
        object.

        """
        return self._Hsize

    @property
    def dsize(self):
        """Total size of the Wigner d matrix

        This incorporates the ell_min, mp_max, and ell_max information associated with
        this object.

        """
        return self._dsize

    @property
    def Dsize(self):
        """Total size of the Wigner 𝔇 matrix

        This incorporates the ell_min, mp_max, and ell_max information associated with
        this object.

        """
        return self._Dsize

    @property
    def Ysize(self):
        """Total size of the spherical-harmonic array

        This incorporates the ell_min and ell_max information associated with this
        object.

        """
        return self._Ysize

    def Hindex(self, ell, mp, m):
        """Compute index into Wigner H matrix accounting for symmetries

        Parameters
        ----------
        ell : int
            Integer satisfying ell_min <= ell <= ell_max
        mp : int
            Integer satisfying -min(ell_max, mp_max) <= mp <= min(ell_max, mp_max)
        m : int
            Integer satisfying -ell <= m <= ell

        Returns
        -------
        i : int
            Index into Wigner H matrix arranged as described below

        See Also
        --------
        Hsize : Total size of the H matrix

        Notes
        -----
        This assumes that the Wigner H matrix is arranged as

            [
                H(ℓ, mp, m)
                for ℓ in range(ell_min, ell_max+1)
                for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
                for m in range(abs(mp), ℓ+1)
            ]

        """
        return WignerHindex(ell, mp, m, self.mp_max)

    def dindex(self, ell, mp, m):
        """Compute index into Wigner d matrix

        Parameters
        ----------
        ell : int
            Integer satisfying ell_min <= ell <= ell_max
        mp : int
            Integer satisfying -min(ell_max, mp_max) <= mp <= min(ell_max, mp_max)
        m : int
            Integer satisfying -ell <= m <= ell

        Returns
        -------
        i : int
            Index into Wigner d matrix arranged as described below

        See Also
        --------
        dsize : Total size of the d matrix

        Notes
        -----
        This assumes that the Wigner d matrix is arranged as

            [
                d(ℓ, mp, m)
                for ℓ in range(ell_min, ell_max+1)
                for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
                for m in range(-ℓ, ℓ+1)
            ]

        """
        return self.Dindex(ell, mp, m)

    def Dindex(self, ell, mp, m):
        """Compute index into Wigner 𝔇 matrix

        Parameters
        ----------
        ell : int
            Integer satisfying ell_min <= ell <= ell_max
        mp : int
            Integer satisfying -min(ell_max, mp_max) <= mp <= min(ell_max, mp_max)
        m : int
            Integer satisfying -ell <= m <= ell

        Returns
        -------
        i : int
            Index into Wigner 𝔇 matrix arranged as described below

        See Also
        --------
        Dsize : Total size of the 𝔇 matrix

        Notes
        -----
        This assumes that the Wigner 𝔇 matrix is arranged as

            [
                𝔇(ℓ, mp, m)
                for ℓ in range(ell_min, ell_max+1)
                for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
                for m in range(-ℓ, ℓ+1)
            ]

        """
        return WignerDindex(self.ell_min, self.mp_max, self.ell_max, ell, mp, m)

    def Yindex(self, ell, m):
        """Compute index into array of mode weights

        Parameters
        ----------
        ell : int
            Integer satisfying ell_min <= ell <= ell_max
        m : int
            Integer satisfying -ell <= m <= ell

        Returns
        -------
        i : int
            Index of a particular element of the mode-weight array as described below

        See Also
        --------
        Ysize : Total size of array of mode weights

        Notes
        -----
        This assumes that the modes are arranged (with fixed s value) as

            [
                Y(s, ℓ, m)
                for ℓ in range(ell_min, ell_max+1)
                for m in range(-ℓ, ℓ+1)
            ]

        """
        return Yindex(self.ell_min, ell, m)

    def H(self, expiβ):
        """Compute a quarter of the H matrix

        Parameters
        ----------
        expiβ : array_like
            Values of exp(i*β) on which to evaluate the H matrix.

        Returns
        -------
        Hwedge : array
            This is a 1-dimensional array of floats; see below.

        See Also
        --------
        d : Compute the full Wigner d matrix
        D : Compute the full Wigner 𝔇 matrix
        rotate : Avoid computing the full 𝔇 matrix and rotate modes directly
        evaluate : Avoid computing the full 𝔇 matrix and evaluate modes directly

        Notes
        -----
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
        _step_1(self.Hwedge)
        _step_2(self.g, self.h, self.n_max, self.mp_max, self.Hwedge, self.Hextra, self.Hv, expiβ.real)
        _step_3(self.a, self.b, self.n_max, self.mp_max, self.Hwedge, self.Hextra, expiβ.real, expiβ.imag)
        _step_4(self.d, self.n_max, self.mp_max, self.Hwedge, self.Hv)
        _step_5(self.d, self.n_max, self.mp_max, self.Hwedge, self.Hv)
        return self.Hwedge

    def d(self, expiβ, out=None):
        """Compute Wigner's d matrix dˡₘₚ,ₘ(β)

        Parameters
        ----------
        expiβ : array_like
            Values of expi(i*β) on which to evaluate the d matrix.
        out : array_like, optional
            Array into which the d values should be written.  It should be an array of
            floats, with size `self.dsize`.  If not present, the array will be created.
            In either case, the array will also be returned.

        Returns
        -------
        d : array
            This is a 1-dimensional array of floats; see below.

        See Also
        --------
        H : Compute a portion of the H matrix
        D : Compute the full Wigner 𝔇 matrix
        rotate : Avoid computing the full 𝔇 matrix and rotate modes directly
        evaluate : Avoid computing the full 𝔇 matrix and evaluate modes directly

        Notes
        -----
        This function is the preferred method of computing the d matrix for large ell
        values.  In particular, above ell≈32 standard formulas become completely
        unusable because of numerical instabilities and overflow.  This function uses
        stable recursion methods instead, and should be usable beyond ell≈1000.

        The result is returned in a 1-dimensional array ordered as

            [
                d(ell, mp, m, β)
                for ell in range(ell_max+1)
                for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
                for m in range(-ell, ell+1)
            ]

        """
        Hwedge = self.H(expiβ)
        d = out or np.empty(self.dsize, dtype=float)
        _fill_wigner_d(self.ell_min, self.ell_max, self.mp_max, d, self.Hwedge)
        return d

    def D(self, R, out=None):
        """Compute Wigner's 𝔇 matrix

        Parameters
        ----------
        R : array_like
            Array to be interpreted as a quaternionic array (thus its final dimension
            must have size 4), representing the rotations on which the 𝔇 matrix will be
            evaluated.
        out : array_like, optional
            Array into which the 𝔇 values should be written.  It should be an array of
            complex, with size `self.Dsize`.  If not present, the array will be
            created.  In either case, the array will also be returned.

        Returns
        -------
        D : array
            This is a 1-dimensional array of complex; see below.

        See Also
        --------
        H : Compute a portion of the H matrix
        d : Compute the full Wigner d matrix
        rotate : Avoid computing the full 𝔇 matrix and rotate modes directly
        evaluate : Avoid computing the full 𝔇 matrix and evaluate modes directly

        Notes
        -----
        This function is the preferred method of computing the 𝔇 matrix for large ell
        values.  In particular, above ell≈32 standard formulas become completely
        unusable because of numerical instabilities and overflow.  This function uses
        stable recursion methods instead, and should be usable beyond ell≈1000.

        This function computes 𝔇ˡₘₚ,ₘ(R).  The result is returned in a 1-dimensional
        array ordered as

            [
                𝔇(ell, mp, m, R)
                for ell in range(ell_max+1)
                for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
                for m in range(-ell, ell+1)
            ]

        """
        R = quaternionic.array(R)
        z = R.to_euler_phases
        Hwedge = self.H(z[1])
        𝔇 = out or np.empty(self.Dsize, dtype=complex)
        zₐpowers = complex_powers(z[0], ell_max)
        zᵧpowers = complex_powers(z[2], ell_max)
        _fill_wigner_D(self.ell_min, self.ell_max, self.mp_max, 𝔇, self.Hwedge, zₐpowers, zᵧpowers)
        return 𝔇

    def sYlm(self, s, R, out=None):
        """Evaluate (possibly spin-weighted) spherical harmonic

        Parameters
        ----------
        R : array_like
            Array to be interpreted as a quaternionic array (thus its final dimension
            must have size 4), representing the rotations on which the sYlm will be
            evaluated.
        out : array_like, optional
            Array into which the d values should be written.  It should be an array of
            complex, with size `self.Ysize`.  If not present, the array will be
            created.  In either case, the array will also be returned.

        Returns
        -------
        Y : array
            This is a 1-dimensional array of complex; see below.

        See Also
        --------
        H : Compute a portion of the H matrix
        d : Compute the full Wigner d matrix
        D : Compute the full Wigner 𝔇 matrix
        rotate : Avoid computing the full 𝔇 matrix and rotate modes directly
        evaluate : Avoid computing the full 𝔇 matrix and evaluate modes directly

        Notes
        -----
        This function is the preferred method of computing the 𝔇 matrix for large ell
        values.  In particular, above ell≈32 standard formulas become completely
        unusable because of numerical instabilities and overflow.  This function uses
        stable recursion methods instead, and should be usable beyond ell≈1000.

        This function computes ₛYₗₘ(R).  The result is returned in a 1-dimensional
        array ordered as

            [
                Y(s, ell, m, R)
                for ell in range(ell_max+1)
                for m in range(-ell, ell+1)
            ]

        """
        if abs(s) > self.mp_max:
            raise ValueError(
                f"This object has mp_max={self.mp_max}, which is not "
                f"sufficient to compute sYlm values for spin weight s={s}"
            )
        R = quaternionic.array(R)
        z = R.to_euler_phases
        Hwedge = self.H(z[1])
        Y = out or np.empty(self.Dsize, dtype=complex)
        zₐpowers = complex_powers(z[0], ell_max)
        zᵧpowers = complex_powers(z[2], ell_max)
        _fill_sYlm(self.ell_min, self.ell_max, s, Y, self.Hwedge, zₐpowers, zᵧpowers)
        return Y

    def rotate(self, modes, R):


    def evaluate(self, modes, R, out=None):
        """Evaluate Modes object as function of rotations

        Parameters
        ----------
        modes : Modes object
        R : quaternionic.array
            Arbitrarily shaped array of quaternions.  All modes in the input will be
            evaluated on each of these quaternions.  Note that it is fairly standard to
            construct these quaternions from spherical coordinates, as with the
            function `quaternionic.array.from_spherical_coordinates`.
        out : array_like, optional
            Array into which the function values should be written.  It should be an
            array of complex, with shape `modes.shape[:-1]+R.shape[:-1]`.  If not
            present, the array will be created.  In either case, the array will also be
            returned.

        Returns
        -------
        f : array_like
            This array holds the complex function values.  Its shape is
            modes.shape[:-1]+R.shape[:-1].

        """
        spin_weight = modes.spin_weight
        ell_min = modes.ell_min
        ell_max = modes.ell_max

        if abs(spin_weight) > self.mp_max:
            raise ValueError(
                f"This object has mp_max={self.mp_max}, which is not "
                f"sufficient to compute sYlm values for spin weight s={spin_weight}"
            )

        if max(abs(spin_weight), ell_min) < self.ell_min:
            raise ValueError(
                f"This object has ell_min={self.ell_min}, which is not "
                f"sufficient for the requested spin weight s={spin_weight} and ell_min={ell_min}"
            )

        if ell_max > self.ell_max:
            raise ValueError(
                f"This object has ell_max={self.ell_max}, which is not "
                f"sufficient for the input modes object with ell_max={ell_max}"
            )

        # Reinterpret inputs as 2-d np.arrays
        mode_weights = modes.ndarray.reshape((-1, modes.shape[-1]))
        quaternions = quaternionic.array(R).ndarray.reshape((-1, 4))

        # Construct storage space
        z = np.empty(3, dtype=complex)
        function_values = out or np.zeros(mode_weights.shape[:-1] + quaternions.shape[:-1], dtype=complex)

        # Loop over all input quaternions
        for i_R in range(quaternions.shape[0]):
            # Compute phases exp(iα), exp(iβ), exp(iγ) from quaternion, storing in z
            quaternionic.converters._to_euler_phases(quaternions[i_R], z)

            # Compute Wigner H elements for this quaternion
            Hwedge = self.H(z[1])

            raise NotImplementedError("Need separate arguments and logic for ell_min/max of H and of modes")
            _evaluate(mode_weights, function_values[:, i_R], spin_weight, ell_min, ell_max, abs(spin_weight), Hwedge, z[0], z[2])

        return function_values.reshape(modes.shape[:-1] + R.shape[:-1])


@jit
def _evaluate(mode_weights, function_values, spin_weight, ell_min, ell_max, mp_max, Hwedge, zₐ, zᵧ):
    """Helper function for `evaluate`"""
    z̄ₐ = zₐ.conjugate()

    coefficient = (-1)**spin_weight * ϵ(spin_weight) * zᵧ.conjugate()**spin_weight

    # Loop over all input sets of modes
    for i_modes in range(mode_weights.shape[0]):
        f = function_values[i_modes:i_modes+1]
        fₗₘ = mode_weights[i_modes]

        raise NotImplementedError("Need separate arguments and logic for ell_min/max of H and of modes")
        for ell in range(max(ell_min, abs(spin_weight)), ell_max+1):
            # Establish some base indices, relative to which offsets are simple
            i_fₗₘ = LM_index(ell, 0, ell_min)
            i_H = _WignerHindex(ell, 0, abs(spin_weight), mp_max)
            i_Hp = _WignerHindex(ell, spin_weight, abs(spin_weight), mp_max)
            i_Hm = _WignerHindex(ell, -spin_weight, abs(spin_weight), mp_max)

            # Initialize with m=0 term
            f_tmp = fₗₘ[i_fₗₘ] * Hwedge[i_H]  # H(ell, -s, 0)

            if ell > 0:

                ϵ_m = (-1)**ell

                # Compute dˡₘ₋ₛ terms recursively for 0<m<l, using symmetries for negative m, and
                # simultaneously add the mode weights times zᵧᵐ=exp[i(ϕₛ-ϕₐ)m] to the result using
                # Horner form
                negative_terms = fₗₘ[i_fₗₘ-ell] * Hwedge[i_Hp + ell - abs(spin_weight)]  # H(ell, -s, -ell)
                positive_terms = ϵ_m * fₗₘ[i_fₗₘ+ell] * Hwedge[i_Hm + ell - abs(spin_weight)]  # H(ell, -s, ell)
                for m in range(ell-1, max(0, abs(spin_weight)-1), -1):
                    ϵ_m *= -1
                    negative_terms *= z̄ₐ
                    negative_terms += fₗₘ[i_fₗₘ-m] * Hwedge[i_Hp + m - abs(spin_weight)]  # H(ell, -s, -m)
                    positive_terms *= zₐ
                    positive_terms += ϵ_m * fₗₘ[i_fₗₘ+m] * Hwedge[i_Hm + m - abs(spin_weight)]  # H(ell, -s, m)
                if spin_weight >= 0:
                    for m in range(max(0, abs(spin_weight)-1), 0, -1):
                        ϵ_m *= -1
                        negative_terms *= z̄ₐ
                        negative_terms += fₗₘ[i_fₗₘ-m] * Hwedge[_WignerHindex(ell, m, spin_weight, mp_max)]  # H(ell, -s, -m)
                        positive_terms *= zₐ
                        positive_terms += ϵ_m * fₗₘ[i_fₗₘ+m] * Hwedge[_WignerHindex(ell, -m, spin_weight, mp_max)]  # H(ell, -s, m)
                else:
                    for m in range(max(0, abs(spin_weight)-1), 0, -1):
                        ϵ_m *= -1
                        negative_terms *= z̄ₐ
                        negative_terms += fₗₘ[i_fₗₘ-m] * Hwedge[_WignerHindex(ell, -m, -spin_weight, mp_max)]  # H(ell, -s, -m)
                        positive_terms *= zₐ
                        positive_terms += ϵ_m * fₗₘ[i_fₗₘ+m] * Hwedge[_WignerHindex(ell, m, -spin_weight, mp_max)]  # H(ell, -s, m)
                f_tmp += negative_terms * z̄ₐ
                f_tmp += positive_terms * zₐ

            f_tmp *= np.sqrt((2 * ell + 1) * inverse_4pi)
            f += f_tmp

        f *= coefficient


@jit
def _fill_wigner_d(ell_min, ell_max, mp_max, d, Hwedge):
    """Helper function for Wigner.d"""
    for ell in range(ell_min, ell_max+1):
        for mp in range(-ell, ell+1):
            for m in range(-ell, ell+1):
                i_d = LMpM_index(ell, mp, m, ell_min)
                i_H = wedge_index(ell, mp, m, mp_max)
                d[i_d] = ϵ(mp) * ϵ(-m) * Hwedge[i_H]


@jit
def _fill_wigner_D(ell_min, ell_max, mp_max, 𝔇, Hwedge, zₐpowers, zᵧpowers):
    """Helper function for Wigner.D"""
    # 𝔇ˡₘₚ,ₘ(R) = dˡₘₚ,ₘ(R) exp[iϕₐ(m-mp)+iϕₛ(m+mp)] = dˡₘₚ,ₘ(R) exp[i(ϕₛ+ϕₐ)m+i(ϕₛ-ϕₐ)mp]
    # exp[iϕₛ] = R̂ₛ = hat(R[0] + 1j * R[3]) = zp
    # exp[iϕₐ] = R̂ₐ = hat(R[2] + 1j * R[1]) = zm.conjugate()
    # exp[i(ϕₛ+ϕₐ)] = zp * zm.conjugate() = z[2] = zᵧ
    # exp[i(ϕₛ-ϕₐ)] = zp * zm = z[0] = zₐ
    for ell in range(ell_min, ell_max+1):
        for mp in range(-ell, 0):
            i_D = LMpM_index(ell, mp, -ell, ell_min)
            for m in range(-ell, 0):
                i_H = wedge_index(ell, mp, m, mp_max)
                𝔇[i_D] = ϵ(mp) * ϵ(-m) * Hwedge[i_H] * zᵧpowers[-m].conjugate() * zₐpowers[-mp].conjugate()
                i_D += 1
            for m in range(0, ell+1):
                i_H = wedge_index(ell, mp, m, mp_max)
                𝔇[i_D] = ϵ(mp) * ϵ(-m) * Hwedge[i_H] * zᵧpowers[m] * zₐpowers[-mp].conjugate()
                i_D += 1
        for mp in range(0, ell+1):
            i_D = LMpM_index(ell, mp, -ell, ell_min)
            for m in range(-ell, 0):
                i_H = wedge_index(ell, mp, m, mp_max)
                𝔇[i_D] = ϵ(mp) * ϵ(-m) * Hwedge[i_H] * zᵧpowers[-m].conjugate() * zₐpowers[mp]
                i_D += 1
            for m in range(0, ell+1):
                i_H = wedge_index(ell, mp, m, mp_max)
                𝔇[i_D] = ϵ(mp) * ϵ(-m) * Hwedge[i_H] * zᵧpowers[m] * zₐpowers[mp]
                i_D += 1


@jit
def _fill_sYlm(ell_min, ell_max, s, Y, Hwedge, zₐpowers, zᵧpowers):
    """Helper function for Wigner.sYlm"""
    mp = -s
    for ell in range(ell_min, ell_max+1):
        coefficient = (-1)**s * np.sqrt((2 * ell + 1) * inverse_4pi)
        i_D = LMpM_index(ell, mp, -ell, ell_min)
        for m in range(-ell, 0):
            i_H = wedge_index(ell, mp, m, mp_max)
            Y[i_D] = coefficient * ϵ(mp) * ϵ(-m) * Hwedge[i_H] * zᵧpowers[-m].conjugate() * zₐpowers[-mp].conjugate()
            i_D += 1
        for m in range(0, ell+1):
            i_H = wedge_index(ell, mp, m, mp_max)
            Y[i_D] = coefficient * ϵ(mp) * ϵ(-m) * Hwedge[i_H] * zᵧpowers[m] * zₐpowers[-mp].conjugate()
            i_D += 1
        i_D = LMpM_index(ell, mp, -ell, ell_min)
        for m in range(-ell, 0):
            i_H = wedge_index(ell, mp, m, mp_max)
            Y[i_D] = coefficient * ϵ(mp) * ϵ(-m) * Hwedge[i_H] * zᵧpowers[-m].conjugate() * zₐpowers[mp]
            i_D += 1
        for m in range(0, ell+1):
            i_H = wedge_index(ell, mp, m, mp_max)
            Y[i_D] = coefficient * ϵ(mp) * ϵ(-m) * Hwedge[i_H] * zᵧpowers[m] * zₐpowers[mp]
            i_D += 1
