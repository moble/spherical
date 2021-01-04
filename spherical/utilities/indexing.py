# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

"""Utilities for computing indexes and total sizes of H, d, and 𝔇 matrices

"""

from .. import jit


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
