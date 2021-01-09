# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

"""Utilities for computing indexes and total sizes of H, d, and 𝔇 matrices

"""

import numpy as np
from .. import jit


@jit
def WignerHsize(mp_max, ell_max=-2):
    """Total size of array of wedges of width mp_max up to ell_max

    Parameters
    ----------
    ell_max : int
    mp_max : int, optional
        If None, it is assumed to be at least ell

    See Also
    --------
    WignerHrange : Array of (ℓ, m', m) indices corresponding to this wedge
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
    if ell_max == -2:
        ell_max = mp_max
    elif ell_max < 0:
        return 0
    if mp_max is None or mp_max > ell_max:
        return (ell_max+1) * (ell_max+2) * (2*ell_max+3) // 6
    else:
        return ((ell_max+1)*(ell_max+2)*(2*ell_max+3) - 2*(ell_max-mp_max)*(ell_max-mp_max+1)*(ell_max-mp_max+2)) // 6


@jit
def WignerHrange(mp_max, ell_max=-1):
    """Create an array of (ℓ, m', m) indices as in H array

    Parameters
    ----------
    ell_max : int
    mp_max : int, optional
        If None, it is assumed to be at least ell

    See Also
    --------
    WignerHsize : Total size of wedge array
    WignerHindex : Index inside these wedges

    Notes
    -----
    Here, it is assumed that only data with m≥|m'| are stored, and only
    corresponding values are passed.  We also assume |m|≤ℓ and |m'|≤ℓ.  Neither of
    these are checked.  The wedge array that this function indexes is ordered as

        [
            H(ell, mp, m) for ell in range(ell_max+1)
            for mp in range(-min(ell, mp_max), min(ell, mp_max)+1)
            for m in range(abs(mp), ell+1)
        ]

    """
    if ell_max < 0:
        ell_max = mp_max
    r = np.zeros((WignerHsize(mp_max, ell_max), 3), dtype=np.int64)
    i = 0
    for ell in range(ell_max+1):
        for mp in range(-min(ell, mp_max), min(ell, mp_max)+1):
            for m in range(abs(mp), ell+1):
                r[i, 0] = ell
                r[i, 1] = mp
                r[i, 2] = m
                i += 1
    return r


@jit
def _WignerHindex(ell, mp, m, mp_max):
    """Helper function for `WignerHindex`"""
    i = WignerHsize(mp_max, ell-1)  # total size of everything with smaller ell
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
    WignerHrange : Array of (ℓ, m', m) indices corresponding to this wedge

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
def WignerDsize(ell_min, mp_max, ell_max=-1):
    """Compute total size of Wigner 𝔇 matrix

    Parameters
    ----------
    ell_min : int
        Integer satisfying 0 <= ell_min <= ell_max
    mp_max : int, optional
        Integer satisfying 0 <= mp_max.  Defaults to ell_max.
    ell_max : int
        Integer satisfying 0 <= ell_min <= ell_max

    Returns
    -------
    i : int
        Total size of Wigner 𝔇 matrix arranged as described below

    See Also
    --------
    WignerDrange : Array of (ℓ, m', m) indices corresponding to the 𝔇 matrix
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
    if ell_max < 0:
        ell_max = mp_max
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
def WignerDrange(ell_min, mp_max, ell_max=-1):
    """Create an array of (ℓ, m', m) indices as in 𝔇 array

    Parameters
    ----------
    ell_min : int
        Integer satisfying 0 <= ell_min <= ell_max
    mp_max : int, optional
        Integer satisfying 0 <= mp_max.  Default is ell_max.
    ell_max : int
        Integer satisfying 0 <= ell_min <= ell_max

    See Also
    --------
    WignerDsize : Total size of 𝔇 array
    WignerDindex : Index inside these wedges

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
    if ell_max < 0:
        ell_max = mp_max
    r = np.zeros((WignerDsize(ell_min, mp_max, ell_max), 3), dtype=np.int64)
    i = 0
    for ell in range(ell_min, ell_max+1):
        for mp in range(-min(ell, mp_max), min(ell, mp_max)+1):
            for m in range(-ell, ell+1):
                r[i, 0] = ell
                r[i, 1] = mp
                r[i, 2] = m
                i += 1
    return r


@jit
def WignerDindex(ell, mp, m, ell_min=0, mp_max=-1):
    """Compute index into Wigner 𝔇 matrix

    Parameters
    ----------
    ell : int
        Integer satisfying ell_min <= ell <= ell_max
    mp : int
        Integer satisfying -min(ell_max, mp_max) <= mp <= min(ell_max, mp_max)
    m : int
        Integer satisfying -ell <= m <= ell
    ell_min : int, optional
        Integer satisfying 0 <= ell_min <= ell_max.  Defaults to 0.
    mp_max : int, optional
        Integer satisfying 0 <= mp_max.  Defaults to ell.

    Returns
    -------
    i : int
        Index into Wigner 𝔇 matrix arranged as described below

    See Also
    --------
    WignerDsize : Total size of the 𝔇 matrix
    WignerDrange : Array of (ℓ, m', m) indices corresponding to the 𝔇 matrix

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
    if mp_max < 0:
        mp_max = ell
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
    Yrange : Array of (ℓ, m) indices corresponding to this array
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
def Yrange(ell_min, ell_max):
    """Create an array of (ℓ, m) indices as in Y array

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
    Ysize : Total size of array of mode weights
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
    r = np.zeros((Ysize(ell_min, ell_max), 2), dtype=np.int64)
    i = 0
    for ell in range(ell_min, ell_max+1):
        for m in range(-ell, ell+1):
            r[i, 0] = ell
            r[i, 1] = m
            i += 1
    return r


@jit
def Yindex(ell, m, ell_min=0):
    """Compute index into array of mode weights

    Parameters
    ----------
    ell : int
        Integer satisfying ell_min <= ell <= ell_max
    m : int
        Integer satisfying -ell <= m <= ell
    ell_min : int, optional
        Integer satisfying 0 <= ell_min.  Defaults to 0.

    Returns
    -------
    i : int
        Index of a particular element of the mode-weight array as described below

    See Also
    --------
    Ysize : Total size of array of mode weights
    Yrange : Array of (ℓ, m) indices corresponding to this array

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


def theta_phi(n_theta, n_phi):
    """Construct (theta, phi) grid

    This grid is in the order expected by spinsfast

    Parameters
    ----------
    n_theta : int
        Number of points in the theta direction
    n_phi : int
        Number of points in the phi direction

    Returns
    -------
    theta_phi_grid : ndarray
        Array of pairs of floats giving the respective [theta, phi] pairs.  The
        shape of this array is (n_theta, n_phi, 2).

    Notes
    -----
    The array looks like

        [
            [θ, ϕ]
            for ϕ ∈ [0, 2π)
            for θ ∈ [0, π]
        ]

    (note the open and closed endpoints, respectively), where ϕ and θ are uniformly
    sampled in their respective ranges.

    """
    return np.array([[[theta, phi]
                      for phi in np.linspace(0.0, 2*np.pi, num=n_phi, endpoint=False)]
                     for theta in np.linspace(0.0, np.pi, num=n_theta, endpoint=True)])

