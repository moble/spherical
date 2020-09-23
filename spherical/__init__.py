# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

"""Evaluating Wigner D matrices, spin-weighted spherical harmonics, and related.

This module contains code for evaluating the Wigner 3j symbols, the Wigner D
matrices, scalar spherical harmonics, and spin-weighted spherical harmonics.
The code is wrapped by numba where possible, allowing the results to be
delivered at speeds approaching or exceeding speeds attained by pure C code.

"""

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:  # pragma: no cover
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)

__all__ = ['Wigner3j', 'Wigner_D_element', 'Wigner_D_matrices', 'SWSH', 'SWSH_grid',
           'factorial', 'binomial_coefficient', 'ladder_operator_coefficient']

import functools
import numpy as np
import numba as nb
from math import factorial
import os.path

#jit = njit = functools.partial(nb.jit, forceobj=True)
jit = njit = functools.partial(nb.njit, cache=True)
jitclass = nb.experimental.jitclass

# Module constants
ell_max = 32  # More than 29, and you get roundoff building quickly
epsilon = 1.e-15
error_on_bad_indices = True

# The coefficient files
#   binomial_coefficients.npy
#   ladder_operator_coefficients.npy
#   Wigner_coefficients.npy
# were originally produced with the code in `_generate_coefficients.py`.


# Factorial
factorials = np.array([float(factorial(i)) for i in range(171)])


@njit('f8(i8)')
def factorial(i):
    return factorials[i]


# Binomial coefficients
_binomial_coefficients = np.load(os.path.join(os.path.dirname(__file__), 'binomial_coefficients.npy'))

@njit('f8(i8,i8)')
def binomial_coefficient(n, k):
    return _binomial_coefficients[(n * (n + 1)) // 2 + k]


# Ladder-operator coefficients
_ladder_operator_coefficients = np.load(os.path.join(os.path.dirname(__file__), 'ladder_operator_coefficients.npy'))

@njit('f8(i8,i8)')
def _ladder_operator_coefficient(twoell, twom):
    return _ladder_operator_coefficients[((twoell + 2) * twoell + twom) // 2]

@njit('f8(f8,f8)')
def ladder_operator_coefficient(ell, m):
    return _ladder_operator_coefficient(round(2*ell), round(2*m))


# Coefficients used in constructing the Wigner D matrices
_Wigner_coefficients = np.load(os.path.join(os.path.dirname(__file__), 'Wigner_coefficients.npy'))

@njit('i8(i8,i8,i8)')
def _Wigner_index(twoell, twomp, twom):
    return twoell*((2*twoell + 3)*twoell + 1) // 6 + (twoell + twomp)//2 * (twoell + 1) + (twoell + twom)//2

@njit('f8(i8,i8,i8)')
def _Wigner_coefficient(twoell, twomp, twom):
    return _Wigner_coefficients[_Wigner_index(twoell, twomp, twom)]

@njit('f8(f8,f8,f8)')
def Wigner_coefficient(ell, mp, m):
    return _Wigner_coefficient(round(2*ell), round(2*mp), round(2*m))


def LM_range(ell_min, ell_max):
    """Array of (ell,m) indices in standard order

    This function returns an array of essentially

    [[ell,m] for ell in range(ell_min, ell_max+1)
             for m in range(-ell,ell+1)]

    This is, for example, the order assumed for mode data in the `waveforms`
    module.

    """
    # # Sympy commands to calculate the total size:
    # from sympy import symbols, summation
    # ell_min,ell,ell_max = symbols('ell_min,ell,ell_max', integer=True)
    # summation((2*ell + 1), (ell, ell_min, ell_max))
    LM = np.empty((ell_max * (ell_max + 2) - ell_min ** 2 + 1, 2), dtype=np.int64)
    _LM_range(ell_min, ell_max, LM)
    return LM


@njit('void(i8,i8,i8[:,:])')
def _LM_range(ell_min, ell_max, LM):
    i = 0
    for ell in range(ell_min, ell_max + 1):
        for m in range(-ell, ell + 1):
            LM[i, 0] = ell
            LM[i, 1] = m
            i += 1


@njit('i8(i8,i8,i8)')
def LM_index(ell, m, ell_min):
    """Array index for given (ell,m) mode

    Assuming an array of

    [[ell,m] for ell in range(ell_min, ell_max+1)
             for m in range(-ell,ell+1)]

    this function returns the index of the (ell,m) element.  (Note that
    ell_max doesn't actually come into this calculation, so it is not taken
    as an argument to the function.)

    This can be calculated in sympy as

      from sympy import symbols, summation
      ell,m,ell_min, = symbols('ell,m,ell_min,', integer=True)
      summation(2*ell + 1, (ell, ell_min, ell-1)) + (ell+m)

    """
    return ell * (ell + 1) - ell_min ** 2 + m


@njit('i8(i8,i8)')
def LM_total_size(ell_min, ell_max):
    """Total array size of (ell,m) components

    Assuming an array of

    [[ell,m] for ell in range(ell_min, ell_max+1)
             for m in range(-ell,ell+1)]

    this function returns the total size of that array.

    This can be calculated in sympy as

      from sympy import symbols, summation
      ell,ell_min,ell_max = symbols('ell,ell_min,ell_max', integer=True)
      summation(2*ell + 1, (ell, ell_min, ell_max))

    """
    return ell_max * (ell_max + 2) - ell_min ** 2 + 1


def LM_deduce_ell_max(size, ell_min=0):
    ell_max = int(np.sqrt(size + ell_min**2) - 1)
    if ell_max * (ell_max + 2) - ell_min ** 2 + 1 != size:
        raise ValueError(f"The input size {size} does not correspond to a possible array of modes with ell_min {ell_min}.")
    return ell_max


def LMpM_range(ell_min, ell_max):
    """Array of (ell,mp,m) indices in standard order

    This function returns an array of essentially

    [[ell,mp,m] for ell in range(ell_min, ell_max+1)
                for mp in range(-ell,ell+1)
                for m in range(-ell,ell+1)]

    This is, for instance, the array of indices of the Wigner D matrices
    constructed by this module.

    """
    # # Sympy commands to calculate the total size:
    # from sympy import symbols, summation
    # ell_min,ell,ell_max = symbols('ell_min,ell,ell_max', integer=True)
    # summation((2*ell + 1)**2, (ell, ell_min, ell_max))
    if abs(round(ell_max)-ell_max) > 1e-10 or abs(round(ell_min)-ell_min) > 1e-10:
        error = ("LMpM_range is only intended for integer values of ell.\n"
                 + "Input values ell_min={0} and ell_max={1} are not valid.\n".format(ell_min, ell_max)
                 + "Try `LMpM_range_half_integer` instead.")
        raise ValueError(error)
    LMpM = np.empty(((ell_max * (11 + ell_max * (12 + 4 * ell_max)) + ell_min * (1 - 4 * ell_min ** 2) + 3) // 3, 3),
                    dtype=np.int64)
    _LMpM_range(ell_min, ell_max, LMpM)
    return LMpM

@njit('void(i8,i8,i8[:,:])')
def _LMpM_range(ell_min, ell_max, LMpM):
    i = 0
    for ell in range(ell_min, ell_max + 1):
        for mp in range(-ell, ell + 1):
            for m in range(-ell, ell + 1):
                LMpM[i, 0] = ell
                LMpM[i, 1] = mp
                LMpM[i, 2] = m
                i += 1


def LMpM_range_half_integer(ell_min, ell_max):
    """Array of (ell,mp,m) indices in standard order, including half-integer values

    This function returns an array of essentially

    [[twoell/2, twomp/2, twom/2]
     for twoell in range(2*ell_min, 2(ell_max+1)
     for twomp in range(-twoell, twoell+1, 2)
     for twom in range(-twoell, twoell+1, 2)]

    See also: LMpM_range

    """
    # # Sympy commands to calculate the total size:
    # from sympy import symbols, summation
    # twoell_min,twoell,twoell_max = symbols('twoell_min,twoell,twoell_max', integer=True)
    # summation((twoell + 1)**2, (twoell, twoell_min, twoell_max))
    LMpM = np.empty((int(((8*ell_max + 18)*ell_max + 13)*ell_max + 3 - ((8 * ell_min + 6) * ell_min + 1)*ell_min) // 3,
                     3), dtype=np.float64)
    _LMpM_range_half_integer(round(2*ell_min), round(2*ell_max), LMpM)
    return LMpM

@njit('void(i8,i8,f8[:,:])')
def _LMpM_range_half_integer(twoell_min, twoell_max, LMpM):
    i = 0
    for twoell in range(twoell_min, twoell_max + 1):
        for twomp in range(-twoell, twoell + 1, 2):
            for twom in range(-twoell, twoell + 1, 2):
                LMpM[i, 0] = twoell / 2
                LMpM[i, 1] = twomp / 2
                LMpM[i, 2] = twom / 2
                i += 1


@njit('i8(i8,i8,i8,i8)')
def LMpM_index(ell, mp, m, ell_min):
    """Array index for given (ell,mp,m) mode

    Assuming an array (e.g., Wigner D matrices) in the order

    [[ell,mp,m] for ell in range(ell_min, ell_max+1)
                for mp in range(-ell,ell+1)
                for m in range(-ell,ell+1)]

    this function returns the index of the (ell,mp,m) element.  (Note that
    ell_max doesn't actually come into this calculation, so it is not taken
    as an argument to the function.)

    This can be calculated in sympy as

      from sympy import symbols, summation
      ell,mp,m,ell_min, = symbols('ell,mp,m,ell_min,', integer=True)
      summation((2*ell + 1)**2, (ell, ell_min, ell-1)) + (2*ell+1)*(ell+mp) + (ell+m)

    """
    # raw output is: 4*ell**3/3 + 2*ell**2 + 2*ell*mp + 5*ell/3 - 4*ell_min**3/3 + ell_min/3 + m + mp
    # We rearrange that to act more nicely
    return (((4 * ell + 6) * ell + 6 * mp + 5) * ell + ell_min * (1 - 4 * ell_min ** 2) + 3 * (m + mp)) // 3


@njit('i8(i8, i8)')
def LMpM_total_size(ell_min, ell_max):
    """Total array size of Wigner D matrix

    Assuming an array (e.g., Wigner D matrices) in the order

    [[ell,mp,m] for ell in range(ell_min, ell_max+1)
                for mp in range(-ell,ell+1)
                for m in range(-ell,ell+1)]

    this function returns the total size of that array.

    This can be calculated in sympy as

      from sympy import symbols, summation
      ell,ell_min,ell_max = symbols('ell,ell_min,ell_max', integer=True)
      summation((2*ell + 1)**2, (ell, ell_min, ell_max))

    """
    # raw output is: 4*ell_max**3/3 + 4*ell_max**2 + 11*ell_max/3 - 4*ell_min**3/3 + ell_min/3 + 1
    # We rearrange that to act more nicely
    return (((4 * ell_max + 12) * ell_max + 11) * ell_max + (-4 * ell_min ** 2 + 1) * ell_min + 3) // 3


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
        Array of pairs of floats giving the respective [theta, phi] pairs.  The shape of this array
        is (n_theta, n_phi, 2).

    """
    return np.array([[[theta, phi]
                      for phi in np.linspace(0.0, 2*np.pi, num=n_phi, endpoint=False)]
                     for theta in np.linspace(0.0, np.pi, num=n_theta, endpoint=True)])


from .WignerD import (Wigner_D_element, _Wigner_D_element,
                      Wigner_D_matrices, _Wigner_D_matrices,
                      _linear_matrix_index, _linear_matrix_diagonal_index,
                      _linear_matrix_offset, _total_size_D_matrices)
from .SWSH import SWSH, SWSH_grid, _SWSH  # sYlm, Ylm
from .SWSH_modes import Modes
from .SWSH_grids import Grid
from .mode_conversions import (constant_as_ell_0_mode, constant_from_ell_0_mode,
                               vector_as_ell_1_modes, vector_from_ell_1_modes,
                               eth_GHP, ethbar_GHP, eth_NP, ethbar_NP,
                               ethbar_inverse_NP)
from .multiplication import multiply
from .recursions import complex_powers, Wigner3jCalculator, Wigner3j, clebsch_gordan
