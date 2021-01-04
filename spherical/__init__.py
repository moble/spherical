# Copyright (c) 2021, Michael Boyle
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

import functools
import numpy as np
import numba as nb
from math import factorial
import os.path

jit = njit = functools.partial(nb.njit, cache=True)
jitclass = nb.experimental.jitclass


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

from .modes import Modes
from .grid import Grid
from .utilities.mode_conversions import (
    constant_as_ell_0_mode, constant_from_ell_0_mode,
    vector_as_ell_1_modes, vector_from_ell_1_modes,
)
from .utilities.operators import (
    eth_GHP, ethbar_GHP, eth_NP, ethbar_NP, ethbar_inverse_NP
)
from .utilities.indexing import (
    WignerHsize, WignerHrange, WignerHindex,
    WignerDsize, WignerDrange, WignerDindex,
    Ysize, Yrange, Yindex,
)
LMpM_total_size, LMpM_range, LMpM_index = WignerDsize, WignerDrange, WignerDindex
LM_total_size, LM_range, LM_index = Ysize, Yrange, Yindex
from .multiplication import multiply
from .recursions import Wigner3jCalculator, Wigner3j, clebsch_gordan
from .recursions.complex_powers import complex_powers
from .wigner import Wigner
