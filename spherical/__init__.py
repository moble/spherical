# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

"""Evaluating Wigner 𝔇 matrices, spin-weighted spherical harmonics, and related.

This module contains code for evaluating the Wigner 3j symbols, the Wigner 𝔇
matrices, scalar spherical harmonics, and spin-weighted spherical harmonics.

The implementations are in terms of recursions, which allow the calculations to
be very accurate and stable to high indices.  Whereas straightforward
implementation of the usual formulas for these objects quickly leads to
instabilities and overflow at ℓ values as low as 29, this implementation allows
accurate computation beyond ℓ values of 1000.

The code is wrapped by numba where possible, allowing the results to be
delivered at speeds approaching or exceeding speeds attained by pure C code.

Two useful classes are also provided to encapsulate functions decomposed in
mode weights or by their values on spherical grids.

"""

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:  # pragma: no cover
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)

import functools
import numpy as np
import numba as nb

jit = njit = functools.partial(nb.njit, cache=True)
jitclass = nb.experimental.jitclass

from .utilities.indexing import (
    WignerHsize, WignerHrange, WignerHindex,
    WignerDsize, WignerDrange, WignerDindex,
    Ysize, Yrange, Yindex,
    theta_phi,
)
LMpM_total_size, LMpM_range, LMpM_index = WignerDsize, WignerDrange, WignerDindex
LM_total_size, LM_range, LM_index = Ysize, Yrange, Yindex
from .utilities.mode_conversions import (
    constant_as_ell_0_mode, constant_from_ell_0_mode,
    vector_as_ell_1_modes, vector_from_ell_1_modes,
)
from .utilities.operators import (
    eth_GHP, ethbar_GHP, eth_NP, ethbar_NP, ethbar_inverse_NP
)

from .recursions.complex_powers import complex_powers
from .recursions.wigner3j import Wigner3jCalculator, Wigner3j, clebsch_gordan

from .wigner import Wigner, wigner_d, wigner_D

from .multiplication import multiply

from .modes import Modes
from .grid import Grid
