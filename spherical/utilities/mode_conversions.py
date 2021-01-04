# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

from math import sqrt, pi
import numpy as np
from .. import jit


@jit
def constant_as_ell_0_mode(constant):
    """Express constant as Y_{0,0} mode weight"""
    return constant * sqrt(4 * pi)


@jit
def constant_from_ell_0_mode(modes):
    """Express Y_{0,0} mode as constant

    This is just the inverse of the `constant_as_ell_0_mode` function.  Note that this does not assume that the
    output constant should be real.  If you want it to be, you must take the real part manually.

    """
    return modes / sqrt(4 * pi)


@jit
def vector_as_ell_1_modes(vector):
    """Express vector as Y_{1,m} mode weights

    A vector can be represented as a linear combination of weights of the ell=1 scalar spherical harmonics.
    Explicitly, if nhat is the usual unit vector in the (theta, phi) direction, then we can define a function
      v(theta, phi) = vector . nhat
    where the `.` represents the dot product, and v(theta, phi) is a pure ell=1 function.  This function simply
    returns the weights of that representation.

    Parameter
    ---------
    vector : float array of length 3
        The input should be an iterable containing the [v_x, v_y, v_z] components of the vector in that order

    Returns
    -------
    float array
       The returned object contains the (1,-1), (1,0), and (1,1) modes in that order

    """
    return np.stack((np.asarray((vector[..., 0] + 1j * vector[..., 1]) * sqrt(2 * pi / 3.)),
                     np.asarray(vector[..., 2] * sqrt(4 * pi / 3.)),
                     np.asarray((-vector[..., 0] + 1j * vector[..., 1]) * sqrt(2 * pi / 3.))), axis=-1)


@jit
def vector_from_ell_1_modes(modes):
    """Express Y_{1,m} modes as vector

    This is just the inverse of the `vector_as_ell_1_modes` function.  Note that this does not assume that the vector
    should be real-valued.  If you want it to be, you must call the `.real` method of the output array manually.

    """
    return np.stack((np.asarray((modes[..., 0] - modes[..., 2]) / (2 * sqrt(2 * pi / 3.))),
                     np.asarray((modes[..., 0] + modes[..., 2]) / (2j * sqrt(2 * pi / 3.))),
                     np.asarray(modes[..., 1] / sqrt(4 * pi / 3.))), axis=-1)
