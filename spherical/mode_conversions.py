# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/waveforms/blob/master/LICENSE>

from math import sqrt, pi
import numpy as np
from . import LM_total_size, jit


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


@jit
def eth_GHP(modes, spin_weight, ell_min=0):
    """Spin-raising eth operator as defined by Geroch-Held-Penrose

    N.B.: For our purposes, eth_GHP is the same as eth_NP/sqrt(2).

    This operator is defined in J. Math. Phys. 14, 874 (1973) <http://link.aip.org/link/?JMP/14/874/1>. The eth
    operator was originally defined by Newman and Penrose in J. Math. Phys. 7, 863 (1966)
    <http://link.aip.org/link/?JMP/7/863/1>, and discussed further by Goldberg et al., J. Math. Phys. 8, 2155 (1967)
    <http://link.aip.org/link/?JMP/8/2155/1>.  In the case of a spacelike 2-surface (which is all we treat in this
    function), the GHP eth operator reduces almost to the original.  By comparing GHP's Eq. (3.8) to Eq. (2.13) of
    Goldberg et al., we can see the difference.  As GHP say below their Eq. (3.11) that "for complete agreement in
    the case of a general 2-surface metric we would have (strictly speaking) to multiply up our eth operator by a
    factor sqrt(2)."

    Parameters
    ----------
    modes : 1-d complex array
        This array contains the modes starting from ell=ell_min, and continuing in standard order (as in sf.LM_range).
    spin_weight : int
        Spin weight of the input field.  The eth operators raise the spin weight by 1.
    ell_min : int, optional
        Smallest ell value present in input `modes`.  Defaults to 0.

    Returns
    -------
    1-d complex array
        The output has the same size as the input `modes`, and corresponds to the same (ell,m) values.  Note, however,
        that these modes have spin weight greater than the input by 1.

    """
    ell_max = int(sqrt(len(modes) + LM_total_size(0, ell_min - 1))) - 1
    eth_modes = np.copy(modes)
    i_mode = 0
    for ell in range(ell_min, ell_max + 1):
        factor = (0.0 if ell < abs(spin_weight + 1) else sqrt((ell - spin_weight) * (ell + spin_weight + 1.) / 2.))
        for m in range(-ell, ell + 1):
            eth_modes[i_mode] *= factor
            i_mode += 1
    return eth_modes


@jit
def ethbar_GHP(modes, spin_weight, ell_min=0):
    """Spin-lowering \bar{eth} operator as defined by Geroch-Held-Penrose

    N.B.: For our purposes, eth_GHP is the same as eth_NP/sqrt(2).

    This is the complex conjugate of the `eth_GHP` operator.  See that function's docstring for more information.

    Parameters
    ----------
    modes : 1-d complex array
        This array contains the modes starting from ell=ell_min, and continuing in standard order (as in sf.LM_range).
    spin_weight : int
        Spin weight of the input field.  The \bar{eth} operators lower the spin weight by 1.
    ell_min : int, optional
        Smallest ell value present in input `modes`.  Defaults to 0.

    Returns
    -------
    1-d complex array
        The output has the same size as the input `modes`, and corresponds to the same (ell,m) values.  Note, however,
        that these modes have spin weight less than the input by 1.

    """
    ell_max = int(sqrt(len(modes) + LM_total_size(0, ell_min - 1))) - 1
    ethbar_modes = np.copy(modes)
    i_mode = 0
    for ell in range(ell_min, ell_max + 1):
        factor = (0.0 if ell < abs(spin_weight - 1) else -sqrt((ell + spin_weight) * (ell - spin_weight + 1.) / 2.))
        for m in range(-ell, ell + 1):
            ethbar_modes[i_mode] *= factor
            i_mode += 1
    return ethbar_modes


@jit
def eth_NP(modes, spin_weight, ell_min=0):
    """Spin-raising eth operator as defined by Newman and Penrose

    N.B.: For our purposes, eth_GHP is the same as eth_NP/sqrt(2).

    This is the original eth operator defined by Newman and Penrose.  See the documentation for eth_GHP for more detail.

    Parameters
    ----------
    modes : 1-d complex array
        This array contains the modes starting from ell=ell_min, and continuing in standard order (as in sf.LM_range).
    spin_weight : int
        Spin weight of the input field.  The eth operators raise the spin weight by 1.
    ell_min : int, optional
        Smallest ell value present in input `modes`.  Defaults to 0.

    Returns
    -------
    1-d complex array
        The output has the same size as the input `modes`, and corresponds to the same (ell,m) values.  Note, however,
        that these modes have spin weight greater than the input by 1.

    """
    ell_max = int(sqrt(len(modes) + LM_total_size(0, ell_min - 1))) - 1
    eth_modes = np.copy(modes)
    i_mode = 0
    for ell in range(ell_min, ell_max + 1):
        factor = (0.0 if ell < abs(spin_weight + 1) else sqrt((ell - spin_weight) * (ell + spin_weight + 1.)))
        for m in range(-ell, ell + 1):
            eth_modes[i_mode] *= factor
            i_mode += 1
    return eth_modes


@jit
def ethbar_NP(modes, spin_weight, ell_min=0):
    """Spin-lowering \bar{eth} operator as defined by Newman and Penrose

    N.B.: For our purposes, eth_GHP is the same as eth_NP/sqrt(2).

    This is the complex conjugate of the `eth_NP` operator.  See that function's docstring for more information.

    Parameters
    ----------
    modes : 1-d complex array
        This array contains the modes starting from ell=ell_min, and continuing in standard order (as in sf.LM_range).
    spin_weight : int
        Spin weight of the input field.  The \bar{eth} operators lower the spin weight by 1.
    ell_min : int, optional
        Smallest ell value present in input `modes`.  Defaults to 0.

    Returns
    -------
    1-d complex array
        The output has the same size as the input `modes`, and corresponds to the same (ell,m) values.  Note, however,
        that these modes have spin weight less than the input by 1.

    """
    ell_max = int(sqrt(len(modes) + LM_total_size(0, ell_min - 1))) - 1
    ethbar_modes = np.copy(modes)
    i_mode = 0
    for ell in range(ell_min, ell_max + 1):
        factor = (0.0 if ell < abs(spin_weight - 1) else -sqrt((ell + spin_weight) * (ell - spin_weight + 1.)))
        for m in range(-ell, ell + 1):
            ethbar_modes[i_mode] *= factor
            i_mode += 1
    return ethbar_modes


@jit
def ethbar_inverse_NP(modes, spin_weight, ell_min=0):
    """Inverse of the spin-lowering \bar{eth} operator as defined by Newman and Penrose

    This function acts as a (partial) inverse or integral of the `ethbar_NP` operator.  (See that function's
    docstring for more information, including the calling signature.)  In particular, composing the two functions in
    either order should give (almost) the identity function.  Essentially, this is an integral of the ethbar
    function; so this is only a partial inverse because constants of integration could be added in some cases.  The
    difference between this function and `eth_NP` is mostly in the normalization.

    """
    ell_max = int(sqrt(len(modes) + LM_total_size(0, ell_min - 1))) - 1
    ethbar_inverse_modes = np.copy(modes)
    i_mode = 0
    for ell in range(ell_min, ell_max + 1):
        term = (ell + spin_weight + 1.) * (ell - spin_weight)
        if term > 0.0:
            factor = -sqrt(term)
            for m in range(-ell, ell + 1):
                ethbar_inverse_modes[i_mode] /= factor
                i_mode += 1
        else:
            i_mode += 2*ell+1
    return ethbar_inverse_modes
