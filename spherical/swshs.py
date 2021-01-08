# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import numpy as np
import quaternionic


def SWSH(R, s, indices, ell_min=0):
    """Evaluate spin-weighted spherical harmonic functions

    Parameters
    ----------
    R : quaternionic.array
        Rotors on which to evaluate the SWSH function (could be just one quaternion)
    s : int
        Spin weight of the field to evaluate
    indices : int, pair of ints, or 2-d array of ints
        The (ell, m) indices to evaluate.  If this is a single int, all modes up to
        and including this ℓ value are output.  If this is a 2-d array, its second
        dimension must have size 2.
    ell_min : int, optional
        If `indices` is a single integer, the output will contain ℓ values starting
        with this number.

    Returns
    -------
    complex array
        The shape of this array is `R.shape[:-1] + (N_lm,)`, where N_lm is the
        number of (ell, m) pairs given or implied by the `indices` argument.

    Notes
    -----
    This function is more general than standard Yₗₘ and ₛYₗₘ functions of angles
    because it uses quaternion rotors instead of angles, and is slightly faster as
    a result.  The core calculation is performed by the `spherical.Wigner` object,
    which uses recursion for speed and accuracy far beyond those achievable by more
    standard approaches.

    """
    from . import Yrange, Wigner

    # Process input arguments
    R = quaternionic.array(R)
    shape = R.shape[:-1]
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim == 0:
        indices = Yrange(ell_min, indices[()])
        shape = R.shape[:-1] + indices.shape[:-1]
    elif indices.shape == (2,):
        indices = np.atleast_2d(indices)
    elif indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(
            f"Input `indices` argument must be a single integer, a pair of integers, or a 2-d array\n"
            f"of integers with second dimension of size 2; it is {indices}."
        )
    else:
        shape = R.shape[:-1] + indices.shape[:-1]
    ell_max = np.max(indices[:, 0])
    if ell_min > ell_max:
        raise ValueError(f"Maximum ell value in input is {ell_max}, which must be greater than ell_min={ell_min}")
    if abs(s) > ell_max:
        raise ValueError(f"Spin weight s={s} will return no values for requested maximum ell value {ell_max}.")

    # Set up input, outputs, and Wigner
    rotors = R.reshape(-1, 4)
    output = np.zeros((rotors.shape[0], indices.shape[0]), dtype=complex)
    wigner = Wigner(ell_max, ell_min, mp_max=abs(s))
    Y = np.zeros(wigner.Ysize, dtype=complex)

    # Loop through each calculation
    for i in range(rotors.shape[0]):
        wigner.sYlm(s, rotors[i], out=Y)
        for j, (ell, m) in enumerate(indices):
            output[i, j] = Y[wigner.Yindex(ell, m)]

    return output.reshape(R.shape[:-1] + indices.shape[:-1])


SWSH_grid = SWSH
