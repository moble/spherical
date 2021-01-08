# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import numpy as np
from .. import jit

def complex_powers(z, M, zpowers=None):
    """Compute integer powers of z=exp(iθ) recursively

    Parameters
    ----------
    z : complex
        Complex number to take integer powers of.  Note that this is assumed to be
        normalized, but can be an array of such numbers.
    M : int
        Highest power to compute
    zpowers : array of complex or None (defaults to None)
        If present, the output will be written into this array, which must have
        shape z.shape+(M+1,).  Pre-allocating this argument can reduce the time
        spent in this function by up to 60%.

    Returns
    -------
    zpowers : array of complex
        Powers z⁰, z¹, ..., zᴹ.  The last dimension has size M+1, any preceding
        dimensions are the same as the input `z` array.

    Notes
    -----
    This algorithm is mostly due to Stoer and Bulirsch in "Introduction to
    Numerical Analysis" (page 24) — with a little help from de Moivre's formula,
    which is essentially exp(iθ)ⁿ = exp(inθ), as well as my own alterations to deal
    with different behaviors in different quadrants.

    There isn't usually a huge advantage to using this specialized function.  If
    you just need a particular power, it will generally be far more efficient and
    just as accurate to compute either exp(iθ)ⁿ or exp(inθ) explicitly.  However,
    if you need all powers from 0 to M, this function is about 10 or 5 times faster
    than those options, respectively.  Like those options, this function is
    numerically stable, in the sense that its errors are usually smaller than the
    error from machine-precision errors in the input argument — or at worst about
    30% larger, around π/2.

    """
    z = np.asarray(z, dtype=complex)
    zravel = z.ravel()
    zpowers = zpowers if zpowers is not None else np.zeros((z.size, M+1), dtype=np.complex128)
    _complex_powers(zravel, M, zpowers)
    return zpowers.reshape(z.shape+(M+1,))


@jit
def _complex_powers(zravel, M, zpowers):
    """Helper function for complex_powers(z, M)"""
    for i in range(zravel.size):
        zpowers[i, 0] = 1.0 + 0.0j
        if M > 0:
            z = zravel[i]
            θ = 1
            while z.real<0 or z.imag<0:
                θ *= 1j
                z /= 1j
            zpowers[i, 1] = z
            clock = θ
            dc = -2 * np.sqrt(z).imag ** 2
            t = 2 * dc
            dz = dc * (1 + 2 * zpowers[i, 1]) + 1j * np.sqrt(-dc * (2 + dc))
            for m in range(2, M+1):
                zpowers[i, m] = zpowers[i, m-1] + dz
                dz += t * zpowers[i, m]
                zpowers[i, m-1] *= clock
                clock *= θ
            zpowers[i, M] *= clock
