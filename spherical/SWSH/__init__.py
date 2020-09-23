# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import math
import cmath
import numpy as np
import quaternionic
from .. import (Wigner_coefficient as coeff, epsilon, LM_range, jit)


def SWSH(R, s, indices):
    """Spin-weighted spherical harmonic calculation from rotor

    Note that this function is more general than standard Ylm and sYlm functions of
    angles because it uses quaternion rotors instead of angle, and is slightly
    faster as a result.

    This function can be called in either of two ways:
      1) with an array of quaternions, and a single (ell,m) pair, or
      2) with a single quaternion, and an array of (ell,m) values

    Parameters
    ----------
    R : unit quaternion or array of unit quaternions
        Rotor on which to evaluate the SWSH function.
    s : int
        Spin weight of the field to evaluate
    indices : 2-d array of int or pair of ints
        Array of (ell,m) values to evaluate

    Returns
    -------
    ndarray (complex)
        The shape of this array is `indices.shape[0]`, and contains the values of
        the SWSH for the (ell,m) values specified in `indices`.

    """
    indices = np.asarray(indices, dtype=np.int64)
    R = quaternionic.array(R)
    if indices.size > 2 or not isinstance(R, np.ndarray):
        values = np.empty((indices.shape[0],), dtype=np.complex128)
        Rs = R.two_spinor
        _SWSH(Rs.a, Rs.b, s, indices, values)
    else:
        indices = indices.flatten()
        values = np.empty((R.size//4,), dtype=np.complex128)
        _SWSHs(quaternionic.array(R).flattened.ndarray, s, indices[0], indices[1], values)
        values = values.reshape(R.shape[:-1])
    return values


def SWSH_grid(R_grid, s, ell_max):
    """Spin-weighted spherical harmonic calculation from rotors representing a grid

    This function is similar to the `SWSH` function, but assumes that the input is
    an array of rotors, representing various points and/or orientations for which
    to evaluate the SWSH values.  It simply iterates over each of the quaternions,
    and evaluates the separately for each rotor.

    Note that this function takes `ell_max` as its argument, rather than a set of
    indices.  The output of this function is simply the entire range of indices.

    Also note that his function is more general than standard Ylm and sYlm
    functions of angles because it uses quaternion rotors instead of angle, and is
    slightly faster as a result.

    Parameters
    ----------
    R_grid : array of unit quaternions
        Rotors on which to evaluate the SWSH function.  Note that, for speed, these
        are assumed to be normalized.
    s : int
        Spin weight of the field to evaluate
    ell_max : int
        Largest ell value in output arrays.  Note that this should probably be
        `ell_max >= abs(s)`, but the output array will contain values corresponding
        to `ell < abs(s)`.  Those values will be 0.0, but must be present for
        compatibility with `spinsfast`.

    Returns
    -------
    array of complex
        The shape of this array is `R_grid.shape[:-1]`, with an extra dimension of
        length N_lm appended.  That extra dimension corresponds to the various
        (ell,m) values in standard order (e.g., as given by LM_range), starting
        from ell=0 for compatibility with spinsfast.

    """
    indices = LM_range(0, ell_max)
    R_grid = quaternionic.array(R_grid)
    values = np.zeros(R_grid.shape[:-1] + (indices.shape[0],), dtype=np.complex128)
    R = R_grid.flattened
    v = values.reshape((-1, indices.shape[0]))
    for i in range(R.shape[0]):
        Rs_i = R[i].two_spinor
        _SWSH(Rs_i.a, Rs_i.b, s, indices, v[i])
    return values


@jit('void(complex128, complex128, int64, int64[:,:], complex128[:])')
def _SWSH(Ra, Rb, s, indices, values):
    """Compute spin-weighted spherical harmonics from rotor components

    This is the core function that does all the work in the computation, but it is
    strict about its inputs, and does not check them for validity -- though numba
    provides some degree of safety.

    _SWSH(Ra, Rb, s, indices, values)

    Parameters
    ----------
    Ra : complex
        Component `a` of the rotor
    Rb : complex
        Component `b` of the rotor
    s : int
        Spin weight of the field to evaluate
    indices : 2-d array of int
        Array of (ell,m) values to evaluate
    values : 1-d array of complex
        Output array to contain values.  Length must equal first dimension of
        `indices`.  Needed because numba cannot create arrays at the moment.

    Returns
    -------
    void
        The input/output array `values` is modified in place.

    """
    N = indices.shape[0]

    # These constants are the recurring quantities in the computation
    # of the matrix values, so we calculate them here just once

    ra, phia = cmath.polar(Ra)
    rb, phib = cmath.polar(Rb)

    if ra <= epsilon:
        for i in range(N):
            ell, m = indices[i, 0:2]
            if (m != s or abs(m) > ell or abs(s) > ell):
                values[i] = 0.0j
            else:
                if (ell) % 2 == 0:
                    values[i] = math.sqrt((2 * ell + 1) / (4 * np.pi)) * Rb ** (-2 * s)
                else:
                    values[i] = -math.sqrt((2 * ell + 1) / (4 * np.pi)) * Rb ** (-2 * s)

    elif rb <= epsilon:
        for i in range(N):
            ell, m = indices[i, 0:2]
            if (m != -s or abs(m) > ell or abs(s) > ell):
                values[i] = 0.0j
            else:
                if (s) % 2 == 0:
                    values[i] = math.sqrt((2 * ell + 1) / (4 * np.pi)) * Ra ** (-2 * s)
                else:
                    values[i] = -math.sqrt((2 * ell + 1) / (4 * np.pi)) * Ra ** (-2 * s)

    elif ra < rb:
        # We have to have these two versions (both this ra<rb branch,
        # and ra>=rb below) to avoid overflows and underflows
        absRRatioSquared = -ra * ra / (rb * rb)
        for i in range(N):
            ell, m = indices[i, 0:2]
            if (abs(m) > ell or abs(s) > ell):
                values[i] = 0.0j
            else:
                rhoMin = max(0, -m + s)
                # Protect against overflow by decomposing Ra,Rb as
                # abs,angle components and pulling out the factor of
                # absRRatioSquared**rhoMin.  Here, ra might be quite
                # small, in which case ra**(-s+m) could be enormous
                # when the exponent (-s+m) is very negative; adding
                # 2*rhoMin to the exponent ensures that it is always
                # positive, which protects from overflow.  Meanwhile,
                # underflow just goes to zero, which is fine since
                # nothing else should be very large.
                Prefactor = cmath.rect(
                    coeff(ell, -m, -s) * rb ** (2 * ell + s - m - 2 * rhoMin) * ra ** (-s + m + 2 * rhoMin),
                    phib * (-s - m) + phia * (-s + m))
                if (Prefactor == 0.0j):
                    values[i] = 0.0j
                else:
                    if ((ell + rhoMin) % 2 != 0):
                        Prefactor *= -1
                    rhoMax = min(ell - m, ell + s)
                    N1 = ell - m + 1
                    N2 = ell + s + 1
                    M = -s + m
                    Sum = 1.0
                    for rho in range(rhoMax, rhoMin, -1):
                        Sum *= absRRatioSquared * ((N1 - rho) * (N2 - rho)) / (rho * (M + rho))
                        Sum += 1
                    # Sum = 0.0
                    # for rho in range(rhoMax, rhoMin-1, -1):
                    # Sum = (  binomial_coefficient(ell-m,rho) * binomial_coefficient(ell+m, ell-rho+s)
                    # + Sum * absRRatioSquared )
                    values[i] = math.sqrt((2 * ell + 1) / (4 * np.pi)) * Prefactor * Sum

    else:  # ra >= rb
        # We have to have these two versions (both this ra>=rb branch,
        # and ra<rb above) to avoid overflows and underflows
        absRRatioSquared = -rb * rb / (ra * ra)
        for i in range(N):
            ell, m = indices[i, 0:2]
            if (abs(m) > ell or abs(s) > ell):
                values[i] = 0.0j
            else:
                rhoMin = max(0, m + s)
                # Protect against overflow by decomposing Ra,Rb as
                # abs,angle components and pulling out the factor of
                # absRRatioSquared**rhoMin.  Here, rb might be quite
                # small, in which case rb**(-s-m) could be enormous
                # when the exponent (-s-m) is very negative; adding
                # 2*rhoMin to the exponent ensures that it is always
                # positive, which protects from overflow.  Meanwhile,
                # underflow just goes to zero, which is fine since
                # nothing else should be very large.
                Prefactor = cmath.rect(
                    coeff(ell, m, -s) * ra ** (2 * ell + s + m - 2 * rhoMin) * rb ** (-s - m + 2 * rhoMin),
                    phia * (-s + m) + phib * (-s - m))
                if (Prefactor == 0.0j):
                    values[i] = 0.0j
                else:
                    if ((rhoMin + s) % 2 != 0):
                        Prefactor *= -1
                    rhoMax = min(ell + m, ell + s)
                    N1 = ell + m + 1
                    N2 = ell + s + 1
                    M = -s - m
                    Sum = 1.0
                    for rho in range(rhoMax, rhoMin, -1):
                        Sum *= absRRatioSquared * ((N1 - rho) * (N2 - rho)) / (rho * (M + rho))
                        Sum += 1
                    # Sum = 0.0
                    # for rho in range(rhoMax, rhoMin-1, -1):
                    # Sum = (  binomial_coefficient(ell+m,rho) * binomial_coefficient(ell-m, ell-rho+s)
                    # + Sum * absRRatioSquared )
                    values[i] = math.sqrt((2 * ell + 1) / (4 * np.pi)) * Prefactor * Sum


@jit('void(float64[:,:], int64, int64, int64, complex128[:])')
def _SWSHs(Rs, s, ell, m, values):
    """Compute spin-weighted spherical harmonics from rotor components

    This is the core function that does all the work in the computation, but it is
    strict about its inputs, and does not check them for validity -- though numba
    provides some degree of safety.

    _SWSHs(Rs, s, ell, m, values)

    Parameters
    ----------
    Rs : 2-d array of float
        Components of the rotors, with the 0 index iterating over rotor, and the 1
        index iterating over component.
    s : int
        Spin weight of the field to evaluate
    ell : int
    m : int
        Values of (ell,m) to output
    values : 1-d array of complex
        Output array to contain values.  Length must equal 0 dimension of `Rs`.
        Needed because numba cannot create arrays at the moment.

    Returns
    -------
    void
        The input/output array `values` is modified in place.

    """
    N = Rs.shape[0]

    if (abs(m) > ell or abs(s) > ell):
        for i in range(N):
            values[i] = 0.0j

    else:

        constant = math.sqrt((2 * ell + 1) / (4 * np.pi))
        ell_even = (ell % 2 == 0)
        s_even = (s % 2 == 0)
        rhoMin_a = max(0, m + s)
        rhoMax_a = min(ell + m, ell + s)
        coefficient_a = coeff(ell, m, -s)
        if ((rhoMin_a + s) % 2 != 0):
            coefficient_a *= -1
        N1_a = ell + m + 1
        N2_a = ell + s + 1
        M_a = -s - m
        rhoMin_b = max(0, -m + s)
        rhoMax_b = min(ell - m, ell + s)
        coefficient_b = coeff(ell, -m, -s)
        if ((ell + rhoMin_b) % 2 != 0):
            coefficient_b *= -1
        N1_b = ell - m + 1
        N2_b = ell + s + 1
        M_b = -s + m

        for i in range(N):
            Ra = complex(Rs[i, 0], Rs[i, 3])
            ra, phia = cmath.polar(Ra)

            Rb = complex(Rs[i, 2], Rs[i, 1])
            rb, phib = cmath.polar(Rb)

            if ra <= epsilon:
                if m != s:
                    values[i] = 0.0j
                elif ell_even:
                    values[i] = constant * Rb ** (-2 * s)
                else:
                    values[i] = -constant * Rb ** (-2 * s)

            elif rb <= epsilon:
                if m != -s:
                    values[i] = 0.0j
                elif s_even:
                    values[i] = constant * Ra ** (-2 * s)
                else:
                    values[i] = -constant * Ra ** (-2 * s)

            elif ra < rb:
                if (coefficient_b == 0.0j):
                    values[i] = 0.0j
                else:
                    absRRatioSquared = -ra * ra / (rb * rb)
                    Prefactor = cmath.rect(
                        coefficient_b * rb ** (2 * ell + s - m - 2 * rhoMin_b) * ra ** (-s + m + 2 * rhoMin_b),
                        phib * (-s - m) + phia * (-s + m))
                    Sum = 1.0
                    for rho in range(rhoMax_b, rhoMin_b, -1):
                        Sum *= absRRatioSquared * ((N1_b - rho) * (N2_b - rho)) / (rho * (M_b + rho))
                        Sum += 1
                    values[i] = constant * Prefactor * Sum

            else:  # ra >= rb
                if (coefficient_a == 0.0j):
                    values[i] = 0.0j
                else:
                    absRRatioSquared = -rb * rb / (ra * ra)
                    Prefactor = cmath.rect(
                        coefficient_a * ra ** (2 * ell + s + m - 2 * rhoMin_a) * rb ** (-s - m + 2 * rhoMin_a),
                        phia * (-s + m) + phib * (-s - m))
                    Sum = 1.0
                    for rho in range(rhoMax_a, rhoMin_a, -1):
                        Sum *= absRRatioSquared * ((N1_a - rho) * (N2_a - rho)) / (rho * (M_a + rho))
                        Sum += 1
                    values[i] = constant * Prefactor * Sum
