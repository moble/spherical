# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

"""Module for computing Wigner D functions.

This module contains various functions such as Wigner_D_element and
Wigner_D_matrices (as well as faster variants, designated by underscores,
which assume correct inputs), which calculate the elements and complete
matrices (respectively) of Wigner's D symbols.

"""

import numbers
import cmath
import numpy as np
import numba as nb
from .. import (jit, _Wigner_coefficient as _coeff,
                Wigner_coefficient as coeff,
                epsilon, error_on_bad_indices, LMpM_total_size,
                ell_max as sf_ell_max)


@jit('b1(i8,i8,i8)')
def _check_valid_indices(twoell, twomp, twom):
    if (twoell > 2*sf_ell_max or abs(twomp) > twoell or abs(twom) > twoell):
        return False
    return True


def Wigner_D_element(*args):
    """Return elements of the Wigner D matrices

    The conventions used for this function are discussed more fully on
    <http://moble.github.io/spherical/>.

    Input arguments
    ---------------
    The input can be in any of the following forms:

    Wigner_D_element(R, ell, mp, m)
    Wigner_D_element(Rs, ell, mp, m)
    Wigner_D_element(R, indices)
    Wigner_D_element(Ra, Rb, ell, mp, m)
    Wigner_D_element(Ra, Rb, indices)
    Wigner_D_element(alpha, beta, gamma, ell, mp, m)
    Wigner_D_element(alpha, beta, gamma, indices)

    Where
      * R is a unit quaternion (no checking of norm is done)
      * Rs is an array of unit quaternions (no checking of norm is done)
      * Ra and Rb are the complex parts of a unit quaternion
      * alpha, beta, gamma are the Euler angles [shudder...]
      * ell, mp, m are the integer or half-integer indices of the
        D matrix element
      * indices is an array of [ell,mp,m] indices as above, or simply
        a list of ell modes, in which case all valid [mp,m] values
        will be returned

    Note that, by default, a ValueError will be raised if the input
    (ell, mp, m) values are not valid.  (For example, |m|>ell.)
    If instead, you would simply like a return value of 0.0, after
    importing this module as sf, simply evaluate

        >>> sf.error_on_bad_indices = False

    Return value
    ------------
    One complex number is returned for each component requested.  If a
    single quaternion and the (ell,mp,m) arguments were given explicitly,
    this means that a single complex scalar is returned.  If more than
    one component was requested, a one-dimensional numpy array of complex
    scalars is returned, in the same order as the input.

    """
    import quaternionic
    if len(args) == 0:
        raise ValueError("No input")
    # Find the rotation from the args
    if isinstance(args[0], np.ndarray) and args[0].shape[-1] == 4:
        if len(args) not in [2, 4]:
            raise ValueError(f"Don't understand {len(args)} arguments with first argument a quaternion")
        elif len(args) == 2:
            R = quaternionic.array(args[0])
            Rs = R.two_spinor
            Ra = Rs.a
            Rb = Rs.b
            mode_offset = 1
        else:
            R = quaternionic.array(np.atleast_2d(args[0]))
            elements = np.atleast_1d(np.empty(args[0].shape[:-1], dtype=np.complex128))
            _Wigner_D_elements(R.ndarray, args[1], args[2], args[3], elements)
            if elements.size == 1 and args[0].ndim == 1:
                return elements[0]
            return elements
    elif all(isinstance(args[i], numbers.Number) for i in range(3)):
        # UUUUGGGGLLLLYYYY.  The rotation is input as Euler angles
        R = quaternionic.array.from_euler_angles(args[0], args[1], args[2])
        Rs = R.two_spinor
        Ra = Rs.a
        Rb = Rs.b
        mode_offset = 3
    elif isinstance(args[0], numbers.Complex) and isinstance(args[1], numbers.Complex):
        # The rotation is input as the two parts of a single quaternion
        Ra = args[0]
        Rb = args[1]
        mode_offset = 2
    else:
        raise ValueError("Can't understand input rotation")

    # Find the indices
    return_scalar = False
    if (len(args) - mode_offset == 3):
        # Assume these are the (ell, mp, m) indices
        ell, mp, m = args[mode_offset:]
        indices = np.array([[round(2*ell), round(2*mp), round(2*m)], ], dtype=np.int64)
        if (error_on_bad_indices and not _check_valid_indices(*(indices[0]))):
            raise ValueError(
                "(ell,mp,m)=({0},{1},{2})".format(ell, mp, m)
                + " is not a valid set of indices for Wigner's D matrix")
        return_scalar = True
    elif (len(args) - mode_offset == 1):
        indices = np.round(2*np.asarray(args[mode_offset])).astype(np.int64)
        if (indices.ndim == 0 and indices.size == 1):
            # This was just a single ell value
            twoell = indices[0]  # already multiplied by 2
            indices = np.array([[twoell, twomp, twom]
                                for twomp in range(-twoell, twoell + 1, 2)
                                for twom in range(-twoell, twoell + 1, 2)])
            if (twoell == 0):
                return_scalar = True
        elif (indices.ndim == 1 and indices.size > 0):
            # This a list of ell values
            indices = np.array(
                [[twoell, twomp, twom] for twoell in indices
                 for twomp in range(-twoell, twoell + 1, 2)
                 for twom in range(-twoell, twoell + 1, 2)])
        elif (indices.ndim == 2):
            # This is an array of [ell,mp,m] values
            if (error_on_bad_indices):
                for twoell, twomp, twom in indices:
                    if not _check_valid_indices(twoell, twomp, twom):
                        raise ValueError(
                            "(ell,mp,m)=({0},{1},{2}) is not a valid set".format(twoell/2, twomp/2, twom/2)
                            + " of indices for Wigner's D matrix")
        else:
            raise ValueError("Can't understand input indices")
    else:
        raise ValueError("Can't understand input indices")

    elements = np.empty((len(indices),), dtype=np.complex128)
    _Wigner_D_element(Ra, Rb, indices, elements)

    if (return_scalar):
        return elements[0]
    return elements

@jit('void(complex128, complex128, int64[:,:], complex128[:])')
def _Wigner_D_element(Ra, Rb, indices, elements):
    """Main work function for computing Wigner D matrix elements

    This is the core function that does all the work in the
    computation, but it is strict about its input, and does not check
    them for validity.  Note that the indices should be integers,
    representing the (2*ell, 2*mp, 2*m) values, meaning that
    (ell, mp, m) can be half-integer.

    Input arguments
    ---------------
    _Wigner_D_element(Ra, Rb, indices, elements)

      * Ra, Rb are the complex components of the rotor
      * indices is an array of integer sets [2*ell, 2*mp, 2*m]
      * elements is an array of complex with length equal to the first
        dimension of indices

    The `elements` variable is needed because numba cannot create
    arrays at the moment, but this is modified in place.

    """
    N = indices.shape[0]

    # These constants are the recurring quantities in the computation
    # of the matrix elements, so we calculate them here just once
    ra, phia = cmath.polar(Ra)
    rb, phib = cmath.polar(Rb)

    if (ra <= epsilon):
        for i in range(N):
            twoell, twomp, twom = indices[i, 0:3]
            if (twomp != -twom or abs(twomp) > twoell or abs(twom) > twoell):
                elements[i] = 0.0j
            else:
                if (twoell - twom) % 4 == 0:
                    elements[i] = Rb ** twom
                else:
                    elements[i] = -Rb ** twom

    elif (rb <= epsilon):
        for i in range(N):
            twoell, twomp, twom = indices[i, 0:3]
            if (twomp != twom or abs(twomp) > twoell or abs(twom) > twoell):
                elements[i] = 0.0j
            else:
                elements[i] = Ra ** twom

    elif (ra < rb):
        # We have to have these two versions (both this ra<rb branch,
        # and ra>=rb below) to avoid overflows and underflows
        absRRatioSquared = -ra * ra / (rb * rb)
        for i in range(N):
            twoell, twomp, twom = indices[i, 0:3]
            if (abs(twomp) > twoell or abs(twom) > twoell):
                elements[i] = 0.0j
            else:
                tworhoMin = max(0, -twomp - twom)
                # Protect against overflow by decomposing Ra,Rb as
                # abs,angle components and pulling out the factor of
                # absRRatioSquared**rhoMin.  Here, ra might be quite
                # small, in which case ra**(m+mp) could be enormous
                # when the exponent (m+mp) is very negative; adding
                # 2*rhoMin to the exponent ensures that it is always
                # positive, which protects from overflow.  Meanwhile,
                # underflow just goes to zero, which is fine since
                # nothing else should be very large.
                Prefactor = cmath.rect(
                    _coeff(twoell, -twomp, twom)
                        * rb ** (twoell - (twom + twomp)/2 - tworhoMin)
                        * ra ** ((twom + twomp)/2 + tworhoMin),
                    phib * (twom - twomp)/2 + phia * (twom + twomp)/2)
                if (Prefactor == 0.0j):
                    elements[i] = 0.0j
                else:
                    if ((twoell - twom - tworhoMin) % 4 != 0):
                        Prefactor *= -1
                    tworhoMax = min(twoell - twomp, twoell - twom)
                    twoN1 = twoell - twomp + 2
                    twoN2 = twoell - twom + 2
                    twoM = twom + twomp
                    Sum = 1.0
                    for tworho in range(tworhoMax, tworhoMin, -2):
                        Sum *= absRRatioSquared * ((twoN1 - tworho) * (twoN2 - tworho)) / (tworho * (twoM + tworho))
                        Sum += 1
                    elements[i] = Prefactor * Sum

    else:  # ra >= rb
        # We have to have these two versions (both this ra>=rb branch,
        # and ra<rb above) to avoid overflows and underflows
        absRRatioSquared = -rb * rb / (ra * ra)
        for i in range(N):
            twoell, twomp, twom = indices[i, 0:3]
            if (abs(twomp) > twoell or abs(twom) > twoell):
                elements[i] = 0.0j
            else:
                tworhoMin = max(0, twomp - twom)
                # Protect against overflow by decomposing Ra,Rb as
                # abs,angle components and pulling out the factor of
                # absRRatioSquared**rhoMin.  Here, rb might be quite
                # small, in which case rb**(m-mp) could be enormous
                # when the exponent (m-mp) is very negative; adding
                # 2*rhoMin to the exponent ensures that it is always
                # positive, which protects from overflow.  Meanwhile,
                # underflow just goes to zero, which is fine since
                # nothing else should be very large.
                Prefactor = cmath.rect(
                    _coeff(twoell, twomp, twom)
                        * ra ** (twoell - twom/2 + twomp/2 - tworhoMin)
                        * rb ** (twom/2 - twomp/2 + tworhoMin),
                    phia * (twom + twomp)/2 + phib * (twom - twomp)/2)
                if (Prefactor == 0.0j):
                    elements[i] = 0.0j
                else:
                    if (tworhoMin % 4 != 0):
                        Prefactor *= -1
                    tworhoMax = min(twoell + twomp, twoell - twom)
                    twoN1 = twoell + twomp + 2
                    twoN2 = twoell - twom + 2
                    twoM = twom - twomp
                    Sum = 1.0
                    for tworho in range(tworhoMax, tworhoMin, -2):
                        Sum *= absRRatioSquared * ((twoN1 - tworho) * (twoN2 - tworho)) / (tworho * (twoM + tworho))
                        Sum += 1
                    elements[i] = Prefactor * Sum


@jit('int64(int64, int64, int64)')
def _linear_matrix_index(ell, mp, m):
    """Index of array corresponding to matrix element

    This gives the index based at the first element of the matrix, so
    if the array is actually a series of matrices (linearized), then
    that initial index for this matrix must be added.

    This assumes that the input array corresponds to

    [[ell,mp,m] for ell in range(ell_min,ell_max+1)
                for mp in range(-ell,ell+1)
                for m in range(-ell,ell+1)]

    """
    return (ell + m) + (ell + mp) * (2 * ell + 1)


@jit('int64(int64, int64)')
def _linear_matrix_diagonal_index(ell, mpm):
    """Index of array corresponding to matrix diagonal element

    This gives the index based at the first element of the matrix, so
    if the array is actually a series of matrices (linearized), then
    that initial index for this matrix must be added.

    This assumes that the input array corresponds to

    [[ell,mp,m] for ell in range(ell_min,ell_max+1)
                for mp in range(-ell,ell+1)
                for m in range(-ell,ell+1)]

    """
    return (ell + mpm) * (2 * ell + 2)


@jit('int64(int64, int64)')
def _linear_matrix_offset(ell, ell_min):
    """Index of initial element in linear array of D matrices

    This gives the index based at the first element of the matrix with
    weight `ell`.  This is the quantity to be added to the result of
    `_linear_matrix_index`.

    This assumes that the input array corresponds to

    [[ell,mp,m] for ell in range(ell_min,ell_max+1)
                for mp in range(-ell,ell+1)
                for m in range(-ell,ell+1)]

    This result can be calculated from sympy as

    from sympy import symbols, summation
    ell_min,ell = symbols('ell_min,ell', integer=True)
    summation((2*ell + 1)**2, (ell, ell_min, ell-1))

    """
    return ( (4 * ell ** 2 - 1) * ell - (4 * ell_min ** 2 - 1) * ell_min ) // 3


@jit('int64(int64,int64)')
def _total_size_D_matrices(ell_min, ell_max):
    return ( ((4 * ell_max + 12) * ell_max + 11) * ell_max + 3 - (4 * ell_min ** 2 - 1) * ell_min ) // 3


@jit('complex128(complex128)')
def conjugate(z):
    return z.conjugate()


def Wigner_D_matrices(R, ell_min, ell_max):
    """Return linear array of Wigner D matrix elements for range of integer ell values

    Note that this only accepts and outputs integer values of ell; for half-integer values,
    use `Wigner_D_element` with an explicit array for the `indices` argument.

    Parameters
    ----------
    R : quaternion
        The rotor for the D matrices
    ell_min : int
        Lowest ell value included in array
    ell_max : int
        Highest ell value included in array

    Returns
    -------
    numpy.ndarray
        Linear array of all matrix elements

        The array is in the standard order, essentially of the form

            [D(ell,mp,m) for ell in range(ell_min, ell_max+1)
                         for mp in range(-ell,ell+1)
                         for m in range(-ell,ell+1)]

    See Also
    --------
    LMpM_total_size: Calculate total size of this array
    LMpM_index: Find index in this array of element (ell,mp,m)
    LMpM_range: Construct list of corresponding (ell,mp,m) values

    """
    if abs(round(ell_max)-ell_max) > 1e-10 or abs(round(ell_min)-ell_min) > 1e-10:
        error = ("Wigner_D_matrices is only implemented for integer values of ell.\n"
                 + "Input values ell_min={0} and ell_max={1} are not valid.\n".format(ell_min, ell_max)
                 + "Try `Wigner_D_element` with an explicit array of `indices` for half-integers.")
        raise ValueError(error)
    matrices = np.empty((LMpM_total_size(ell_min, ell_max),), dtype=np.complex128)
    _Wigner_D_matrices(R.a, R.b, ell_min, ell_max, matrices)
    return matrices


@jit('void(complex128, complex128, int64, int64, complex128[:])',
      locals={'Prefactor1': nb.complex128, 'Prefactor2': nb.complex128})
def _Wigner_D_matrices(Ra, Rb, ell_min, ell_max, matrices):
    """Main work function for `Wigner_D_matrices`

    This is the core function that does all the work in the
    computation, but it is strict about its input, and does not check
    them for validity.

    Input arguments
    ---------------
    _Wigner_D_matrices(Ra, Rb, ell_min, ell_max, elements)

      * Ra, Rb are the complex components of the rotor
      * ell_min, ell_max are the limits of the matrices
      * matrix is a one-dimensional array of complex numbers to be
        filled with the elements of the matrices; the correct shape is
        assumed

    The `matrices` variable is needed because numba cannot create
    arrays at the moment, but this is modified in place, so after
    calling this function, the input array will contain the correct
    values.

    """

    # These constants are the recurring quantities in the computation
    # of the matrix elements, so we calculate them here just once
    ra, phia = cmath.polar(Ra)
    rb, phib = cmath.polar(Rb)

    if (ra <= epsilon):
        for ell in range(ell_min, ell_max + 1):
            i_ell = _linear_matrix_offset(ell, ell_min)
            for i in range((2 * ell + 1) ** 2):
                matrices[i_ell + i] = 0j
            for mpmm in range(-ell, ell + 1):
                i_mpmm = _linear_matrix_index(ell, mpmm, -mpmm)
                if (ell + mpmm) % 2 == 0:
                    matrices[i_ell + i_mpmm] = Rb ** (-2 * mpmm)
                else:
                    matrices[i_ell + i_mpmm] = -(Rb ** (-2 * mpmm))

    elif (rb <= epsilon):
        for ell in range(ell_min, ell_max + 1):
            i_ell = _linear_matrix_offset(ell, ell_min)
            for i in range((2 * ell + 1) ** 2):
                matrices[i_ell + i] = 0j
            for mpm in range(-ell, ell + 1):
                i_mpm = _linear_matrix_diagonal_index(ell, mpm)
                matrices[i_ell + i_mpm] = Ra ** (2 * mpm)

    elif (ra < rb):
        # We have to have these two versions (both this ra<rb branch,
        # and ra>=rb below) to avoid overflows and underflows
        absRRatioSquared = -ra * ra / (rb * rb)
        for ell in range(ell_min, ell_max + 1):
            i_ell = _linear_matrix_offset(ell, ell_min)
            for mp in range(-ell, 1):
                for m in range(mp, -mp + 1):
                    i_mpm = _linear_matrix_index(ell, mp, m)
                    rhoMin = max(0, -mp - m)
                    # Protect against overflow by decomposing Ra,Rb as
                    # abs,angle components and pulling out the factor of
                    # absRRatioSquared**rhoMin.  Here, ra might be quite
                    # small, in which case ra**(m+mp) could be enormous
                    # when the exponent (m+mp) is very negative; adding
                    # 2*rhoMin to the exponent ensures that it is always
                    # positive, which protects from overflow.  Meanwhile,
                    # underflow just goes to zero, which is fine since
                    # nothing else should be very large.
                    d = coeff(ell, -mp, m) * rb ** (2 * ell - m - mp - 2 * rhoMin) * ra ** (m + mp + 2 * rhoMin)
                    if (d == 0.0j):
                        matrices[i_ell + i_mpm] = 0.0j
                        if (abs(m) != abs(mp)):
                            # D_{-mp,-m}(R) = (-1)^{mp+m} \bar{D}_{mp,m}(R)
                            matrices[i_ell + _linear_matrix_index(ell, -mp, -m)] = 0.0j
                            # D_{m,mp}(R) = \bar{D}_{mp,m}(\bar{R})
                            matrices[i_ell + _linear_matrix_index(ell, m, mp)] = 0.0j
                            # D_{-m,-mp}(R) = (-1)^{mp+m} D_{mp,m}(\bar{R})
                            matrices[i_ell + _linear_matrix_index(ell, -m, -mp)] = 0.0j
                        elif (m != 0):
                            # D_{-mp,-m}(R) = (-1)^{mp+m} \bar{D}_{mp,m}(R)
                            matrices[i_ell + _linear_matrix_index(ell, -mp, -m)] = 0.0j
                    else:
                        if ((ell + m + rhoMin) % 2 != 0):
                            d *= -1
                        Prefactor1 = cmath.rect(d, phib * (m - mp) + phia * (m + mp))
                        Prefactor2 = cmath.rect(d, (phib + np.pi) * (m - mp) - phia * (m + mp))
                        rhoMax = min(ell - mp, ell - m)
                        N1 = ell - mp + 1
                        N2 = ell - m + 1
                        M = m + mp
                        Sum = 1.0
                        for rho in range(rhoMax, rhoMin, -1):
                            Sum *= absRRatioSquared * ((N1 - rho) * (N2 - rho)) / (rho * (M + rho))
                            Sum += 1
                        # Sum *= absRRatioSquared**rhoMin
                        matrices[i_ell + i_mpm] = Prefactor1 * Sum
                        if (abs(m) != abs(mp)):
                            if ((m + mp) % 2 == 0):
                                # D_{-mp,-m}(R) = (-1)^{mp+m} \bar{D}_{mp,m}(R)
                                matrices[i_ell + _linear_matrix_index(ell, -mp, -m)] = Prefactor1.conjugate() * Sum
                                # D_{-m,-mp}(R) = (-1)^{mp+m} D_{mp,m}(\bar{R})
                                matrices[i_ell + _linear_matrix_index(ell, -m, -mp)] = Prefactor2 * Sum
                            else:
                                # D_{-mp,-m}(R) = (-1)^{mp+m} \bar{D}_{mp,m}(R)
                                matrices[i_ell + _linear_matrix_index(ell, -mp, -m)] = -Prefactor1.conjugate() * Sum
                                # D_{-m,-mp}(R) = (-1)^{mp+m} D_{mp,m}(\bar{R})
                                matrices[i_ell + _linear_matrix_index(ell, -m, -mp)] = -Prefactor2 * Sum
                            # D_{m,mp}(R) = \bar{D}_{mp,m}(\bar{R})
                            matrices[i_ell + _linear_matrix_index(ell, m, mp)] = Prefactor2.conjugate() * Sum
                        elif (m != 0):
                            if ((m + mp) % 2 == 0):
                                # D_{-mp,-m}(R) = (-1)^{mp+m} \bar{D}_{mp,m}(R)
                                matrices[i_ell + _linear_matrix_index(ell, -mp, -m)] = Prefactor1.conjugate() * Sum
                            else:
                                # D_{-mp,-m}(R) = (-1)^{mp+m} \bar{D}_{mp,m}(R)
                                matrices[i_ell + _linear_matrix_index(ell, -mp, -m)] = -Prefactor1.conjugate() * Sum

    else:  # ra >= rb
        # We have to have these two versions (both this ra>=rb branch,
        # and ra<rb above) to avoid overflows and underflows
        absRRatioSquared = -rb * rb / (ra * ra)
        for ell in range(ell_min, ell_max + 1):
            i_ell = _linear_matrix_offset(ell, ell_min)
            for mp in range(-ell, 1):
                for m in range(mp, -mp + 1):
                    i_mpm = _linear_matrix_index(ell, mp, m)
                    rhoMin = max(0, mp - m)
                    # Protect against overflow by decomposing Ra,Rb as
                    # abs,angle components and pulling out the factor of
                    # absRRatioSquared**rhoMin.  Here, rb might be quite
                    # small, in which case rb**(m-mp) could be enormous
                    # when the exponent (m-mp) is very negative; adding
                    # 2*rhoMin to the exponent ensures that it is always
                    # positive, which protects from overflow.  Meanwhile,
                    # underflow just goes to zero, which is fine since
                    # nothing else should be very large.
                    d = coeff(ell, mp, m) * ra ** (2 * ell - m + mp - 2 * rhoMin) * rb ** (m - mp + 2 * rhoMin)
                    if (d == 0.0j):
                        matrices[i_ell + i_mpm] = 0.0j
                        if (abs(m) != abs(mp)):
                            # D_{-mp,-m}(R) = (-1)^{mp+m} \bar{D}_{mp,m}(R)
                            matrices[i_ell + _linear_matrix_index(ell, -mp, -m)] = 0.0j
                            # D_{m,mp}(R) = \bar{D}_{mp,m}(\bar{R})
                            matrices[i_ell + _linear_matrix_index(ell, m, mp)] = 0.0j
                            # D_{-m,-mp}(R) = (-1)^{mp+m} D_{mp,m}(\bar{R})
                            matrices[i_ell + _linear_matrix_index(ell, -m, -mp)] = 0.0j
                        elif (m != 0):
                            # D_{-mp,-m}(R) = (-1)^{mp+m} \bar{D}_{mp,m}(R)
                            matrices[i_ell + _linear_matrix_index(ell, -mp, -m)] = 0.0j
                    else:
                        if (rhoMin % 2 != 0):
                            d *= -1
                        Prefactor1 = cmath.rect(d, phia * (m + mp) + phib * (m - mp))
                        Prefactor2 = cmath.rect(d, -phia * (m + mp) + (phib + np.pi) * (m - mp))
                        rhoMax = min(ell + mp, ell - m)
                        N1 = ell + mp + 1
                        N2 = ell - m + 1
                        M = m - mp
                        Sum = 1.0
                        for rho in range(rhoMax, rhoMin, -1):
                            Sum *= absRRatioSquared * ((N1 - rho) * (N2 - rho)) / (rho * (M + rho))
                            Sum += 1
                        # Sum *= absRRatioSquared**rhoMin
                        matrices[i_ell + i_mpm] = Prefactor1 * Sum
                        if (abs(m) != abs(mp)):
                            if (m + mp) % 2 == 0:
                                # D_{-mp,-m}(R) = (-1)^{mp+m} \bar{D}_{mp,m}(R)
                                matrices[i_ell + _linear_matrix_index(ell, -mp, -m)] = Prefactor1.conjugate() * Sum
                                # D_{-m,-mp}(R) = (-1)^{mp+m} D_{mp,m}(\bar{R})
                                matrices[i_ell + _linear_matrix_index(ell, -m, -mp)] = Prefactor2 * Sum
                            else:
                                # D_{-mp,-m}(R) = (-1)^{mp+m} \bar{D}_{mp,m}(R)
                                matrices[i_ell + _linear_matrix_index(ell, -mp, -m)] = -Prefactor1.conjugate() * Sum
                                # D_{-m,-mp}(R) = (-1)^{mp+m} D_{mp,m}(\bar{R})
                                matrices[i_ell + _linear_matrix_index(ell, -m, -mp)] = -Prefactor2 * Sum
                            # D_{m,mp}(R) = \bar{D}_{mp,m}(\bar{R})
                            matrices[i_ell + _linear_matrix_index(ell, m, mp)] = Prefactor2.conjugate() * Sum
                        elif (m != 0):
                            if (m + mp) % 2 == 0:
                                # D_{-mp,-m}(R) = (-1)^{mp+m} \bar{D}_{mp,m}(R)
                                matrices[i_ell + _linear_matrix_index(ell, -mp, -m)] = Prefactor1.conjugate() * Sum
                            else:
                                # D_{-mp,-m}(R) = (-1)^{mp+m} \bar{D}_{mp,m}(R)
                                matrices[i_ell + _linear_matrix_index(ell, -mp, -m)] = -Prefactor1.conjugate() * Sum


@jit('void(float64[:,:], int64, int64, int64, complex128[:])',
      locals={'Prefactor1': nb.complex128, 'Prefactor2': nb.complex128})
def _Wigner_D_elements(Rs, ell, mp, m, values):
    """Main work function for computing Wigner D matrix elements

    This is the core function that does all the work in the
    computation, but it is strict about its input, and does not check
    them for validity.

    Input arguments
    ---------------
    _Wigner_D_matrices(Rs, ell, m, mp, elements)

    The `matrices` variable is needed because numba cannot create
    arrays at the moment, but this is modified in place, so after
    calling this function, the input array will contain the correct
    values.

    """
    N = Rs.shape[0]

    if (abs(m) > ell or abs(mp) > ell):
        for i in range(N):
            values[i] = 0.0j

    else:

        rhoMin_a = max(0, mp - m)
        rhoMax_a = min(ell + mp, ell - m)
        coefficient_a = coeff(ell, mp, m)
        if (rhoMin_a % 2 != 0):
            coefficient_a *= -1
        N1_a = ell + mp + 1
        N2_a = ell - m + 1
        M_a = m - mp
        rhoMin_b = max(0, -m - mp)
        rhoMax_b = min(ell - mp, ell - m)
        coefficient_b = coeff(ell, -mp, m)
        if ((ell + m + rhoMin_b) % 2 != 0):
            coefficient_b *= -1
        N1_b = ell - mp + 1
        N2_b = ell - m + 1
        M_b = m + mp

        for i in range(N):

            Ra = complex(Rs[i, 0], Rs[i, 3])
            ra, phia = cmath.polar(Ra)

            Rb = complex(Rs[i, 2], Rs[i, 1])
            rb, phib = cmath.polar(Rb)

            if ra <= epsilon:
                if mp != -m:
                    values[i] = 0.0j
                elif (ell + mp) % 2 == 0:
                    values[i] = Rb ** (-2 * mp)
                else:
                    values[i] = -(Rb ** (-2 * mp))

            elif rb <= epsilon:
                if mp != m:
                    values[i] = 0.0j
                else:
                    values[i] = Ra ** (2 * mp)

            elif ra < rb:
                absRRatioSquared = -ra * ra / (rb * rb)
                d = coefficient_b * rb ** (2 * ell - m - mp - 2 * rhoMin_b) * ra ** (m + mp + 2 * rhoMin_b)
                if d == 0.0j:
                    values[i] = 0.0j
                else:
                    Prefactor1 = cmath.rect(d, phib * (m - mp) + phia * (m + mp))
                    Prefactor2 = cmath.rect(d, (phib + np.pi) * (m - mp) - phia * (m + mp))
                    Sum = 1.0
                    for rho in range(rhoMax_b, rhoMin_b, -1):
                        Sum *= absRRatioSquared * ((N1_b - rho) * (N2_b - rho)) / (rho * (M_b + rho))
                        Sum += 1
                    values[i] = Prefactor1 * Sum

            else:  # ra >= rb
                absRRatioSquared = -rb * rb / (ra * ra)
                d = coefficient_a * ra ** (2 * ell - m + mp - 2 * rhoMin_a) * rb ** (m - mp + 2 * rhoMin_a)
                if d == 0.0j:
                    values[i] = 0.0j
                else:
                    Prefactor1 = cmath.rect(d, phia * (m + mp) + phib * (m - mp))
                    Prefactor2 = cmath.rect(d, -phia * (m + mp) + (phib + np.pi) * (m - mp))
                    Sum = 1.0
                    for rho in range(rhoMax_a, rhoMin_a, -1):
                        Sum *= absRRatioSquared * ((N1_a - rho) * (N2_a - rho)) / (rho * (M_a + rho))
                        Sum += 1
                    values[i] = Prefactor1 * Sum
