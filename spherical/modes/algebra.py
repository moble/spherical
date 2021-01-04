# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

### NOTE: The functions in this file are intended purely for inclusion in the Modes class.  In
### particular, they assume that the first argument, `self` is an instance of Modes.  They should
### probably not be used outside of that class.

import copy
import numpy as np
from .. import LM_total_size, LM_index
from ..multiplication import _multiplication_helper


def conjugate(self, inplace=False):
    """Return Modes object corresponding to conjugated function

    The operations of conjugation and decomposition into mode weights do not commute.  That is,
    the modes of a conjugated function do not equal the conjugated modes of a function.  So,
    rather than simply returning the conjugate of the data from this Modes object, this function
    returns a Modes object containing the data for the conjugated function.

    If `inplace` is True, then the operation is performed in place, modifying this Modes object
    itself.  Note that some copying is still needed, but only for 2 modes at a time, and those
    copies are freed after this function returns.

    Here is the derivation:

        f = sum(f{s, l,m} * {s}Y{l,m} for l, m in LM)
        conjugate(f) = sum(conjugate(f{s, l,m}) * conjugate({s}Y{l,m}))
                     = sum(conjugate(f{s, l,m}) * (-1)**(s+m) * {-s}Y{l,-m})
                     = sum((-1)**(s+m) * conjugate(f{s, l, -m}) * {-s}Y{l,m})

        conjugate(f){s', l',m'} = integral(
            sum((-1)**(s+m) * conjugate(f{s,l,-m}) * {-s}Y{l,m}) * {s'}Y{l',m'},
            dR  # integration over rotation group
        )
        = sum((-1)**(s+m) * conjugate(f{s,l,-m}) * delta_{-s, s'} * delta_{l, l'} * delta_{m, m'})
        = (-1)**(s'+m') * conjugate(f{-s', l', -m'})

    The result is this:

        conjugate(f){s, l, m} = (-1)**(s+m) * conjugate(f{-s, l, -m})

    """
    s = self.view(np.ndarray)
    c = s if inplace else np.zeros_like(s)
    for ell in range(abs(self.spin_weight), self.ell_max+1):
        i = LM_index(ell, 0, self.ell_min)
        if self.spin_weight%2 == 0:
            c[..., i] = np.conjugate(s[..., i])
        else:
            c[..., i] = -np.conjugate(s[..., i])
        for m in range(1, ell+1):
            i_p, i_n = LM_index(ell, m, self.ell_min), LM_index(ell, -m, self.ell_min)
            if (self.spin_weight+m)%2 == 0:
                c[..., i_p], c[..., i_n] = np.conjugate(s[..., i_n]), np.conjugate(s[..., i_p])
            else:
                c[..., i_p], c[..., i_n] = -np.conjugate(s[..., i_n]), -np.conjugate(s[..., i_p])
    if inplace:
        self._metadata['spin_weight'] = -self.spin_weight
        return self
    metadata = copy.copy(self._metadata)
    metadata['spin_weight'] = -self.spin_weight
    return type(self)(c, **metadata)


@property
def bar(self):
    return self.conjugate()
bar.__doc__ = conjugate.__doc__


def _real_func(self, inplace=False):
    """Return Modes object corresponding to real-valued function

    Note that this only makes sense for functions of spin weight zero; other spins will raise
    ValueErrors.

    The condition that a function `f` be real is given by

        f{l, m} = conjugate(f){l, m} = (-1)**(m) * conjugate(f{l, -m})

    [Note that conj(f){l, m} != conj(f{l, m}).  See Modes.conjugate docstring.]

    We enforce that condition by essentially averaging the two modes:

        f{l, m} = (f{l, m} + (-1)**m * conjugate(f{l, -m})) / 2

    """
    if self.spin_weight != 0:
        raise ValueError("The real part of a function with non-zero spin weight is meaningless")
    s = self.view(np.ndarray)
    c = s if inplace else np.zeros_like(s)
    for ell in range(abs(self.spin_weight), self.ell_max+1):
        i = LM_index(ell, 0, self.ell_min)
        c[..., i] = np.real(s[..., i])
        for m in range(1, ell+1):
            i_p, i_n = LM_index(ell, m, self.ell_min), LM_index(ell, -m, self.ell_min)
            if m%2 == 0:
                c[..., i_p] = (s[..., i_p] + np.conjugate(s[..., i_n])) / 2
                c[..., i_n] = np.conjugate(c[..., i_p])
            else:
                c[..., i_p] = (s[..., i_p] - np.conjugate(s[..., i_n])) / 2
                c[..., i_n] = -np.conjugate(c[..., i_p])
    if inplace:
        return self
    return type(self)(c, **self._metadata)


@property
def real(self):
    return self._real_func(False)


def _imag_func(self, inplace=False):
    """Return Modes object corresponding to imaginary-valued function

    Note that this only makes sense for functions of spin weight zero; other spins will raise
    ValueErrors.

    The condition that a function `f` be purely imaginary is given by

        f = -conjugate(f)

    We take the mode decomposition of each side to find

        f{l, m} = -conjugate(f){l, m} = (-1)**(m+1) * conjugate(f{l, -m})

    [Note that conj(f){l, m} != conj(f{l, m}).  See Modes.conjugate docstring.]

    We enforce that condition by essentially averaging the two modes:

        f{l, m} = (f{l, m} - (-1)**m * conjugate(f{l, -m})) / 2

    Then, we multiply by -1j to ensure that np.imag(f.grid()) equals f.imag.grid().

    """
    if self.spin_weight != 0:
        raise ValueError("The imaginary part of a function with non-zero spin weight is meaningless")
    s = self.view(np.ndarray)
    c = s if inplace else np.zeros_like(s)
    for ell in range(abs(self.spin_weight), self.ell_max+1):
        i = LM_index(ell, 0, self.ell_min)
        c[..., i] = np.imag(s[..., i])
        for m in range(1, ell+1):
            i_p, i_n = LM_index(ell, m, self.ell_min), LM_index(ell, -m, self.ell_min)
            if m%2 == 0:
                c[..., i_p] = -1j * (s[..., i_p] - np.conjugate(s[..., i_n])) / 2
                #c[..., i_n] = -1j * -np.conjugate((s[..., i_p] - np.conjugate(s[..., i_n])) / 2)
                c[..., i_n] = np.conjugate(c[..., i_p])
            else:
                c[..., i_p] = -1j * (s[..., i_p] + np.conjugate(s[..., i_n])) / 2
                #c[..., i_n] = -1j * np.conjugate((s[..., i_p] + np.conjugate(s[..., i_n])) / 2)
                c[..., i_n] = -np.conjugate(c[..., i_p])
    if inplace:
        return self
    return type(self)(c, **self._metadata)


@property
def imag(self):
    return self._imag_func(False)


def norm(self):
    return np.linalg.norm(self.view(np.ndarray), axis=-1)


def add(self, other, subtraction=False):
    if isinstance(other, type(self)):
        if self.spin_weight != other.spin_weight:
            raise ValueError(
                f"Cannot add modes with different spin weights ({self.spin_weight} and {other.spin_weight})"
            )
        s = self.spin_weight
        ell_min = min(self.ell_min, other.ell_min)
        ell_max = max(self.ell_max, other.ell_max)
        shape = np.broadcast(self[..., 0], other[..., 0]).shape + (LM_total_size(ell_min, ell_max),)
        result = np.zeros(shape, dtype=np.complex_)
        i_s1 = LM_total_size(ell_min, self.ell_min-1)
        i_s2 = i_s1+LM_total_size(self.ell_min, self.ell_max)
        i_o1 = LM_total_size(ell_min, other.ell_min-1)
        i_o2 = i_o1+LM_total_size(other.ell_min, other.ell_max)
        result[..., i_s1:i_s2] = self.view(np.ndarray)
        if subtraction:
            result[..., i_o1:i_o2] -= other.view(np.ndarray)
        else:
            result[..., i_o1:i_o2] += other.view(np.ndarray)
        metadata = copy.copy(self._metadata)
        metadata['spin_weight'] = s
        metadata['ell_min'] = ell_min
        metadata['ell_max'] = ell_max
        return type(self)(result, **metadata)
    elif np.any(other):
        raise ValueError(f"It is not permitted to add nonzero scalars to a {type(self).__name__} object")
    else:
        return np.add(self, other)


def subtract(self, other):
    if isinstance(other, type(self)):
        if self.spin_weight != other.spin_weight:
            raise ValueError(
                f"Cannot subtract modes with different spin weights ({self.spin_weight} and {other.spin_weight})"
            )
        return self.add(other, True)
    elif np.any(other):
        raise ValueError(f"It is not permitted to add nonzero scalars to a {type(self).__name__} object")
    else:
        return np.subtract(self, other)


def multiply(self, other, truncator=None):
    """Multiply by another spin-weighted function or a scalar

    For spin-weighted functions, the spin weight of their product is the sum of the spin weights of
    the input.

    If those functions are band limited to maximum ell values, their product has maximum ell value
    given by the sum of the input maximum ell values.  Note that this output ell_max can be
    controlled by the `truncator` argument to this function; if the result is smaller than the sum
    of the input maximum ell values, this is equivalent to performing the full multiplication and
    then setting higher ell modes to zero.  The benefit of this type of truncation is that the
    higher modes don't even need to be computed, and no aliasing will result.

    Parameters
    ==========
    other: Modes, array_like, complex, or float
        Modes objects representing the spin-weighted functions, or an array or float which is
        equivalent to a spin-0 function.
    truncator: None or callable [defaults to None]
        Function to be applied to the tuple (self.ell_max, other.ell_max) to produce the ell_max for
        the resulting function.  Sensible values of the truncator include the built-in python
        functions `min`, `max`, and `sum` -- the latter giving the full "correct" answer.  If you
        want a specific ell value, you can use `lambda tup: max_ell`.  If None, this function falls
        back on the `multiplication_truncator` metadata fields of the input Modes objects, and uses
        the greater of the values that they return.  If either input object is missing the
        `multiplication_truncator` metadata field, it defaults to `sum`.

    """
    if isinstance(other, type(self)):
        s = self.view(np.ndarray)
        o = other.view(np.ndarray)
        new_s = self.spin_weight + other.spin_weight
        new_ell_min = 0
        if truncator is not None:
            new_ell_max = truncator((self.ell_max, other.ell_max))
        else:
            new_ell_max = max(
                truncator((self.ell_max, other.ell_max))
                for truncator in [
                    self._metadata.get('multiplication_truncator', sum),
                    other._metadata.get('multiplication_truncator', sum)
                ]
            )
        new_shape = np.broadcast(s[..., 0], o[..., 0]).shape + (LM_total_size(new_ell_min, new_ell_max),)
        new = np.zeros(new_shape, dtype=np.complex_)
        _multiplication_helper(s, self.ell_min, self.ell_max, self.spin_weight,
                               o, other.ell_min, other.ell_max, other.spin_weight,
                               new, new_ell_min, new_ell_max, new_s)
        metadata = copy.copy(self._metadata)
        metadata['spin_weight'] = new_s
        metadata['ell_min'] = new_ell_min
        metadata['ell_max'] = new_ell_max
        return type(self)(new, **metadata)
    else:
        if self._check_broadcasting(other):
            return self * other
        elif self._check_broadcasting(other, reverse=True):
            return other * self
        else:
            raise ValueError(
                "Cannot broadcast input array to this Modes object.  Note that the input array\n           "
                "must broadcast to all but last dimension of this object; it is not allowed to\n           "
                "multiply each mode weight individually.  If you really want to hack this, view\n           "
                "this Modes object as an ndarray, and don't complain when your results are wrong."
            )


def divide(self, other):
    if isinstance(other, type(self)):
        raise ValueError(f"Cannot divide one Modes object by another")
    else:
        return self / other
