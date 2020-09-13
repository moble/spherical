# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

### NOTE: The functions in this file are intended purely for inclusion in the Grid class.  In
### particular, they assume that the first argument, `self` is an instance of Grid.  They should
### probably not be used outside of that class.

import copy
import numpy as np


def conjugate(self, inplace=False):
    """Return Grid object corresponding to conjugated function

    The result has spin weight equal to the negative of this object's spin weight.

    """
    s = self.view(np.ndarray)
    c = s if inplace else np.empty_like(s)
    np.conjugate(s, out=c)
    if inplace:
        self._metadata['spin_weight'] = -self.s
        return self
    metadata = copy.copy(self._metadata)
    metadata['spin_weight'] = -self.s
    return type(self)(c, **metadata)


@property
def bar(self):
    return self.conjugate()
bar.__doc__ = conjugate.__doc__


@property
def real(self):
    """Return real-valued part of function values on the grid

    This only makes sense when the function has spin weight 0; anything else will raise an error.

    """
    if self.s != 0:
        raise ValueError("The real part of a function with non-zero spin weight is meaningless")
    s = self.view(np.ndarray)
    c = s.real
    return type(self)(c, **self._metadata)


@property
def imag(self):
    """Return imaginary-valued part of function values on the grid

    This only makes sense when the function has spin weight 0; anything else will raise an error.

    """
    if self.s != 0:
        raise ValueError("The imaginary part of a function with non-zero spin weight is meaningless")
    s = self.view(np.ndarray)
    c = s.imag
    return type(self)(c, **self._metadata)


def absolute(self):
    """Return absolute value of function values on the grid"""
    metadata = copy.copy(self._metadata)
    metadata['spin_weight'] = 0
    return type(self)(np.abs(self.view(np.ndarray)), **metadata)


def add(self, other):
    if isinstance(other, type(self)):
        s = self.s
        if s != other.s:
            raise ValueError(f"Cannot add functions with different spin weights ({s} and {other.s})")
        if self.n_theta != other.n_theta or self.n_phi != other.n_phi:
            raise ValueError(f"Shape mismatch: self.shape={self.shape}; other.shape={other.shape}")
        result = self.view(np.ndarray) + other.view(np.ndarray)
        return type(self)(result, **self._metadata)
    elif self.s != 0 and np.any(other):
        raise ValueError(f"It is not permitted to add non-zero scalars to a {type(self).__name__} object of non-zero spin weight")
    else:
        result = self.view(np.ndarray) + other
        return type(self)(result, **self._metadata)


def subtract(self, other):
    if isinstance(other, type(self)):
        if self.s != other.s:
            raise ValueError(f"Cannot subtract functions with different spin weights ({s} and {other.s})")
        if self.n_theta != other.n_theta or self.n_phi != other.n_phi:
            raise ValueError(f"Shape mismatch: self.shape={self.shape}; other.shape={other.shape}")
        result = self.view(np.ndarray) - other.view(np.ndarray)
        return type(self)(result, **self._metadata)
    elif self.s != 0 and np.any(other):
        raise ValueError(f"It is not permitted to subtract non-zero scalars from a {type(self).__name__} object of non-zero spin weight")
    else:
        result = self.view(np.ndarray) - other
        return type(self)(result, **self._metadata)


def multiply(self, other, truncator=None):
    """Multiply by another spin-weighted function or a scalar

    For spin-weighted functions, the spin weight of their product is the sum of the spin weights of
    the input.

    Parameters
    ==========
    other: Grid, array_like, complex, or float
        Grid object representing the spin-weighted functions, or an array or float which is
        equivalent to a spin-0 function.

    """
    if isinstance(other, type(self)):
        if self.n_theta != other.n_theta or self.n_phi != other.n_phi:
            raise ValueError(f"Shape mismatch: self.shape={self.shape}; other.shape={other.shape}")
        result = self.view(np.ndarray) * other.view(np.ndarray)
        result_s = self.s + other.s
        metadata = copy.copy(self._metadata)
        metadata['spin_weight'] = result_s
        return type(self)(result, **metadata)
    else:
        if self._check_broadcasting(other):
            return type(self)(self.view(np.ndarray) * other, **self._metadata)
        elif self._check_broadcasting(other, reverse=True):
            return type(self)(other * self.view(np.ndarray), **self._metadata)
        else:
            raise ValueError(f"Cannot broadcast input array to this {type(self).__name__} object.  Note that the input array\n           "
                             "must broadcast to all but last two dimensions of this object; it is not allowed to\n           "
                             "multiply each grid value individually.  If you really want to hack this, view this\n           "
                             "object as an ndarray, and don't complain if your results are wrong.")


def divide(self, other):
    """Divide by another spin-weighted function or a scalar

    For spin-weighted functions, the spin weight of their ratio is the difference of the spin
    weights of the input.

    Parameters
    ==========
    other: Grid, array_like, complex, or float
        Grid object representing the spin-weighted functions, or an array or float which is
        equivalent to a spin-0 function.

    """
    if isinstance(other, type(self)):
        if self.n_theta != other.n_theta or self.n_phi != other.n_phi:
            raise ValueError(f"Shape mismatch: self.shape={self.shape}; other.shape={other.shape}")
        result = self.view(np.ndarray) / other.view(np.ndarray)
        result_s = self.s - other.s
        metadata = copy.copy(self._metadata)
        metadata['spin_weight'] = result_s
        return type(self)(result, **metadata)
    else:
        if self._check_broadcasting(other):
            return type(self)(self.view(np.ndarray) / other, **self._metadata)
        else:
            raise ValueError(f"Cannot broadcast input array to this {type(self).__name__} object.  Note that the input array\n           "
                             "must broadcast to all but last two dimensions of this object; it is not allowed to\n           "
                             "multiply each grid value individually.  If you really want to hack this, view this\n           "
                             "object as an ndarray, and don't complain if your results are wrong.")
