# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import copy
import math
import numpy as np


class Grid(np.ndarray):
    """Object to store SWHS values on a grid

    This class subclasses numpy's ndarray object, so that it should act like a numpy array in many
    respects, even with functions like np.zeros_like.

    NOTE: The functions `np.copy(grid)` and `np.array(grid, copy=True)` return `ndarray` objects;
    they lose information about the SWSH attributes, and have different ufuncs.  If you wish to keep
    this information, use `grid.copy()`.  Also note that pickling works as expected, as do copy.copy
    and copy.deepcopy.

    The number of dimensions is arbitrary as long as it is 2 or more, but the grid must be stored in
    the last 2 axes.  Specifically, the grid must have shape (n_theta, n_phi).  (See the function
    `spherical.theta_phi` for an example of the actual grid locations expected.)  For
    example, a SWSH function of time may be stored as a 3-d array where the first axis represents
    different times, and the second and third axes represent the function values at each instant of
    time.

    This class also does two important things that are unlike numpy arrays:

    1) It tracks the spin weight of the function represented by this data.

    2) It overrides most of numpy's "universal functions" (ufuncs) to work appropriately for
       spin-weighted functions.  Specifically, these ufuncs are interpreted as acting on the
       spin-weighted function itself, rather than just the grid values.  The returned values are --
       where possible -- Grid objects.  Most importantly, we have these overriden methods:

       a) Multiplying two Grid objects will result in a new Grid object that represents the
          pointwise product of the functions themselves (and will correctly have spin weight given
          by the sum of the spin weights of the first two functions).  Division is only permitted
          when the divisor has spin weight zero.

       b) Addition (and subtraction) is permitted for functions of the same spin weight, but it does
          not make sense to add (or subtract) functions with different spin weights, so any attempt
          to do so raises a ValueError.

       Note that a constant is implicitly a function of spin weight zero, and is treated as such.

       Numerous other ufuncs -- such as log, exp, trigonometric ufuncs, bit-twiddling ufuncs, and so
       on -- are disabled because they don't result in objects of specific spin weights.

    It is possible to treat the underlying data of a Grid object `grid` as an ordinary numpy array
    by taking `grid.view(np.ndarray)`.  However, it is hoped that this class already performs all
    reasonable operations.  If you find a missing feature that requires you to resort to this,
    please feel free to open an issue in this project's github page to discuss it.

    Also, be aware that ndarrays also have various built-in methods that cannot be overridden very
    easily, such as max, copysign, etc.  If you try to use -- even indirectly -- those functions
    that don't have any clear interpretation for spin-weighted functions, things will likely break.


    Constructor parameters
    ======================
    input_array: array_like
        This may be a numpy ndarray, or any subclass.  It must be able to be viewed as complex.  If
        it has a `_metadata` attribute, that field will be copied to the new array; if the following
        parameters are not passed to the constructor, they will be searched for in the metadata.
    spin_weight: int [optional if present in `input_array._metadata`]
        The spin weight of the function that this Modes object describes.  This must be specified
        somehow, whether via a `_metadata` attribute of the input array, or as a keyword argument,
        or as the second positional argument (where the latter will override the former values).

    """

    # https://numpy.org/doc/1.18/user/basics.subclassing.html
    def __new__(cls, input_array, *args, **kwargs):
        if len(args) > 1:
            raise ValueError("Only one positional argument is recognized")
        elif len(args) == 1:
            kwargs['spin_weight'] = args[0]
        metadata = copy.copy(getattr(input_array, '_metadata', {}))
        metadata.update(**kwargs)
        input_array = np.asanyarray(input_array)
        if np.ndim(input_array) < 2:
            raise ValueError(f"Input array must have at least two dimensions; it has shape {input_array.shape}")
        n_theta, n_phi = input_array.shape[-2:]
        spin_weight = metadata.get('spin_weight', None)
        if spin_weight is None:
            raise ValueError("Spin weight must be specified")
        if n_theta < 2*abs(spin_weight)+1 or n_phi < 2*abs(spin_weight)+1:
            raise ValueError(f"Input array must have at least {2*abs(s)+1} points in each direction to have any "
                             f"nontrivial content for a field of spin weight {spin_weight}.")
        obj = input_array.view(cls)
        obj._metadata = metadata
        return obj

    # https://numpy.org/doc/1.18/user/basics.subclassing.html
    def __array_finalize__(self, obj):
        if obj is None: return
        self._metadata = copy.copy(getattr(obj, '_metadata', {}))
        if not 'spin_weight' in self._metadata:
            self._metadata['spin_weight'] = None

    # For pickling
    def __reduce__(self):
        state = super(Modes, self).__reduce__()
        new_attributes = state[2] + (self._metadata,)
        return (state[0], state[1], new_attributes)

    # For unpickling
    def __setstate__(self, state):
        self._metadata = copy.deepcopy(state[-1])
        super(Modes, self).__setstate__(state[:-1])

    @property
    def ndarray(self):
        """View this array as a numpy ndarray"""
        return self.view(np.ndarray)

    @property
    def s(self):
        """Spin weight of this Modes object"""
        return self._metadata['spin_weight']

    spin_weight = s

    @property
    def n_theta(self):
        """Number of elements along the theta axis"""
        return self.shape[-2]

    @property
    def n_phi(self):
        """Number of elements along the phi axis"""
        return self.shape[-1]

    from .algebra import (
        conjugate, bar, real, imag, absolute,
        add, subtract, multiply, divide
    )

    conj = conjugate

    from .utilities import (
        modes, _check_broadcasting
    )

    from .ufuncs import __array_ufunc__
