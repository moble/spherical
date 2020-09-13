# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import copy
import math
import numpy as np
from .. import LM_total_size, LM_deduce_ell_max


class Modes(np.ndarray):
    """Object to store SWHS modes

    This class subclasses numpy's ndarray object, so that it should act like a numpy array in many
    respects, even with functions like np.zeros_like.

    NOTE: The functions `np.copy(modes)` and `np.array(modes, copy=True)` return `ndarray` objects;
    they lose information about the SWSH attributes, and have different ufuncs.  If you wish to keep
    this information, use `modes.copy()`.  Also note that pickling works as expected, as do
    copy.copy and copy.deepcopy.

    The number of dimensions is arbitrary as long as it is at least 1, but the modes must be stored
    in the last axis.  For example, a SWSH function of time may be stored as a 2-d array where the
    first axis represents different times, and the second axis represents the mode weights at each
    instant of time.

    This class also does three important things that are unlike numpy arrays:

    1) It tracks the spin weight and minimum and maximum ell values stored in this data.

    2) It provides additional convenience methods, like `index` to find the index of a particular
       (ell, m) mode; a `grid` method to convert modes to the SWSH values on a grid over the sphere
       (while correctly handling additional dimensions); and the various derivative operators,
       including $\eth$ and $\bar{\eth}$.

    3) It overrides most of numpy's "universal functions" (ufuncs) to work appropriately for
       spin-weighted functions.  Specifically, these ufuncs are interpreted as acting on the
       spin-weighted function itself, rather than just the mode weights.  Most importantly, we have
       these overriden methods:

       a) Conjugating a Modes object will result in a new Modes object that represents the
          conjugated spin-weighted function, rather than simply conjugating the mode-weights of the
          function.

       b) Multiplying two Modes objects will result in a new Modes object that represents the
          pointwise product of the functions themselves (and will correctly have spin weight given
          by the sum of the spin weights of the first two functions), rather than the product of the
          mode weights.  Division is only permitted when the divisor is a constant.

       c) Addition (and subtraction) is permitted for functions of the same spin weight, but it does
          not make sense to add (or subtract) functions with different spin weights, so any attempt
          to do so raises a ValueError.  [Note that adding a constant is equivalent to adding a
          function of spin weight zero, and is treated in the same way.]

       d) The "absolute" ufunc does not return the absolute value of each mode weight (which is
          almost certainly meaningless); it returns the L2 norm of spin-weighted function over the
          sphere -- which happens to equal the sum of the squares of the absolute values of the mode
          weights.

       Numerous other ufuncs -- such as log, exp, trigonometric ufuncs, bit-twiddling ufuncs, and so
       on -- are disabled because they don't make sense when applied to functions.

    It is possible to treat the underlying data of a Modes object `modes` as an ordinary numpy array
    by taking `modes.view(np.ndarray)`.  However, it is hoped that this class already performs all
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
    ell_min: int [defaults to 0]
        The smallest ell value present in the *input* data.  Note that the data will be stored in
        this object with all modes present, starting from ell=0.  This parameter just determines how
        the input data will be copied into the internal representation.
    ell_max: int [optional if ell_min is not passed as positional argument]
        The largest ell value present in the input data.  If this is not passed explicitly, it will
        be inferred from the size of the data.
    multiplication_truncator: None or callable [optional]
        Function to be used by default when multiplying Modes objects together.  See the
        documentation for spherical.Modes.multiply for more details.  The default behavior
        with `sum` is the most correct one -- keeping all ell values that result -- but also the
        most wasteful, and very likely to be overkill.  The user may prefer to use `max`, which will
        just return a product with ell_max equal to the larger of the two input ell_max values.

    """

    # https://numpy.org/doc/1.18/user/basics.subclassing.html
    def __new__(cls, input_array, *args, **kwargs):
        if len(args) == 2 or len(args) > 3:
            raise ValueError("Only one, two, or four positional arguments may be passed")
        if len(args) >= 1:
            kwargs['spin_weight'] = args[0]
        if len(args) == 3:
            kwargs['ell_min'] = args[1]
            kwargs['ell_max'] = args[2]
        ell_min = kwargs.pop('ell_min', 0)
        metadata = copy.copy(getattr(input_array, '_metadata', {}))
        metadata.update(**kwargs)
        input_array = np.asanyarray(input_array).view(complex)
        spin_weight = metadata.get('spin_weight', None)
        if spin_weight is None:
            raise ValueError("Spin weight must be specified")
        ell_max = metadata.get('ell_max', None)
        if ell_max is None:
            if np.ndim(input_array) == 0:
                ell_max = 0
            else:
                ell_max = LM_deduce_ell_max(input_array.shape[-1], ell_min)
            metadata['ell_max'] = ell_max
        if input_array.shape[-1] != LM_total_size(ell_min, ell_max):
            raise ValueError(f"Input array has shape {input_array.shape} when viewed as a complex array.\n            "
                             +f"Its last dimension should have size {LM_total_size(ell_min, ell_max)}, "
                             +f"for consistency with the input ell_min ({ell_min}) and ell_max ({ell_max}).")
        if ell_min == 0:
            obj = input_array.view(cls)
        else:
            insertion_indices = [0,]*LM_total_size(0, ell_min-1)
            obj = np.insert(input_array, insertion_indices, 0.0, axis=-1).view(cls)
        obj[..., :LM_total_size(0, abs(spin_weight)-1)] = 0.0
        obj._metadata = metadata
        return obj

    # https://numpy.org/doc/1.18/user/basics.subclassing.html
    def __array_finalize__(self, obj):
        if obj is None: return
        self._metadata = copy.copy(getattr(obj, '_metadata', {}))
        if not 'spin_weight' in self._metadata:
            self._metadata['spin_weight'] = None
        if not 'ell_max' in self._metadata:
            self._metadata['ell_max'] = None

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
    def ell_min(self):
        """Smallest ell value stored in data [need not equal abs(s)]"""
        return 0

    @property
    def ell_max(self):
        """Largest ell value stored in data"""
        return self._metadata['ell_max']

    @property
    def n_modes(self):
        """Number of elements along the last axis"""
        return self.shape[-1]

    from .algebra import (
        conjugate, bar, _real_func, real, _imag_func, imag, norm,
        add, subtract, multiply, divide
    )

    conj = conjugate

    from .derivatives import (
        Lsquared, Lz, Lplus, Lminus,
        Rsquared, Rz, Rplus, Rminus,
        eth, ethbar
    )

    from .utilities import (
        index, truncate_ell, grid, evaluate, _check_broadcasting
    )

    from .ufuncs import __array_ufunc__
