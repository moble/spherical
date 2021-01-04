# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

### NOTE: The functions in this file are intended purely for inclusion in the Grid class.  In
### particular, they assume that the first argument, `self` is an instance of Grid.  They should
### probably not be used outside of that class.

import copy
import numpy as np


def __array_ufunc__(self, ufunc, method, *args, out=None, **kwargs):
    # These are required for basic support, but can be more-or-less passed through because they return bools
    if ufunc in [np.greater, np.greater_equal, np.less, np.less_equal, np.not_equal, np.equal,
                 np.logical_and, np.logical_or, np.isfinite, np.isinf, np.isnan]:
        args = [arg.view(np.ndarray) if isinstance(arg, type(self)) else arg for arg in args]
        if out is not None:
            kwargs['out'] = tuple(o.view(np.ndarray) if isinstance(o, type(self)) else o for o in out)
        return self.view(np.ndarray).__array_ufunc__(ufunc, method, *args, **kwargs)

    # Here are all the functions we will support directly; all other ufuncs are probably meaningless
    elif ufunc not in [np.positive, np.negative, np.add, np.subtract,
                       np.multiply, np.divide, np.true_divide,
                       np.conj, np.conjugate, np.absolute,
                       np.power, np.sqrt, np.square, np.reciprocal]:
        return NotImplemented

    # We will not be supporting any more ufunc keywords
    if kwargs:
        raise NotImplementedError(f"Unrecognized arguments to {type(self).__name__}.__array_ufunc__: {kwargs}")

    if ufunc in [np.positive, np.negative]:
        out_view = out if out is None else out[0].view(np.ndarray)
        result = type(self)(ufunc(self.view(np.ndarray), out=out_view), **self._metadata)
        if out is not None and isinstance(out[0], type(self)):
            out[0]._metadata = copy.copy(self._metadata)

    elif ufunc in [np.add, np.subtract]:
        if isinstance(args[0], type(self)) and isinstance(args[1], type(self)):
            g1, g2 = args[:2]
            if g1.spin_weight != g2.spin_weight:
                raise ValueError(
                    f"Cannot {ufunc.__name__} grids with different spin weights ({g1.spin_weight}, {g2.spin_weight})"
                )
            if g1.n_theta != g2.n_theta or g1.n_phi != g2.n_phi:
                raise ValueError(f"Shape mismatch: grid1.shape={g1.shape}; grid2.shape={g2.shape}")
            out_view = out if out is None else out[0].view(np.ndarray)
            result = type(self)(ufunc(g1.view(np.ndarray), g2.view(np.ndarray), out=out_view), **self._metadata)
            if out is not None and isinstance(out[0], type(self)):
                metadata = copy.copy(self._metadata)
                out[0]._metadata = metadata
        elif isinstance(args[0], type(self)):
            grid = args[0]
            scalars = np.asanyarray(args[1])
            if (grid.spin_weight!=0 and np.any(scalars)) or not grid._check_broadcasting(scalars):
                return NotImplemented
            out_view = out if out is None else out[0].view(np.ndarray)
            result = type(self)(ufunc(grid.view(np.ndarray), scalars[..., np.newaxis, np.newaxis], out=out_view), **self._metadata)
            if out is not None and isinstance(out[0], type(self)):
                out[0]._metadata = copy.copy(self._metadata)
        elif isinstance(args[1], type(self)):
            grid = args[1]
            scalars = np.asanyarray(args[0])
            if (grid.spin_weight!=0 and np.any(scalars)) or not grid._check_broadcasting(scalars, reverse=True):
                return NotImplemented
            out_view = out if out is None else out[0].view(np.ndarray)
            result = type(self)(ufunc(scalars[..., np.newaxis, np.newaxis], grid.view(np.ndarray), out=out_view), **self._metadata)
            if out is not None and isinstance(out[0], type(self)):
                out[0]._metadata = copy.copy(self._metadata)
        else:
            return NotImplemented

    elif ufunc in [np.multiply, np.divide, np.true_divide]:
        if isinstance(args[0], type(self)) and isinstance(args[1], type(self)):
            g1, g2 = args[:2]
            if g1.n_theta != g2.n_theta or g1.n_phi != g2.n_phi:
                raise ValueError(f"Shape mismatch: grid1.shape={g1.shape}; grid2.shape={g2.shape}")
            result_s = g1.spin_weight + g2.spin_weight if ufunc is np.multiply else g1.spin_weight - g2.spin_weight
            result_metadata = copy.copy(g1._metadata)
            result_metadata['spin_weight'] = result_s
            out_view = out if out is None else out[0].view(np.ndarray)
            result = type(self)(ufunc(g1.view(np.ndarray), g2.view(np.ndarray), out=out_view), **result_metadata)
            if out is not None and isinstance(out[0], type(self)):
                out[0]._metadata = result_metadata
        elif isinstance(args[0], type(self)):
            grid = args[0]
            scalars = np.asanyarray(args[1])
            if not grid._check_broadcasting(scalars):
                return NotImplemented
            result_s = grid.spin_weight
            result_metadata = copy.copy(grid._metadata)
            result_metadata['spin_weight'] = result_s
            out_view = out if out is None else out[0].view(np.ndarray)
            result = type(self)(ufunc(grid.view(np.ndarray), scalars[..., np.newaxis, np.newaxis], out=out_view), **result_metadata)
            if out is not None and isinstance(out[0], type(self)):
                out[0]._metadata = result_metadata
        elif isinstance(args[1], type(self)):
            grid = args[1]
            scalars = np.asanyarray(args[0])
            if not grid._check_broadcasting(scalars, reverse=True):
                return NotImplemented
            result_s = grid.spin_weight if ufunc is np.multiply else - grid.spin_weight
            result_metadata = copy.copy(grid._metadata)
            result_metadata['spin_weight'] = result_s
            out_view = out if out is None else out[0].view(np.ndarray)
            result = type(self)(ufunc(scalars[..., np.newaxis, np.newaxis], grid.view(np.ndarray), out=out_view), **result_metadata)
            if out is not None and isinstance(out[0], type(self)):
                out[0]._metadata = result_metadata
        else:
            return NotImplemented

    elif ufunc in [np.conj, np.conjugate]:
        if isinstance(args[0], type(self)):
            metadata = copy.copy(args[0]._metadata)
            metadata['spin_weight'] = -args[0].spin_weight
            out_view = out if out is None else out[0].view(np.ndarray)
            result = type(self)(np.conjugate(args[0].view(np.ndarray), out=out_view), **metadata)
            if out is not None and isinstance(out[0], type(self)):
                out[0]._metadata = metadata
        else:
            return NotImplemented

    elif ufunc is np.absolute:
        if isinstance(args[0], type(self)):
            metadata = copy.copy(args[0]._metadata)
            metadata['spin_weight'] = 0
            out_view = out if out is None else out[0].view(np.ndarray)
            result = type(self)(np.absolute(args[0].view(np.ndarray), out=out_view), **metadata)
            if out is not None and isinstance(out[0], type(self)):
                out[0]._metadata = metadata
        else:
            return NotImplemented

    elif ufunc is np.power:
        if isinstance(args[0], type(self)):
            try:
                exponent = int(args[1])
            except:
                return NotImplemented
            metadata = copy.copy(args[0]._metadata)
            metadata['spin_weight'] = exponent * args[0].spin_weight
            out_view = out if out is None else out[0].view(np.ndarray)
            result = type(self)(np.power(args[0].view(np.ndarray), exponent, out=out_view), **metadata)
            if out is not None and isinstance(out[0], type(self)):
                out[0]._metadata = metadata
        else:
            return NotImplemented

    elif ufunc in [np.sqrt, np.square, np.reciprocal]:
        if isinstance(args[0], type(self)):
            if ufunc is np.sqrt:
                if args[0].spin_weight % 2 != 0:
                    return NotImplemented
                new_spin = args[0].spin_weight // 2
            elif ufunc is np.square:
                new_spin = args[0].spin_weight * 2
            elif ufunc is np.reciprocal:
                new_spin = -args[0].spin_weight
            metadata = copy.copy(args[0]._metadata)
            metadata['spin_weight'] = new_spin
            out_view = out if out is None else out[0].view(np.ndarray)
            result = type(self)(ufunc(args[0].view(np.ndarray), out=out_view), **metadata)
            if out is not None and isinstance(out[0], type(self)):
                out[0]._metadata = metadata
        else:
            return NotImplemented

    else:
        # I thought we filtered everything else out...
        raise NotImplementedError(f"{type(self).__name__}.__array_ufunc__ has reached a point it should not have for ufunc {ufunc}")

    if result is NotImplemented:
        return NotImplemented

    if method == 'at':
        return

    return result
