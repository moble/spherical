# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

### NOTE: The functions in this file are intended purely for inclusion in the Modes class.  In
### particular, they assume that the first argument, `self` is an instance of Modes.  They should
### probably not be used outside of that class.

import copy
import numpy as np
from .. import LM_total_size
from ..multiplication import _multiplication_helper


def __array_ufunc__(self, ufunc, method, *args, out=None, **kwargs):
    # These are required for basic support, but can be more-or-less passed through because they return bools
    if ufunc in [np.not_equal, np.equal, np.logical_and, np.logical_or, np.isfinite, np.isinf, np.isnan]:
        args = [arg.view(np.ndarray) if isinstance(arg, type(self)) else arg for arg in args]
        if out is not None:
            kwargs['out'] = tuple(o.view(np.ndarray) if isinstance(o, type(self)) else o for o in out)
        return self.view(np.ndarray).__array_ufunc__(ufunc, method, *args, **kwargs)

    # Here are all the functions we will support directly; all other ufuncs are probably meaningless
    elif ufunc not in [np.positive, np.negative, np.add, np.subtract,
                       np.multiply, np.divide, np.true_divide,
                       np.conj, np.conjugate, np.absolute]:
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
            m1, m2 = args[:2]
            if m1.spin_weight != m2.spin_weight:
                return NotImplemented
            s = m1.spin_weight
            ell_min = min(m1.ell_min, m2.ell_min)
            ell_max = max(m1.ell_max, m2.ell_max)
            shape = np.broadcast(m1[..., 0], m2[..., 0]).shape + (LM_total_size(ell_min, ell_max),)
            result = out or np.zeros(shape, dtype=np.complex_)
            if isinstance(result, tuple):
                result = result[0]
            if isinstance(result, type(self)):
                result = result.view(np.ndarray)
            i_s1 = LM_total_size(ell_min, m1.ell_min-1)
            i_s2 = i_s1+LM_total_size(m1.ell_min, m1.ell_max)
            i_o1 = LM_total_size(ell_min, m2.ell_min-1)
            i_o2 = i_o1+LM_total_size(m2.ell_min, m2.ell_max)
            result[..., i_s1:i_s2] = m1.view(np.ndarray)
            if ufunc is np.subtract:
                result[..., i_o1:i_o2] -= m2.view(np.ndarray)
            else:
                result[..., i_o1:i_o2] += m2.view(np.ndarray)
            metadata = copy.copy(self._metadata)
            metadata['spin_weight'] = s
            metadata['ell_min'] = ell_min
            metadata['ell_max'] = ell_max
            result = type(self)(result, **metadata)
            if out is not None and isinstance(out[0], type(self)):
                metadata.pop('ell_min')
                out[0]._metadata = metadata
        elif isinstance(args[0], type(self)):
            modes = args[0]
            scalars = np.asanyarray(args[1])
            if np.any(scalars) or not modes._check_broadcasting(scalars):
                return NotImplemented
            result = type(self)(ufunc(modes.view(np.ndarray), scalars[..., np.newaxis], out=out), **self._metadata)
            if out is not None and isinstance(out[0], type(self)):
                out[0]._metadata = copy.copy(self._metadata)
        elif isinstance(args[1], type(self)):
            scalars = np.asanyarray(args[0])
            modes = args[1]
            if np.any(scalars) or not modes._check_broadcasting(scalars, reverse=True):
                return NotImplemented
            result = type(self)(ufunc(scalars[..., np.newaxis], modes.view(np.ndarray), out=out), **self._metadata)
            if out is not None and isinstance(out[0], type(self)):
                out[0]._metadata = copy.copy(self._metadata)
        else:
            return NotImplemented

    elif ufunc is np.multiply:
        if isinstance(args[0], type(self)) and isinstance(args[1], type(self)):
            s = args[0].view(np.ndarray)
            o = args[1].view(np.ndarray)
            result_s = args[0].spin_weight + args[1].spin_weight
            result_ell_min = 0
            result_ell_max = max(
                truncator((args[0].ell_max, args[1].ell_max))
                for truncator in [
                    args[0]._metadata.get('multiplication_truncator', sum),
                    args[1]._metadata.get('multiplication_truncator', sum)
                ]
            )
            result_shape = np.broadcast(s[..., 0], o[..., 0]).shape + (LM_total_size(result_ell_min, result_ell_max),)
            result = out or np.zeros(result_shape, dtype=np.complex_)
            if isinstance(result, tuple):
                result = result[0]
            if isinstance(result, type(self)):
                result = result.view(np.ndarray)
            _multiplication_helper(s, args[0].ell_min, args[0].ell_max, args[0].spin_weight,
                                   o, args[1].ell_min, args[1].ell_max, args[1].spin_weight,
                                   result, result_ell_min, result_ell_max, result_s)
            metadata = copy.copy(self._metadata)
            metadata['spin_weight'] = result_s
            metadata['ell_min'] = result_ell_min
            metadata['ell_max'] = result_ell_max
            result = type(self)(result, **metadata)
            if out is not None and isinstance(out[0], type(self)):
                metadata.pop('ell_min')
                out[0]._metadata = metadata
        elif isinstance(args[0], type(self)):
            modes = args[0]
            scalars = np.asanyarray(args[1])
            if not modes._check_broadcasting(scalars):
                return NotImplemented
            result = type(self)(ufunc(modes.view(np.ndarray), scalars[..., np.newaxis], out=out), **self._metadata)
            if out is not None and isinstance(out[0], type(self)):
                out[0]._metadata = copy.copy(self._metadata)
        elif isinstance(args[1], type(self)):
            scalars = np.asanyarray(args[0])
            modes = args[1]
            if not modes._check_broadcasting(scalars, reverse=True):
                return NotImplemented
            result = type(self)(ufunc(scalars[..., np.newaxis], modes.view(np.ndarray), out=out), **self._metadata)
            if out is not None and isinstance(out[0], type(self)):
                out[0]._metadata = copy.copy(self._metadata)
        else:
            return NotImplemented

    elif ufunc in [np.divide, np.true_divide]:
        if isinstance(args[1], type(self)) or not isinstance(args[0], type(self)):
            return NotImplemented
        modes = args[0]
        scalars = np.asanyarray(args[1])
        if not modes._check_broadcasting(scalars):
            return NotImplemented
        result = type(self)(ufunc(modes.view(np.ndarray), scalars[..., np.newaxis], out=out), **self._metadata)
        if out is not None and isinstance(out[0], type(self)):
            out[0]._metadata = copy.copy(self._metadata)

    elif ufunc in [np.conj, np.conjugate]:
        if isinstance(args[0], type(self)):
            s = args[0].view(np.ndarray)
            c = np.zeros_like(s) if out is None else out[0]
            for ell in range(abs(args[0].spin_weight), args[0].ell_max+1):
                i = LM_index(ell, 0, args[0].ell_min)
                if args[0].spin_weight%2 == 0:
                    c[..., i] = np.conjugate(s[..., i])
                else:
                    c[..., i] = -np.conjugate(s[..., i])
                for m in range(1, ell+1):
                    i_p, i_n = LM_index(ell, m, args[0].ell_min), LM_index(ell, -m, args[0].ell_min)
                    if (args[0].spin_weight+m)%2 == 0:
                        c[..., i_p], c[..., i_n] = np.conjugate(s[..., i_n]), np.conjugate(s[..., i_p])
                    else:
                        c[..., i_p], c[..., i_n] = -np.conjugate(s[..., i_n]), -np.conjugate(s[..., i_p])
            metadata = copy.copy(args[0]._metadata)
            metadata['spin_weight'] = -args[0].spin_weight
            result = type(self)(c, **metadata)
            if out is not None and isinstance(out[0], type(self)):
                out[0]._metadata = metadata
        else:
            return NotImplemented

    elif ufunc is np.absolute:
        return args[0].norm()

    else:
        # I thought we filtered everything else out...
        raise NotImplementedError(f"{type(self).__name__}.__array_ufunc__ has reached a point it should not have for ufunc {ufunc}")

    if result is NotImplemented:
        return NotImplemented

    if method == 'at':
        return

    return result
