# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import numpy as np
import quaternionic

from . import jit, WignerHsize, WignerHindex, WignerDsize, WignerDindex, Ysize, Yindex
from .recursions.wignerH import ϵ, _step_1, _step_2, _step_3, _step_4, _step_5
from .recursions.complex_powers import _complex_powers
from .utilities.indexing import _WignerHindex

inverse_4pi = 1.0 / (4 * np.pi)

to_euler_phases = quaternionic.converters.ToEulerPhases(jit)


def wigner_d(expiβ, ell_min, ell_max, out=None, workspace=None):
    """Compute Wigner's d matrix dˡₘₚ,ₘ(β)

    This is a simple wrapper for the Wigner.d method.  If you plan on calling this
    function more than once, you should probably construct a Wigner object and call
    the `d` method explicitly.

    See that function's documentation for more details.

    """
    return Wigner(ell_max, ell_min).d(expiβ, out=out, workspace=workspace)


def wigner_D(R, ell_min, ell_max, out=None, workspace=None):
    """Compute Wigner's 𝔇 matrix 𝔇ˡₘₚ,ₘ(R)

    This is a simple wrapper for the Wigner.D method.  If you plan on calling this
    function more than once, you should probably construct a Wigner object and call
    the `D` method explicitly.

    See that function's documentation for more details.

    """
    return Wigner(ell_max, ell_min).D(R, out=out, workspace=workspace)


class Wigner:
    def __init__(self, ell_max, ell_min=0, mp_max=np.iinfo(np.int64).max):
        self.ell_min = int(ell_min)
        self.ell_max = int(ell_max)
        self.mp_max = min(abs(int(mp_max)), self.ell_max)

        if ell_min < 0 or ell_min > ell_max:
            raise ValueError(f"ell_min={ell_min} must be non-negative and no greater than ell_max={ell_max}")
        if ell_max < 0:
            raise ValueError(f"ell_max={ell_max} must be non-negative")

        self._Hsize = WignerHsize(self.mp_max, self.ell_max)
        self._dsize = WignerDsize(self.ell_min, self.mp_max, self.ell_max)
        self._Dsize = WignerDsize(self.ell_min, self.mp_max, self.ell_max)
        self._Ysize = Ysize(self.ell_min, self.ell_max)

        workspace = self.new_workspace()
        self.Hwedge, self.Hv, self.Hextra, self.zₐpowers, self.zᵧpowers, self.z = self._split_workspace(workspace)

        n = np.array([n for n in range(self.ell_max+2) for m in range(-n, n+1)])
        m = np.array([m for n in range(self.ell_max+2) for m in range(-n, n+1)])
        absn = np.array([n for n in range(self.ell_max+2) for m in range(n+1)])
        absm = np.array([m for n in range(self.ell_max+2) for m in range(n+1)])
        self.a = np.sqrt((absn+1+absm) * (absn+1-absm) / ((2*absn+1)*(2*absn+3)))
        self.b = np.sqrt((n-m-1) * (n-m) / ((2*n-1)*(2*n+1)))
        self.b[m<0] *= -1
        self.d = 0.5 * np.sqrt((n-m) * (n+m+1))
        self.d[m<0] *= -1
        with np.errstate(divide='ignore', invalid='ignore'):
            self.g = 2*(m+1) / np.sqrt((n-m)*(n+m+1))
            self.h = np.sqrt((n+m+2)*(n-m-1) / ((n-m)*(n+m+1)))
        if not (
            np.all(np.isfinite(self.a)) and
            np.all(np.isfinite(self.b)) and
            np.all(np.isfinite(self.d))
        ):
            raise ValueError("Found a non-finite value inside this object")

    def new_workspace(self):
        """Return a new empty array providing workspace for calculating H"""
        return np.zeros(
            self.Hsize
            + (self.ell_max+1)**2
            + self.ell_max+2
            + 2*(self.ell_max+1)
            + 2*(self.ell_max+1)
            + 2*3,
            dtype=float
        )

    def _split_workspace(self, workspace):
        size1 = self.Hsize
        size2 = (self.ell_max+1)**2
        size3 = self.ell_max+2
        size4 = 2*(self.ell_max+1)
        size5 = 2*(self.ell_max+1)
        size6 = 2*3
        i1 = size1
        i2 = i1 + size2
        i3 = i2 + size3
        i4 = i3 + size4
        i5 = i4 + size5
        i6 = i5 + size6
        if workspace.size < i6:
            raise ValueError(f"Input workspace has size {workspace.size}, but {i6} is needed")
        Hwedge = workspace[:i1]
        Hv = workspace[i1:i2]
        Hextra = workspace[i2:i3]
        zₐpowers = workspace[i3:i4].view(complex)[np.newaxis]
        zᵧpowers = workspace[i4:i5].view(complex)[np.newaxis]
        z = workspace[i5:i6].view(complex)
        return Hwedge, Hv, Hextra, zₐpowers, zᵧpowers, z

    @property
    def Hsize(self):
        """Total size of Wigner H array

        The H array represents just 1/4 of the total possible indices of the H matrix,
        which are the same as for the Wigner d and 𝔇 matrices.

        This incorporates the mp_max, and ell_max information associated with this
        object.

        """
        return self._Hsize

    @property
    def dsize(self):
        """Total size of the Wigner d matrix

        This incorporates the ell_min, mp_max, and ell_max information associated with
        this object.

        """
        return self._dsize

    @property
    def Dsize(self):
        """Total size of the Wigner 𝔇 matrix

        This incorporates the ell_min, mp_max, and ell_max information associated with
        this object.

        """
        return self._Dsize

    @property
    def Ysize(self):
        """Total size of the spherical-harmonic array

        This incorporates the ell_min and ell_max information associated with this
        object.

        """
        return self._Ysize

    def Hindex(self, ell, mp, m):
        """Compute index into Wigner H matrix accounting for symmetries

        Parameters
        ----------
        ell : int
            Integer satisfying ell_min <= ell <= ell_max
        mp : int
            Integer satisfying -min(ell_max, mp_max) <= mp <= min(ell_max, mp_max)
        m : int
            Integer satisfying -ell <= m <= ell

        Returns
        -------
        i : int
            Index into Wigner H matrix arranged as described below

        See Also
        --------
        Hsize : Total size of the H matrix

        Notes
        -----
        This assumes that the Wigner H matrix is arranged as

            [
                H(ℓ, mp, m)
                for ℓ in range(ell_min, ell_max+1)
                for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
                for m in range(abs(mp), ℓ+1)
            ]

        """
        return WignerHindex(ell, mp, m, self.mp_max)

    def dindex(self, ell, mp, m):
        """Compute index into Wigner d matrix

        Parameters
        ----------
        ell : int
            Integer satisfying ell_min <= ell <= ell_max
        mp : int
            Integer satisfying -min(ell_max, mp_max) <= mp <= min(ell_max, mp_max)
        m : int
            Integer satisfying -ell <= m <= ell

        Returns
        -------
        i : int
            Index into Wigner d matrix arranged as described below

        See Also
        --------
        dsize : Total size of the d matrix

        Notes
        -----
        This assumes that the Wigner d matrix is arranged as

            [
                d(ℓ, mp, m)
                for ℓ in range(ell_min, ell_max+1)
                for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
                for m in range(-ℓ, ℓ+1)
            ]

        """
        return self.Dindex(ell, mp, m)

    def Dindex(self, ell, mp, m):
        """Compute index into Wigner 𝔇 matrix

        Parameters
        ----------
        ell : int
            Integer satisfying ell_min <= ell <= ell_max
        mp : int
            Integer satisfying -min(ell_max, mp_max) <= mp <= min(ell_max, mp_max)
        m : int
            Integer satisfying -ell <= m <= ell

        Returns
        -------
        i : int
            Index into Wigner 𝔇 matrix arranged as described below

        See Also
        --------
        Dsize : Total size of the 𝔇 matrix

        Notes
        -----
        This assumes that the Wigner 𝔇 matrix is arranged as

            [
                𝔇(ℓ, mp, m)
                for ℓ in range(ell_min, ell_max+1)
                for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
                for m in range(-ℓ, ℓ+1)
            ]

        """
        return WignerDindex(ell, mp, m, self.ell_min, self.mp_max)

    def Yindex(self, ell, m):
        """Compute index into array of mode weights

        Parameters
        ----------
        ell : int
            Integer satisfying ell_min <= ell <= ell_max
        m : int
            Integer satisfying -ell <= m <= ell

        Returns
        -------
        i : int
            Index of a particular element of the mode-weight array as described below

        See Also
        --------
        Ysize : Total size of array of mode weights

        Notes
        -----
        This assumes that the modes are arranged (with fixed s value) as

            [
                Y(s, ℓ, m)
                for ℓ in range(ell_min, ell_max+1)
                for m in range(-ℓ, ℓ+1)
            ]

        """
        return Yindex(self.ell_min, ell, m)

    def H(self, expiβ, Hwedge, Hv, Hextra):
        """Compute a quarter of the H matrix

        WARNING: The returned array will be a view into the `workspace` variable (see
        below for an explanation of that).  If you need to call this function again
        using the same workspace before extracting all information from the first call,
        you should use `numpy.copy` to make a separate copy of the result.

        Parameters
        ----------
        expiβ : array_like
            Values of exp(i*β) on which to evaluate the H matrix.

        Returns
        -------
        Hwedge : array
            This is a 1-dimensional array of floats; see below.
        workspace : array_like, optional
            A working array like the one returned by Wigner.new_workspace().  If not
            present, this object's default workspace will be used.  Note that it is not
            safe to use the same workspace on multiple threads.  Also see the WARNING
            above.

        See Also
        --------
        d : Compute the full Wigner d matrix
        D : Compute the full Wigner 𝔇 matrix
        rotate : Avoid computing the full 𝔇 matrix and rotate modes directly
        evaluate : Avoid computing the full 𝔇 matrix and evaluate modes directly

        Notes
        -----
        H is related to Wigner's (small) d via

            dₗⁿᵐ = ϵₙ ϵ₋ₘ Hₗⁿᵐ,

        where

                 ⎧ 1 for k≤0
            ϵₖ = ⎨
                 ⎩ (-1)ᵏ for k>0

        H has various advantages over d, including the fact that it can be efficiently
        and robustly valculated via recurrence relations, and the following symmetry
        relations:

            H^{m', m}_n(β) = H^{m, m'}_n(β)
            H^{m', m}_n(β) = H^{-m', -m}_n(β)
            H^{m', m}_n(β) = (-1)^{n+m+m'} H^{-m', m}_n(π - β)
            H^{m', m}_n(β) = (-1)^{m+m'} H^{m', m}_n(-β)

        Because of these symmetries, we only need to evaluate at most 1/4 of all the
        elements.

        """
        _step_1(Hwedge)
        _step_2(self.g, self.h, self.ell_max, self.mp_max, Hwedge, Hextra, Hv, expiβ)
        _step_3(self.a, self.b, self.ell_max, self.mp_max, Hwedge, Hextra, expiβ)
        _step_4(self.d, self.ell_max, self.mp_max, Hwedge, Hv)
        _step_5(self.d, self.ell_max, self.mp_max, Hwedge, Hv)
        return Hwedge

    def d(self, expiβ, out=None, workspace=None):
        """Compute Wigner's d matrix dˡₘₚ,ₘ(β)

        Parameters
        ----------
        expiβ : array_like
            Values of expi(i*β) on which to evaluate the d matrix.
        out : array_like, optional
            Array into which the d values should be written.  It should be an array of
            floats, with size `self.dsize`.  If not present, the array will be created.
            In either case, the array will also be returned.
        workspace : array_like, optional
            A working array like the one returned by Wigner.new_workspace().  If not
            present, this object's default workspace will be used.  Note that it is not
            safe to use the same workspace on multiple threads.

        Returns
        -------
        d : array
            This is a 1-dimensional array of floats; see below.

        See Also
        --------
        H : Compute a portion of the H matrix
        D : Compute the full Wigner 𝔇 matrix
        rotate : Avoid computing the full 𝔇 matrix and rotate modes directly
        evaluate : Avoid computing the full 𝔇 matrix and evaluate modes directly

        Notes
        -----
        This function is the preferred method of computing the d matrix for large ell
        values.  In particular, above ell≈32 standard formulas become completely
        unusable because of numerical instabilities and overflow.  This function uses
        stable recursion methods instead, and should be usable beyond ell≈1000.

        The result is returned in a 1-dimensional array ordered as

            [
                d(ell, mp, m, β)
                for ell in range(ell_max+1)
                for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
                for m in range(-ell, ell+1)
            ]

        """
        if workspace is not None:
            Hwedge, Hv, Hextra, zₐpowers, zᵧpowers, z = self._split_workspace(workspace)
        else:
            Hwedge, Hv, Hextra, zₐpowers, zᵧpowers, z = (
                self.Hwedge, self.Hv, self.Hextra, self.zₐpowers, self.zᵧpowers, self.z
            )

        Hwedge = self.H(expiβ, Hwedge, Hv, Hextra)
        d = out if out is not None else np.zeros(self.dsize, dtype=float)
        _fill_wigner_d(self.ell_min, self.ell_max, self.mp_max, d, Hwedge)
        return d

    def D(self, R, out=None, workspace=None):
        """Compute Wigner's 𝔇 matrix

        Parameters
        ----------
        R : array_like
            Array to be interpreted as a quaternionic array (thus its final dimension
            must have size 4), representing the rotations on which the 𝔇 matrix will be
            evaluated.
        out : array_like, optional
            Array into which the 𝔇 values should be written.  It should be an array of
            complex, with size `self.Dsize`.  If not present, the array will be
            created.  In either case, the array will also be returned.
        workspace : array_like, optional
            A working array like the one returned by Wigner.new_workspace().  If not
            present, this object's default workspace will be used.  Note that it is not
            safe to use the same workspace on multiple threads.

        Returns
        -------
        D : array
            This is a 1-dimensional array of complex; see below.

        See Also
        --------
        H : Compute a portion of the H matrix
        d : Compute the full Wigner d matrix
        rotate : Avoid computing the full 𝔇 matrix and rotate modes directly
        evaluate : Avoid computing the full 𝔇 matrix and evaluate modes directly

        Notes
        -----
        This function is the preferred method of computing the 𝔇 matrix for large ell
        values.  In particular, above ell≈32 standard formulas become completely
        unusable because of numerical instabilities and overflow.  This function uses
        stable recursion methods instead, and should be usable beyond ell≈1000.

        This function computes 𝔇ˡₘₚ,ₘ(R).  The result is returned in a 1-dimensional
        array ordered as

            [
                𝔇(ell, mp, m, R)
                for ell in range(ell_max+1)
                for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
                for m in range(-ell, ell+1)
            ]

        """
        if workspace is not None:
            Hwedge, Hv, Hextra, zₐpowers, zᵧpowers, z = self._split_workspace(workspace)
        else:
            Hwedge, Hv, Hextra, zₐpowers, zᵧpowers, z = (
                self.Hwedge, self.Hv, self.Hextra, self.zₐpowers, self.zᵧpowers, self.z
            )

        to_euler_phases(R, z)
        Hwedge = self.H(z[1], Hwedge, Hv, Hextra)
        𝔇 = out if out is not None else np.zeros(self.Dsize, dtype=complex)
        _complex_powers(z[0:1], self.ell_max, zₐpowers)
        _complex_powers(z[2:3], self.ell_max, zᵧpowers)
        _fill_wigner_D(self.ell_min, self.ell_max, self.mp_max, 𝔇, Hwedge, zₐpowers[0], zᵧpowers[0])
        return 𝔇

    def sYlm(self, s, R, out=None, workspace=None):
        """Evaluate (possibly spin-weighted) spherical harmonic

        Parameters
        ----------
        R : array_like
            Array to be interpreted as a quaternionic array (thus its final dimension
            must have size 4), representing the rotations on which the sYlm will be
            evaluated.
        out : array_like, optional
            Array into which the d values should be written.  It should be an array of
            complex, with size `self.Ysize`.  If not present, the array will be
            created.  In either case, the array will also be returned.
        workspace : array_like, optional
            A working array like the one returned by Wigner.new_workspace().  If not
            present, this object's default workspace will be used.  Note that it is not
            safe to use the same workspace on multiple threads.

        Returns
        -------
        Y : array
            This is a 1-dimensional array of complex; see below.

        See Also
        --------
        H : Compute a portion of the H matrix
        d : Compute the full Wigner d matrix
        D : Compute the full Wigner 𝔇 matrix
        rotate : Avoid computing the full 𝔇 matrix and rotate modes directly
        evaluate : Avoid computing the full 𝔇 matrix and evaluate modes directly

        Notes
        -----
        The spherical harmonics of spin weight s are related to the 𝔇 matrix as

            ₛYₗₘ(R) = (-1)ˢ √((2ℓ+1)/(4π)) 𝔇ˡₘ₋ₛ(R)
                   = (-1)ˢ √((2ℓ+1)/(4π)) 𝔇̄ˡ₋ₛₘ(R̄)

        This function is the preferred method of computing the sYlm for large ell
        values.  In particular, above ell≈32 standard formulas become completely
        unusable because of numerical instabilities and overflow.  This function uses
        stable recursion methods instead, and should be usable beyond ell≈1000.

        This function computes ₛYₗₘ(R).  The result is returned in a 1-dimensional
        array ordered as

            [
                Y(s, ell, m, R)
                for ell in range(ell_max+1)
                for m in range(-ell, ell+1)
            ]

        """
        if abs(s) > self.mp_max:
            raise ValueError(
                f"This object has mp_max={self.mp_max}, which is not "
                f"sufficient to compute sYlm values for spin weight s={s}"
            )
        if out is not None and out.shape != (self.Ysize,):
            raise ValueError(
                f"Given output array has shape {out.shape}; it should be {(self.Ysize,)}"
            )

        if workspace is not None:
            Hwedge, Hv, Hextra, zₐpowers, zᵧpowers, z = self._split_workspace(workspace)
        else:
            Hwedge, Hv, Hextra, zₐpowers, zᵧpowers, z = (
                self.Hwedge, self.Hv, self.Hextra, self.zₐpowers, self.zᵧpowers, self.z
            )

        to_euler_phases(R, z)

        Hwedge = self.H(z[1], Hwedge, Hv, Hextra)
        Y = out if out is not None else np.zeros(self.Ysize, dtype=complex)
        _complex_powers(z[0:1], self.ell_max, zₐpowers)
        zᵧpower = z[2]**abs(s)
        _fill_sYlm(self.ell_min, self.ell_max, self.mp_max, s, Y, Hwedge, zₐpowers[0], zᵧpower)
        return Y

    def rotate(self, modes, R, out=None, workspace=None):
        """Rotate Modes object

        Parameters
        ----------
        modes : Modes object
        R : quaternionic.array
            Unit quaternion representing the rotation of the frame in which the mode
            weights are measured.
        out : array_like, optional
            Array into which the rotated mode weights should be written.  It should be
            an array of complex with the same shape as `modes`.  If not present, the
            array will be created.  In either case, the array will also be returned.
        workspace : array_like, optional
            A working array like the one returned by Wigner.new_workspace().  If not
            present, this object's default workspace will be used.  Note that it is not
            safe to use the same workspace on multiple threads.

        Returns
        -------
        rotated_modes : array_like
            This array holds the complex function values.  Its shape is
            modes.shape[:-1]+R.shape[:-1].

        """
        ell_min = modes.ell_min
        ell_max = modes.ell_max
        spin_weight = modes.spin_weight

        if ell_max > self.ell_max:
            raise ValueError(
                f"This object has ell_max={self.ell_max}, which is not "
                f"sufficient for the input modes object with ell_max={ell_max}"
            )

        # Reinterpret inputs as 2-d np.arrays
        mode_weights = modes.ndarray.reshape((-1, modes.shape[-1]))
        R = quaternionic.array(R)

        # Construct storage space
        rotated_mode_weights = (
            out
            if out is not None
            else np.zeros_like(mode_weights)
        )

        D = self.D(R, workspace)

        _rotate(
            mode_weights, rotated_mode_weights,
            self.ell_min, self.ell_max, self.mp_max,
            ell_min, ell_max, spin_weight,
            D
        )

        return type(modes)(
            rotated_mode_weights.reshape(modes.shape),
            **modes._metadata
        )


    def evaluate(self, modes, R, out=None, workspace=None):
        """Evaluate Modes object as function of rotations

        Parameters
        ----------
        modes : Modes object
        R : quaternionic.array
            Arbitrarily shaped array of quaternions.  All modes in the input will be
            evaluated on each of these quaternions.  Note that it is fairly standard to
            construct these quaternions from spherical coordinates, as with the
            function `quaternionic.array.from_spherical_coordinates`.
        out : array_like, optional
            Array into which the function values should be written.  It should be an
            array of complex, with shape `modes.shape[:-1]+R.shape[:-1]`.  If not
            present, the array will be created.  In either case, the array will also be
            returned.
        workspace : array_like, optional
            A working array like the one returned by Wigner.new_workspace().  If not
            present, this object's default workspace will be used.  Note that it is not
            safe to use the same workspace on multiple threads.

        Returns
        -------
        f : array_like
            This array holds the complex function values.  Its shape is
            modes.shape[:-1]+R.shape[:-1].

        """
        spin_weight = modes.spin_weight
        ell_min = modes.ell_min
        ell_max = modes.ell_max

        if abs(spin_weight) > self.mp_max:
            raise ValueError(
                f"This object has mp_max={self.mp_max}, which is not "
                f"sufficient to compute sYlm values for spin weight s={spin_weight}"
            )

        if max(abs(spin_weight), ell_min) < self.ell_min:
            raise ValueError(
                f"This object has ell_min={self.ell_min}, which is not "
                f"sufficient for the requested spin weight s={spin_weight} and ell_min={ell_min}"
            )

        if ell_max > self.ell_max:
            raise ValueError(
                f"This object has ell_max={self.ell_max}, which is not "
                f"sufficient for the input modes object with ell_max={ell_max}"
            )

        # Reinterpret inputs as 2-d np.arrays
        mode_weights = modes.ndarray.reshape((-1, modes.shape[-1]))
        quaternions = quaternionic.array(R).ndarray.reshape((-1, 4))

        # Construct storage space
        # z = np.zeros(3, dtype=complex)
        Y = np.zeros(self.Ysize, dtype=complex)
        function_values = (
            out
            if out is not None
            else np.zeros(mode_weights.shape[:-1] + quaternions.shape[:-1], dtype=complex)
        )

        # Loop over all input quaternions
        for i_R in range(quaternions.shape[0]):
            self.sYlm(spin_weight, quaternions[i_R], out=Y)
            np.matmul(mode_weights, Y, out=function_values[..., i_R])

        return function_values.reshape(modes.shape[:-1] + R.shape[:-1])


@jit
def _fill_wigner_d(ell_min, ell_max, mp_max, d, Hwedge):
    """Helper function for Wigner.d"""
    for ell in range(ell_min, ell_max+1):
        for mp in range(-ell, ell+1):
            for m in range(-ell, ell+1):
                i_d = WignerDindex(ell, mp, m, ell_min)
                i_H = WignerHindex(ell, mp, m, mp_max)
                d[i_d] = ϵ(mp) * ϵ(-m) * Hwedge[i_H]


@jit
def _fill_wigner_D(ell_min, ell_max, mp_max, 𝔇, Hwedge, zₐpowers, zᵧpowers):
    """Helper function for Wigner.D"""
    # 𝔇ˡₘₚ,ₘ(R) = dˡₘₚ,ₘ(R) exp[iϕₐ(m-mp)+iϕₛ(m+mp)] = dˡₘₚ,ₘ(R) exp[i(ϕₛ+ϕₐ)m+i(ϕₛ-ϕₐ)mp]
    # exp[iϕₛ] = R̂ₛ = hat(R[0] + 1j * R[3]) = zp
    # exp[iϕₐ] = R̂ₐ = hat(R[2] + 1j * R[1]) = zm.conjugate()
    # exp[i(ϕₛ+ϕₐ)] = zp * zm.conjugate() = z[2] = zᵧ
    # exp[i(ϕₛ-ϕₐ)] = zp * zm = z[0] = zₐ
    for ell in range(ell_min, ell_max+1):
        for mp in range(-ell, 0):
            i_D = WignerDindex(ell, mp, -ell, ell_min)
            for m in range(-ell, 0):
                i_H = WignerHindex(ell, mp, m, mp_max)
                𝔇[i_D] = ϵ(mp) * ϵ(-m) * Hwedge[i_H] * zᵧpowers[-m].conjugate() * zₐpowers[-mp].conjugate()
                i_D += 1
            for m in range(0, ell+1):
                i_H = WignerHindex(ell, mp, m, mp_max)
                𝔇[i_D] = ϵ(mp) * ϵ(-m) * Hwedge[i_H] * zᵧpowers[m] * zₐpowers[-mp].conjugate()
                i_D += 1
        for mp in range(0, ell+1):
            i_D = WignerDindex(ell, mp, -ell, ell_min)
            for m in range(-ell, 0):
                i_H = WignerHindex(ell, mp, m, mp_max)
                𝔇[i_D] = ϵ(mp) * ϵ(-m) * Hwedge[i_H] * zᵧpowers[-m].conjugate() * zₐpowers[mp]
                i_D += 1
            for m in range(0, ell+1):
                i_H = WignerHindex(ell, mp, m, mp_max)
                𝔇[i_D] = ϵ(mp) * ϵ(-m) * Hwedge[i_H] * zᵧpowers[m] * zₐpowers[mp]
                i_D += 1


@jit
def _fill_sYlm(ell_min, ell_max, mp_max, s, Y, Hwedge, zₐpowers, zᵧpower):
    """Helper function for Wigner.sYlm"""
    #  ₛYₗₘ(R) = (-1)ˢ √((2ℓ+1)/(4π)) 𝔇ˡₘ₋ₛ(R)
    ell0 = max(abs(s), ell_min)
    Y[:Yindex(ell0, -ell0, ell_min)] = 0.0
    if s >= 0:
        c1 = zᵧpower.conjugate()
        for ell in range(ell0, ell_max+1):
            i_Y = Yindex(ell, -ell, ell_min)
            c2 = c1 * np.sqrt((2 * ell + 1) * inverse_4pi)
            for m in range(-ell, 0):
                i_H = WignerHindex(ell, m, -s, mp_max)
                Y[i_Y] = c2 * Hwedge[i_H] * zₐpowers[-m].conjugate()
                i_Y += 1
            for m in range(0, ell+1):
                i_H = WignerHindex(ell, m, -s, mp_max)
                Y[i_Y] = c2 * ϵ(m) * Hwedge[i_H] * zₐpowers[m]
                i_Y += 1
    else:  # s < 0
        c1 = (-1)**s * zᵧpower
        for ell in range(ell0, ell_max+1):
            i_Y = Yindex(ell, -ell, ell_min)
            c2 = c1 * np.sqrt((2 * ell + 1) * inverse_4pi)
            for m in range(-ell, 0):
                i_H = WignerHindex(ell, m, -s, mp_max)
                Y[i_Y] = c2 * Hwedge[i_H] * zₐpowers[-m].conjugate()
                i_Y += 1
            for m in range(0, ell+1):
                i_H = WignerHindex(ell, m, -s, mp_max)
                Y[i_Y] = c2 * ϵ(m) * Hwedge[i_H] * zₐpowers[m]
                i_Y += 1


@jit
def _rotate(fₗₘ, fₗₙ, ell_min_w, ell_max_w, mp_max_w, ell_min_m, ell_max_m, spin_weight_m, 𝔇):
    """Helper function for Wigner.rotate"""
    for ell in range(max(abs(spin_weight_m), ell_min_m), ell_max_m+1):
        i1 = Yindex(ell, -ell, ell_min_m)
        i2 = Yindex(ell, ell, ell_min_m) + 1
        𝔇ˡ = 𝔇[WignerDindex(ell, -ell, -ell, ell_min_w):WignerDindex(ell, ell, ell, ell_min_w)+1]
        𝔇ˡ = 𝔇ˡ.reshape(2*ell+1, 2*ell+1)
        for i in range(fₗₙ.shape[0]):
            fₗₙ[i, i1:i2] = fₗₘ[i, i1:i2] @ 𝔇ˡ
