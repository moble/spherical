# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

### NOTE: The functions in this file are intended purely for inclusion in the Modes class.  In
### particular, they assume that the first argument, `self` is an instance of Modes.  They should
### probably not be used outside of that class.

import copy
import numpy as np


def Lsquared(self):
    """Total angular-momentum operator

    This is the standard L^2 operator, familiar from basic physics, extended to
    work with SWSHs.  It is also known as the Casimir operator, and is equal to

        L^2 = LxLx + LyLy + LzLz = 0.5(L+L- + L-L+) + LzLz

    Note that these are the left Lie derivatives, but L^2 = R^2, where R is the
    right Lie derivative.

    The left Lie derivative of a function f(Q) over the unit quaternions with
    respect to a generator of rotation g is defined as

        Lg(f){Q} = -0.5j df{exp(t*g) * Q} / dt |t=0

    This agrees with the usual angular-momentum operators familiar from
    spherical-harmonic theory, and reduces to it when the function has spin weight
    0, but also applies to functions of general spin weight.  In terms of the
    SWSHs, we can write the action of Lsquared as

        Lsquared {s}Y{l,m} = l * (l+1) * {s}Y{l,m}

    """
    import numpy as np
    d = self.copy()
    s = self.view(np.ndarray)
    for ell in range(abs(self.s), self.ell_max+1):
        d[..., d.index(ell, -ell):d.index(ell, ell)+1] = (
            ell * (ell+1)
            * s[..., d.index(ell, -ell):d.index(ell, ell)+1]
        )
    return d


def Lz(self):
    """Left Lie derivative with respect to rotation about z

    The left Lie derivative of a function f(Q) over the unit quaternions with
    respect to a generator of rotation g is defined as

        Lg(f){Q} = -0.5j df{exp(t*g) * Q} / dt |t=0

    This agrees with the usual angular-momentum operators familiar from
    spherical-harmonic theory, and reduces to it when the function has spin weight
    0, but also applies to functions of general spin weight.  In terms of the
    SWSHs, we can write the action of Lz as

        Lz {s}Y{l,m} = m * {s}Y{l,m}

    """
    import numpy as np
    d = self.copy()
    s = self.view(np.ndarray)
    for ell in range(abs(self.s), self.ell_max+1):
        for m in range(-ell, ell+1):
            d[..., d.index(ell, m)] = m * s[..., self.index(ell, m)]
    return d


def Lplus(self):
    """Raising operator for Lz

    We define Lplus to be the raising operator for the left Lie derivative with
    respect to rotation about z, Lz.  By definition, this means that [Lz, Lplus] =
    Lplus, which allows us to derive Lplus = Lx + 1j * Ly.  In terms of the SWSHs,
    we can write the action of Lplus as

        Lplus {s}Y{l,m} = sqrt((l-m)*(l+m+1)) {s}Y{l,m+1}

    Consequently, the modes of a function are affected as

        {Lplus f}{s, l, m} = sqrt((l+m)*(l-m+1)) * f{s,l,m-1}

    """
    # sYlm = (-1)**s sqrt((2ell+1)/(4pi)) D{l,m,-s}
    # Lplus {s}Y{l,m} = (-1)**s sqrt((2ell+1)/(4pi)) Lplus D{l,m,-s}
    #                 = (-1)**s sqrt((2ell+1)/(4pi)) sqrt((l-m)(l+m+1)) D{l,m+1,-s}
    #                 = sqrt((l-m)(l+m+1)) (-1)**s sqrt((2ell+1)/(4pi)) D{l,m+1,-s}
    #                 = sqrt((l-m)(l+m+1)) {s}Y{l,m+1}
    # {L+ f}{s', l', m'}
    #    = integral(L+ f {s'}Ybar{l',m'})  # Integral over rotation group
    #    = integral(L+ sum(f{s,l,m}{s}Y{l,m}) {s'}Ybar{l',m'})
    #    = sum(f{s,l,m} integral(L+ {s}Y{l,m} {s'}Ybar{l',m'}))
    #    = sum(f{s,l,m} integral(sqrt((l-m)(l+m+1)) {s}Y{l,m+1} {s'}Ybar{l',m'}))
    #    = sum(sqrt((l-m)(l+m+1)) f{s,l,m} integral({s}Y{l,m+1} {s'}Ybar{l',m'}))
    #    = sum(sqrt((l-m)(l+m+1)) f{s,l,m} delta{s, s'} delta{m+1, m'} delta{l, l'}
    #    = sqrt((l'-(m'-1))(l'+(m'-1)+1)) f{s',l',m'-1}
    #    = sqrt((l'+m')(l'-m'+1)) f{s',l',m'-1}
    # {L+ f}{s, l, m} = sqrt((l+m)(l-m+1)) f{s,l,m-1}
    import math
    import numpy as np
    d = np.zeros_like(self)
    s = self.view(np.ndarray)
    for ell in range(abs(self.s), self.ell_max+1):
        d[..., self.index(ell, -ell)] = 0.0
        for m in range(-ell+1, ell+1):
            d[..., d.index(ell, m)] = math.sqrt((ell+m)*(ell-m+1)) * s[..., self.index(ell, m-1)]
    return d


def Lminus(self):
    """Lowering operator for Lz

    We define Lminus to be the lowering operator for the left Lie derivative with
    respect to rotation about z, Lz.  By definition, this means that [Lz, Lminus] =
    -Lminus, which allows us to derive Lminus = Lx - 1j * Ly.  In terms of the
    SWSHs, we can write the action of Lminus as

        Lminus {s}Y{l,m} = sqrt((l+m)*(l-m+1)) * {s}Y{l,m-1}

    Consequently, the modes of a function are affected as

        {Lminus f}{s, l, m} = sqrt((l-m)*(l+m+1)) * f{s,l,m+1}

    """
    # sYlm = (-1)**s sqrt((2ell+1)/(4pi)) D{l,m,-s}
    # Lminus {s}Y{l,m} = (-1)**s sqrt((2ell+1)/(4pi)) Lminus D{l,m,-s}
    #                  = (-1)**s sqrt((2ell+1)/(4pi)) sqrt((l+m)(l-m+1)) D{l,m-1,-s}
    #                  = sqrt((l+m)(l-m+1)) (-1)**s sqrt((2ell+1)/(4pi)) D{l,m-1,-s}
    #                  = sqrt((l+m)(l-m+1)) {s}Y{l,m-1}
    # {L- f}{s', l', m'}
    #    = integral(L- f {s'}Ybar{l',m'})  # Integral over rotation group
    #    = integral(L- sum(f{s,l,m}{s}Y{l,m}) {s'}Ybar{l',m'})
    #    = sum(f{s,l,m} integral(L- {s}Y{l,m} {s'}Ybar{l',m'}))
    #    = sum(f{s,l,m} integral(sqrt((l+m)(l-m+1)) {s}Y{l,m-1} {s'}Ybar{l',m'}))
    #    = sum(sqrt((l+m)(l-m+1)) f{s,l,m} integral({s}Y{l,m-1} {s'}Ybar{l',m'}))
    #    = sum(sqrt((l+m)(l-m+1)) f{s,l,m} delta{s, s'} delta{m-1, m'} delta{l, l'}
    #    = sqrt((l'+(m'+1))(l'-(m'+1)+1)) f{s',l',m'+1}
    #    = sqrt((l'-m')(l'+m'+1)) f{s',l',m'+1}
    # {L- f}{s, l, m} = sqrt((l-m)(l+m+1)) f{s,l,m+1}
    import math
    import numpy as np
    d = np.zeros_like(self)
    s = self.view(np.ndarray)
    for ell in range(abs(self.s), self.ell_max+1):
        for m in range(-ell, ell):
            d[..., self.index(ell, m)] = math.sqrt((ell-m)*(ell+m+1)) * s[..., self.index(ell, m+1)]
        d[..., self.index(ell, ell)] = 0.0
    return d


def Rsquared(self):
    """Total angular-momentum operator

    This is the R^2 operator, much like the L^2 operator familiar from basic
    physics, but using the right Lie derivative, and extended to work with SWSHs.
    It is also known as the Casimir operator, and is equal to

        R^2 = RxRx + RyRy + RzRz = 0.5(R+R- + R-R+) + RzRz

    Note that these are the right Lie derivatives, but L^2 = R^2, where L is the
    left Lie derivative.

    The right Lie derivative of a function f(Q) over the unit quaternions with
    respect to a generator of rotation g is defined as

        Rg(f){Q} = -0.5j df{Q * exp(t*g)} / dt |t=0

    This is unlike the usual angular-momentum operators Lz, etc., familiar from
    spherical-harmonic theory because the exponential is on the right-hand side of
    the argument.  This operator is less common in physics because it represents
    the dependence of the function on the choice of frame.  In terms of the SWSHs,
    we can write the action of Rsquared as

        Rsquared {s}Y{l,m} = l * (l+1) * {s}Y{l,m}

    """
    return self.Lsquared()


def Rz(self):
    """Right Lie derivative with respect to rotation about z

    The right Lie derivative of a function f(Q) over the unit quaternions with
    respect to a generator of rotation g is defined as

        Rg(f){Q} = -0.5j df{Q * exp(t*g)} / dt |t=0

    This is unlike the usual angular-momentum operators Lz, etc., familiar from
    spherical-harmonic theory because the exponential is on the right-hand side of
    the argument.  This operator is less common in physics because it represents
    the dependence of the function on the choice of frame.  In terms of the SWSHs,
    we can write the action of Rz as

        Rz {s}Y{l,m} = -s * {s}Y{l,m}

    Equivalently, the modes of a function are affected as

        {Rz f} {s,l,m} = -s * f{s,l,m}

    Note the unfortunate sign of `s`, which seems to be opposite to what we expect,
    and arises from the choice of definition of `s` in the original paper by Newman
    and Penrose.

    """
    # {Rzf}{s', l', m'}
    #    = integral(Rz f {s'}Ybar{l',m'})  # Integral over rotation group
    #    = integral(Rz sum(f{s,l,m}{s}Y{l,m}) {s'}Ybar{l',m'})
    #    = sum(f{s,l,m} integral(Rz {s}Y{l,m} {s'}Ybar{l',m'}))
    #    = sum(f{s,l,m} integral(-s {s}Y{l,m} {s'}Ybar{l',m'}))
    #    = sum(-s f{s,l,m} integral({s}Y{l,m} {s'}Ybar{l',m'}))
    #    = sum(-s f{s,l,m} delta{s, s'} delta{m, m'} delta{l, l'}
    #    = -s f{s',l',m'}
    # {Rzf}{s, l, m} = -s f{s,l,m}
    import numpy as np
    return type(self)(-self.s * self.view(np.ndarray), **self._metadata)


def Rplus(self):
    """Raising operator for Rz

    We define Rplus to be the raising operator for the right Lie derivative with
    respect to rotation about z, Rz.  By definition, this means that [Rz, Rplus] =
    Rplus, which allows us to derive Rplus = Rx - 1j * Ry.  In terms of the SWSHs,
    we can write the action of Rplus as

        Rplus {s}Y{l,m} = sqrt((l+s)(l-s+1)) {s-1}Y{l,m}

    Consequently, the modes of a function are affected as

        {Rplus f} {s,l,m} = sqrt((l-s)(l+s+1)) f{s+1,l,m}

    Again, because of the unfortunate choice of the sign of `s` made in the
    original paper by Newman and Penrose, this looks like a lowering operator for
    `s`.  But it really is a raising operator for Rz, and raises the eigenvalue of
    the corresponding Wigner matrix - though that lowers the value of `s`.

    """
    # sYlm = (-1)**s sqrt((2ell+1)/(4pi)) D{l,m,-s}
    # Rplus {s}Y{l,m} = (-1)**s sqrt((2ell+1)/(4pi)) Rplus D{l,m,-s}
    #                 = (-1)**s sqrt((2ell+1)/(4pi)) sqrt((l+s)(l-s+1)) D{l,m,-s+1}
    #                 = sqrt((l+s)(l-s+1)) (-1)**s sqrt((2ell+1)/(4pi)) D{l,m,-s+1}
    #                 = sqrt((l+s)(l-s+1)) {s-1}Y{l,m}
    # {R+f}{s', l', m'}
    #    = integral(R+ f {s'}Ybar{l',m'})  # Integral over rotation group
    #    = integral(R+ sum(f{s,l,m}{s}Y{l,m}) {s'}Ybar{l',m'})
    #    = sum(f{s,l,m} integral(R+ {s}Y{l,m} {s'}Ybar{l',m'}))
    #    = sum(f{s,l,m} integral(sqrt((l+s)(l-s+1)) {s-1}Y{l,m} {s'}Ybar{l',m'}))
    #    = sum(sqrt((l+s)(l-s+1)) f{s,l,m} integral({s-1}Y{l,m} {s'}Ybar{l',m'}))
    #    = sum(sqrt((l+s)(l-s+1)) f{s,l,m} delta{s-1, s'} delta{m, m'} delta{l, l'}
    #    = sqrt((l'+s'+1)(l'-(s'+1)+1) f{s'+1,l',m'}
    # {R+f}{s, l, m} = sqrt((l-s)(l+s+1)) f{s+1,l,m}
    import math
    import numpy as np
    metadata = copy.copy(self._metadata)
    metadata['spin_weight'] = self.s-1
    metadata['ell_min'] = min(abs(self.s-1), self.ell_min)
    metadata['ell_max'] = self.ell_max
    d = type(self)(np.zeros_like(self.view(np.ndarray)), **metadata)
    s = self.view(np.ndarray)
    for ell in range(max(abs(d.s), abs(self.s)), d.ell_max+1):
        if ell >= self.ell_min:
            d[..., d.index(ell, -ell):d.index(ell, ell)+1] = (
                math.sqrt((ell-d.s)*(ell+d.s+1))
                * s[..., self.index(ell, -ell):self.index(ell, ell)+1]
            )
    return d


def Rminus(self):
    """Lowering operator for Rz

    We define Rminus to be the lowering operator for the right Lie derivative with
    respect to rotation about z, Rz.  By definition, this means that [Rz, Rminus] =
    -Rminus, which allows us to derive Rminus = Rx + 1j * Ry.  In terms of the
    SWSHs, we can write the action of Rminus as

        Rminus {s}Y{l,m} = sqrt((l-s)(l+s+1)) {s+1}Y{l,m}

    Consequently, the modes of a function are affected as

        {Rminus f} {s,l,m} = sqrt((l+s)(l-s+1)) f{s-1,l,m}

    Again, because of the unfortunate choice of the sign of `s` made in the
    original paper by Newman and Penrose, this looks like a raising operator for
    `s`.  But it really is a lowering operator for Rz, and lowers the eigenvalue of
    the corresponding Wigner matrix - though that raises the value of `s`.

    """
    # sYlm = (-1)**s sqrt((2ell+1)/(4pi)) D{l,m,-s}
    # Rminus {s}Y{l,m} = (-1)**s sqrt((2ell+1)/(4pi)) Rminus D{l,m,-s}
    #                  = (-1)**s sqrt((2ell+1)/(4pi)) sqrt((l-s)(l+s+1)) D{l,m,-s-1}
    #                  = sqrt((l-s)(l+s+1)) (-1)**s sqrt((2ell+1)/(4pi)) D{l,m,-s-1}
    #                  = sqrt((l-s)(l+s+1)) {s+1}Y{l,m}
    # {R- f}{s', l', m'}
    #    = integral(R- f {s'}Ybar{l',m'})  # Integral over rotation group
    #    = integral(R- sum(f{s,l,m}{s}Y{l,m}) {s'}Ybar{l',m'})
    #    = sum(f{s,l,m} integral(R- {s}Y{l,m} {s'}Ybar{l',m'}))
    #    = sum(f{s,l,m} integral(sqrt((l-s)(l+s+1)) {s+1}Y{l,m} {s'}Ybar{l',m'}))
    #    = sum(sqrt((l-s)(l+s+1)) f{s,l,m} integral({s+1}Y{l,m} {s'}Ybar{l',m'}))
    #    = sum(sqrt((l-s)(l+s+1)) f{s,l,m} delta{s+1, s'} delta{m, m'} delta{l, l'}
    #    = sqrt((l'-(s'-1))(l'+(s'-1)+1)) f{s'-1,l',m'}
    #    = sqrt((l'-s'+1)(l'+s')) f{s'-1,l',m'}
    # {R- f}{s, l, m} = sqrt((l+s)(l-s+1)) f{s-1,l,m}
    import math
    import numpy as np
    metadata = copy.copy(self._metadata)
    metadata['spin_weight'] = self.s+1
    metadata['ell_min'] = min(abs(self.s+1), self.ell_min)
    metadata['ell_max'] = self.ell_max
    d = type(self)(np.zeros_like(self.view(np.ndarray)), **metadata)
    s = self.view(np.ndarray)
    for ell in range(max(abs(d.s), abs(self.s)), d.ell_max+1):
        if ell >= self.ell_min:
            d[..., d.index(ell, -ell):d.index(ell, ell)+1] = (
                math.sqrt((ell+d.s)*(ell-d.s+1))
                * s[..., self.index(ell, -ell):self.index(ell, ell)+1]
            )
    return d


@property
def eth(self):
    """Spin-raising derivative operator defined by Newman and Penrose

    Note that this is identical to `Rminus`.

    By definition, the spin-raising operator satisfies [S, eth] = eth (where S is
    the spin operator, which just multiplies the function by its spin).  In terms
    of the SWSHs, we can write the action of eth as

        eth {s}Y{l,m} = sqrt((l-s)(l+s+1)) {s+1}Y{l,m}

    Consequently, the modes of a function are affected as

        {eth f} {s,l,m} = sqrt((l+s)(l-s+1)) f{s-1,l,m}

    """
    return self.Rminus()


@property
def ethbar(self):
    """Spin-lowering conjugate-derivative operator defined by Newman and Penrose

    Note that this is identical to `Rplus`, except with an opposite sign, to follow
    the conventions laid out by Newman and Penrose.

    By definition, the spin-lowering operator satisfies [S, ethbar] = -ethbar
    (where S is the spin operator, which just multiplies the function by its spin).
    In terms of the SWSHs, we can write the action of ethbar as

        ethbar {s}Y{l,m} = -sqrt((l+s)(l-s+1)) {s-1}Y{l,m}

    Consequently, the modes of a function are affected as

        {ethbar f} {s,l,m} = -sqrt((l-s)(l+s+1)) f{s+1,l,m}

    """
    return -self.Rplus()
