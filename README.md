[![Test Status](https://github.com/moble/spherical/workflows/tests/badge.svg)](https://github.com/moble/spherical/actions)
[![Test Coverage](https://codecov.io/gh/moble/spherical/branch/main/graph/badge.svg?token=zIw5m2Gs68)](https://codecov.io/gh/moble/spherical)
[![Documentation Status](https://readthedocs.org/projects/spherical/badge/?version=main)](https://spherical.readthedocs.io/en/main/?badge=main)
[![PyPI Version](https://img.shields.io/pypi/v/spherical?color=)](https://pypi.org/project/spherical/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/spherical.svg?color=)](https://anaconda.org/conda-forge/spherical)
[![DOI](https://zenodo.org/badge/295054260.svg)](https://zenodo.org/badge/latestdoi/295054260)


# Spherical Functions

Python/numba package for evaluating and transforming Wigner's 𝔇 matrices,
Wigner's 3-j symbols, and spin-weighted (and scalar) spherical harmonics.
These functions are evaluated directly in terms of quaternions, as well as in
the more standard forms of spherical coordinates and Euler
angles.<sup>[1](#1-euler-angles-are-awful)</sup>

These quantities are computed using recursion relations, which makes it
possible to compute to very high ℓ values.  Unlike direct evaluation of
individual elements, which will generally cause overflow or underflow beyond
ℓ≈30, these recursion relations should be accurate for ℓ values beyond 1000.

The conventions for this package are described in detail on
[this page](http://moble.github.io/spherical/).

## Installation

Because this package is pure python code, installation is very simple.  In
particular, with a reasonably modern installation, you can just run a command
like

```bash
conda install -c conda-forge spherical
```

or

```bash
python -m pip install spherical
```

Either of these will download and install the package.


## Usage

#### Functions of angles or rotations

Currently, due to the nature of recursions, this module does not allow
calculation of individual elements, but returns ranges of results.  For
example, when computing Wigner's 𝔇 matrix, all matrices up to a given ℓ will be
returned; when evaluating a spin-weighted spherical harmonic, all harmonics up
to a given ℓ will be returned.  Fortunately, this is usually what is required
in any case.

To calculate Wigner's d or 𝔇 matrix or spin-weighted spherical harmonics, first
construct a `Wigner` object.

```python
import quaternionic
import spherical
ell_max = 16  # Use the largest ℓ value you expect to need
wigner = spherical.Wigner(ell_max)
```

This module takes input as quaternions.  The `quaternionic` module has [various
ways of constructing
quaternions](https://quaternionic.readthedocs.io/en/latest/#rotations),
including direct construction or conversion from rotation matrices, axis-angle
representation, Euler angles,<sup>[1](#euler-angles-are-awful)</sup> or
spherical coordinates, among others:

```python
R = quaternionic.array([1, 2, 3, 4]).normalized
R = quaternionic.array.from_axis_angle(vec)
R = quaternionic.array.from_euler_angles(alpha, beta, gamma)
R = quaternionic.array.from_spherical_coordinates(theta, phi)
```

Mode weights can be rotated as

```python
wigner.rotate(modes, R)
```

or evaluated as

```python
wigner.evaluate(modes, R)
```

We can compute the 𝔇 matrix as

```python
D = wigner.D(R)
```

which can be indexed as

```python
D[wigner.Dindex(ell, mp, m)]
```

or we can compute the spin-weighted spherical harmonics as

```python
Y = wigner.sYlm(s, R)
```

which can be indexed as

```python
Y[wigner.Yindex(ell, m)]
```

Note that, if relevant, it is probably more efficient to use the `rotate` and
`evaluate` methods than to use `D` or `Y`.



#### Clebsch-Gordan and 3-j symbols

It is possible to compute individual values of the 3-j or Clebsch-Gordan
symbols:

```python
w3j = spherical.Wigner3j(j_1, j_2, j_3, m_1, m_2, m_3)
cg = spherical.clebsch_gordan(j_1, m_1, j_2, m_2, j_3, m_3)
```

However, when more than one element is needed (as is typically the case), it is
much more efficient to compute a range of values:

```python
calc3j = spherical.Wigner3jCalculator(j2_max, j3_max)
w3j = calc3j.calculate(j2, j3, m2, m3)
```


## Acknowledgments

I very much appreciate Barry Wardell's help in sorting out the relationships
between my conventions and those of other people and software packages
(especially Mathematica's crazy conventions).

This code is, of course, hosted on github.  Because it is an open-source
project, the hosting is free, and all the wonderful features of github are
available, including free wiki space and web page hosting, pull requests, a
nice interface to the git logs, etc.

The work of creating this code was supported in part by the Sherman Fairchild
Foundation and by NSF Grants No. PHY-1306125 and AST-1333129.


<br/>

---

###### <sup>1</sup> Euler angles are awful

Euler angles are pretty much
[the worst things ever](http://moble.github.io/spherical/#euler-angles)
and it makes me feel bad even supporting them.  Quaternions are
faster, more accurate, basically free of singularities, more
intuitive, and generally easier to understand.  You can work entirely
without Euler angles (I certainly do).  You absolutely never need
them.  But if you're so old fashioned that you really can't give them
up, they are fully supported.
