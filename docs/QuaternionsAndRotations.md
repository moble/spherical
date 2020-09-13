## Quaternions, rotations, spherical coordinates

A unit quaternion (or "rotor") $\rotor{R}$ can rotate a vector
$\vec{v}$ into a new vector $\Rotated{\vec{v}}$ according to the
formula
\begin{equation\*}
  \Rotated{\vec{v}} = \rotor{R}\, \vec{v}\, \rotor{R}^{-1}.
\end{equation\*}
In principle, a unit quaternion obeys $\co{\rotor{R}} =
\rotor{R}^{-1}$.  In practice, however, there are cases where the
system is (slightly slower, but) more stable numerically if the
explicit inversion is used.  And since the inversion is such a simple
operation, we just use it.

The Wigner $\mathfrak{D}$ matrices can be derived directly in terms of
components of the unit rotor corresponding to the rotation of which
$\mathfrak{D}$ is a representation.  That is, we don't need to use
Euler angles or rotation matrices at all.  In fact, the resulting
expressions are at least as simple as those using Euler angles, but
are faster and more accurate to compute.  The derivation is shown on
[this page](WignerDMatrices.html).

![Spherical-coordinate system; By Andeggs, via Wikimedia Commons]({{ site.url }}/spherical/images/3D_Spherical_Coords.svg){: style="float:right;height:200px"}

When necessary to make contact with previous literature, we will use
the standard spherical coordinates $(\vartheta, \varphi)$, in the
physics convention.  To explain this, we first establish our
right-handed basis vectors: $(\basis{x}, \basis{y}, \basis{z})$, which
remain fixed at all times.  Thus, $\vartheta$ measures the angle from
the positive $\basis{z}$ axis and $\varphi$ measures the angle of the
projection into the $\basis{x}$-$\basis{y}$ plane from the positive
$\basis{x}$ axis, as shown in the figure.

Now, if $\hat{n}$ is the unit vector in that direction, we construct a
related rotor
\begin{equation\*}
  \rthetaphi = e^{\varphi \basis{z}/2}\, e^{\vartheta \basis{y}/2}.
\end{equation\*}
This can be obtained as a `quaternionic.array` object in python as

```python
>>> import quaternionic
>>> vartheta, varphi = 0.1, 0.2
>>> R_tp = quaternionic.array.from_spherical_coordinates(vartheta, varphi)
```

Here, rotations are given assuming the right-hand screw rule, so that
this corresponds to an initial rotation through the angle $\vartheta$
about $\basis{y}$, followed by a rotation through $\varphi$ about
$\basis{z}$.  (The factors of $1/2$ are present because $\rthetaphi$
is essentially the square-root of the rotation; the effect of this
rotor produces the full rotations through $\vartheta$ and $\varphi$.)
This rotor is particularly useful, because $\hat{n}$ and the two
standard tangent vectors at that point are all given by rotations of
the basis vectors $(\basis{x}, \basis{y}, \basis{z})$.  Specifically,
we have
\begin{align\*}
  \hat{\vartheta} &= \rthetaphi\,\basis{x}\,\rthetaphi^{-1}, \\\\\\\\
  \hat{\varphi} &= \rthetaphi\, \basis{y}\, \rthetaphi^{-1}, \\\\\\\\
  \hat{n} &= \rthetaphi\, \basis{z}\, \rthetaphi^{-1}.
\end{align\*}
Note that we interpret the rotations in the active sense, where a
point moves, while our coordinate system and basis vectors are left
fixed.  Hopefully, this won't cause too much confusion if we express
the original point with the vector $\basis{z}$, for example, and the
rotated point is therefore given by some $\rotor{R}\, \basis{z}\,
\rotor{R}$.
