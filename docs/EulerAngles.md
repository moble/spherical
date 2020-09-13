# Euler angles

Euler angles are a terrible set of coordinates for the rotation group.
Compared to the other three standard presentations of rotations (rotation
matrices, axis-angle form, and the closely related unit quaternions), Euler
angles present no advantages and many severe disadvantages.  Composition of
rotations is complicated, numerically slow and inaccurate, and essentially
requires transformation to a different presentation anyway.  Interpolation of
Euler angles is meaningless and prone to severe distortions.  Most damningly of
all are the coordinate singularities (gimbal lock).  To summarize, Euler angles
are absolutely — and by a wide margin — the worst way to deal with
rotations.

We can work entirely without Euler angles.  (My own work simply never uses
them; the $\mathfrak{D}$ matrices are written directly in terms of rotors.)
Nonetheless, they are ubiquitous throughout the literature.  And while we can
work entirely without Euler angles, it can sometimes be useful to compare to
other results.  So, to make contact with that literature, we will need to
choose a convention for constructing a rotation from a triple of angles
$(\alpha, \beta, \gamma)$.  We therefore define the rotor
\begin{equation}
  R\_{(\alpha, \beta, \gamma)} = e^{\alpha\, \vec{z}/2}\, e^{\beta\,
  \vec{y}/2}\, e^{\gamma\, \vec{z}/2}.
\end{equation}
This can be obtained as a `quaternionic.array` object in python as

```python
>>> import quaternionic
>>> alpha, beta, gamma = 0.1, 0.2, 0.3
>>> R_euler = quaternionic.array.from_euler_angles(alpha, beta, gamma)
```

Note that the rotations are always taken about the *fixed* axes $\vec{z}$ and
$\vec{y}$.  Also, this is in operator form, so it must be read from right to
left: The rotation is given by an initial rotation through $\gamma$ about the
$\vec{z}$ axis, followed by a rotation through $\beta$ about the $\vec{y}$
axis, followed by a final rotation through $\alpha$ about the $\vec{z}$ axis.
This may seem slightly backwards, but it is a common convention --- in
particular, it is the one adopted by Wikipedia in its
[Wigner-D article](https://en.wikipedia.org/wiki/Wigner_D-matrix#Definition_of_the_Wigner_D-matrix).

It is worth noting that the standard right-handed basis vectors $(\vec{x},
\vec{y}, \vec{z})$ can be identified with generators of rotations usually
seen in quantum mechanics (or generally just special-function theory) according
to the rule
\begin{align}
  \frac{\vec{x}}{2} &\mapsto -i\, J\_x, \\\\\\\\
  \frac{\vec{y}}{2} &\mapsto -i\, J\_y, \\\\\\\\
  \frac{\vec{z}}{2} &\mapsto -i\, J\_z.
\end{align}
This is important when relating quaternion expressions to expressions more
commonly seen in the literature.  In particular, with this identification, we
have the usual commutation relations
\begin{align}
  \left[\frac{\vec{x}}{2}, \frac{\vec{y}}{2}\right] = \frac{\vec{z}}{2} &\mapsto [J_x, J_y] = i\, J_z, \\\\\\\\
  \left[\frac{\vec{y}}{2}, \frac{\vec{z}}{2}\right] = \frac{\vec{x}}{2} &\mapsto [J_y, J_z] = i\, J_x, \\\\\\\\
  \left[\frac{\vec{z}}{2}, \frac{\vec{x}}{2}\right] = \frac{\vec{y}}{2} &\mapsto [J_z, J_x] = i\, J_y.
\end{align}
And in any case, this certainly clarifies what to do with expressions like the
following from Wikipedia:
\begin{equation}
  \mathcal{R}(\alpha,\beta,\gamma) = e^{-i\alpha\, J_z}\,
  e^{-i\beta\, J_y} e^{-i\gamma\, J_z},
\end{equation}
which shows that my interpretation of Euler angles is the same as Wikipedia's.
