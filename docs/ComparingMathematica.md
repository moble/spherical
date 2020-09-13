## Comparison with Mathematica

In the "Applications" section of Mathematica's documentation page for
`WignerD`, the rotation matrix is constructed from Euler angles
$(\psi,\theta,\phi)$ according to the expression

```
RotationMatrix[-phi, {0, 0, 1}].RotationMatrix[-theta, {0, 1, 0}].RotationMatrix[-psi, {0, 0, 1}]
```

This is the inverse of the matrix given by

```
RotationMatrix[psi, {0, 0, 1}].RotationMatrix[theta, {0, 1, 0}].RotationMatrix[phi, {0, 0, 1}]
```

The latter, of course, would be equivalent to a rotor
$e^{\psi\vec{z}/2}\, e^{\theta\vec{y}/2}\, e^{\phi\vec{z}/2}$,
which is what I would have denoted $\mathbf{R}_{(\psi, \theta,
\phi)}$.  So my rotor gives the inverse rotation of Mathematica's
Euler angles.  This could also be viewed as a disagreement over active
and passive transformations.

However, there are still other disagreements.  The same page states
that

```
WignerD[{j,m1,m2},psi,theta,phi]
```

gives the function $\mathfrak{D}^j_{m_1,m_2}(\psi,\theta,\phi)$, and
the spherical harmonics are related to $\mathfrak{D}$ by

\begin{equation}
  \mathfrak{D}^\ell_{0,m}(0, \theta, \phi) =
  \sqrt{\frac{4\pi}{2\ell+1}} Y_{\ell,m} (\theta, \phi).
\end{equation}

