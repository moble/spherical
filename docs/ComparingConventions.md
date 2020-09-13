# Comparing conventions

!!! important
    This is very much a work in progress.  Pull requests are welcome!

## Goldstein-Poole-Safko "Classical Mechanics"

### Euler angles

 1. $\phi$ about $z$
 2. $\theta$ about $x'$
 3. $\psi$ about $z''$

Goldstein (p. 151) claims that this convention is widely used in
celestial mechanics and applied mechanics, and frequently in molecular
and solid-state physics.  Goldstein also has an Appendix on the
various conventions.  I would write Goldstein's rotation as
\begin{equation}
  \label{eq:GoldsteinEulerAngles}
  e^{\phi \vec{z}/2}\, e^{\theta \vec{x}/2}\, e^{\psi \vec{z}/2}.
\end{equation}
That's right, he uses $x$!  I think this means that my coordinate
$\gamma$ would be related to his $\psi$ according to $\gamma = \psi -
\pi/2$.

### Wigner $\mathfrak{D}$; harmonics

I know of nowhere that Goldstein uses either

## Varshalovich et al.

"Quantum theory of angular momentum"




## Edmonds (1974)

### Euler angles

In Sec. 1.3, Edmonds gives rotations "to be performed successively in the
order:"

  1. A rotation $\alpha$ about the $z$ axis
  2. A rotation $\beta$ about the $y'$ axis
  3. A rotation $\gamma$ about the $z''$ axis

He notes that these are positive, right-handed rotations about the relevant
axes, and the coordinate system is right-handed.  Moreover, this rotation
describes the rotation of a rigid body about a fixed point, where $z$, $y'$,
and $z''$ move with the body.  He points out that the net rotation is
equivalent to

  1. A rotation $\gamma$ about the $z$ axis
  2. A rotation $\beta$ about the $y$ axis
  3. A rotation $\alpha$ about the $z$ axis

where the axes are fixed with respect to the inertial frame.


### Wigner $\mathfrak{D}$; harmonics

This is where things get ugly with Edmonds.  He seems to switch from an active
transformation to a passive one, but he still seems to have made an error in
writing down the transformation matrix.  Basically, all the angles should have
their signs reversed.  In this version of the book, Edmonds mentions the paper
by
[Wolf (1969)](http://scitation.aip.org/content/aapt/journal/ajp/37/5/10.1119/1.1975665),
which sorted through various conventions and pointed out an error in older
versions of Edmonds.  But I think there's still an error.


## Devanathan

"Angular Momentum Techniques in Quantum Mechanics"


\begin{align\*}
  R_{\text{Devanathan}}(\alpha, \beta, \gamma)
  &=
  e^{\gamma \vec{z}''/2}\, e^{\beta \vec{y}'/2}\, e^{\alpha \vec{z}/2} \\\\\\\\
  &=
  e^{\gamma \vec{z}''/2}\, e^{\beta e^{\alpha \vec{z}/2}\, \vec{y}\,
  e^{-\alpha \vec{z}/2}/2}\, e^{\alpha \vec{z}/2} \\\\\\\\
  &=
  e^{\gamma \vec{z}''/2}\, e^{\alpha \vec{z}/2}\, e^{\beta
  \vec{y}/2}\, e^{-\alpha \vec{z}/2}\, e^{\alpha \vec{z}/2} \\\\\\\\
  &=
  e^{\gamma \vec{z}''/2}\, e^{\alpha \vec{z}/2}\, e^{\beta
  \vec{y}/2} \\\\\\\\
  &=
  e^{\gamma e^{\alpha \vec{z}/2}\, e^{\beta \vec{y}/2} \vec{z}\,
  e^{-\beta \vec{y}/2}\, e^{-\alpha \vec{z}/2}/2}\, e^{\alpha
  \vec{z}/2}\, e^{\beta \vec{y}/2} \\\\\\\\
  &=
  e^{\alpha \vec{z}/2}\, e^{\beta \vec{y}/2}\, e^{\gamma \vec{z}/2}\,
  e^{-\beta \vec{y}/2}\, e^{-\alpha \vec{z}/2}\, e^{\alpha
  \vec{z}/2}\, e^{\beta \vec{y}/2} \\\\\\\\
  &=
  e^{\alpha \vec{z}/2}\, e^{\beta \vec{y}/2}\, e^{\gamma \vec{z}/2}
  \\\\\\\\
  &= R_{\text{spherical}}(\gamma, \beta, \alpha)
\end{align\*}



## Shankar


## Sakurai


## Wikipedia


## Mathematica

The Euler angles correspond to what I would have considered the
*inverse* rotation.

## Sympy

### Euler angles

The
[`sympy.physics.quantum.spin.Rotation`](http://docs.sympy.org/dev/modules/physics/quantum/spin.html#sympy.physics.quantum.spin.Rotation)
class uses the $z''$-$y'$-$z$ convention (which the documentation refers to
as the "passive $z$-$y$-$z$" convention).  This basically means that I have
to swap the $\alpha$ and $\gamma$ arguments to keep mine consistent with
sympy:
\begin{equation\*}
  R_{\text{sympy}}(\alpha,\beta,\gamma) =
  R_{\text{spherical}}(\gamma,\beta,\alpha).
\end{equation\*}

### $\mathfrak{D}$ matrices

[Sympy](http://sympy.org/en/index.html) (the symbolic math package for
python) implements the $\mathfrak{D}$ matrices as the function
`sympy.physics.quantum.spin.WignerD`.  Note that
[the documentation](http://docs.sympy.org/latest/modules/physics/quantum/spin.html#sympy.physics.quantum.spin.WignerD)
swaps the symbols $m$ and $m'$ in their documentation, relative to the
usual order.  Nonetheless, the arguments to the function are in the
standard order $(m',m)$.  That is, if you call the function using what
the rest of this module (and other items on this page) regard as $m'$
and $m$, you will get the expected result---up to the interpretation
of the Euler angles.  Don't be alarmed by the fact that the
documentation lists the arguments as $(m,m')$.

```python
alpha,beta,gamma = R.euler_angles()
Dsympy = sympy.physics.quantum.spin.Rotation.D(ell, mp, m, gamma, beta, alpha).doit().evalf(n=32).conjugate()
Dsf = sf.Wigner_D_element(R, ell, mp, m)
abs(Dsympy-Dsf) < 1e-13 # True for ell<29
```

## Wigner

(As translated by J. J. Griffin)

The $\mathfrak{D}$ matrix is given on page 167, Eq. (15.27).
