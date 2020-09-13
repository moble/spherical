# Wigner $\mathfrak{D}$ matrices

The Wigner $\mathfrak{D}$ matrices allow us to represent the rotation
group by means of finite-dimensional matrices.  They reduce to
spherical functions in special cases, which further allows us to
derive the rotation of those spherical functions---as described in
more detail on [this page](SWSHs.html).  They are derived in terms of
a particular split of the quaternion group into two parts.

Explicitly, a quaternion $\quat{Q}$ can be expressed in terms of two
complex numbers $\quat{Q}_a = \quat{Q}_1 + i\, \quat{Q}_z$ and
$\quat{Q}_b = \quat{Q}_y + i\, \quat{Q}_x$.[^1] This is only
important because it allows us to verify the multiplication law
\begin{align}
  \label{eq:QuaternionComponentProducts}
  (\quat{P}\,\quat{Q})_a &= \quat{P}_a\, \quat{Q}_a - \co{\quat{P}}_b\, \quat{Q}_b, \\\\
  (\quat{P}\,\quat{Q})_b &= \quat{P}_b\, \quat{Q}_a + \co{\quat{P}}_a\, \quat{Q}_b.
\end{align}
Given a rotor $\rotor{R}$, these two complex numbers are the
quantities actually used in computing the $\mathfrak{D}$ matrix
elements.  Note that there is a series of basic choices leading to
both the decomposition of a quaternion into two complex numbers, and
the product law.  None of these choices is set in stone; we just
choose something with an eye to the desired result.

The following is essentially the same as Wigner's original derivation,
but with more elegance, and more sensitivity to numerical issues and
special cases.  This version of the derivation comes from
[a paper](http://arxiv.org/abs/1302.2919) I wrote a couple years ago,
and is the source of the code used in this module.

The basic idea of the derivation is to construct a
$(2\ell+1)$-dimensional vector space of homogeneous polynomials in
these complex numbers $\quat{Q}_a$ and $\quat{Q}_b$.  To make that a
little more concrete, the basis of this vector space is
\begin{equation*}
  \label{WignerBasisComponent}
  \mathbf{e}_{(m)}(\quat{Q}) \defined
    \frac{\quat{Q}_{a}^{\ell+m}\, \quat{Q}_{b}^{\ell-m}}
    {\sqrt{ (\ell+m)!\, (\ell-m)! }}.
\end{equation*}
Here, $m$ ranges from $-\ell$ to $\ell$ in steps of $1$, but $\ell$
(and hence $m$) can be a half-integer.  Now, the key idea is that a
rotation of $\quat{Q}$ by some new rotor $\quat{R}$ gives us a new
vector basis, which we can represent in terms of the basis shown
above.  In fact, we get a matrix transforming one set of basis vectors
to another.  We'll write this as
\begin{equation*}
  \mathbf{e}_{(m')}(\quat{R}\, \quat{Q})
  = \sum_{m} \mathfrak{D}^{(\ell)}_{m',m}(\quat{R})\, \mathbf{e}_{(m)}(\quat{Q}).
\end{equation*}

So now, we've defined the $\mathfrak{D}$ matrices.  But we can also
plug $\quat{R}\, \quat{Q}$ into the original expression for
$\mathbf{e}$, and figure out what $\mathfrak{D}$ should actually be.
We'll have to use the expressions for $(\quat{R}\, \quat{Q})_a$ and
$(\quat{R}\, \quat{Q})_b$ given above, and we'll find that we have
polynomials with terms given as sums of two different things.

This brings us to the first fork in the road.  If either $\quat{R}_a$
or $\quat{R}_b$ is tiny (i.e., the absolute value is numerically
small), we can (and in fact must) treat these as special cases.
First, if $\lvert \quat{R}_a \rvert \lesssim 10^{-15}$, for example,
we can just ignore it; since $\lvert \quat{R} \rvert=1$ (within
numerical precision), we are assured that $\lvert \quat{R}_b \rvert
\approx 1$.  Thus, we get
\begin{align*}
  \mathbf{e}_{(m')}(\quat{R}\, \quat{Q})
  &\approx \frac{ (- \co{\quat{R}}_{b}\, \quat{Q}_{b})^{\ell+m'}\,
    (\quat{R}_{b}\, \quat{Q}_{a})^{\ell-m'} } { \sqrt{
      (\ell+m')!\, (\ell-m')! } }, \\\\
  &\approx (- \co{\quat{R}}_{b})^{\ell+m'}\,
  (\quat{R}_{b})^{\ell-m'}\, \mathbf{e}_{(-m')}(\quat{Q}).
\end{align*}
In this case, it's not hard to see that the expression for the
$\mathfrak{D}$ matrix is
\begin{equation}
  \label{eq:D_RaApprox0}
  \mathfrak{D}^{(\ell)}_{m',m}(\quat{R})
  = (-1)^{\ell+m'}\, \quat{R}_b^{-2m'} \delta_{-m',m}
  = (-1)^{\ell+m}\, \quat{R}_b^{2m} \delta_{-m',m}.
\end{equation}
In the same way, we can calculate this for $\lvert \quat{R}_b \rvert
\lesssim 10^{-15}$:
{::comment}
\begin{align*}
  \mathbf{e}_{(m')}(\quat{R}\, \quat{Q})
  &\approx \frac{ (\quat{R}_{a}\, \quat{Q}_{a})^{\ell+m'}\,
    (\co{\quat{R}}_{a}\, \quat{Q}_{b})^{\ell-m'} } { \sqrt{
      (\ell+m')!\, (\ell-m')! } }, \\\\
  &\approx (\quat{R}_{a})^{\ell+m'}\,
  (\co{\quat{R}}_{a})^{\ell-m'}\, \mathbf{e}_{(m')}(\quat{Q}).
\end{align*}
{:/comment}
\begin{equation}
  \label{eq:D_RbApprox0}
  \mathfrak{D}^{(\ell)}_{m',m}(\quat{R})
  = \quat{R}_a^{2m'} \delta_{m',m}
  = \quat{R}_a^{2m} \delta_{m',m}.
\end{equation}

Now, the other fork in that road is the general case, when both
components have larger magnitudes.  We have powers of the sum of those
terms in Eq. \eqref{eq:QuaternionComponentProducts}.  This leads us to
use (two applications of) the
[binomial expansion](https://en.wikipedia.org/wiki/Binomial_theorem).
After [a little simplification](WignerDDerivation.html), we can
express the result as
\begin{multline}
  \label{eq:DAnalytically}
  \mathfrak{D}^{(\ell)}_{m',m}(\quat{R})
  = \sum_{\rho} \binom{\ell+m'} {\rho}\, \binom{\ell-m'}
  {\ell-\rho-m}\, (-1)^{\rho}\, \\\\ \times \quat{R}_{a}^{\ell+m'-\rho}\,
  \co{\quat{R}}_{a}^{\ell-\rho-m}\,
  \quat{R}_{b}^{\rho-m'+m}\, \co{\quat{R}}_{b}^{\rho}\,
  \sqrt{ \frac{ (\ell+m)!\, (\ell-m)! } { (\ell+m')!\,
      (\ell-m')! } }.
\end{multline}
It turns out that this expression is not the best way to implement the
calculation in the code.  The reason is that we would need to take a
bunch of exponents of complex numbers, and there's the possibility
that the sum would cancel out to give a very small number, which would
be polluted with roundoff, etc.  So we manipulate it to put it in a
better form.  For example, we can simplify the above as
\begin{align}
  \nonumber
  \mathfrak{D}^{(\ell)}_{m',m}(\quat{R})
  &=
  \sqrt{ \frac{ (\ell+m)!\, (\ell-m)! } { (\ell+m')!\, (\ell-m')! } }\,
  \quat{R}_{a}^{\ell+m'}\,
  \co{\quat{R}}_{a}^{\ell-m}\,
  \quat{R}_{b}^{-m'+m} \\\\
  \nonumber
  &\qquad
  \times \sum_{\rho}
  \binom{\ell+m'} {\rho}\, \binom{\ell-m'} {\ell-\rho-m}\,
  (-1)^{\rho}\,
  \quat{R}_{a}^{-\rho}\,
  \co{\quat{R}}_{a}^{-\rho}\,
  \quat{R}_{b}^{\rho}\,
  \co{\quat{R}}_{b}^{\rho}, \\\\
  \nonumber
  &=
  \sqrt{ \frac{ (\ell+m)!\, (\ell-m)! } { (\ell+m')!\, (\ell-m')! } }\,
  \lvert \quat{R}_{a} \rvert^{2\ell-2m}\,
  \quat{R}_{a}^{m'+m}\,
  \quat{R}_{b}^{-m'+m} \\\\
  \label{eq:D_RaGeqRb}
  &\qquad
  \times \sum_{\rho}
  \binom{\ell+m'} {\rho}\, \binom{\ell-m'} {\ell-\rho-m}\,
  \left( - \frac{\lvert \quat{R}_{b} \rvert^2}
  {\lvert \quat{R}_{a} \rvert^2} \right)^{\rho}.
\end{align}
Typically, this last expression can be evaluated pretty efficiently.

But to make it really fast, accurate, and robust, we have to use some
tricks.  First of all, the various combinatorical factors are
expensive to re-evaluate each time.  The obvious way to handle this is
just to pre-compute them, and have a function returning the
appropriate values with some simple indexing tricks.  These functions
are implemented in this module's
[initialization code](https://github.com/moble/spherical/blob/master/__init__.py#L18).

Now, that sum is essentially a polynomial, and the best way to evaluate a
polynomial uses
[Horner form](http://reference.wolfram.com/language/ref/HornerForm.html) ---
which is both faster and more accurate than the naive approach.  Also, since
the coefficients involve factorials of the summation index, we can factor
out the $\rho_{\text{min}}$ binomials, and be left with just a few
factorials, which we can then
[evaluate more efficiently](PolynomialsWithFactorials.html).  This also
allows us to pull out the lowest-order coefficient of that polynomial:
$(-\lvert R_b \rvert / \lvert R_a \rvert)^{2 \rho_{\mathrm{min}}}$.  We will
see momentarily that being able to do this is very important.

That deals very nicely with the sum, but we also need to deal with the
factor in front of the sum.  And there are some fairly subtle
complications to deal with when evaluating that factor.  First of all,
complex exponentials are slow.  Second, there are cases where those
terms can become really huge, only to be multiplied by something
really tiny, leaving us reasonable values in the neighborhood of 1.
But those exponents can be very large; if we're looking for matrix
elements for $\ell=32$, the exponents will range from $-64$ all the
way up to $64$.  And since the largest and smallest numbers python can
represent are in the neighborhood of $10^{323}$ and $10^{-308}$,
respectively, we can easily get `nan` or `0` for calculations that
should actually give us something that is neither of those values ---
for example, if $\lvert R_b \rvert \approx 10^{-6}$ and $-m'=m=-32$.
But these terms only appear due to our separation of powers into the
sum.  If we now bring out the lowest factor from the sum, we can make
those exponents disappear.  The problem now is to cancel the exponents
in a mixture of exponents of complex numbers and exponents of real
numbers, and to do so efficiently.

The solution to this problem is simple and, miraculously, makes the
code 2.3 times faster!  We decompose $R_a$ and $R_b$ into magnitude
and phase, calculate the exponents to which those must be raised, and
then produce the result as a complex number from the separate
magnitude and phase.  There are fast functions in the standard python
module `cmath` that do these operations, and they are directly
supported in `numba` since version 0.15, so this is quite a simple
solution to implement, also.  The calculation of this prefactor is
given in this module
[here](https://github.com/moble/spherical/blob/master/WignerD.py#L225).

But this doesn't solve all our problems.  The same issues can creep up
if $\lvert R_a \rvert$ is small (though it requires a smaller
magnitude, and occurs for a smaller subset of $m',m$ values).  In this
case, it would be better if we could find an expression that sort of
reverses the roles of $a$ and $b$.  It turns out that this isn't hard.
In deriving Eq. \eqref{eq:DAnalytically}, a choice was made regarding
the summation variable.  We can simply transform that summation
variable as $\rho \mapsto \ell-m-\rho$, and obtain
\begin{multline*}
  \mathfrak{D}^{(\ell)}_{m',m}(\quat{R})
  = \sum_{\rho} \binom{\ell+m'} {\ell-m-\rho}\, \binom{\ell-m'}
  {\rho}\, (-1)^{\ell-m-\rho}\, \\\\ \times
  \quat{R}_{a}^{m'+m+\rho}\, \co{\quat{R}}_{a}^{\rho}\,
  \quat{R}_{b}^{\ell-m'-\rho}\, \co{\quat{R}}_{b}^{\ell-m-\rho}\,
  \sqrt{ \frac{ (\ell+m)!\, (\ell-m)! } { (\ell+m')!\,
      (\ell-m')! } }.
\end{multline*}
It's interesting to note the symmetry with the earlier version of this
equation; we've essentially just exchanged the labels $a$ and $b$,
while also reversing the sign of $m'$, and multiplying by an overall
factor of $(-1)^{\ell+m}$.

In any case, we can apply the same simplification to this expression
as before:
\begin{align}
  \nonumber
  \mathfrak{D}^{(\ell)}_{m',m}(\quat{R})
  &=
  (-1)^{\ell-m}
  \sqrt{ \frac{ (\ell+m)!\, (\ell-m)! } { (\ell+m')!\, (\ell-m')! } }
  \quat{R}_{a}^{m'+m}\,
  \quat{R}_{b}^{\ell-m'}\,
  \co{\quat{R}}_{b}^{\ell-m}
  \\\\
  \nonumber
  &\qquad
  \times \sum_{\rho}
  \binom{\ell+m'} {\ell-m-\rho}\, \binom{\ell-m'} {\rho}\,
  (-1)^\rho\,
  \quat{R}_{a}^{\rho}\,
  \co{\quat{R}}_{a}^{\rho}\,
  \quat{R}_{b}^{-\rho}\,
  \co{\quat{R}}_{b}^{-\rho}
  \\\\
  \nonumber
  &=
  (-1)^{\ell-m}
  \sqrt{ \frac{ (\ell+m)!\, (\ell-m)! } { (\ell+m')!\, (\ell-m')! } }
  \quat{R}_{a}^{m'+m}\,
  \quat{R}_{b}^{m-m'}\,
  \lvert \quat{R}_{b} \rvert^{2\ell-2m}
  \\\\
  \label{eq:D_RaLeqRb}
  &\qquad
  \times \sum_{\rho}
  \binom{\ell+m'} {\ell-m-\rho}\, \binom{\ell-m'} {\rho}\,
  \left( - \frac{ \lvert \quat{R}_{a} \rvert^2 }
  { \lvert \quat{R}_{b} \vert^2 } \right)^{\rho}
\end{align}
And again, we evaluate this cleverly, as above.

So we get four branches in our logic, with a different expression for
$\mathfrak{D}$ in each branch:

  1. When $\lvert \quat{R}_a \rvert \lesssim 10^{-15}$, use
     Eq. \eqref{eq:D_RaApprox0}.
  2. When $\lvert \quat{R}_b \rvert \lesssim 10^{-15}$, use
     Eq. \eqref{eq:D_RbApprox0}.
  3. When $\lvert \quat{R}_a \rvert \geq \lvert \quat{R}_b \rvert$,
     use Eq. \eqref{eq:D_RaGeqRb}.
  4. When $\lvert \quat{R}_a \rvert < \lvert \quat{R}_b \rvert$, use
     Eq. \eqref{eq:D_RaLeqRb}.

Note that these expressions are valid even for half-integer values of
$\ell$, noting that if $\ell$ is half-integer, then so must $m'$ and
$m$ be.  However, in the interests of fast implementation, the code in
this module assumes integer values.  (It would be simple for me to
implement the more general case.  If this is a functionality you need,
please feel free to
[open an issue](https://github.com/moble/spherical/issues)
on this module's github page to request it.)



## Relation to the antiquated form of $\mathfrak{D}$ using Euler angles

I hope I don't have to repeat my utter disdain for the use of Euler
angles.  However, it is important to make contact with other
literature to be able to compare conventions.  As noted
[previously](index.html#euler-angles), the rotation performed by the
set $(\alpha, \beta, \gamma)$ of Euler angles (using conventions to
agree with
[Wikipedia's page](https://en.wikipedia.org/wiki/Wigner_D-matrix#Definition_of_the_Wigner_D-matrix)
on $\mathfrak{D}$ matrices) can be written in quaternion form as
\begin{align*}
  \quat{R}_{(\alpha, \beta, \gamma)} &= e^{\alpha\, \basis{z}/2}\,
  e^{\beta\, \basis{y}/2}\, e^{\gamma\, \basis{z}/2}, \\\\
  &= \left(
      \cos \frac{\alpha}{2}\, \cos \frac{\beta}{2}\, \cos \frac{\gamma}{2}
      -\sin \frac{\alpha}{2}\, \cos \frac{\beta}{2}\, \sin \frac{\gamma}{2}
    \right) \\\\
  &\qquad + \basis{x} \left(
      \cos \frac{\alpha}{2}\, \sin \frac{\beta}{2}\, \sin \frac{\gamma}{2}
      -\sin \frac{\alpha}{2}\, \sin \frac{\beta}{2}\, \cos \frac{\gamma}{2}
    \right) \\\\
  &\qquad + \basis{y} \left(
      \cos \frac{\alpha}{2}\, \sin \frac{\beta}{2}\, \cos \frac{\gamma}{2}
      +\sin \frac{\alpha}{2}\, \sin \frac{\beta}{2}\, \sin \frac{\gamma}{2}
    \right) \\\\
  &\qquad + \basis{z} \left(
      \sin \frac{\alpha}{2}\, \cos \frac{\beta}{2}\, \cos \frac{\gamma}{2}
      +\cos \frac{\alpha}{2}\, \cos \frac{\beta}{2}\, \sin \frac{\gamma}{2}
    \right).
\end{align*}
Taking the complex components of this, we have
\begin{equation*}
  \quat{R}_a = e^{i\,\alpha/2}\, \cos\frac{\beta}{2}\, e^{i\,\gamma/2},
  \qquad
  \quat{R}_b = e^{-i\,\alpha/2}\, \sin\frac{\beta}{2}\, e^{i\,\gamma/2}.
\end{equation*}
We can plug these values into, e.g., Eq. \eqref{eq:D_RaGeqRb}, and
get the standard, hideous, reprehensible form of the $\mathfrak{D}$
matrices in terms of Euler angles.

Again, of course, this is not a form that should be used for
calculations, but could be useful for comparing conventions with
antiquated research.


---

[^1]: This, of course, is not a productive way of *thinking about*
      quaternions, but it is a very useful way of *calculating with*
      quaternions, since complex numbers are already built in to many
      languages.  That is, this decomposition into two complex
      components is something that the user probably does not need to
      worry about.  It is, however, an isomorphism between quaternions
      and the usual (awful) presentation of Pauli spinors as
      two-component complex vectors, which is closer to the flavor of
      Wigner's original derivation.  It is also how the code is
      actually implemented.

