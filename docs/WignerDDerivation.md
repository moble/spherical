---
---

This is just the long-form derivation of the formula for Wigner's
$\mathfrak{D}$ matrices, discussed more on
[this page](WignerDMatrices.html).

\begin{align\*}
  \mathbf{e}\_{(\emprime)}(\quat{R}\, \quat{Q})
  &=
  \frac{(\quat{R}\, \quat{Q})\_{a}^{\ell+\emprime}\, (\quat{R}\, \quat{Q})\_{b}^{\ell-\emprime}}
  {\sqrt{ (\ell+\emprime)!\, (\ell-\emprime)! }} \\\\\\\\
  &=
  \frac{(\quat{R}\_a\, \quat{Q}\_a - \co{\quat{R}}\_b\, \quat{Q}\_b)^{\ell+\emprime}\,
    (\quat{R}\, \quat{Q})\_{b}^{\ell-\emprime}}
  {\sqrt{ (\ell+\emprime)!\, (\ell-\emprime)! }} \\\\\\\\
  &=
  \sum\_{\rho} \binom{\ell+m'} {\rho}
    \frac{(\quat{R}\_a\, \quat{Q}\_a)^{\ell+\emprime-\rho} (- \co{\quat{R}}\_b\, \quat{Q}\_b)^{\rho}\,
    (\quat{R}\_b\, \quat{Q}\_a + \co{\quat{R}}\_a\, \quat{Q}\_b)^{\ell-\emprime}} {\sqrt{ (\ell+\emprime)!\, (\ell-\emprime)! }} \\\\\\\\
  &=
  \sum\_{\rho,\rho'} \binom{\ell+m'} {\rho} \binom{\ell-m'} {\rho'}
    \frac{(\quat{R}\_a\, \quat{Q}\_a)^{\ell+\emprime-\rho} (- \co{\quat{R}}\_b\, \quat{Q}\_b)^{\rho}\,
    (\quat{R}\_b\, \quat{Q}\_a)^{\ell-\emprime-\rho'} (\co{\quat{R}}\_a\, \quat{Q}\_b)^{\rho'}}
    {\sqrt{ (\ell+\emprime)!\, (\ell-\emprime)! }} \\\\\\\\
  &=
  \sum\_{\rho,\em} \binom{\ell+m'} {\rho} \binom{\ell-m'} {\ell-\em-\rho}
    \frac{(\quat{R}\_a\, \quat{Q}\_a)^{\ell+\emprime-\rho} (- \co{\quat{R}}\_b\, \quat{Q}\_b)^{\rho}\,
    (\quat{R}\_b\, \quat{Q}\_a)^{\em-\emprime+\rho} (\co{\quat{R}}\_a\, \quat{Q}\_b)^{\ell-\em-\rho}}
    {\sqrt{ (\ell+\emprime)!\, (\ell-\emprime)! }} \\\\\\\\
  &=
  \sum\_{\rho,\em} \binom{\ell+m'} {\rho} \binom{\ell-m'} {\ell-\em-\rho}
    \quat{R}\_a^{\ell+\emprime-\rho} (- \co{\quat{R}}\_b)^{\rho}\, \quat{R}\_b^{\em-\emprime+\rho} \co{\quat{R}}\_a^{\ell-\em-\rho}
    \frac{\quat{Q}\_a^{\ell+\em} \quat{Q}\_b^{\ell-\em}} {\sqrt{ (\ell+\emprime)!\, (\ell-\emprime)! }} \\\\\\\\
  &=
  \sum\_{\em} \mathbf{e}\_{(\em)}(\quat{Q}) \sum\_{\rho} \binom{\ell+m'} {\rho} \binom{\ell-m'} {\ell-\em-\rho}
    \quat{R}\_a^{\ell+\emprime-\rho} (- \co{\quat{R}}\_b)^{\rho}\, \quat{R}\_b^{\em-\emprime+\rho} \co{\quat{R}}\_a^{\ell-\em-\rho}
    \frac{\sqrt{ (\ell+\em)!\, (\ell-\em)! }} {\sqrt{ (\ell+\emprime)!\, (\ell-\emprime)! }} \\\\\\\\
\end{align\*}

We have introduced a new summation variable $\em$ and used the
substitution $\rho' \mapsto \ell-\em-\rho$ to bring this into the form
we need to express Wigner's $\mathfrak{D}$ matrix.  Alternatively, we
could have made an equivalent substitution for $\rho$, so that
$\mathfrak{D}$ would be given as a sum over $\rho'$.  This would have
the effect of reversing the roles of $a$ and $b$, which is what we do
on [this page](WignerDMatrices.html) when $\lvert \quat{R}\_a \rvert <
\lvert \quat{R}\_b \rvert$.
