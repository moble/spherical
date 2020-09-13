# Wigner 𝔇 Derivation

This is just the long-form derivation of the formula for Wigner's
$\mathfrak{D}$ matrices, discussed more on
[this page](/WignerDMatrices).

\begin{align}
  \mathbf{e}_{(m')}(\mathbf{R}\, \mathbf{Q})
  &=
  \frac{(\mathbf{R}\, \mathbf{Q})_{a}^{\ell+m'}\, (\mathbf{R}\, \mathbf{Q})_{b}^{\ell-m'}}
  {\sqrt{ (\ell+m')!\, (\ell-m')! }} \\\\
  &=
  \frac{(\mathbf{R}_a\, \mathbf{Q}_a - \bar{\mathbf{R}}_b\, \mathbf{Q}_b)^{\ell+m'}\,
    (\mathbf{R}\, \mathbf{Q})_{b}^{\ell-m'}}
  {\sqrt{ (\ell+m')!\, (\ell-m')! }} \\\\
  &=
  \sum_{\rho} \binom{\ell+m'} {\rho}
    \frac{(\mathbf{R}_a\, \mathbf{Q}_a)^{\ell+m'-\rho} (- \bar{\mathbf{R}}_b\, \mathbf{Q}_b)^{\rho}\,
    (\mathbf{R}_b\, \mathbf{Q}_a + \bar{\mathbf{R}}_a\, \mathbf{Q}_b)^{\ell-m'}} {\sqrt{ (\ell+m')!\, (\ell-m')! }} \\\\
  &=
  \sum_{\rho,\rho'} \binom{\ell+m'} {\rho} \binom{\ell-m'} {\rho'}
    \frac{(\mathbf{R}_a\, \mathbf{Q}_a)^{\ell+m'-\rho} (- \bar{\mathbf{R}}_b\, \mathbf{Q}_b)^{\rho}\,
    (\mathbf{R}_b\, \mathbf{Q}_a)^{\ell-m'-\rho'} (\bar{\mathbf{R}}_a\, \mathbf{Q}_b)^{\rho'}}
    {\sqrt{ (\ell+m')!\, (\ell-m')! }} \\\\
  &=
  \sum_{\rho,m} \binom{\ell+m'} {\rho} \binom{\ell-m'} {\ell-m-\rho}
    \frac{(\mathbf{R}_a\, \mathbf{Q}_a)^{\ell+m'-\rho} (- \bar{\mathbf{R}}_b\, \mathbf{Q}_b)^{\rho}\,
    (\mathbf{R}_b\, \mathbf{Q}_a)^{m-m'+\rho} (\bar{\mathbf{R}}_a\, \mathbf{Q}_b)^{\ell-m-\rho}}
    {\sqrt{ (\ell+m')!\, (\ell-m')! }} \\\\
  &=
  \sum_{\rho,m} \binom{\ell+m'} {\rho} \binom{\ell-m'} {\ell-m-\rho}
    \mathbf{R}_a^{\ell+m'-\rho} (- \bar{\mathbf{R}}_b)^{\rho}\, \mathbf{R}_b^{m-m'+\rho} \bar{\mathbf{R}}_a^{\ell-m-\rho}
    \frac{\mathbf{Q}_a^{\ell+m} \mathbf{Q}_b^{\ell-m}} {\sqrt{ (\ell+m')!\, (\ell-m')! }} \\\\
  &=
  \sum_{m} \mathbf{e}_{(m)}(\mathbf{Q}) \sum_{\rho} \binom{\ell+m'} {\rho} \binom{\ell-m'} {\ell-m-\rho}
    \mathbf{R}_a^{\ell+m'-\rho} (- \bar{\mathbf{R}}_b)^{\rho}\, \mathbf{R}_b^{m-m'+\rho} \bar{\mathbf{R}}_a^{\ell-m-\rho}
    \frac{\sqrt{ (\ell+m)!\, (\ell-m)! }} {\sqrt{ (\ell+m')!\, (\ell-m')! }} \\\\
\end{align}

We have introduced a new summation variable $m$ and used the
substitution $\rho' \mapsto \ell-m-\rho$ to bring this into the form
we need to express Wigner's $\mathfrak{D}$ matrix.  Alternatively, we
could have made an equivalent substitution for $\rho$, so that
$\mathfrak{D}$ would be given as a sum over $\rho'$.  This would have
the effect of reversing the roles of $a$ and $b$, which is what we do
on [this page](/WignerDMatrices) when $\lvert \mathbf{R}_a \rvert <
\lvert \mathbf{R}_b \rvert$.
