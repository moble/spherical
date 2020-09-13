# Polynomials with factorials

We know that Horner form is a very efficient way to evaluate polynomials,
but if the coefficients of that polynomial involve factorials of the index
itself, we can incorporate a further simplification.  The standard Horner
form of a general polynomial is
\begin{equation}
  \sum_{\rho=\rho_0} c_\rho x^\rho = x^{\rho_0} \left( c_{\rho_0} + x
  \left(c_{\rho_0+1} + x \left( c_{\rho_0+2} + \ldots \right) \right)
  \right).
\end{equation}
If `c` is a function that returns the appropriate coefficient, we can
implement this with code like

```python
tot = 0.0
for i in range(rho_max, rho_min-1, -1):
    tot = c(i) + x * tot
tot *= x**(rho_min)
```

Note that `c(i)` is always *added to* the current total in this algorithm.

Now, if instead, the coefficients are all $\rho!$, we have
\begin{equation}
  \sum_{\rho=\rho_0} \rho! x^\rho = (\rho_0)! x^{\rho_0} \left( 1 +
  (\rho_0+1) x \left(1 + (\rho_0+2) x \left( 1 + \ldots \right) \right)
  \right).
\end{equation}
The efficient code for this is

```python
tot = 1.0
for rho in range(rho_max, rho_min, -1):
    tot = 1 + rho * x * tot
tot *= factorial(rho_min)*x**rho_min
```

Here, the coefficient *multiplies* the current total.

Now, a more interesting case for our purposes involves a more complicated
coefficient:
\begin{equation}
  \sum_{\rho=\rho_0} \frac{(M+\rho_0)!} {(M+\rho)!} x^\rho = x^{\rho_0}
  \left( 1 + \frac{1} {M+\rho_0+1} x \left(1 + \frac{1} {M+\rho_0+2} x
  \left( 1 + \ldots \right) \right) \right).
\end{equation}
Here again, we can turn this into a simple algorithm:

```python
tot = 1.0
for rho in range(rho_max, rho_min, -1):
    tot = 1 + x * tot / (M+rho)
tot *= x**rho_min
```

Another interesting case for our purposes involves $-\rho$:
\begin{equation}
  \sum_{\rho=\rho_0} \frac{(N-\rho_0)!} {(N-\rho)!} x^\rho = x^{\rho_0}
  \left( 1 + (N-\rho_0) x \left(1 + (N-\rho_0-1) x
  \left( 1 + \ldots \right) \right) \right).
\end{equation}
Here again, we can turn this into a simple algorithm:

```python
tot = 1.0
for rho in range(rho_max, rho_min, -1):
    tot = 1 + x * tot * (N-rho+1)
tot *= x**rho_min
```

And, of course, we can combine them---which is the really interesting case:
\begin{equation}
  \sum_{\rho=\rho_0}
  \frac{\rho_0!} {\rho!}
  \frac{(N_1-\rho_0)!} {(N_1-\rho)!}
  \frac{(M+\rho_0)!} {(M+\rho)!}
  \frac{(N_2-\rho_0)!} {(N_2-\rho)!}
  x^\rho
\end{equation}

```python
tot = 1.0
for rho in range(rho_max, rho_min, -1):
    tot = 1 + x * tot * (N1-rho+1) * (N2-rho+1) / (rho * (M+rho))
tot *= x**rho_min
```

Applied to the case of calculating the Wigner $\mathfrak{D}$ matrices,
we have $N_{1} = \ell+m'$, $M = m-m'$, and $N_{2} = 2\ell - m - m'$,
and something like $x = - \lvert R_{a} / R_{b} \rvert^{2}$.  This
results in the algorithm


Applied to the case of calculating the Wigner $\mathfrak{D}$ matrices,
for $\lvert R_{a} \rvert \geq \lvert R_{b} \rvert$, we have
$x = - \lvert R_{b} / R_{a} \rvert^{2}$ and
\begin{gather}
  N_{1} = \ell + m' \\\\\\\\
  N_{2} = \ell - m \\\\\\\\
  M = (\ell-m') - (\ell-m) = m - m'
\end{gather}
For $\lvert R_{a} \rvert < \lvert R_{b} \rvert$, we have
$x = - \lvert R_{a} / R_{b} \rvert^{2}$ and
\begin{gather}
  N_{1} = \ell - m' \\\\\\\\
  N_{2} = \ell - m \\\\\\\\
  M = (\ell+m') - (\ell-m) = m + m'.
\end{gather}

This still can't solve the instability for values of $x$ close to -1, but it
might least make the process a bit faster.  And it might be nicer if I ever get
around to using `gmpy2` (or something) to compute this loop, which should allow
me to go to arbitrarily high $\ell$ (though at some cost).

In particular, I could potentially move this entire loop into an external C
call, which would only take a few integer arguments and a single float, rather
than trying to access the arrays for the binomial coefficients.  That loop
would include some of the gmp library, presumably.  But that might be simple
enough, assuming gmpy2 is installed, and just using its copy of the library.
