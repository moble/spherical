# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import numpy as np
import numba

"""Associated Legendre function recursion

All references are to "A unified approach to the Clenshaw summation and the recursive computation of
very high degree and order normalised associated Legendre functions" by Holmes and Featherstone
(2002).  The text suggests that the best approach -- both in terms of speed and accuracy -- is the
"modified forward row method" described in Sec. 2.5.

Define k=1 for m=0 and k=2 for m>0, j=2 for m=0 and j=1 for m>0, and

\hat{P}_{n, m}(\theta) = \sqrt{ \frac{k(2n+1)(n-m)!} {(n+m)!} } P_{n,m}(\theta) / \sin^m(\theta)

Now, define the coefficients

g_{n,m} = \frac{2(m+1)} {\sqrt{(n-m)(n+m+1)}},

and

h_{n,m} = \sqrt{ \frac{(n+m+2)(n-m-1)} {(n-m)(n+m+1)} }.

Noting that h_{n,n-1} is always 0, we formulate the recursion as

\hat{P}_{1,1} = \sqrt{3},

\hat{P}_{n,n} = \sqrt{\frac{2n+1} {2n}} \hat{P}_{n-1,n-1} \text{ for $n>1},

\hat{P}_{n,m} = \frac{1}{\sqrt{j}} \left(g_{n,m} \cos\theta \hat{P}_{n,m+1} - h_{n,m} \sin^2\theta \hat{P}_{n,m+2}\right),

where the last term in the last equation above drops out for $m=n-1$.
"""


def P̆ₙₘ(θ, sinθ, cosθ, max_n, g, h):
    """Compute √[ (kₘ(2n+1)(n-m)!)((n+m)!) Pₙₘ(θ) / sinᵐθ] recursively

    Note that this function is primarily a helper method for computing the standard associated
    Legendre functions Pₙₘ(θ), the Wigner 𝔇 matrices, or related quantities.

    Here, k=1 for m=0 and k=2 for m>0.  The recursion uses the "modified forward row method"
    described in Sec. 2.5 of Holmes and Featherstone (2002).

    """



class P̆Calculator(object):
    def __init__(self, n_max):
        self.n_max = int(n_max)
        if self.n_max < 0:
            raise ValueError(f'Nonsensical value for n_max = {self.n_max}')
        self.nm_total_size = sf.LMpM_total_size(0, self.n_max)
        self.nmpm_total_size_plus1 = sf.LMpM_total_size(0, self.n_max+1)
        n = np.array([n for n in range(self.n_max+2) for m in range(-n, n+1)])
        m = np.array([m for n in range(self.n_max+2) for m in range(-n, n+1)])
        absn = np.array([n for n in range(self.n_max+2) for m in range(n+1)])
        absm = np.array([m for n in range(self.n_max+2) for m in range(n+1)])
        self.sqrt_factorial_ratio = (-1)**absm * np.sqrt(
            np.asarray(factorial(absn-absm, exact=True) / factorial(absn+absm, exact=True), dtype=float))
        self.a = np.sqrt((absn+1+absm) * (absn+1-absm) / ((2*absn+1)*(2*absn+3)))
        self.b = np.sqrt((n-m-1) * (n-m) / ((2*n-1)*(2*n+1)))
        self.b[m<0] *= -1
        self.d = 0.5 * np.sqrt((n-m) * (n+m+1))
        self.d[m<0] *= -1
        with np.errstate(divide='ignore', invalid='ignore'):
            self.g = 2*(m+1) / np.sqrt((n-m)*(n+m+1))
            self.h = np.sqrt((n+m+2)*(n-m-1) / ((n-m)*(n+m+1)))
        self.absm = absm
        if not (
            np.all(np.isfinite(self.sqrt_factorial_ratio)) and
            np.all(np.isfinite(self.a)) and
            np.all(np.isfinite(self.b)) and
            np.all(np.isfinite(self.d)) and
            np.all(np.isfinite(self.absm))
        ):
            raise ValueError("Found a non-finite value inside this object")

    
    
