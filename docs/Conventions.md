# Conventions

Defined exclusively in terms of quaternions, the Wigner $\mathfrak{D}$
matrices and spherical-harmonic functions (spin-weighted and scalar)
are pretty simple, and easy to make internally consistent.  However,
it is important to establish which conventions are in use —
especially in comparison to other sources.  Here, I carefully examine
all the assumptions built in to the conventions for the
`spherical` package, and relate these choices to those made
by other authors.

On this page, we simply introduce the basic conventions and notation
for rotations in terms of the beautiful, elegant, efficient, and
highly intuitive presentation with quaternions.  We then compare to
uglier, old-fashioned presentations in terms of spherical coordinates
and the profoundly hideous Euler angles.  On
[another page](/WignerDMatrices), we will see how to get directly
from quaternions to highly efficient and accurate formulas for the
Wigner $\mathfrak{D}$ matrices, with no use of Euler angles
whatsoever.  Similarly, [we will be able](/SWSHs) to express
spin-weighted spherical harmonics directly in terms of quaternions,
though with a simple translation to and from standard spherical
coordinates.  This will allow us to derive simple rotation laws for
the SWSHs and modes of a general decomposition in terms of SWSHs.
