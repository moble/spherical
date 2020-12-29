
import numpy as np
from .. import jit

# # The formulas used in these functions were calculated in sympy using the expressions in comments
# # throughout this file.  To start, evaluate these lines:
#
# from sympy import symbols, summation, horner
# ell,mp,m,ell_min,ell_max,mp_max = symbols('ell,mp,m,ell_min,ell_max,mp_max', integer=True)


@jit
def WignerDsize(ell_max):
    """Compute total size of Wigner 𝔇 matrix

    Parameters
    ----------
    ell_max : int
        Integer satisfying 0 <= ell_max

    Returns
    -------
    i : int
        Total size of Wigner 𝔇 matrix arranged as in Notes

    See Also
    --------
    WignerDsize_mpmax_ellmin : For general ell_min and mp_max arguments
    WignerDsize_mpmax : Assuming ell_min=0
    WignerDsize_ellmin : Assuming mp_max=0

    Notes
    -----
    This assumes that the Wigner 𝔇 matrix is arranged as

        [
            𝔇(ℓ, mp, m)
            for ℓ in range(ell_max+1)
            for mp in range(-ℓ, ℓ+1)
            for m in range(-ℓ, ℓ+1)
        ]

    """
    # # Assuming mp_max = ell_max and ell_min=0, this is the calculation we need:
    # horner(
    #     (
    #         # sum over all mp=0 elements
    #         summation(summation(summation(1, (m, -ell, ell)), (mp, 0, 0)), (ell, 0, ell_max))
    #         # sum over all ell_min <= ell <= ell_max elements
    #         + summation(summation(summation(2, (m, -ell, ell)), (mp, 1, ell)), (ell, 0, ell_max))
    #     ).expand().simplify(),
    #     (ell_max)
    # )
    return (ell_max * (ell_max * (4 * ell_max + 12) + 11) + 3) // 3


@jit
def WignerDsize_ellmin(ell_min, ell_max):
    """Compute total size of Wigner 𝔇 matrix

    Parameters
    ----------
    ell_min : int
        Integer satisfying 0 <= ell_min <= ell_max
    ell_max : int
        Integer satisfying 0 <= ell_max

    Returns
    -------
    i : int
        Total size of Wigner 𝔇 matrix arranged as in Notes

    See Also
    --------
    WignerDsize_mpmax_ellmin : For general ell_min and mp_max arguments
    WignerDsize_mpmax : Assuming ell_min=0
    WignerDsize : Assuming mp_max=0 and ell_min=0

    Notes
    -----
    This assumes that the Wigner 𝔇 matrix is arranged as

        [
            𝔇(ℓ, mp, m)
            for ℓ in range(ell_min, ell_max+1)
            for mp in range(-ℓ, ℓ+1)
            for m in range(-ℓ, ℓ+1)
        ]

    """
    # # Assuming mp_max = ell_max, this is the calculation we need:
    # horner(
    #     (
    #         # sum over all mp=0 elements
    #         summation(summation(summation(1, (m, -ell, ell)), (mp, 0, 0)), (ell, ell_min, ell_max))
    #         # sum over all ell_min <= ell <= ell_max elements
    #         + summation(summation(summation(2, (m, -ell, ell)), (mp, 1, ell)), (ell, ell_min, ell_max))
    #     ).expand().simplify(),
    #     (ell_min, ell_max)
    # )
    return (
        ell_max * (ell_max * (4 * ell_max + 12) + 11)
        + ell_min * (1 - 4 * ell_min**2)
        + 3
    ) // 3


@jit
def WignerDsize_mpmax(ell_max, mp_max):
    """Compute total size of Wigner 𝔇 matrix

    Parameters
    ----------
    ell_max : int
        Integer satisfying 0 <= ell_max
    mp_max : int
        Integer satisfying 0 <= mp_max >= ell_min

    Returns
    -------
    i : int
        Total size of Wigner 𝔇 matrix arranged as in Notes

    See Also
    --------
    WignerDsize_mpmax_ellmin : For general ell_min and mp_max arguments
    WignerDsize_ellmin : Assuming mp_max=0
    WignerDsize : Assuming mp_max=0 and ell_min=0

    Notes
    -----
    This assumes that the Wigner 𝔇 matrix is arranged as

        [
            𝔇(ℓ, mp, m)
            for ℓ in range(ell_max+1)
            for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
            for m in range(-ℓ, ℓ+1)
        ]

    """
    # # Assuming ell_min=0, this is the calculation we need:
    # horner(
    #     (
    #         # sum over all mp=0 elements
    #         summation(summation(summation(1, (m, -ell, ell)), (mp, 0, 0)), (ell, 0, ell_max))
    #         # sum over all ell <= mp_max elements
    #         + summation(summation(summation(2, (m, -ell, ell)), (mp, 1, ell)), (ell, 0, mp_max))
    #         # sum over all ell >= mp_max elements
    #         + summation(summation(summation(2, (m, -ell, ell)), (mp, 1, mp_max)), (ell, mp_max+1, ell_max))
    #     ).expand().simplify(),
    #     (mp_max, ell_max)
    # )
    return (
        3 * ell_max * (ell_max + 2)
        + mp_max * (
            3 * ell_max * (2 * ell_max + 4)
            + mp_max * (-2 * mp_max - 3) + 5
        )
        + 3
    ) // 3


@jit
def WignerDsize_mpmax_ellmin(ell_min, ell_max, mp_max):
    """Compute total size of Wigner 𝔇 matrix

    Parameters
    ----------
    ell_min : int
        Integer satisfying 0 <= ell_min <= ell_max
    ell_max : int
        Integer satisfying 0 <= ell_max
    mp_max : int
        Integer satisfying 0 <= ell_min <= mp_max

    Returns
    -------
    i : int
        Total size of Wigner 𝔇 matrix arranged as in Notes

    See Also
    --------
    WignerDsize_mpmax_ellmin : For general ell_min and mp_max arguments
    WignerDsize_mpmax : Assuming ell_min=0
    WignerDsize_ellmin : Assuming mp_max=0
    WignerDsize : Assuming mp_max=0 and ell_min=0

    Notes
    -----
    This assumes that the Wigner 𝔇 matrix is arranged as

        [
            𝔇(ℓ, mp, m)
            for ℓ in range(ell_min, ell_max+1)
            for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
            for m in range(-ℓ, ℓ+1)
        ]

    """
    # # Assuming mp_max >= ell_min, this is the calculation we need:
    # horner(
    #     (
    #         # sum over all mp=0 elements
    #         summation(summation(summation(1, (m, -ell, ell)), (mp, 0, 0)), (ell, ell_min, ell_max))
    #         # sum over all ell <= mp_max elements
    #         + summation(summation(summation(2, (m, -ell, ell)), (mp, 1, ell)), (ell, ell_min, mp_max))
    #         # sum over all ell >= mp_max elements
    #         + summation(summation(summation(2, (m, -ell, ell)), (mp, 1, mp_max)), (ell, mp_max+1, ell_max))
    #     ).expand().simplify(),
    #     (mp_max, ell_min, ell_max)
    # )
    # # If mp_max <= ell_min, we need this calculation:
    # horner(
    #     (
    #         # sum over all mp=0 elements
    #         summation(summation(summation(1, (m, -ell, ell)), (mp, 0, 0)), (ell, ell_min, ell_max))
    #         # sum over all remaining |mp| <= mp_max elements
    #         + summation(summation(summation(2, (m, -ell, ell)), (mp, 1, mp_max)), (ell, ell_min, ell_max))
    #     ).expand().simplify(),
    #     (mp_max, ell_min, ell_max)
    # )
    if mp_max > ell_min:
        return (
            3 * ell_max * (ell_max + 2)
            + ell_min * (1 - 4 * ell_min**2)
            + mp_max * (
                3 * ell_max * (2 * ell_max + 4)
                + mp_max * (-2 * mp_max - 3) + 5
            )
            + 3
        ) // 3
    else:
        return (ell_max * (ell_max + 2) - ell_min**2) * (1 + 2 * mp_max) + 2 * mp_max + 1


@jit
def LMpM_index(ell_min, ell_max, mp_max, ell, mp, m):
    assert ell_min <= ell <= ell_max
    assert abs(mp) <= min(ell, mp_max)
    assert abs(m) <= ell
    # total size of everything with smaller ℓ
    i = LMpM_total_size(ell_min, ell-1, mp_max)
    # total size of everything at this ℓ with smaller mp
    i += (mp + min(mp_max, ell)) * (2 * ell + 1)
    # total size of everything at this ℓ and mp with smaller m
    i += m + ell

    return i


class WignerD:
    def __init__(self, ell_max, ell_min=0, mp_max=np.iinfo(np.int64).max):
        self.ell_min = int(ell_min)
        self.ell_max = int(ell_max)
        self.mp_max = abs(int(mp_max))

        if ell_min < 0 or ell_min > ell_max:
            raise ValueError(f"ell_min={ell_min} must be non-negative and no greater than ell_max={ell_max}")
        if ell_max < 0:
            raise ValueError(f"ell_max={ell_max} must be non-negative")

        self.total_size = self._total_size

        if mp_max >= ell_max:
            self.index = self._index
        else:
            self.index = self._index_mp_max

    def _total_size(self, ell_max):
        if self.mp_max > self.ell_min:
            return (
                3 * ell_max * (ell_max + 2)
                + self.ell_min * (1 - 4 * self.ell_min**2)
                + self.mp_max * (
                    3 * ell_max * (2 * ell_max + 4)
                    + self.mp_max * (-2 * self.mp_max - 3) + 5
                )
                + 3
            ) // 3
        else:
            return (
                (ell_max * (ell_max + 2) - self.ell_min**2) * (1 + 2 * self.mp_max)
                + 2 * self.mp_max + 1
            )

    @jit
    def _index(self, ell, mp, m):
        # assert self.ell_min <= ell <= self.ell_max
        # assert abs(mp) <= min(ell, self.mp_max)
        # assert abs(m) <= ell

        # total size of everything with smaller ℓ
        i = LMpM_total_size(self.ell_min, ell-1, self.mp_max)
        # total size of everything at this ℓ with smaller mp
        i += (mp + min(self.mp_max, ell)) * (2 * ell + 1)
        # total size of everything at this ℓ and mp with smaller m
        i += m + ell

        return i

    @jit
    def _index_mp_max(self, ell, mp, m):
        pass


class WignerCalculator:
    pass
