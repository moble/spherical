import numpy as np


def wignerD_indices(ell_min, ell_max, mp_max):
    return [
        (ℓ, mp, m) for ℓ in range(ell_min, ell_max+1)
        for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
        for m in range(-ℓ, ℓ+1)
    ]


def test_WignerDsize_mpmax_ellmin():
    from spherical.recursions.wigner import WignerDsize_mpmax_ellmin
    # k = 0
    for ell_max in range(24+1):
        for ell_min in range(ell_max+1):
            for mp_max in range(ell_max+1):
                a = WignerDsize_mpmax_ellmin(ell_min, ell_max, mp_max)
                b = len(wignerD_indices(ell_min, ell_max, mp_max))
                assert a == b, ((ell_min, ell_max, mp_max), a, b, k)
    #             k += 1
    # print(f"Total sizes checked: {k:_}")


def test_WignerDsize_mpmax():
    from spherical.recursions.wigner import WignerDsize_mpmax
    # k = 0
    for ell_max in range(32+1):
        for ell_min in [0]:
            for mp_max in range(ell_max+1):
                a = WignerDsize_mpmax(ell_max, mp_max)
                b = len(wignerD_indices(ell_min, ell_max, mp_max))
                assert a == b, ((ell_min, ell_max, mp_max), a, b, k)
    #             k += 1
    # print(f"Total sizes checked: {k:_}")


def test_WignerDsize_ellmin():
    from spherical.recursions.wigner import WignerDsize_ellmin
    # k = 0
    for ell_max in range(32+1):
        for ell_min in range(ell_max+1):
            for mp_max in [ell_max]:
                a = WignerDsize_ellmin(ell_min, ell_max)
                b = len(wignerD_indices(ell_min, ell_max, mp_max))
                assert a == b, ((ell_min, ell_max, mp_max), a, b, k)
    #             k += 1
    # print(f"Total sizes checked: {k:_}")


def test_WignerDsize():
    from spherical.recursions.wigner import WignerDsize
    # k = 0
    for ell_max in range(64+1):
        for ell_min in [0]:
            for mp_max in [ell_max]:
                a = WignerDsize(ell_max)
                b = len(wignerD_indices(ell_min, ell_max, mp_max))
                assert a == b, ((ell_min, ell_max, mp_max), a, b, k)
    #             k += 1
    # print(f"Total sizes checked: {k:_}")
