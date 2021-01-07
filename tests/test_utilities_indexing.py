#!/usr/bin/env python

# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import numpy as np
import spherical as sf
import pytest

slow = pytest.mark.slow


def test_WignerHrange(ell_max):
    def r(mp_max, ell_max):
        return [
            (ell, mp, m) for ell in range(ell_max+1)
            for mp in range(-min(ell, mp_max), min(ell, mp_max)+1)
            for m in range(abs(mp), ell+1)
        ]
    for ell_max in range(ell_max+1):
        a = sf.WignerHrange(ell_max)  # Implicitly, mp_max=ell_max
        b = r(ell_max, ell_max)
        assert np.array_equal(a, b), ((ell_max, ell_max), a, b)
        for mp_max in range(ell_max+1):
            a = sf.WignerHrange(mp_max, ell_max)
            b = r(mp_max, ell_max)
            assert np.array_equal(a, b), ((mp_max, ell_max), a, b)


def test_WignerHsize(ell_max):
    # k = 0
    for ell_max in range(ell_max+1):
        a = sf.WignerHsize(ell_max)
        b = len(sf.WignerHrange(ell_max, ell_max))
        assert a == b, ((ell_max, ell_max), a, b)#, k)
        for mp_max in range(ell_max+1):
            a = sf.WignerHsize(mp_max, ell_max)
            b = len(sf.WignerHrange(mp_max, ell_max))
            assert a == b, ((mp_max, ell_max), a, b)#, k)
    #       k += 1
    # print(f"Total sizes checked: {k:_}")


@slow
def test_WignerHindex(ell_max_slow):
    # k = 0
    def fold_H_indices(ell, mp, m):
        if m < -mp:
            if m < mp:
                return [ell, -mp, -m]
            else:
                return [ell, -m, -mp]
        else:
            if m < mp:
                return [ell, m, mp]
            else:
                return [ell, mp, m]
    for ell_max in range(ell_max_slow+1):
        r = sf.WignerHrange(ell_max)
        for ell in range(ell_max+1):
            for mp in range(-ell, ell+1):
                for m in range(-ell, ell+1):
                    i = sf.WignerHindex(ell, mp, m)
                    assert np.array_equal(r[i], fold_H_indices(ell, mp, m)), ((ell, mp, m), i, ell_max, r)
                    # k += 1
        for mp_max in range(ell_max+1):
            r = sf.WignerHrange(mp_max, ell_max)
            for ell in range(ell_max+1):
                for mp in range(-ell, ell+1):
                    for m in range(-ell, ell+1):
                        if abs(mp) > mp_max and abs(m) > mp_max:
                            continue
                        i = sf.WignerHindex(ell, mp, m, mp_max)
                        assert np.array_equal(r[i], fold_H_indices(ell, mp, m)), ((ell, mp, m), i, ell_max, mp_max, r)
    #                     k += 1
    # print(f"\nTotal indices checked: {k:_}")


@slow
def test_WignerDrange(ell_max_slow):
    def r(ell_min, mp_max, ell_max):
        return [
            (ℓ, mp, m) for ℓ in range(ell_min, ell_max+1)
            for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
            for m in range(-ℓ, ℓ+1)
        ]
    for ell_max in range(ell_max_slow//2+1):
        for ell_min in range(ell_max+1):
            a = sf.WignerDrange(ell_min, ell_max)  # Implicitly, mp_max=ell_max
            b = r(ell_min, ell_max, ell_max)
            assert np.array_equal(a, b), ((ell_min, mp_max, ell_max), a, b)
            for mp_max in range(ell_max+1):
                a = sf.WignerDrange(ell_min, mp_max, ell_max)
                b = r(ell_min, mp_max, ell_max)
                assert np.array_equal(a, b), ((ell_min, mp_max, ell_max), a, b)


def test_WignerDsize_mpmax_ellmin(ell_max):
    # k = 0
    for ell_max in range(ell_max+1):
        for ell_min in range(ell_max+1):
            for mp_max in range(ell_max+1):
                a = sf.WignerDsize(ell_min, mp_max, ell_max)
                b = len(sf.WignerDrange(ell_min, mp_max, ell_max))
                assert a == b, ((ell_min, mp_max, ell_max), a, b)#, k)
    #             k += 1
    # print(f"Total sizes checked: {k:_}")


def test_WignerDsize_mpmax(ell_max):
    # k = 0
    for ell_max in range(ell_max+1):
        for ell_min in [0]:
            for mp_max in range(ell_max+1):
                a = sf.WignerDsize(ell_min, mp_max, ell_max)
                #a = sf.WignerDsize_mpmax(ell_max, mp_max)
                b = len(sf.WignerDrange(ell_min, mp_max, ell_max))
                assert a == b, ((ell_min, mp_max, ell_max), a, b)#, k)
    #             k += 1
    # print(f"Total sizes checked: {k:_}")


def test_WignerDsize_ellmin(ell_max):
    # k = 0
    for ell_max in range(ell_max+1):
        for ell_min in range(ell_max+1):
            for mp_max in [ell_max]:
                a = sf.WignerDsize(ell_min, mp_max, ell_max)
                # a = sf.WignerDsize_ellmin(ell_min, ell_max)
                b = len(sf.WignerDrange(ell_min, mp_max, ell_max))
                assert a == b, ((ell_min, mp_max, ell_max), a, b)#, k)
    #             k += 1
    # print(f"Total sizes checked: {k:_}")


def test_WignerDsize(ell_max):
    # k = 0
    for ell_max in range(ell_max+1):
        for ell_min in [0]:
            for mp_max in [ell_max]:
                a = sf.WignerDsize(ell_min, mp_max, ell_max)
                # a = sf.WignerDsize(ell_max)
                b = len(sf.WignerDrange(ell_min, mp_max, ell_max))
                assert a == b, ((ell_min, mp_max, ell_max), a, b)#, k)
    #             k += 1
    # print(f"Total sizes checked: {k:_}")


@slow
def test_WignerDindex(ell_max_slow):
    k = 0
    for ell_max in range(ell_max_slow+1):
        r = sf.WignerDrange(0, ell_max)
        for ell in range(ell_max+1):
            for mp in range(-ell, ell+1):
                for m in range(-ell, ell+1):
                    i = sf.WignerDindex(ell, mp, m)
                    assert np.array_equal(r[i], [ell, mp, m]), ((ell, mp, m), i, ell_max, r)
                    k += 1
        for ell_min in range(ell_max+1):
            r = sf.WignerDrange(ell_min, ell_max)
            for ell in range(ell_min, ell_max+1):
                for mp in range(-ell, ell+1):
                    for m in range(-ell, ell+1):
                        i = sf.WignerDindex(ell, mp, m, ell_min)
                        assert np.array_equal(r[i], [ell, mp, m]), ((ell, mp, m), i, ell_min, ell_max, r)
                        k += 1
            for mp_max in range(ell_max+1):
                r = sf.WignerDrange(ell_min, mp_max, ell_max)
                for ell in range(ell_min, ell_max+1):
                    for mp in range(-min(ell, mp_max), min(ell, mp_max)+1):
                        for m in range(-ell, ell+1):
                            i = sf.WignerDindex(ell, mp, m, ell_min, mp_max)
                            assert np.array_equal(r[i], [ell, mp, m]), ((ell, mp, m), i, ell_min, mp_max, ell_max, r)
                        k += 1
    # print(f"\nTotal indices checked: {k:_}")


def test_Yrange(ell_max):
    def r(ell_min, ell_max):
        return [
            (ℓ, m) for ℓ in range(ell_min, ell_max+1)
            for m in range(-ℓ, ℓ+1)
        ]
    for ell_max in range(ell_max+1):
        for ell_min in range(ell_max+1):
            a = sf.Yrange(ell_min, ell_max)
            b = r(ell_min, ell_max)
            assert np.array_equal(a, b), ((ell_min, ell_max), a, b)


def test_Ysize(ell_max):
    # k = 0
    for ell_max in range(ell_max+1):
        for ell_min in range(ell_max+1):
            a = sf.Ysize(ell_min, ell_max)
            b = len(sf.Yrange(ell_min, ell_max))
            assert a == b, ((ell_min, ell_max), a, b)#, k)
    #        k += 1
    # print(f"Total sizes checked: {k:_}")


def test_Yindex(ell_max):
    k = 0
    for ell_max in range(ell_max+1):
        for ell_min in [0]:
            r = sf.Yrange(ell_min, ell_max)
            for ell in range(ell_min, ell_max+1):
                for m in range(-ell, ell+1):
                    i = sf.Yindex(ell, m)
                    assert np.array_equal(r[i], [ell, m]), ((ell, m), i, ell_min, ell_max, r)
                    k += 1
        for ell_min in range(ell_max+1):
            r = sf.Yrange(ell_min, ell_max)
            for ell in range(ell_min, ell_max+1):
                for m in range(-ell, ell+1):
                    i = sf.Yindex(ell, m, ell_min)
                    assert np.array_equal(r[i], [ell, m]), ((ell, m), i, ell_min, ell_max, r)
                    k += 1
    # print(f"\nTotal indices checked: {k:_}")



#####################################################
## Older versions kept for backwards compatibility ##
#####################################################


def test_LM_range(ell_max):
    for l_max in range(ell_max + 1):
        for l_min in range(l_max + 1):
            assert np.array_equal(sf.LM_range(l_min, l_max),
                                  np.array([[ell, m] for ell in range(l_min, l_max + 1) for m in range(-ell, ell + 1)]))


def test_LM_index(ell_max):
    for ell_min in range(ell_max + 1):
        LM = sf.LM_range(ell_min, ell_max)
        for ell in range(ell_min, ell_max + 1):
            for m in range(-ell, ell + 1):
                assert np.array_equal(np.array([ell, m]), LM[sf.LM_index(ell, m, ell_min)])


def test_LM_total_size(ell_max):
    for l_min in range(ell_max + 1):
        for l_max in range(l_min, ell_max + 1):
            assert sf.LM_index(l_max + 1, -(l_max + 1), l_min) == sf.LM_total_size(l_min, l_max)


def test_LMpM_range(ell_max):
    for l_max in range(ell_max + 1):
        assert np.array_equal(sf.LMpM_range(l_max, l_max),
                              np.array([[l_max, mp, m]
                                        for mp in range(-l_max, l_max + 1)
                                        for m in range(-l_max, l_max + 1)]))
        for l_min in range(l_max + 1):
            assert np.array_equal(sf.LMpM_range(l_min, l_max),
                                  np.array([[ell, mp, m]
                                            for ell in range(l_min, l_max + 1)
                                            for mp in range(-ell, ell + 1)
                                            for m in range(-ell, ell + 1)]))


@slow
def test_LMpM_index(ell_max_slow):
    for ell_min in range(ell_max_slow + 1):
        LMpM = sf.LMpM_range(ell_min, ell_max_slow)
        for ell in range(ell_min, ell_max_slow + 1):
            for mp in range(-ell, ell + 1):
                for m in range(-ell, ell + 1):
                    assert np.array_equal(np.array([ell, mp, m]), LMpM[sf.LMpM_index(ell, mp, m, ell_min)])


def test_LMpM_total_size(ell_max):
    for l_min in range(ell_max + 1):
        for l_max in range(l_min, ell_max + 1):
            assert sf.LMpM_index(l_max + 1, -(l_max + 1), -(l_max + 1), l_min) == sf.LMpM_total_size(l_min, l_max)
