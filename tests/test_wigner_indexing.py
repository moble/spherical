#!/usr/bin/env python

# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import numpy as np
import spherical as sf


def test_Wigner_indexing():
    ell_max_max = 12
    # print()
    for ell_max in range(ell_max_max+1):
        # print(f"\tell_max={ell_max} (<{total_ell_max+1})")
        for ell_min in range(ell_max+1):
            for mp_max in range(ell_max+1):
                wigner = sf.Wigner(ell_max, ell_min, mp_max)

                H_indices = [
                    [ℓ, mp, m]
                    for ℓ in range(ell_max+1)
                    for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
                    for m in range(abs(mp), ℓ+1)
                ]
                assert wigner.Hsize == len(H_indices)
                assert wigner.Hsize == sf.WignerHsize(wigner.mp_max, wigner.ell_max)
                for ℓ in range(ell_max+1):
                    for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1):
                        for m in range(abs(mp), ℓ+1):
                            assert H_indices[wigner.Hindex(ℓ, mp, m)] == [ℓ, mp, m]

                d_indices = [
                    [ℓ, mp, m]
                    for ℓ in range(ell_min, ell_max+1)
                    for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
                    for m in range(-ℓ, ℓ+1)
                ]
                assert wigner.dsize == len(d_indices)
                assert wigner.dsize == sf.WignerDsize(wigner.ell_min, wigner.mp_max, wigner.ell_max)
                for ℓ in range(ell_min, ell_max+1):
                    for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1):
                        for m in range(-ℓ, ℓ+1):
                            assert d_indices[wigner.dindex(ℓ, mp, m)] == [ℓ, mp, m]
        
                𝔇_indices = [
                    [ℓ, mp, m]
                    for ℓ in range(ell_min, ell_max+1)
                    for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1)
                    for m in range(-ℓ, ℓ+1)
                ]
                assert wigner.Dsize == len(𝔇_indices)
                assert wigner.Dsize == sf.WignerDsize(wigner.ell_min, wigner.mp_max, wigner.ell_max)
                for ℓ in range(ell_min, ell_max+1):
                    for mp in range(-min(ℓ, mp_max), min(ℓ, mp_max)+1):
                        for m in range(-ℓ, ℓ+1):
                            assert 𝔇_indices[wigner.Dindex(ℓ, mp, m)] == [ℓ, mp, m]

                for s in range(-mp_max, mp_max+1):
                    Y_indices = [
                        [s, ℓ, m]
                        for ℓ in range(ell_min, ell_max+1)
                        for m in range(-ℓ, ℓ+1)
                    ]
                    assert wigner.Ysize == len(Y_indices)
                    assert wigner.Ysize == sf.Ysize(wigner.ell_min, wigner.ell_max)
                    for ℓ in range(ell_min, ell_max+1):
                        for m in range(-ℓ, ℓ+1):
                            assert Y_indices[wigner.Yindex(ℓ, m)] == [s, ℓ, m]
