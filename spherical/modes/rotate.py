"""Evaluate modes as functions of rotations


"""

import numpy as np
from .. import jit


@jit
def rotate(modes, rotor, ell_min, ell_max, H, workspace):
    fₗₙ = np.asarray(modes, dtype=complex).reshape(-1, modes.shape[-1])
    fₗₘ = np.zeros_like(modes)
    negative_terms = np.zeros(fₗₙ.shape[:-1], dtype=fₗₙ.dtype)
    positive_terms = np.zeros(fₗₙ.shape[:-1], dtype=fₗₙ.dtype)

    zᵧ, zᵦ, zₐ = quaternion_angles(quaternion.as_float_array(rotor))

    # Compute H elements (basically Wigner's d functions)
    Hˡₙₘ = H(np.array([[zᵦ.real]]), np.array([[zᵦ.imag]]), workspace)[:, 0, 0]

    # Pre-compute zₐᵐ=exp[i(ϕₛ+ϕₐ)m] for all *positive* values of m
    zₐᵐ = complex_powers(zₐ, ell_max)

    for ell in range(ell_min, ell_max+1):
        for m in range(-ell, ell+1):
            # fₗₘ = ϵ₋ₘ zₐᵐ {fₗ₀ Hˡ₀ₘ(R) + Σₚₙ [fₗ₋ₙ Hˡ₋ₙₘ(R) / zᵧⁿ + fₗₙ (-1)ⁿ Hˡₙₘ(R) zᵧⁿ]}
            iₘ = LM_index(ell, m, ell_min)

            # Initialize with n=0 term
            fₗₘ[first_slice, iₘ] = (
                fₗₙ[first_slice, LM_index(ell, 0, ell_min)]
                * Hˡₙₘ[wedge_index(*wedgeify_index(ell, 0, m))]
            )

            if ell > 0:

                # Compute dˡₙₘ terms recursively for 0<n<l, using symmetries for negative n, and
                # simultaneously add the mode weights times zᵧⁿ=exp[i(ϕₛ-ϕₐ)n] to the result using
                # Horner form
                negative_terms[first_slice] = (
                    fₗₙ[first_slice, LM_index(ell, -ell, ell_min)]
                    * Hˡₙₘ[wedge_index(*wedgeify_index(ell, -ell, m))]
                )
                positive_terms[first_slice] = (
                    (-1)**ell
                    * fₗₙ[first_slice, LM_index(ell, ell, ell_min)]
                    * Hˡₙₘ[wedge_index(*wedgeify_index(ell, ell, m))]
                )
                for n in range(ell-1, 0, -1):
                    negative_terms *= zᵧ.conjugate()
                    negative_terms += (
                        fₗₙ[first_slice, LM_index(ell, -n, ell_min)]
                        * Hˡₙₘ[wedge_index(*wedgeify_index(ell, -n, m))]
                    )
                    positive_terms *= zᵧ
                    positive_terms += (
                        (-1)**n
                        * fₗₙ[first_slice, LM_index(ell, n, ell_min)]
                        * Hˡₙₘ[wedge_index(*wedgeify_index(ell, n, m))]
                    )
                fₗₘ[first_slice, iₘ] += negative_terms * zᵧ.conjugate()
                fₗₘ[first_slice, iₘ] += positive_terms * zᵧ

            # Finish calculation of fₗₘ by multiplying by zₐᵐ=exp[i(ϕₛ+ϕₐ)m]
            if m >= 0:
                fₗₘ[first_slice, iₘ] *= ϵ(-m) * zₐᵐ[m]
            else:
                fₗₘ[first_slice, iₘ] *= ϵ(-m) * zₐᵐ[-m].conjugate()

    fₗₘ = fₗₘ.reshape(modes.shape)
    return fₗₘ
