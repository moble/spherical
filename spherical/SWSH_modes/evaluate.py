"""Evaluate modes as functions of rotations


"""

import numpy as np
from .. import jit

one_over_4pi = 1.0 / (4 * np.pi)


def evaluate(modes, R):
    spin_weight = modes.spin_weight
    ell_min = modes.ell_min
    ell_max = modes.ell_max
    
    # Reinterpret inputs as 2-d np.arrays
    mode_weights = modes.ndarray.reshape((-1, modes.shape[-1]))
    quaternions = R.ndarray.reshape((-1, 4))
    
    # Prepare to compute Wigner elements (H is roughly Wigner's d function with nicer properties)
    H = spherical.recursions.HCalculator(ell_max)#, abs(spin_weight))

    # Construct storage space
    workspace = H.workspace([1.0])
    # print(workspace.shape)
    z = np.empty(3, dtype=complex)
    function_values = np.zeros(mode_weights.shape[:-1] + quaternions.shape[:-1], dtype=complex)
    
    # Loop over all input quaternions
    for i_R in range(quaternions.shape[0]):
        # Compute phases exp(iα), exp(iβ), exp(iγ) from quaternion, storing in z
        _quaternion_phases(quaternions[i_R], z)

        # Compute all integer powers zαᵏ for k ∈ [0, ell_max]
        zαpowers = complex_powers(z[0], ell_max)
        # print("zαpowers", np.any(np.isnan(zαpowers)))

        # Compute Wigner H elements for this quaternion
        Hwedge = H(z[1].real, z[1].imag, workspace)[:, 0]
        # print(Hwedge.shape, Hwedge.dtype)
        # print("Hwedge", np.any(np.isnan(Hwedge)))
        # print(f"mode_weights", np.any(np.isnan(mode_weights)))

        _evaluate(mode_weights, function_values[:, i_R], spin_weight, ell_min, ell_max, abs(spin_weight), Hwedge, zαpowers, z[2])
        
    return function_values.reshape(modes.shape[:-1] + R.shape[:-1])

@jit
def _evaluate(mode_weights, function_values, spin_weight, ell_min, ell_max, mp_max, Hwedge, zαpowers, zγ):
    """Helper function for `evaluate`"""
    z̄ᵧ = zᵧ.conjugate()

    # Loop over all input sets of modes
    for i_modes in range(mode_weights.shape[0]):
        f = function_values[i_modes:i_modes+1]
        fₗₘ = mode_weights[i_modes]
        
        ### TODO:
        # 0. Use newer H with narrow wedges (mp_max)
        # 1. Reduce LM_index uses to 1, and then index relative to that one
        # 2. Manually replace wedge_index calls with calls into the existing wedge
        # 3. Call wedge_index helper function directly
        # 4. Replace (-1)**k expressions with tracked signs

        for ell in range(ell_min, ell_max+1):
            # Initialize with m=0 term
            f_tmp = (
                fₗₘ[LM_index(ell, 0, ell_min)]
                * Hwedge[wedge_index(ell, 0, -spin_weight)]#, mp_max)]
            )

            if ell > 0:

                # Compute dˡₘ₋ₛ terms recursively for 0<m<l, using symmetries for negative m, and
                # simultaneously add the mode weights times zᵧᵐ=exp[i(ϕₛ-ϕₐ)m] to the result using
                # Horner form
                negative_terms = (
                    fₗₘ[LM_index(ell, -ell, ell_min)]
                    * Hwedge[wedge_index(ell, -ell, -spin_weight)]#, mp_max)]
                )
                positive_terms = (
                    (-1)**ell
                    * fₗₘ[LM_index(ell, ell, ell_min)]
                    * Hwedge[wedge_index(ell, ell, -spin_weight)]#, mp_max)]
                )
                for m in range(ell-1, 0, -1):
                    negative_terms *= z̄ᵧ
                    negative_terms += (
                        fₗₘ[LM_index(ell, -m, ell_min)]
                        * Hwedge[wedge_index(ell, -m, -spin_weight)]#, mp_max)]
                    )
                #for m in range(ell-1, 0, -1):
                    positive_terms *= zᵧ
                    positive_terms += (
                        (-1)**m
                        * fₗₘ[LM_index(ell, m, ell_min)]
                        * Hwedge[wedge_index(ell, m, -spin_weight)]#, mp_max)]
                    )
                f_tmp += negative_terms * z̄ᵧ
                f_tmp += positive_terms * zᵧ

            f_tmp *= np.sqrt((2 * ell + 1) * one_over_4pi)
            f += f_tmp

        # Finish calculation of f by multiplying by zₐ⁻ˢ = exp[i(ϕₛ+ϕₐ)*(-s)]
        if -spin_weight >= 0:
            f *= (-1)**spin_weight * ϵ(spin_weight) * zαpowers[-spin_weight]
        else:
            f *= (-1)**spin_weight * ϵ(spin_weight) * zαpowers[spin_weight].conjugate()

