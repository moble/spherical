"""Evaluate modes as functions of rotations


"""

import numpy as np
import quaternionic

from .. import jit, LM_index
from ..recursions import complex_powers
from ..recursions.wignerH import wedge_index, HCalculator, ϵ

one_over_4pi = 1.0 / (4 * np.pi)


#  Compute f = Σₗₘ fₗₘ ₛYₗₘ = Σₗₘ fₗₘ (-1)ˢ √(2ℓ+1)/(4π) 𝔇ˡₘ,₋ₛ(R), where f is a
#  (possibly spin-weighted) function, fₗₘ are its mode weights in the current
#  frame, s is the spin weight, and f is the function value at R.
#
#    f = Σₗₘ fₗₘ ₛYₗₘ
#      = Σₗₘ fₗₘ (-1)ˢ √(2ℓ+1)/(4π) 𝔇ˡₘ,₋ₛ(R)
#      = Σₗₘ fₗₘ (-1)ˢ √(2ℓ+1)/(4π) dˡₘ₋ₛ(R) exp[iϕₐ(-s-m)+iϕₛ(-s+m)]
#      = Σₗₘ fₗₘ (-1)ˢ √(2ℓ+1)/(4π) dˡₘ₋ₛ(R) exp[-i(ϕₛ+ϕₐ)s+i(ϕₛ-ϕₐ)m]
#      = (-1)ˢ Σₗ √(2ℓ+1)/(4π) exp[-i(ϕₛ+ϕₐ)s] Σₘ fₗₘ dˡₘ₋ₛ(R) exp[i(ϕₛ-ϕₐ)m]
#      = (-1)ˢ zₚ⁻ˢ Σₗ √(2ℓ+1)/(4π) Σₘ fₗₘ dˡₘ₋ₛ(R) zₘᵐ
#      = (-1)ˢ zₚ⁻ˢ Σₗ √(2ℓ+1)/(4π) {fₗ₀ dˡ₀₋ₛ(R) + Σₚₘ [fₗₘ dˡₘ₋ₛ(R) zₘᵐ + fₗ₋ₘ dˡ₋ₘ₋ₛ(R) / zₘᵐ]}
#      = (-1)ˢ zₚ⁻ˢ Σₗ √(2ℓ+1)/(4π) {fₗ₀ ϵₛ Hˡ₀₋ₛ(R) + Σₚₘ [fₗₘ ϵₘ ϵₛ Hˡₘ₋ₛ(R) zₘᵐ + fₗ₋ₘ ϵ₋ₘ ϵₛ Hˡ₋ₘ₋ₛ(R) / zₘᵐ]}
#      = (-1)ˢ ϵₛ zₚ⁻ˢ Σₗ √(2ℓ+1)/(4π) {fₗ₀ Hˡ₀₋ₛ(R) + Σₚₘ [fₗₘ (-1)ᵐ Hˡₘ₋ₛ(R) zₘᵐ + fₗ₋ₘ Hˡ₋ₘ₋ₛ(R) / zₘᵐ]}
#
#     # Σₙ fₗₙ 𝔇ˡₙₘ(R) = ϵ₋ₘ zₚᵐ {fₗ₀ Hˡ₀ₘ(R) + Σₚₙ [fₗₙ (-1)ⁿ Hˡₙₘ(R) zₘⁿ + fₗ₋ₙ Hˡ₋ₙₘ(R) / zₘⁿ]}

def evaluate(modes, R):
    """Evaluate Modes object as function of rotations

    Parameters
    ----------
    modes : Modes object
    R : quaternionic.array
        Arbitrarily shaped array of quaternions.  All modes in the input will be
        evaluated on each of these quaternions.  Note that it is fairly standard to
        construct these quaternions from spherical coordinates, as with the
        function `quaternionic.array.from_spherical_coordinates`.

    Returns
    -------
    f : array_like
        This array holds the complex function values.  Its shape is
        modes.shape[:-1]+R.shape[:-1].

    """
    spin_weight = modes.spin_weight
    ell_min = modes.ell_min
    ell_max = modes.ell_max

    # Reinterpret inputs as 2-d np.arrays
    mode_weights = modes.ndarray.reshape((-1, modes.shape[-1]))
    quaternions = R.ndarray.reshape((-1, 4))

    # Prepare to compute Wigner elements (H is roughly Wigner's d function with nicer properties)
    H = HCalculator(ell_max)#, abs(spin_weight))

    # Construct storage space
    workspace = H.workspace([1.0])
    z = np.empty(3, dtype=complex)
    function_values = np.zeros(mode_weights.shape[:-1] + quaternions.shape[:-1], dtype=complex)

    # Loop over all input quaternions
    for i_R in range(quaternions.shape[0]):
        # Compute phases exp(iα), exp(iβ), exp(iγ) from quaternion, storing in z
        quaternionic.converters._to_euler_phases(quaternions[i_R], z)

        # Compute Wigner H elements for this quaternion
        Hwedge = H(z[1].real, z[1].imag, workspace)[:, 0]

        _evaluate(mode_weights, function_values[:, i_R], spin_weight, ell_min, ell_max, abs(spin_weight), Hwedge, z[0], z[2])

    return function_values.reshape(modes.shape[:-1] + R.shape[:-1])


@jit
def _evaluate(mode_weights, function_values, spin_weight, ell_min, ell_max, mp_max, Hwedge, zₐ, zᵧ):
    """Helper function for `evaluate`"""
    # z̄ᵧ = zᵧ.conjugate()
    z̄ₐ = zₐ.conjugate()
    
    coefficient = (-1)**spin_weight * ϵ(spin_weight) * zᵧ.conjugate()**spin_weight

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
                    # negative_terms *= z̄ᵧ
                    negative_terms *= z̄ₐ
                    negative_terms += (
                        fₗₘ[LM_index(ell, -m, ell_min)]
                        * Hwedge[wedge_index(ell, -m, -spin_weight)]#, mp_max)]
                    )
                    # positive_terms *= zᵧ
                    positive_terms *= zₐ
                    positive_terms += (
                        (-1)**m
                        * fₗₘ[LM_index(ell, m, ell_min)]
                        * Hwedge[wedge_index(ell, m, -spin_weight)]#, mp_max)]
                    )
                # f_tmp += negative_terms * z̄ᵧ
                # f_tmp += positive_terms * zᵧ
                f_tmp += negative_terms * z̄ₐ
                f_tmp += positive_terms * zₐ

            f_tmp *= np.sqrt((2 * ell + 1) * one_over_4pi)
            f += f_tmp

        # # Finish calculation of f by multiplying by zₐ⁻ˢ = exp[i(ϕₛ+ϕₐ)*(-s)]
        # if -spin_weight >= 0:
        #     f *= (-1)**spin_weight * ϵ(spin_weight) * zᵧpowers[-spin_weight]
        # else:
        #     f *= (-1)**spin_weight * ϵ(spin_weight) * zᵧpowers[spin_weight].conjugate()
        f *= coefficient
