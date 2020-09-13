import numpy as np
from numpy import *
import pytest
import spherical as sf

multiplication_functions = [sf.multiply,]

try:
    import spinsfast
    spinsfast_not_present = False

    def spinsfast_multiply(f, ellmin_f, ellmax_f, s_f, g, ellmin_g, ellmax_g, s_g):
        """Multiply functions pointwise

        This function takes the same arguments as the spherical.multiply function and
        returns the same quantities.  However, this performs the multiplication simply by
        transforming the modes to function values on a (theta, phi) grid, multiplying those values
        at each point, and then transforming back to modes.

        """
        s_fg = s_f + s_g
        ellmin_fg = 0
        ellmax_fg = ellmax_f + ellmax_g
        n_theta = 2*ellmax_fg + 1
        n_phi = n_theta
        f_map = spinsfast.salm2map(f, s_f, ellmax_f, n_theta, n_phi)
        g_map = spinsfast.salm2map(g, s_g, ellmax_g, n_theta, n_phi)
        fg_map = f_map * g_map
        fg = spinsfast.map2salm(fg_map, s_fg, ellmax_fg)
        return fg, ellmin_fg, ellmax_fg, s_fg

    multiplication_functions += [spinsfast_multiply]

except ImportError:
    spinsfast_not_present = True

requires_spinsfast = pytest.mark.skipif(spinsfast_not_present, reason="Requires spinsfast to be importable")


@pytest.mark.parametrize("multiplication_function", multiplication_functions)
def test_trivial_multiplication(multiplication_function):
    """Test 1*g and g*1

    We test trivial multiplication by 1.  The modes of the `1` function are just sqrt(4*pi) for the
    (ell,m)=(0,0) mode, and 0 for everything else.  Even though this can be described by just the
    first element, we test this for each ellmax_f up to 8 just to make sure the multiplication
    function works as expected.  This is multiplied by a function with random values input for its
    modes.

    """
    np.random.seed(1234)
    atol = 2e-15
    rtol = 2e-15
    ellmax_g = 8
    for ellmax_f in range(1, ellmax_g):
        f = np.zeros(sf.LM_total_size(0, ellmax_f), dtype=np.complex)
        f[0] = np.sqrt(4*np.pi)
        ellmin_f = 0
        s_f = 0
        ellmin_g = 0
        for s_g in range(1-ellmax_g, ellmax_g):
            i_max = sf.LM_index(ellmax_g, ellmax_g, ellmin_g)
            g = np.random.rand(sf.LM_total_size(0, ellmax_g)) + 1j*np.random.rand(sf.LM_total_size(0, ellmax_g))
            g[:sf.LM_total_size(0, abs(s_g)-1)+1] = 0.0
            fg, ellmin_fg, ellmax_fg, s_fg = multiplication_function(f, ellmin_f, ellmax_f, s_f, g, ellmin_g, ellmax_g, s_g)
            assert s_fg == s_f + s_g
            assert ellmin_fg == 0
            assert ellmax_fg == ellmax_f + ellmax_g
            assert np.allclose(fg[:i_max+1], g[:i_max+1], atol=atol, rtol=rtol)
            gf, ellmin_gf, ellmax_gf, s_gf = multiplication_function(g, ellmin_g, ellmax_g, s_g, f, ellmin_f, ellmax_f, s_f)
            assert s_gf == s_g + s_f
            assert ellmin_gf == 0
            assert ellmax_gf == ellmax_g + ellmax_f
            assert np.allclose(gf[:i_max+1], g[:i_max+1], atol=atol, rtol=rtol)


@requires_spinsfast
@pytest.mark.parametrize("multiplication_function", multiplication_functions)
def test_SWSH_multiplication_formula(multiplication_function):
    """Test formula for multiplication of SWSHs

    Much of the analysis is based on the formula

    s1Yl1m1 * s2Yl2m2 = sum([
        s3Yl3m3 * (-1)**(l1+l2+l3+s3+m3) * sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*pi))
            * Wigner3j(l1, l2, l3, s1, s2, -s3) * Wigner3j(l1, l2, l3, m1, m2, -m3)
        for s3 in [s1+s2]
        for l3 in range(abs(l1-l2), l1+l2+1)
        for m3 in [m1+m2]
    ])

    This test evaluates each side of this formula, and compares the values at all collocation points
    on the sphere.  This is tested for each possible value of (s1, l1, m1, s2, l2, m2) up to l1=4
    and l2=4 [the number of items to test scales very rapidly with ell], and tested again for each
    (0, 1, m1, s2, l2, m2) up to l2=8.

    """
    atol=2e-15
    rtol=2e-15
    ell_max = 4
    for ell1 in range(ell_max+1):
        for s1 in range(-ell1, ell1+1):
            for m1 in range(-ell1, ell1+1):
                for ell2 in range(ell_max+1):
                    for s2 in range(-ell2, ell2+1):
                        for m2 in range(-ell2, ell2+1):
                            swsh1 = np.zeros(sf.LM_total_size(0, ell_max), dtype=np.complex)
                            swsh1[sf.LM_index(ell1, m1, 0)] = 1.0
                            swsh2 = np.zeros(sf.LM_total_size(0, ell_max), dtype=np.complex)
                            swsh2[sf.LM_index(ell2, m2, 0)] = 1.0
                            swsh3, ell_min_3, ell_max_3, s3 = multiplication_function(swsh1, 0, ell_max, s1, swsh2, 0, ell_max, s2)
                            assert s3 == s1 + s2
                            assert ell_min_3 == 0
                            assert ell_max_3 == 2*ell_max
                            n_theta = 2*ell_max_3 + 1
                            n_phi = n_theta
                            swsh3_map = spinsfast.salm2map(swsh3, s3, ell_max_3, n_theta, n_phi)
                            swsh4_map = np.zeros_like(swsh3_map)
                            for ell4 in range(abs(ell1-ell2), ell1+ell2+1):
                                for s4 in [s1+s2]:
                                    for m4 in [m1+m2]:
                                        swsh4_k = np.zeros_like(swsh3)
                                        swsh4_k[sf.LM_index(ell4, m4, 0)] = (
                                            (-1)**(ell1+ell2+ell4+s4+m4)
                                            * np.sqrt((2*ell1+1)*(2*ell2+1)*(2*ell4+1)/(4*np.pi))
                                            * sf.Wigner3j(ell1, ell2, ell4, s1, s2, -s4)
                                            * sf.Wigner3j(ell1, ell2, ell4, m1, m2, -m4)
                                        )
                                        swsh4_map[:] += spinsfast.salm2map(swsh4_k, s4, ell_max_3, n_theta, n_phi)
                            assert np.allclose(swsh3_map, swsh4_map, atol=atol, rtol=rtol)

    atol=8e-15
    rtol=8e-15
    ell_max = 8
    for ell1 in [1]:
        for s1 in [0]:
            for m1 in [-1, 0, 1]:
                for ell2 in range(ell_max+1):
                    for s2 in range(-ell2, ell2+1):
                        for m2 in range(-ell2, ell2+1):
                            swsh1 = np.zeros(sf.LM_total_size(0, ell_max), dtype=np.complex)
                            swsh1[sf.LM_index(ell1, m1, 0)] = 1.0
                            swsh2 = np.zeros(sf.LM_total_size(0, ell_max), dtype=np.complex)
                            swsh2[sf.LM_index(ell2, m2, 0)] = 1.0
                            swsh3, ell_min_3, ell_max_3, s3 = multiplication_function(swsh1, 0, ell_max, s1, swsh2, 0, ell_max, s2)
                            assert s3 == s1 + s2
                            assert ell_min_3 == 0
                            assert ell_max_3 == 2*ell_max
                            n_theta = 2*ell_max_3 + 1
                            n_phi = n_theta
                            swsh3_map = spinsfast.salm2map(swsh3, s3, ell_max_3, n_theta, n_phi)
                            swsh4_map = np.zeros_like(swsh3_map)
                            for ell4 in [ell2-1, ell2, ell2+1]:
                                if ell4 < 0:
                                    continue
                                swsh4_k = np.zeros_like(swsh3)
                                # swsh4_k[sf.LM_index(ell4, m1+m2, 0)] = (
                                #     (-1)**(1+ell2+ell4+s2+m1+m2)
                                #     * np.sqrt(3*(2*ell2+1)*(2*ell4+1)/(4*np.pi))
                                #     * sf.Wigner3j(1, ell2, ell4, 0, s2, -s2)
                                #     * sf.Wigner3j(1, ell2, ell4, m1, m2, -m1-m2)
                                # )
                                # swsh4_map[:] += (
                                #     spinsfast.salm2map(swsh4_k, s2, ell_max_3, n_theta, n_phi)
                                # )
                                swsh4_k[sf.LM_index(ell4, m1+m2, 0)] = (
                                    (-1)**(ell2+ell4+m1)
                                    * np.sqrt((2*ell4+1))
                                    * sf.Wigner3j(1, ell2, ell4, 0, s2, -s2)
                                    * sf.Wigner3j(1, ell2, ell4, m1, m2, -m1-m2)
                                )
                                swsh4_map[:] += (
                                    (-1)**(s2+m2+1)
                                    * np.sqrt(3*(2*ell2+1)/(4*np.pi))
                                    * spinsfast.salm2map(swsh4_k, s2, ell_max_3, n_theta, n_phi)
                                )
                            assert np.allclose(swsh3_map, swsh4_map, atol=atol, rtol=rtol), np.max(np.abs(swsh3_map-swsh4_map))


@pytest.mark.parametrize("multiplication_function", multiplication_functions)
def test_first_nontrivial_multiplication(multiplication_function):
    """Test f*g where f is an ell=1, s=0 function

    We can write out the expression for f*g in the case where f is a pure ell=1 function, so we can
    check the multiplication function against that formula.  As in `test_trivial_multiplication`, we
    test with multiple values of ellmax_f, and construct `g` from random values.

    """
    def specialized_multiplication(f, ellmin_f, ellmax_f, s_f, g, ellmin_g, ellmax_g, s_g):
        ellmin_fg = 0
        ellmax_fg = ellmax_f + ellmax_g
        s_fg = s_f + s_g
        fg = np.zeros(sf.LM_total_size(0, ellmax_fg), dtype=np.complex)
        # for ell in range(ellmin_g, ellmax_g+1):
        #     for m in range(-ell, ell+1):
        #         i_g = sf.LM_index(ell, m, ellmin_g)
        #         for ellprime in [ell-1, ell, ell+1]:
        #             if ellprime < ellmin_g or ellprime > ellmax_g or ellprime < abs(s_g):
        #                 continue
        #             for mprime in [m-1, m, m+1]:
        #                 if mprime < -ellprime or mprime > ellprime:
        #                     continue
        #                 i_fg = sf.LM_index(ellprime, mprime, ellmin_fg)
        #                 m1 = mprime - m
        #                 fg[i_fg] += (
        #                     # (-1)**(s_g+m+1)
        #                     # * np.sqrt(3*(2*ell+1)/(4*np.pi))
        #                     # * (-1)**(m1)
        #                     # * np.sqrt(2*ellprime+1)
        #                     (-1)**(1 + ell + ellprime + s_g + m + m1)
        #                     * np.sqrt(3*(2*ell+1)*(2*ellprime+1)/(4*np.pi))
        #                     * sf.Wigner3j(1, ell, ellprime, 0, s_g, -s_g)
        #                     * sf.Wigner3j(1, ell, ellprime, m1, m, -mprime)
        #                     * f[sf.LM_index(1, m1, 0)]
        #                     * g[i_g]
        #                 )
        ####################################################################################
        # for ell in range(ellmin_g, ellmax_g+1):
        #     for m in range(-ell, ell+1):
        #         i_g = sf.LM_index(ell, m, ellmin_g)
        #         coefficient = (-1)**(s_g + m) * np.sqrt(3*(2*ell+1)/(4*np.pi))
        #         for ellprime in [ell-1, ell, ell+1]:
        #             if ellprime < ellmin_g or ellprime > ellmax_g or ellprime < abs(s_g):
        #                 continue
        #             if m-1 >= -ellprime and m-1 <= ellprime:
        #                 i_fg = sf.LM_index(ellprime, m-1, ellmin_fg)
        #                 fg[i_fg] += (
        #                     coefficient
        #                     * (-1)**(ell + ellprime)
        #                     * np.sqrt((2*ellprime+1))
        #                     * sf.Wigner3j(1, ell, ellprime, 0, s_g, -s_g)
        #                     * sf.Wigner3j(1, ell, ellprime, -1, m, -m+1)
        #                     * f[sf.LM_index(1, -1, 0)]
        #                     * g[i_g]
        #                 )
        #             if m >= -ellprime and m <= ellprime:
        #                 i_fg = sf.LM_index(ellprime, m, ellmin_fg)
        #                 fg[i_fg] += (
        #                     coefficient
        #                     * (-1)**(ell + ellprime)
        #                     * -np.sqrt(2*ellprime+1)
        #                     * sf.Wigner3j(1, ell, ellprime, 0, s_g, -s_g)
        #                     * sf.Wigner3j(1, ell, ellprime, 0, m, -m)
        #                     * f[sf.LM_index(1, 0, 0)]
        #                     * g[i_g]
        #                 )
        #             if m+1 >= -ellprime and m+1 <= ellprime:
        #                 i_fg = sf.LM_index(ellprime, m+1, ellmin_fg)
        #                 fg[i_fg] += (
        #                     coefficient
        #                     * (-1)**(ell + ellprime)
        #                     * np.sqrt(2*ellprime+1)
        #                     * sf.Wigner3j(1, ell, ellprime, 0, s_g, -s_g)
        #                     * sf.Wigner3j(1, ell, ellprime, 1, m, -m-1)
        #                     * f[sf.LM_index(1, 1, 0)]
        #                     * g[i_g]
        #                 )
        ####################################################################################
        # for ell in range(ellmin_g, ellmax_g+1):
        #     for m in range(-ell, ell+1):
        #         i_g = sf.LM_index(ell, m, ellmin_g)
        #         coefficient = (-1)**(s_g + m + 1) * np.sqrt(3*(2*ell+1)/(4*np.pi))
        #         if ell-1 >= ellmin_g and ell-1 <= ellmax_g and ell-1 >= abs(s_g):
        #             if m-1 >= -ell+1 and m-1 <= ell-1:
        #                 i_fg = sf.LM_index(ell-1, m-1, ellmin_fg)
        #                 fg[i_fg] += (
        #                     coefficient
        #                     * np.sqrt(2*ell-1)
        #                     * sf.Wigner3j(1, ell, ell-1, 0, s_g, -s_g)
        #                     * sf.Wigner3j(1, ell, ell-1, -1, m, -m+1)
        #                     * f[sf.LM_index(1, -1, 0)]
        #                     * g[i_g]
        #                 )
        #             if m >= -ell+1 and m <= ell-1:
        #                 i_fg = sf.LM_index(ell-1, m, ellmin_fg)
        #                 fg[i_fg] += (
        #                     -coefficient
        #                     * np.sqrt(2*ell-1)
        #                     * sf.Wigner3j(1, ell, ell-1, 0, s_g, -s_g)
        #                     * sf.Wigner3j(1, ell, ell-1, 0, m, -m)
        #                     * f[sf.LM_index(1, 0, 0)]
        #                     * g[i_g]
        #                 )
        #             if m+1 >= -ell+1 and m+1 <= ell-1:
        #                 i_fg = sf.LM_index(ell-1, m+1, ellmin_fg)
        #                 fg[i_fg] += (
        #                     coefficient
        #                     * np.sqrt(2*ell-1)
        #                     * sf.Wigner3j(1, ell, ell-1, 0, s_g, -s_g)
        #                     * sf.Wigner3j(1, ell, ell-1, 1, m, -m-1)
        #                     * f[sf.LM_index(1, 1, 0)]
        #                     * g[i_g]
        #                 )
        #         if ell >= ellmin_g and ell <= ellmax_g and ell >= abs(s_g):
        #             if m-1 >= -ell and m-1 <= ell:
        #                 i_fg = sf.LM_index(ell, m-1, ellmin_fg)
        #                 fg[i_fg] += (
        #                     -coefficient
        #                     * np.sqrt(2*ell+1)
        #                     * sf.Wigner3j(1, ell, ell, 0, s_g, -s_g)
        #                     * sf.Wigner3j(1, ell, ell, -1, m, -m+1)
        #                     * f[sf.LM_index(1, -1, 0)]
        #                     * g[i_g]
        #                 )
        #             if m >= -ell and m <= ell:
        #                 i_fg = sf.LM_index(ell, m, ellmin_fg)
        #                 fg[i_fg] += (
        #                     coefficient
        #                     * np.sqrt(2*ell+1)
        #                     * sf.Wigner3j(1, ell, ell, 0, s_g, -s_g)
        #                     * sf.Wigner3j(1, ell, ell, 0, m, -m)
        #                     * f[sf.LM_index(1, 0, 0)]
        #                     * g[i_g]
        #                 )
        #             if m+1 >= -ell and m+1 <= ell:
        #                 i_fg = sf.LM_index(ell, m+1, ellmin_fg)
        #                 fg[i_fg] += (
        #                     -coefficient
        #                     * np.sqrt(2*ell+1)
        #                     * sf.Wigner3j(1, ell, ell, 0, s_g, -s_g)
        #                     * sf.Wigner3j(1, ell, ell, 1, m, -m-1)
        #                     * f[sf.LM_index(1, 1, 0)]
        #                     * g[i_g]
        #                 )
        #         if ell+1 >= ellmin_g and ell+1 <= ellmax_g and ell+1 >= abs(s_g):
        #             if m-1 >= -ell-1 and m-1 <= ell+1:
        #                 i_fg = sf.LM_index(ell+1, m-1, ellmin_fg)
        #                 fg[i_fg] += (
        #                     coefficient
        #                     * np.sqrt(2*ell+3)
        #                     * sf.Wigner3j(1, ell, ell+1, 0, s_g, -s_g)
        #                     * sf.Wigner3j(1, ell, ell+1, -1, m, -m+1)
        #                     * f[sf.LM_index(1, -1, 0)]
        #                     * g[i_g]
        #                 )
        #             if m >= -ell-1 and m <= ell+1:
        #                 i_fg = sf.LM_index(ell+1, m, ellmin_fg)
        #                 fg[i_fg] += (
        #                     -coefficient
        #                     * np.sqrt(2*ell+3)
        #                     * sf.Wigner3j(1, ell, ell+1, 0, s_g, -s_g)
        #                     * sf.Wigner3j(1, ell, ell+1, 0, m, -m)
        #                     * f[sf.LM_index(1, 0, 0)]
        #                     * g[i_g]
        #                 )
        #             if m+1 >= -ell-1 and m+1 <= ell+1:
        #                 i_fg = sf.LM_index(ell+1, m+1, ellmin_fg)
        #                 fg[i_fg] += (
        #                     coefficient
        #                     * np.sqrt(2*ell+3)
        #                     * sf.Wigner3j(1, ell, ell+1, 0, s_g, -s_g)
        #                     * sf.Wigner3j(1, ell, ell+1, 1, m, -m-1)
        #                     * f[sf.LM_index(1, 1, 0)]
        #                     * g[i_g]
        #                 )
        ####################################################################################
        for ell in range(ellmin_g, ellmax_g+1):
            for m in range(-ell, ell+1):
                i_fg = sf.LM_index(ell, m, ellmin_fg)
                coefficient = (-1)**(s_g + m) * np.sqrt(3*(2*ell+1)/(4*np.pi))
                if ell+1 >= ellmin_g and ell+1 <= ellmax_g and ell+1 >= abs(s_g):
                    if m+1 >= -ell-1 and m+1 <= ell+1:
                        fg[i_fg] += (
                            coefficient
                            * np.sqrt(2*ell+3)
                            * sf.Wigner3j(1, ell+1, ell, 0, s_g, -s_g)
                            * sf.Wigner3j(1, ell+1, ell, -1, m+1, -m)
                            * f[sf.LM_index(1, -1, 0)]
                            * g[sf.LM_index(ell+1, m+1, ellmin_g)]
                        )
                    if m >= -ell-1 and m <= ell+1:
                        fg[i_fg] += (
                            coefficient
                            * np.sqrt(2*ell+3)
                            * sf.Wigner3j(1, ell+1, ell, 0, s_g, -s_g)
                            * sf.Wigner3j(1, ell+1, ell, 0, m, -m)
                            * f[sf.LM_index(1, 0, 0)]
                            * g[sf.LM_index(ell+1, m, ellmin_g)]
                        )
                    if m-1 >= -ell-1 and m-1 <= ell+1:
                        fg[i_fg] += (
                            coefficient
                            * np.sqrt(2*ell+3)
                            * sf.Wigner3j(1, ell+1, ell, 0, s_g, -s_g)
                            * sf.Wigner3j(1, ell+1, ell, 1, m-1, -m)
                            * f[sf.LM_index(1, 1, 0)]
                            * g[sf.LM_index(ell+1, m-1, ellmin_g)]
                        )
                if ell >= ellmin_g and ell <= ellmax_g and ell >= abs(s_g):
                    if m+1 >= -ell and m+1 <= ell:
                        fg[i_fg] += (
                            -coefficient
                            * np.sqrt(2*ell+1)
                            * sf.Wigner3j(1, ell, ell, 0, s_g, -s_g)
                            * sf.Wigner3j(1, ell, ell, -1, m+1, -m)
                            * f[sf.LM_index(1, -1, 0)]
                            * g[sf.LM_index(ell, m+1, ellmin_g)]
                        )
                    if m >= -ell and m <= ell:
                        fg[i_fg] += (
                            -coefficient
                            * np.sqrt(2*ell+1)
                            * sf.Wigner3j(1, ell, ell, 0, s_g, -s_g)
                            * sf.Wigner3j(1, ell, ell, 0, m, -m)
                            * f[sf.LM_index(1, 0, 0)]
                            * g[sf.LM_index(ell, m, ellmin_g)]
                        )
                    if m-1 >= -ell and m-1 <= ell:
                        fg[i_fg] += (
                            -coefficient
                            * np.sqrt(2*ell+1)
                            * sf.Wigner3j(1, ell, ell, 0, s_g, -s_g)
                            * sf.Wigner3j(1, ell, ell, 1, m-1, -m)
                            * f[sf.LM_index(1, 1, 0)]
                            * g[sf.LM_index(ell, m-1, ellmin_g)]
                        )
                if ell-1 >= ellmin_g and ell-1 <= ellmax_g and ell-1 >= abs(s_g):
                    if m+1 >= -ell+1 and m+1 <= ell-1:
                        fg[i_fg] += (
                            coefficient
                            * np.sqrt(2*ell-1)
                            * sf.Wigner3j(1, ell-1, ell, 0, s_g, -s_g)
                            * sf.Wigner3j(1, ell-1, ell, -1, m+1, -m)
                            * f[sf.LM_index(1, -1, 0)]
                            * g[sf.LM_index(ell-1, m+1, ellmin_g)]
                        )
                    if m >= -ell+1 and m <= ell-1:
                        fg[i_fg] += (
                            coefficient
                            * np.sqrt(2*ell-1)
                            * sf.Wigner3j(1, ell-1, ell, 0, s_g, -s_g)
                            * sf.Wigner3j(1, ell-1, ell, 0, m, -m)
                            * f[sf.LM_index(1, 0, 0)]
                            * g[sf.LM_index(ell-1, m, ellmin_g)]
                        )
                    if m-1 >= -ell+1 and m-1 <= ell-1:
                        fg[i_fg] += (
                            coefficient
                            * np.sqrt(2*ell-1)
                            * sf.Wigner3j(1, ell-1, ell, 0, s_g, -s_g)
                            * sf.Wigner3j(1, ell-1, ell, 1, m-1, -m)
                            * f[sf.LM_index(1, 1, 0)]
                            * g[sf.LM_index(ell-1, m-1, ellmin_g)]
                        )
        return fg, ellmin_fg, ellmax_fg, s_fg

    np.random.seed(1234)
    atol = 2e-15
    rtol = 2e-15
    ellmax_g = 8
    #print()
    for ellmax_f in range(1, ellmax_g+1):
        # print(ellmax_f)
        f = np.zeros(sf.LM_total_size(0, ellmax_f), dtype=np.complex)
        f[1:4] = np.random.rand(3) + 1j*np.random.rand(3)
        ellmin_f = 0
        s_f = 0
        ellmin_g = 0
        for s_g in range(1-ellmax_g, ellmax_g):
            # print('\t', s_g)
            i_max = sf.LM_index(ellmax_g, ellmax_g, ellmin_g)
            g = np.random.rand(sf.LM_total_size(0, ellmax_g)) + 1j*np.random.rand(sf.LM_total_size(0, ellmax_g))
            g[:sf.LM_total_size(0, abs(s_g)-1)+1] = 0.0
            fg, ellmin_fg, ellmax_fg, s_fg = multiplication_function(f, ellmin_f, ellmax_f, s_f, g, ellmin_g, ellmax_g, s_g)
            fg2, ellmin_fg2, ellmax_fg2, s_fg2 = specialized_multiplication(f, ellmin_f, ellmax_f, s_f, g, ellmin_g, ellmax_g, s_g)
            assert s_fg == s_f + s_g
            assert ellmin_fg == 0
            assert ellmax_fg == ellmax_f + ellmax_g
            import pprint
            assert np.allclose(fg[:i_max+1], fg2[:i_max+1], atol=atol, rtol=rtol), pprint.pformat(list(fg))+'\n\n'+pprint.pformat(list(fg2))
            gf, ellmin_gf, ellmax_gf, s_gf = multiplication_function(g, ellmin_g, ellmax_g, s_g, f, ellmin_f, ellmax_f, s_f)
            assert s_gf == s_g + s_f
            assert ellmin_gf == 0
            assert ellmax_gf == ellmax_g + ellmax_f
            assert np.allclose(gf[:i_max+1], fg2[:i_max+1], atol=atol, rtol=rtol)
