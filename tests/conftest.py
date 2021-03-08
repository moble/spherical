# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical/blob/master/LICENSE>

import os
import pytest
import numpy as np
import quaternionic

ell_max_default = 36

try:
    import spinsfast
    requires_spinsfast = lambda f: f
except:
    requires_spinsfast = pytest.mark.skip(reason="spinsfast is missing")

try:
    import scipy
    requires_scipy = lambda f: f
except:
    requires_scipy = pytest.mark.skip(reason="scipy is missing")

try:
    import sympy
    requires_sympy = lambda f: f
except:
    requires_sympy = pytest.mark.skip(reason="sympy is missing")



def pytest_addoption(parser):
    parser.addoption("--ell_max", action="store", type=int, default=ell_max_default,
                     help="Maximum ell value to test")
    parser.addoption("--ell_max_slow", action="store", type=int, default=ell_max_default // 2,
                     help="Maximum ell value to test with slow tests")
    parser.addoption("--run_slow_tests", action="store_true", default=False,
                     help="Run all tests, including slow ones")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_slow_tests"):
        return
    skip_slow = pytest.mark.skip(reason="need --run_slow_tests option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getoption("--run_slow_tests"):
        pytest.skip("Need `--run_slow_tests` command-line argument to run")


@pytest.fixture
def ell_max(request):
    return request.config.getoption("--ell_max")


@pytest.fixture
def ell_max_slow(request):
    return request.config.getoption("--ell_max_slow")


@pytest.fixture
def special_angles():
    return np.arange(-1 * np.pi, 1 * np.pi + 0.1, np.pi / 4.)


@pytest.fixture
def on_windows():
    from sys import platform
    return 'win' in platform.lower() and not 'darwin' in platform.lower()


@pytest.fixture
def eps():
    return np.finfo(float).eps


def quaternion_sampler():
    Qs_array = quaternionic.array([
        [np.nan, 0., 0., 0.],
        [np.inf, 0., 0., 0.],
        [-np.inf, 0., 0., 0.],
        [0., 0., 0., 0.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [1.1, 2.2, 3.3, 4.4],
        [-1.1, -2.2, -3.3, -4.4],
        [1.1, -2.2, -3.3, -4.4],
        [
            0.18257418583505537115232326093360,
            0.36514837167011074230464652186720,
            0.54772255750516611345696978280080,
            0.73029674334022148460929304373440
        ],
        [1.7959088706354, 0.515190292664085, 0.772785438996128, 1.03038058532817],
        [2.81211398529184, -0.392521193481878, -0.588781790222817, -0.785042386963756],
    ])
    names = type("QNames", (object,), dict())()
    names.q_nan1 = 0
    names.q_inf1 = 1
    names.q_minf1 = 2
    names.q_0 = 3
    names.q_1 = 4
    names.x = 5
    names.y = 6
    names.z = 7
    names.Q = 8
    names.Qneg = 9
    names.Qbar = 10
    names.Qnormalized = 11
    names.Qlog = 12
    names.Qexp = 13
    return Qs_array, names


@pytest.fixture
def Qs():
    return quaternion_sampler()[0]


@pytest.fixture
def Q_names():
    return quaternion_sampler()[1]


@pytest.fixture
def Q_conditions():
    Qs_array, names = quaternion_sampler()
    conditions = type("QConditions", (object,), dict())()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conditions.zero = np.arange(len(Qs_array))[Qs_array == Qs_array[names.q_0]]
        conditions.nonzero = np.arange(len(Qs_array))[np.nonzero(Qs_array)]
        conditions.nan = np.arange(len(Qs_array))[np.isnan(Qs_array)]
        conditions.nonnan = np.arange(len(Qs_array))[~np.isnan(Qs_array)]
        conditions.nonnannonzero = np.arange(len(Qs_array))[~np.isnan(Qs_array) & (Qs_array != Qs_array[names.q_0])]
        conditions.inf = np.arange(len(Qs_array))[np.isinf(Qs_array)]
        conditions.noninf = np.arange(len(Qs_array))[~np.isinf(Qs_array)]
        conditions.noninfnonzero = np.arange(len(Qs_array))[~np.isinf(Qs_array) & (Qs_array != Qs_array[names.q_0])]
        conditions.finite = np.arange(len(Qs_array))[np.isfinite(Qs_array)]
        conditions.nonfinite = np.arange(len(Qs_array))[~np.isfinite(Qs_array)]
        conditions.finitenonzero = np.arange(len(Qs_array))[np.isfinite(Qs_array) & (Qs_array != Qs_array[names.q_0])]
    return conditions


@pytest.fixture
def Rs():
    np.random.seed(1842)
    ones = [0, -1., 1.]
    rs = [[w, x, y, z] for w in ones for x in ones for y in ones for z in ones][1:]
    rs = rs + [r for r in [quaternionic.array(np.random.uniform(-1, 1, size=4)) for _ in range(20)]]
    return quaternionic.array(rs).normalized
