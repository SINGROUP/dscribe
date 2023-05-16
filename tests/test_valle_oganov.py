import pytest
import numpy as np
from conftest import (
    assert_n_features,
    assert_no_system_modification,
    assert_sparse,
    assert_parallellization,
    assert_symmetries,
    assert_derivatives,
    assert_derivatives_include,
    assert_derivatives_exclude,
    get_simple_finite,
)
from dscribe.descriptors import ValleOganov, MBTR


# =============================================================================
# Utilities
default_k2 = {"function": "distance", "sigma": 10 ** (-0.625), "n": 10, "r_cut": 2.1}
default_k3 = {"function": "angle", "sigma": 10 ** (-0.625), "n": 10, "r_cut": 2.1}


def valle_oganov(**kwargs):
    """Returns a function that can be used to create a valid ValleOganov
    descriptor for a dataset.
    """

    def func(systems=None):
        species = set()
        for system in systems:
            species.update(system.get_atomic_numbers())
        final_kwargs = {
            "species": species,
        }
        final_kwargs.update(kwargs)
        return ValleOganov(**final_kwargs)

    return func


def valle_oganov_default_k2(**kwargs):
    return valle_oganov(**default_k2, **kwargs)


# =============================================================================
# Common tests with parametrizations that may be specific to this descriptor
@pytest.mark.parametrize(
    "setup, n_features",
    [
        (default_k2, 1 / 2 * 2 * (2 + 1) * 10),  # K2
        (default_k3, 1 / 2 * 2 * 2 * (2 + 1) * 10),  # K3
    ],
)
def test_number_of_features(setup, n_features):
    assert_n_features(valle_oganov(**setup), n_features)


@pytest.mark.parametrize("n_jobs", (1, 2))
@pytest.mark.parametrize("sparse", (True, False))
def test_parallellization(n_jobs, sparse):
    assert_parallellization(valle_oganov_default_k2, n_jobs, sparse=sparse)


def test_no_system_modification():
    assert_no_system_modification(valle_oganov_default_k2)


def test_sparse():
    assert_sparse(valle_oganov_default_k2)


def test_symmetries():
    """Tests the symmetries of the descriptor."""
    assert_symmetries(valle_oganov_default_k2())


def test_derivatives_numerical():
    assert_derivatives(valle_oganov_default_k2(), "numerical", False)


def test_derivatives_analytical():
    assert_derivatives(valle_oganov_default_k2(), "analytical", False)


@pytest.mark.parametrize("method", ("numerical",))
def test_derivatives_include(method):
    assert_derivatives_include(valle_oganov_default_k2(), method)


@pytest.mark.parametrize("method", ("numerical",))
def test_derivatives_exclude(method):
    assert_derivatives_exclude(valle_oganov_default_k2(), method)


# =============================================================================
# Tests that are specific to this descriptor.
@pytest.mark.parametrize(
    "vo_setup, mbtr_setup",
    [
        pytest.param(
            {"function": "distance", "sigma": 0.1, "n": 20, "r_cut": 5},
            {
                "geometry": {"function": "distance"},
                "grid": {"min": 0, "max": 5, "sigma": 0.1, "n": 20},
                "weighting": {"function": "inverse_square", "r_cut": 5},
            },
            id="distance",
        ),
        pytest.param(
            {"function": "angle", "sigma": 0.1, "n": 20, "r_cut": 5},
            {
                "geometry": {"function": "angle"},
                "grid": {"min": 0, "max": 180, "sigma": 0.1, "n": 20},
                "weighting": {"function": "smooth_cutoff", "r_cut": 5},
            },
            id="angle",
        ),
    ],
)
def test_vs_mbtr(vo_setup, mbtr_setup):
    """Tests that the ValleOganov subclass gives the same output as MBTR with
    the corresponding parameters.
    """
    system = get_simple_finite()
    desc = ValleOganov(species=[1, 8], **vo_setup)
    feat = desc.create(system)

    desc2 = MBTR(
        species=[1, 8],
        periodic=True,
        **mbtr_setup,
        normalization="valle_oganov",
        sparse=False,
    )
    feat2 = desc2.create(system)

    assert np.array_equal(feat, feat2)
