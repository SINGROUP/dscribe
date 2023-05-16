import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk
from dscribe.descriptors import LMBTR

from conftest import (
    assert_n_features,
    assert_dtype,
    assert_basis,
    assert_no_system_modification,
    assert_sparse,
    assert_parallellization,
    assert_symmetries,
    assert_normalization,
    assert_centers,
    assert_derivatives,
    assert_derivatives_exclude,
    assert_derivatives_include,
    assert_mbtr_location,
    assert_mbtr_location_exception,
    assert_mbtr_peak,
    assert_systems,
    assert_systems,
    get_simple_finite,
)

# =============================================================================
# Utilities
default_k2 = {
    "geometry": {"function": "inverse_distance"},
    "grid": {"min": 0, "max": 1 / 0.7, "sigma": 0.1, "n": 50},
    "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-2},
}

default_k3 = {
    "geometry": {"function": "angle"},
    "grid": {"min": 0, "max": 180, "sigma": 2, "n": 50},
    "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-2},
}


def lmbtr(**kwargs):
    """Returns a function that can be used to create a valid MBTR
    descriptor for a dataset.
    """

    def func(systems=None):
        species = set()
        for system in systems or []:
            species.update(system.get_atomic_numbers())
        final_kwargs = {
            "species": species,
        }
        final_kwargs.update(kwargs)
        return LMBTR(**final_kwargs)

    return func


def lmbtr_default_k2(**kwargs):
    return lmbtr(**default_k2, **kwargs)


# =============================================================================
# Common tests with parametrizations that may be specific to this descriptor
@pytest.mark.parametrize(
    "setup, n_features",
    [
        pytest.param(default_k2, 3 * default_k2["grid"]["n"], id="K2"),
        pytest.param(
            default_k3,
            3 * (3 * 3 - 1) * default_k3["grid"]["n"] / 2,
            id="K3",
        ),
    ],
)
def test_number_of_features(setup, n_features):
    assert_n_features(lmbtr(**setup), n_features)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("sparse", [True, False])
def test_dtype(dtype, sparse):
    assert_dtype(lmbtr_default_k2, dtype, sparse)


@pytest.mark.parametrize("n_jobs", (1,))
@pytest.mark.parametrize("sparse", (True, False))
@pytest.mark.parametrize(
    "centers",
    [
        "all",
        "indices_fixed",
        "indices_variable",
        "cartesian_fixed",
        "cartesian_variable",
    ],
)
def test_parallellization(n_jobs, sparse, centers):
    assert_parallellization(lmbtr_default_k2, n_jobs, sparse, centers)


def test_no_system_modification():
    assert_no_system_modification(lmbtr_default_k2)


@pytest.mark.parametrize(
    "pbc, cell",
    [
        pytest.param(
            [True, True, True], [5, 5, 5], id="Fully periodic system with cell"
        ),
        pytest.param([False, False, False], None, id="Unperiodic system with no cell"),
        pytest.param(
            [False, False, False], [0, 0, 0], id="Unperiodic system with no cell"
        ),
        pytest.param(
            [True, False, False], [5, 5, 5], id="Partially periodic system with cell"
        ),
        pytest.param(
            [True, True, True],
            [[0.0, 5.0, 5.0], [5.0, 0.0, 5.0], [5.0, 5.0, 0.0]],
            id="Fully periodic system with non-cubic cell",
        ),
    ],
)
def test_systems(pbc, cell):
    assert_systems(lmbtr_default_k2(periodic=True), pbc, cell)


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(default_k2, id="K2"),
        pytest.param(default_k3, id="K3"),
    ],
)
def test_basis(setup):
    assert_basis(lmbtr(**setup, periodic=True))


def test_sparse():
    assert_sparse(lmbtr_default_k2)


def test_symmetries():
    # Local descriptors are not permutation symmetric.
    assert_symmetries(lmbtr(**default_k2), True, True, False)


@pytest.mark.parametrize("pbc", (False, True))
@pytest.mark.parametrize("attach", (False, True))
def test_derivatives_numerical(pbc, attach):
    assert_derivatives(lmbtr_default_k2(), "numerical", pbc, attach=attach)


@pytest.mark.parametrize("method", ("numerical",))
def test_derivatives_include(method):
    assert_derivatives_include(lmbtr_default_k2(), method, False)


@pytest.mark.parametrize("method", ("numerical",))
def test_derivatives_exclude(method):
    assert_derivatives_exclude(lmbtr_default_k2(), method, False)


@pytest.mark.parametrize(
    "normalization, norm_rel, norm_abs",
    [
        ("l2", None, 1),
    ],
)
def test_normalization(normalization, norm_rel, norm_abs):
    """Tests that the normalization works correctly."""
    system = bulk("Cu", "fcc", cubic=True, a=3.6)
    assert_normalization(lmbtr_default_k2, system, normalization, norm_rel, norm_abs)


def test_centers():
    assert_centers(lmbtr_default_k2)


@pytest.mark.parametrize("k", [2, 3])
def test_location(k):
    assert_mbtr_location(lmbtr, k)


@pytest.mark.parametrize(
    "location",
    [
        pytest.param(["X", "G"], id="invalid species"),
        pytest.param(["X", "O"], id="species not specified"),
        pytest.param(["X", "H", "H"], id="invalid k"),
    ],
)
def test_location_exceptions(location):
    assert_mbtr_location_exception(lmbtr(**default_k2, species=["H"])(), location)


@pytest.mark.parametrize(
    "system,k,geometry,grid,weighting,periodic,peaks,prominence",
    [
        pytest.param(
            get_simple_finite(),
            2,
            {"function": "distance"},
            {"min": -1, "max": 3, "sigma": 0.5, "n": 1000},
            {"function": "unity"},
            False,
            [(("X", "H"), [1.4984985], [1]), (("X", "O"), [0.94994995], [1])],
            0.5,
            id="k2 finite",
        ),
        pytest.param(
            Atoms(
                cell=[
                    [10, 0, 0],
                    [10, 10, 0],
                    [10, 0, 10],
                ],
                symbols=["H", "C"],
                scaled_positions=[
                    [0.1, 0.5, 0.5],
                    [0.9, 0.5, 0.5],
                ],
                pbc=True,
            ),
            2,
            {"function": "distance"},
            {"min": 0, "max": 10, "sigma": 0.5, "n": 1000},
            {"function": "exp", "scale": 0.8, "threshold": 1e-3},
            True,
            [(("X", "C"), [2, 8], np.exp(-0.8 * np.array([2, 8])))],
            0.001,
            id="k2 periodic",
        ),
        pytest.param(
            get_simple_finite(),
            3,
            {"function": "angle"},
            {"min": -10, "max": 180, "sigma": 5, "n": 2000},
            {"function": "unity"},
            False,
            [
                (("X", "H", "O"), [38], 1),
                (("X", "O", "H"), [104], 1),
                (("H", "X", "O"), [38], 1),
            ],
            0.5,
            id="k3 finite",
        ),
        pytest.param(
            Atoms(
                cell=[
                    [10, 0, 0],
                    [0, 10, 0],
                    [0, 0, 10],
                ],
                symbols=3 * ["H"],
                scaled_positions=[
                    [0.05, 0.40, 0.5],
                    [0.05, 0.60, 0.5],
                    [0.95, 0.5, 0.5],
                ],
                pbc=True,
            ),
            3,
            {"function": "angle"},
            {"min": 0, "max": 180, "sigma": 5, "n": 2000},
            {"function": "exp", "scale": 0.85, "threshold": 1e-3},
            True,
            [
                (("X", "H", "H"), [45, 90], np.exp([-0.85 * (2 + 2 * np.sqrt(2))] * 2)),
                (("H", "X", "H"), [45], np.exp([-0.85 * (2 + 2 * np.sqrt(2))])),
            ],
            0.01,
            id="k3 periodic",
        ),
    ],
)
def test_peaks(system, k, geometry, grid, weighting, periodic, peaks, prominence):
    """Tests the correct peak locations and intensities are found."""
    assert_mbtr_peak(
        lmbtr, system, k, grid, geometry, weighting, periodic, peaks, prominence
    )


# =============================================================================
# Tests that are specific to this descriptor.
def test_exceptions():
    """Tests different invalid parameters that should raise an
    exception.
    """
    # Cannot use n_atoms normalization
    with pytest.raises(ValueError) as excinfo:
        LMBTR(
            species=[1],
            **default_k2,
            periodic=False,
            normalization="n_atoms",
        )

    msg = (
        "Unknown normalization option given. Please use one of the following: l2, none."
    )
    assert msg == str(excinfo.value)
