import pytest
from dscribe.descriptors import LMBTR

from conftest import (
    assert_n_features,
    assert_dtype,
    assert_basis,
    assert_no_system_modification,
    assert_sparse,
    assert_parallellization,
    assert_symmetries,
    assert_mbtr_location,
    assert_mbtr_location_exception,
    assert_systems,
)

# =============================================================================
# Utilities
default_k2 = {
    "k2": {
        "geometry": {"function": "inverse_distance"},
        "grid": {"min": 0, "max": 1 / 0.7, "sigma": 0.1, "n": 50},
        "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-2},
    }
}

default_k3 = {
    "k3": {
        "geometry": {"function": "angle"},
        "grid": {"min": 0, "max": 180, "sigma": 2, "n": 50},
        "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-2},
    }
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
        pytest.param(default_k2, 3 * default_k2["k2"]["grid"]["n"], id="K2"),
        pytest.param(
            default_k3,
            3 * (3 * 3 - 1) * default_k3["k3"]["grid"]["n"] / 2,
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


@pytest.mark.parametrize(
    "n_jobs, flatten, sparse",
    [
        (1, True, False),  # Serial job, flattened, dense
        (2, True, False),  # Parallel job, flattened, dense
        (1, True, True),  # Serial job, flattened, sparse
        (2, True, True),  # Parallel job, flattened, sparse
    ],
)
def test_parallellization(n_jobs, flatten, sparse):
    assert_parallellization(lmbtr_default_k2, n_jobs, flatten, sparse)


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
            k2=default_k2["k2"],
            periodic=False,
            normalization="n_atoms",
        )

    msg = "Unknown normalization option given. Please use one of the following: l2_each, none."
    assert msg == str(excinfo.value)
