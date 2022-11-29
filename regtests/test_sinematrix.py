import pytest
import numpy as np
from numpy.random import RandomState
from ase import Atoms
from conftest import (
    assert_matrix_descriptor_exceptions,
    assert_matrix_descriptor_flatten,
    assert_matrix_descriptor_sorted,
    assert_matrix_descriptor_eigenspectrum,
    assert_matrix_descriptor_random,
    assert_no_system_modification,
    assert_sparse,
    assert_parallellization,
    assert_symmetries,
    assert_derivatives,
    big_system,
    water,
)
from dscribe.descriptors import SineMatrix


# =============================================================================
# Utilities
def sine_matrix(**kwargs):
    def func(systems=None):
        n_atoms_max = None if systems is None else max([len(s) for s in systems])
        final_kwargs = {
            "n_atoms_max": n_atoms_max,
            "permutation": "none",
            "flatten": True,
        }
        final_kwargs.update(kwargs)
        if (
            final_kwargs["permutation"] == "random"
            and final_kwargs.get("sigma") is None
        ):
            final_kwargs["sigma"] = 2
        return SineMatrix(**final_kwargs)

    return func


# =============================================================================
# Common tests with parametrizations that may be specific to this descriptor
def test_matrix_descriptor_exceptions():
    assert_matrix_descriptor_exceptions(sine_matrix)


def test_matrix_descriptor_flatten():
    assert_matrix_descriptor_flatten(sine_matrix)


def test_matrix_descriptor_sorted():
    assert_matrix_descriptor_sorted(sine_matrix)


def test_matrix_descriptor_eigenspectrum():
    assert_matrix_descriptor_eigenspectrum(sine_matrix)


def test_matrix_descriptor_random():
    assert_matrix_descriptor_random(sine_matrix)


@pytest.mark.parametrize(
    "n_jobs, flatten, sparse",
    [
        (1, True, False),  # Serial job, flattened, dense
        (2, True, False),  # Parallel job, flattened, dense
        (2, False, False),  # Unflattened output, dense
        (1, True, True),  # Serial job, flattened, sparse
        (2, True, True),  # Parallel job, flattened, sparse
        (2, False, True),  # Unflattened output, sparse
    ],
)
def test_parallellization(n_jobs, flatten, sparse):
    assert_parallellization(sine_matrix, n_jobs, flatten, sparse)


def test_no_system_modification():
    assert_no_system_modification(sine_matrix)


def test_sparse():
    assert_sparse(sine_matrix)


@pytest.mark.parametrize(
    "permutation_option, translation, rotation, permutation",
    [
        ("none", True, True, False),
        ("eigenspectrum", True, True, True),
        ("sorted_l2", False, False, False),
    ],
)
def test_symmetries(permutation_option, translation, rotation, permutation):
    """Tests the symmetries of the descriptor. Notice that sorted_l2 is not
    guaranteed to have any of the symmetries due to numerical issues with rows
    that have nearly equal norm."""
    assert_symmetries(
        sine_matrix(permutation=permutation_option), translation, rotation, permutation
    )

@pytest.mark.parametrize(
    "permutation, method",
    [
        ("none", "numerical"),
        ("eigenspectrum", "numerical"),
        ("sorted_l2", "numerical"),
    ],
)
def test_derivatives(permutation, method):
    assert_derivatives(sine_matrix(permutation=permutation), method, True)

# =============================================================================
# Tests that are specific to this descriptor.
@pytest.mark.parametrize(
    "permutation, n_features",
    [
        ("none", 25),
        ("eigenspectrum", 5),
        ("sorted_l2", 25),
    ],
)
def test_number_of_features(permutation, n_features):
    """Tests that the reported number of features is correct."""
    desc = SineMatrix(n_atoms_max=5, permutation="none", flatten=False)
    n_features = desc.get_number_of_features()
    assert n_features == 25


@pytest.mark.parametrize(
    "permutation",
    [
        ("none"),
        ("eigenspectrum"),
        ("sorted_l2"),
    ],
)
def test_features(permutation):
    """Tests that the correct features are present in the desciptor."""
    desc = SineMatrix(n_atoms_max=2, permutation="none", flatten=False)

    # Test that without cell the matrix cannot be calculated
    system = Atoms(
        positions=[[0, 0, 0], [1.0, 1.0, 1.0]],
        symbols=["H", "H"],
    )
    with pytest.raises(ValueError):
        desc.create(system)

    # Test that periodic boundaries are considered by seeing that an atom
    # in the origin is replicated to the  corners
    system = Atoms(
        cell=[
            [10, 10, 0],
            [0, 10, 0],
            [0, 0, 10],
        ],
        scaled_positions=[[0, 0, 0], [1.0, 1.0, 1.0]],
        symbols=["H", "H"],
        pbc=True,
    )
    # from ase.visualize import view
    # view(system)
    matrix = desc.create(system)

    # The interaction between atoms 1 and 2 should be infinite due to
    # periodic boundaries.
    assert matrix[0, 1] == float("Inf")

    # The interaction of an atom with itself is always 0.5*Z**2.4
    atomic_numbers = system.get_atomic_numbers()
    for i, i_diag in enumerate(np.diag(matrix)):
        assert i_diag == 0.5 * atomic_numbers[i] ** 2.4


def test_unit_cells():
    """Tests if arbitrary unit cells are accepted"""
    desc = SineMatrix(n_atoms_max=3, permutation="none", flatten=False)

    molecule = water()

    molecule.set_cell([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        nocell = desc.create(molecule)

    molecule.set_pbc(True)
    molecule.set_cell([[20.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 40.0]])

    largecell = desc.create(molecule)

    molecule.set_cell([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])

    cubic_cell = desc.create(molecule)

    molecule.set_cell([[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]])

    triclinic_smallcell = desc.create(molecule)
