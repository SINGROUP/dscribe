import time
import pytest
import numpy as np
from numpy.random import RandomState
from conftest import (
    assert_n_features,
    assert_matrix_descriptor_exceptions,
    assert_matrix_descriptor_sorted,
    assert_matrix_descriptor_eigenspectrum,
    assert_matrix_descriptor_random,
    assert_no_system_modification,
    assert_sparse,
    assert_parallellization,
    assert_symmetries,
    assert_derivatives,
    assert_derivatives_include,
    assert_derivatives_exclude,
    get_complex_periodic,
    get_simple_periodic,
    get_simple_finite,
)
from dscribe.descriptors import CoulombMatrix


# =============================================================================
# Utilities
def cm_python(system, n_atoms_max, permutation, sigma=None):
    """Calculates a python reference value for the Coulomb matrix."""
    pos = system.get_positions()
    n = len(system)
    distances = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    q = system.get_atomic_numbers()
    qiqj = q[None, :] * q[:, None]
    np.fill_diagonal(distances, 1)
    cm = qiqj / distances
    np.fill_diagonal(cm, 0.5 * q**2.4)
    random_state = RandomState(42)

    # Permutation option
    if permutation == "eigenspectrum":
        eigenvalues = np.linalg.eigvalsh(cm)
        abs_values = np.absolute(eigenvalues)
        sorted_indices = np.argsort(abs_values)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        padded = np.zeros((n_atoms_max))
        padded[:n] = eigenvalues
    else:
        if permutation == "sorted_l2":
            norms = np.linalg.norm(cm, axis=1)
            sorted_indices = np.argsort(norms, axis=0)[::-1]
            cm = cm[sorted_indices]
            cm = cm[:, sorted_indices]
        elif permutation == "random":
            norms = np.linalg.norm(cm, axis=1)
            noise_norm_vector = random_state.normal(norms, sigma)
            indexlist = np.argsort(noise_norm_vector)
            indexlist = indexlist[::-1]  # Order highest to lowest
            cm = cm[indexlist][:, indexlist]
        elif permutation == "none":
            pass
        else:
            raise ValueError("Unknown permutation option")

        # Flattening
        padded = np.zeros((n_atoms_max, n_atoms_max))
        padded[:n, :n] = cm
        padded = padded.flatten()

    return padded


def coulomb_matrix(**kwargs):
    """Returns a function that can be used to create a valid CoulombMatrix
    descriptor for a dataset.
    """

    def func(systems=None):
        n_atoms_max = None if systems is None else max([len(s) for s in systems])
        final_kwargs = {
            "n_atoms_max": n_atoms_max,
            "permutation": "none",
        }
        final_kwargs.update(kwargs)
        if (
            final_kwargs["permutation"] == "random"
            and final_kwargs.get("sigma") is None
        ):
            final_kwargs["sigma"] = 2
        return CoulombMatrix(**final_kwargs)

    return func


# =============================================================================
# Common tests with parametrizations that may be specific to this descriptor
@pytest.mark.parametrize(
    "permutation, n_features",
    [
        ("none", 9),
        ("eigenspectrum", 3),
        ("sorted_l2", 9),
    ],
)
def test_number_of_features(permutation, n_features):
    assert_n_features(coulomb_matrix(permutation=permutation), n_features)


def test_matrix_descriptor_exceptions():
    assert_matrix_descriptor_exceptions(coulomb_matrix)


def test_matrix_descriptor_sorted():
    assert_matrix_descriptor_sorted(coulomb_matrix)


def test_matrix_descriptor_eigenspectrum():
    assert_matrix_descriptor_eigenspectrum(coulomb_matrix)


def test_matrix_descriptor_random():
    assert_matrix_descriptor_random(coulomb_matrix)


@pytest.mark.parametrize("n_jobs", (1, 2))
@pytest.mark.parametrize("sparse", (True, False))
def test_parallellization(n_jobs, sparse):
    assert_parallellization(coulomb_matrix, n_jobs, sparse)


def test_no_system_modification():
    assert_no_system_modification(coulomb_matrix)


def test_sparse():
    assert_sparse(coulomb_matrix)


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
        coulomb_matrix(permutation=permutation_option),
        translation,
        rotation,
        permutation,
    )


@pytest.mark.parametrize("permutation", ("none", "eigenspectrum", "sorted_l2"))
def test_derivatives_numerical(permutation):
    assert_derivatives(coulomb_matrix(permutation=permutation), "numerical", False)


@pytest.mark.parametrize("method", ("numerical",))
def test_derivatives_include(method):
    assert_derivatives_include(coulomb_matrix(), method)


@pytest.mark.parametrize("method", ("numerical",))
def test_derivatives_exclude(method):
    assert_derivatives_exclude(coulomb_matrix(), method)


# =============================================================================
# Tests that are specific to this descriptor.
@pytest.mark.parametrize(
    "permutation",
    [
        ("none"),
        ("eigenspectrum"),
        ("sorted_l2"),
    ],
)
def test_features(permutation):
    n_atoms_max = 5
    system = get_simple_finite()
    desc = CoulombMatrix(n_atoms_max=n_atoms_max, permutation=permutation)
    cm = desc.create(system)
    cm_assumed = cm_python(system, n_atoms_max, permutation, False)
    assert np.allclose(cm, cm_assumed)


def test_periodicity():
    """Tests that periodicity is not taken into account in Coulomb matrix
    even if the system is set as periodic.
    """
    desc = CoulombMatrix(n_atoms_max=5, permutation="none")
    bulk_system = get_simple_periodic()
    cm = desc.create(bulk_system)
    pos = bulk_system.get_positions()
    assumed = 1 * 1 / np.linalg.norm((pos[0] - pos[1]))
    assert cm[1] == assumed


@pytest.mark.parametrize(
    "permutation",
    [
        "none",
        "eigenspectrum",
        "sorted_l2",
        "random",
    ],
)
def test_performance(permutation):
    """Tests that the C++ code performs better than the numpy version."""
    n_iter = 10
    system = get_complex_periodic()
    start = time
    n_atoms_max = len(system)
    descriptor = coulomb_matrix(permutation=permutation)([system])

    # Measure C++ time
    start = time.time()
    for i in range(n_iter):
        descriptor.create(system)
    end = time.time()
    elapsed_cpp = end - start

    # Measure Python time
    start = time.time()
    for i in range(n_iter):
        cm_python(system, n_atoms_max, permutation, True)
    end = time.time()
    elapsed_python = end - start

    assert elapsed_python > elapsed_cpp
