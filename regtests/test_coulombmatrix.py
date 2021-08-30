import pytest
import numpy as np
import sparse
from ase.build import molecule
from conftest import (
    check_symmetry_rotation,
    check_symmetry_translation,
    check_derivatives_include,
    check_derivatives_exclude,
    check_derivatives_numerical,
)
from dscribe.descriptors import CoulombMatrix


def test_exceptions(H2O):
    # Unknown permutation option
    with pytest.raises(ValueError):
        CoulombMatrix(n_atoms_max=5, permutation="unknown")
    # Negative n_atom_max
    with pytest.raises(ValueError):
        CoulombMatrix(n_atoms_max=-1)
    # System has more atoms that the specifed maximum
    with pytest.raises(ValueError):
        cm = CoulombMatrix(n_atoms_max=2)
        cm.create([H2O])


def test_number_of_features():
    desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=False)
    n_features = desc.get_number_of_features()
    assert n_features == 25


def test_periodicity(bulk):
    """Tests that periodicity is not taken into account in Coulomb matrix
    even if the system is set as periodic.
    """
    desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=False)
    cm = desc.create(bulk)
    pos = bulk.get_positions()
    assumed = 1 * 1 / np.linalg.norm((pos[0] - pos[1]))
    assert cm[0, 1] == assumed


def test_flatten(H2O):
    # Unflattened
    desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=False)
    cm = desc.create(H2O)
    assert cm.shape == (5, 5)

    # Flattened
    desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=True)
    cm = desc.create(H2O)
    assert cm.shape == (25,)


def test_sparse(H2O):
    # Dense
    desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=False, sparse=False)
    vec = desc.create(H2O)
    assert type(vec) == np.ndarray

    # Sparse
    desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=True, sparse=True)
    vec = desc.create(H2O)
    assert type(vec) == sparse.COO


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
def test_parallel(n_jobs, flatten, sparse):
    """Tests creating dense output parallelly."""
    samples = [molecule("CO"), molecule("N2O")]
    n_atoms_max = 5
    desc = CoulombMatrix(
        n_atoms_max=n_atoms_max, permutation="none", flatten=flatten, sparse=sparse
    )
    n_features = desc.get_number_of_features()

    output = desc.create(system=samples, n_jobs=n_jobs)
    assumed = (
        np.empty((2, n_features))
        if flatten
        else np.empty((2, n_atoms_max, n_atoms_max))
    )
    a = desc.create(samples[0])
    b = desc.create(samples[1])
    if sparse:
        output = output.todense()
        a = a.todense()
        b = b.todense()
    assumed[0, :] = a
    assumed[1, :] = b
    assert np.allclose(output, assumed)


def test_features(H2O):
    """Tests that the correct features are present in the desciptor."""
    desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=False)
    cm = desc.create(H2O)

    # Test against assumed values
    q = H2O.get_atomic_numbers()
    p = H2O.get_positions()
    norm = np.linalg.norm
    assumed = np.array(
        [
            [
                0.5 * q[0] ** 2.4,
                q[0] * q[1] / (norm(p[0] - p[1])),
                q[0] * q[2] / (norm(p[0] - p[2])),
            ],
            [
                q[1] * q[0] / (norm(p[1] - p[0])),
                0.5 * q[1] ** 2.4,
                q[1] * q[2] / (norm(p[1] - p[2])),
            ],
            [
                q[2] * q[0] / (norm(p[2] - p[0])),
                q[2] * q[1] / (norm(p[2] - p[1])),
                0.5 * q[2] ** 2.4,
            ],
        ]
    )
    zeros = np.zeros((5, 5))
    zeros[:3, :3] = assumed
    assumed = zeros
    assert np.array_equal(cm, assumed)


def descriptor_for_system(systems):
    n_atoms_max = max([len(s) for s in systems])
    desc = CoulombMatrix(n_atoms_max=n_atoms_max, permutation="none", flatten=True)
    return desc


def test_symmetries():
    """Tests the symmetries of the descriptor."""
    check_symmetry_translation(descriptor_for_system)
    check_symmetry_rotation(descriptor_for_system)


def test_derivatives():
    methods = ["numerical"]
    check_derivatives_include(descriptor_for_system, methods)
    check_derivatives_exclude(descriptor_for_system, methods)
    # check_derivatives_numerical(descriptor_for_system)
