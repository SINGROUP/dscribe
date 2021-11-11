import pytest
import numpy as np
import scipy
from numpy.random import RandomState
import time
import sparse
from ase.build import molecule
from ase import Atoms
from conftest import (
    assert_symmetry_rotation,
    assert_symmetry_translation,
    assert_symmetry_permutation,
    assert_derivatives_include,
    assert_derivatives_exclude,
    assert_derivatives_numerical,
    big_system,
)
from dscribe.descriptors import CoulombMatrix

random_state = RandomState(42)


def cm_python(system, n_atoms_max, permutation, flatten, sigma=None):
    """Calculates a python reference value for the Coulomb matrix."""
    pos = system.get_positions()
    n = len(system)
    distances = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    q = system.get_atomic_numbers()
    qiqj = q[None, :] * q[:, None]
    np.fill_diagonal(distances, 1)
    cm = qiqj / distances
    np.fill_diagonal(cm, 0.5 * q ** 2.4)

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
            raise ValueError("Unkown permutation option")
        # Flattening
        if flatten:
            cm = cm.flatten()
            padded = np.zeros((n_atoms_max ** 2))
            padded[: n ** 2] = cm
        else:
            padded = np.zeros((n_atoms_max, n_atoms_max))
            padded[:n, :n] = cm

    return padded


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


@pytest.mark.parametrize(
    "permutation, n_features",
    [
        ("none", 25),
        ("eigenspectrum", 5),
        ("sorted_l2", 25),
    ],
)
def test_number_of_features(permutation, n_features):
    desc = CoulombMatrix(n_atoms_max=5, permutation=permutation, flatten=False)
    assert n_features == desc.get_number_of_features()


def test_periodicity(bulk):
    """Tests that periodicity is not taken into account in Coulomb matrix
    even if the system is set as periodic.
    """
    desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=False)
    cm = desc.create(bulk)
    pos = bulk.get_positions()
    assumed = 1 * 1 / np.linalg.norm((pos[0] - pos[1]))
    assert cm[0, 1] == assumed


@pytest.mark.parametrize(
    "permutation",
    [
        ("none"),
        ("eigenspectrum"),
        ("sorted_l2"),
    ],
)
def test_features(permutation, H2O):
    n_atoms_max = 5
    desc = CoulombMatrix(
        n_atoms_max=n_atoms_max, permutation=permutation, flatten=False
    )
    n_features = desc.get_number_of_features()
    cm = desc.create(H2O)
    cm_assumed = cm_python(H2O, n_atoms_max, permutation, False)
    assert np.allclose(cm, cm_assumed)


def test_random():
    """Tests if the random sorting obeys a gaussian distribution. Could
    possibly fail even though everything is OK.

    Measures how many times the two rows with biggest norm exchange place when
    random noise is added. This should correspond to the probability P(X > Y),
    where X = N(\mu_1, \sigma^2), Y = N(\mu_2, \sigma^2). This probability can
    be reduced to P(X > Y) = P(X-Y > 0) = P(N(\mu_1 - \mu_2, \sigma^2 +
    sigma^2) > 0). See e.g.
    https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables
    """
    HHe = Atoms(
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        positions=[
            [0, 0, 0],
            [0.71, 0, 0],
        ],
        symbols=["H", "He"],
    )

    # Get the mean value to compare to
    sigma = 5
    n_atoms_max = 2
    desc = CoulombMatrix(
        n_atoms_max=n_atoms_max, permutation="sorted_l2", flatten=False
    )
    cm = desc.create(HHe)
    means = np.linalg.norm(cm, axis=1)
    mu2 = means[0]
    mu1 = means[1]

    desc = CoulombMatrix(
        n_atoms_max=n_atoms_max,
        permutation="random",
        sigma=sigma,
        seed=42,
        flatten=False,
    )
    count = 0
    rand_instances = 20000
    for i in range(0, rand_instances):
        cm = desc.create(HHe)
        i_means = np.linalg.norm(cm, axis=1)
        if i_means[0] < i_means[1]:
            count += 1

    # The expected probability is calculated from the cumulative
    # distribution function.
    expected = 1 - scipy.stats.norm.cdf(0, mu1 - mu2, np.sqrt(sigma ** 2 + sigma ** 2))
    observed = count / rand_instances

    assert abs(expected - observed) <= 1e-2


def test_flatten(H2O):
    # Unflattened
    desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=False)
    cm_unflattened = desc.create(H2O)
    assert cm_unflattened.shape == (5, 5)

    # Flattened
    desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=True)
    cm_flattened = desc.create(H2O)
    assert cm_flattened.shape == (25,)

    # Check that flattened and unflattened versions contain same values
    assert np.array_equal(cm_flattened[:9], cm_unflattened[:3, :3].ravel())
    assert np.all((cm_flattened[9:] == 0))
    cm_unflattened[:3, :3] = 0
    assert np.all((cm_unflattened == 0))


def test_sparse(H2O):
    # Dense
    desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=True, sparse=False)
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


def descriptor_func(permutation):
    """Returns a function which produces a descriptor object when given a
    dataset to work on.
    """

    def descriptor(systems):
        n_atoms_max = max([len(s) for s in systems])
        sigma = 2 if permutation == "random" else None
        return CoulombMatrix(
            n_atoms_max=n_atoms_max, permutation=permutation, flatten=True, sigma=sigma
        )

    return descriptor


@pytest.mark.parametrize(
    "permutation, test_permutation",
    [
        ("none", False),
        ("eigenspectrum", True),
        ("sorted_l2", False),
    ],
)
def test_symmetries(permutation, test_permutation):
    """Tests the symmetries of the descriptor."""
    func = descriptor_func(permutation)
    assert_symmetry_translation(func)
    assert_symmetry_rotation(func)
    if test_permutation:
        assert_symmetry_permutation(func)


@pytest.mark.parametrize(
    "permutation, method",
    [
        ("none", "numerical"),
        ("eigenspectrum", "numerical"),
        ("sorted_l2", "numerical"),
    ],
)
def test_derivatives(permutation, method):
    func = descriptor_func(permutation)
    assert_derivatives_include(func, method)
    assert_derivatives_exclude(func, method)
    if method == "numerical":
        assert_derivatives_numerical(func)
    else:
        raise Exception("Not implemented yet")


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
    system = big_system()
    times = []
    start = time
    n_atoms_max = len(system)
    descriptor = descriptor_func(permutation)([system])

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
