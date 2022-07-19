import math
import numpy as np
import sparse
import pytest
import scipy
from ase import Atoms
from ase.build import molecule, bulk
from ase.visualize import view
from dscribe.descriptors import CoulombMatrix, SineMatrix, EwaldSumMatrix

"""
Contains a set of shared test functions.
"""


def big_system():
    """ "Elaborate test system with multiple species, non-cubic cell, and
    close-by atoms.
    """
    a = 1
    return Atoms(
        symbols=["C", "C", "C"],
        cell=[[0, a, a], [a, 0, a], [a, a, 0]],
        scaled_positions=[
            [0, 0, 0],
            [1 / 3, 1 / 3, 1 / 3],
            [2 / 3, 2 / 3, 2 / 3],
        ],
        pbc=[True, True, True],
    ) * (3, 3, 3)


@pytest.fixture()
def H2O():
    """The H2O molecule."""
    return water()


def water():
    """The H2O molecule in a cell."""
    return Atoms(
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        positions=[
            [0, 0, 0],
            [0.95, 0, 0],
            [
                0.95 * (1 + math.cos(76 / 180 * math.pi)),
                0.95 * math.sin(76 / 180 * math.pi),
                0.0,
            ],
        ],
        symbols=["H", "O", "H"],
    )

def molecule_complex():
    """Acetyl fluoride molecule in a cell with no periodicity."""
    mol = molecule("CH3COF")
    mol.set_cell([5, 5, 5])
    mol.center()
    return mol


@pytest.fixture()
def bulk_system():
    """Simple bulk system."""
    return Atoms(
        cell=[5, 5, 5],
        scaled_positions=[
            [0.1, 0, 0],
            [0.9, 0, 0],
        ],
        symbols=["H", "H"],
        pbc=True,
    )


def assert_symmetries(
    descriptor_func, translation=True, rotation=True, permutation=True
):
    if translation:
        assert_symmetry_translation(descriptor_func)
    if rotation:
        assert_symmetry_rotation(descriptor_func)
    if permutation:
        assert_symmetry_permutation(descriptor_func)


def assert_symmetry_rotation(descriptor_func):
    """Tests whether the descriptor output is invariant to rotations of the
    original system.
    """
    system = water()
    descriptor = descriptor_func([system])
    features = descriptor.create(system)
    is_rot_sym = True

    # Rotational Check
    for rotation in ["x", "y", "z"]:
        i_system = system.copy()
        i_system.rotate(45, rotation, rotate_cell=True)
        i_features = descriptor.create(i_system)
        deviation = np.max(np.abs(features - i_features))
        if deviation > 1e-4:
            is_rot_sym = False
    assert is_rot_sym


def assert_symmetry_translation(descriptor_func):
    """Tests whether the descriptor output is invariant to translations of
    the original system.

    Args:
        create(function): A function that when given an Atoms object
        returns a final descriptor vector for it.
    """
    system = water()
    descriptor = descriptor_func([system])
    features = descriptor.create(system)
    is_trans_sym = True

    # Rotational Check
    for translation in [[1.0, 1.0, 1.0], [-5.0, 5.0, -5.0], [1.0, 1.0, -10.0]]:
        i_system = system.copy()
        i_system.translate(translation)
        i_features = descriptor.create(i_system)
        deviation = np.max(np.abs(features - i_features))
        if deviation > 1e-4:
            is_trans_sym = False

    assert is_trans_sym


def assert_symmetry_permutation(descriptor_func):
    """Tests whether the descriptor output is invariant to permutation of
    atom indexing.
    """
    system = water()
    descriptor = descriptor_func([system])
    features = descriptor.create(system)
    is_perm_sym = True

    for permutation in ([0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1]):
        i_system = system[permutation]
        i_features = descriptor.create(i_system)
        deviation = np.max(np.abs(features - i_features))
        if deviation > 1e-7:
            is_perm_sym = False

    assert is_perm_sym


def assert_derivatives(descriptor_func, method):
    assert_derivatives_include(descriptor_func, method)
    assert_derivatives_exclude(descriptor_func, method)
    if method == "numerical":
        assert_derivatives_numerical(descriptor_func)


def assert_derivatives_include(descriptor_func, method):
    H2O = molecule("H2O")
    CO2 = molecule("CO2")
    descriptor = descriptor_func([H2O, CO2])

    # Invalid include options
    with pytest.raises(ValueError):
        descriptor.derivatives(H2O, include=[], method=method)
    with pytest.raises(ValueError):
        descriptor.derivatives(H2O, include=[3], method=method)
    with pytest.raises(ValueError):
        descriptor.derivatives(H2O, include=[-1], method=method)

    # Test that correct atoms are included and in the correct order
    D1, d1 = descriptor.derivatives(H2O, include=[2, 0], method=method)
    D2, d2 = descriptor.derivatives(H2O, method=method)
    assert np.array_equal(D1[0, :], D2[2, :])
    assert np.array_equal(D1[1, :], D2[0, :])

    # Test that using multiple samples and single include works
    D1, d1 = descriptor.derivatives([H2O, CO2], include=[1, 0], method=method)
    D2, d2 = descriptor.derivatives([H2O, CO2], method=method)
    assert np.array_equal(D1[:, 0, :], D2[:, 1, :])
    assert np.array_equal(D1[:, 1, :], D2[:, 0, :])

    # Test that using multiple samples and multiple includes
    D1, d1 = descriptor.derivatives([H2O, CO2], include=[[0], [1]], method=method)
    D2, d2 = descriptor.derivatives([H2O, CO2], method=method)
    assert np.array_equal(D1[0, 0, :], D2[0, 0, :])
    assert np.array_equal(D1[1, 0, :], D2[1, 1, :])


def assert_derivatives_exclude(descriptor_func, method):
    H2O = molecule("H2O")
    CO2 = molecule("CO2")
    descriptor = descriptor_func([H2O, CO2])

    # Invalid exclude options
    with pytest.raises(ValueError):
        descriptor.derivatives(H2O, exclude=[3], method=method)
    with pytest.raises(ValueError):
        descriptor.derivatives(H2O, exclude=[-1], method=method)

    # Test that correct atoms are excluded and in the correct order
    D1, d1 = descriptor.derivatives(H2O, exclude=[1], method=method)
    D2, d2 = descriptor.derivatives(H2O, method=method)
    assert np.array_equal(D1[0, :], D2[0, :])
    assert np.array_equal(D1[1, :], D2[2, :])

    # Test that using single list and multiple samples works
    D1, d1 = descriptor.derivatives([H2O, CO2], exclude=[1], method=method)
    D2, d2 = descriptor.derivatives([H2O, CO2], method=method)
    assert np.array_equal(D1[:, 0, :], D2[:, 0, :])
    assert np.array_equal(D1[:, 1, :], D2[:, 2, :])


def assert_derivatives_numerical(descriptor_func):
    """Test numerical values against a naive python implementation."""
    # Elaborate test system with multiple species, non-cubic cell, and close-by
    # atoms.
    system = big_system()
    h = 0.0001
    n_atoms = len(system)
    n_comp = 3
    descriptor = descriptor_func([system])

    # The maximum error depends on how big the system is. With a small system
    # the error is smaller for non-periodic systems than the corresponding
    # error when periodicity is turned on. The errors become equal (~1e-5) when
    # the size of the system is increased.
    n_features = descriptor.get_number_of_features()
    derivatives_python = np.zeros((n_atoms, n_comp, n_features))
    d0 = descriptor.create(system)
    coeffs = [-1.0 / 2.0, 1.0 / 2.0]
    deltas = [-1.0, 1.0]
    for i_atom in range(len(system)):
        for i_comp in range(3):
            for i_stencil in range(2):
                system_disturbed = system.copy()
                i_pos = system_disturbed.get_positions()
                i_pos[i_atom, i_comp] += h * deltas[i_stencil]
                system_disturbed.set_positions(i_pos)
                d1 = descriptor.create(system_disturbed)
                derivatives_python[i_atom, i_comp, :] += coeffs[i_stencil] * d1 / h

    # Calculate with central finite difference implemented in C++.
    derivatives_cpp, d_cpp = descriptor.derivatives(system, method="numerical")

    # Compare descriptor values
    assert np.allclose(d0, d_cpp, atol=1e-6)

    # Compare derivative values
    assert np.allclose(derivatives_python, derivatives_cpp, atol=2e-5)


def assert_sparse(descriptor_func):
    """Test that sparse output is created upon request in the correct format."""
    system = water()

    # Dense
    desc = descriptor_func(flatten=True, sparse=False)([system])
    features = desc.create(system)
    assert type(features) == np.ndarray

    # Sparse
    desc = descriptor_func(flatten=True, sparse=True)([system])
    features = desc.create(system)
    assert type(features) == sparse.COO


def assert_parallellization(descriptor_func, n_jobs, flatten, sparse):
    """Tests creating output parallelly."""
    samples = [bulk("NaCl", "rocksalt", a=5.64), bulk("Fe", "bcc", a=3.8)]
    desc = descriptor_func(flatten=flatten, sparse=sparse)(samples)
    n_features = desc.get_number_of_features()

    output = desc.create(system=samples, n_jobs=n_jobs)
    a = desc.create(samples[0])
    b = desc.create(samples[1])
    if sparse:
        output = output.todense()
        a = a.todense()
        b = b.todense()
    assumed = np.array([a, b])
    assert np.allclose(output, assumed)


def assert_no_system_modification(descriptor_func):
    """Tests that the descriptor does not modify the system that is given as
    input.
    """

    def check_modifications(system):
        cell = np.array(system.get_cell())
        pos = np.array(system.get_positions())
        pbc = np.array(system.get_pbc())
        atomic_numbers = np.array(system.get_atomic_numbers())
        symbols = np.array(system.get_chemical_symbols())
        descriptor = descriptor_func()([system])
        features = descriptor.create(system)
        assert np.array_equal(cell, system.get_cell())
        assert np.array_equal(pos, system.get_positions())
        assert np.array_equal(pbc, system.get_pbc())
        assert np.array_equal(atomic_numbers, system.get_atomic_numbers())
        assert np.array_equal(symbols, system.get_chemical_symbols())

    # Try separately for periodic and non-periodic systems
    check_modifications(bulk("Cu", "fcc", a=3.6))
    check_modifications(water())


def assert_matrix_descriptor_exceptions(descriptor_func):
    system = water()

    # Unknown permutation option
    with pytest.raises(ValueError):
        descriptor_func(n_atoms_max=5, permutation="unknown")()
    # Negative n_atom_max
    with pytest.raises(ValueError):
        descriptor_func(n_atoms_max=-1)()
    # System has more atoms than the specified maximum
    with pytest.raises(ValueError):
        cm = descriptor_func(n_atoms_max=2)()
        cm.create([system])


def assert_matrix_descriptor_flatten(descriptor_func):
    system = water()

    # Unflattened
    desc = descriptor_func(n_atoms_max=5, flatten=False)([system])
    unflattened = desc.create(system)
    assert unflattened.shape == (5, 5)

    # Flattened
    desc = descriptor_func(n_atoms_max=5, flatten=True)([system])
    flattened = desc.create(system)
    assert flattened.shape == (25,)

    # Check that flattened and unflattened versions contain same values
    assert np.array_equal(flattened.reshape((5, 5)), unflattened)

    # Check that the arrays are zero-padded correctly
    unflattened[:3, :3] = 0
    assert np.all((unflattened == 0))


def assert_matrix_descriptor_sorted(descriptor_func):
    """Tests that sorting using row norm works as expected"""
    system = molecule_complex()
    desc = descriptor_func(permutation="sorted_l2", flatten=False)([system])
    features = desc.create(system)

    # Check that norms are ordered correctly
    lens = np.linalg.norm(features, axis=1)
    old_len = lens[0]
    for length in lens[1:]:
        assert length <= old_len
        old_len = length

    # Check that the matrix is symmetric
    assert np.allclose(features, features.T, rtol=0, atol=1e-13)


def assert_matrix_descriptor_eigenspectrum(descriptor_func):
    """Tests that the eigenvalues are sorted correctly and that the output is
    zero-padded.
    """
    system = water()
    desc = descriptor_func(n_atoms_max=5, permutation="eigenspectrum")([system])
    features = desc.create(system)

    assert features.shape == (5,)

    # Test that eigenvalues are in decreasing order when looking at absolute
    # value
    prev_eig = float("Inf")
    for eigenvalue in features[: len(system)]:
        assert abs(eigenvalue) <= abs(prev_eig)
        prev_eig = eigenvalue

    # Test that array is zero-padded
    assert np.array_equal(features[len(system) :], [0, 0])


def assert_matrix_descriptor_random(descriptor_func):
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
    desc = descriptor_func(permutation="sorted_l2", flatten=False)([HHe])
    features = desc.create(HHe)
    means = np.linalg.norm(features, axis=1)
    mu2 = means[0]
    mu1 = means[1]

    desc = descriptor_func(
        permutation="random",
        sigma=sigma,
        seed=42,
        flatten=False,
    )([HHe])
    count = 0
    rand_instances = 20000
    for i in range(0, rand_instances):
        features = desc.create(HHe)
        i_means = np.linalg.norm(features, axis=1)
        if i_means[0] < i_means[1]:
            count += 1

    # The expected probability is calculated from the cumulative
    # distribution function.
    expected = 1 - scipy.stats.norm.cdf(0, mu1 - mu2, np.sqrt(sigma**2 + sigma**2))
    observed = count / rand_instances

    assert abs(expected - observed) <= 1e-2
