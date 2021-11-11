import math
import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule


def big_system():
    """ "Elaborate test system with multiple species, non-cubic cell, and
    close-by atoms.
    """
    a = 1
    return (
        Atoms(
            symbols=["C", "C", "C"],
            cell=[[0, a, a], [a, 0, a], [a, a, 0]],
            scaled_positions=[
                [0, 0, 0],
                [1 / 3, 1 / 3, 1 / 3],
                [2 / 3, 2 / 3, 2 / 3],
            ],
            pbc=[True, True, True],
        )
        * (3, 3, 3)
    )


@pytest.fixture()
def H2O():
    """The H2O molecule."""
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


@pytest.fixture()
def bulk():
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


def assert_symmetry_rotation(descriptor_func):
    """Tests whether the descriptor output is invariant to rotations of the
    original system.
    """
    system = molecule("H2O")
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
    system = molecule("H2O")
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
    system = molecule("H2O")
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
