import math
import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule


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


def check_symmetry_rotation(create):
    """Tests whether the descriptor output is invariant to rotations of the
    original system.
    """
    system = molecule("H2O")
    features = create(system)
    is_rot_sym = True

    # Rotational Check
    for rotation in ["x", "y", "z"]:
        i_system = system.copy()
        i_system.rotate(45, rotation, rotate_cell=True)
        i_features = create(i_system)
        deviation = np.max(np.abs(features - i_features))
        if deviation > 1e-4:
            is_rot_sym = False
    assert is_rot_sym


def check_symmetry_translation(create):
    """Tests whether the descriptor output is invariant to translations of
    the original system.

    Args:
        create(function): A function that when given an Atoms object
        returns a final descriptor vector for it.
    """
    system = molecule("H2O")
    features = create(system)
    is_trans_sym = True

    # Rotational Check
    for translation in [[1.0, 1.0, 1.0], [-5.0, 5.0, -5.0], [1.0, 1.0, -10.0]]:
        i_system = system.copy()
        i_system.translate(translation)
        i_features = create(i_system)
        deviation = np.max(np.abs(features - i_features))
        if deviation > 1e-4:
            is_trans_sym = False

    assert is_trans_sym


def check_symmetry_permutation(create):
    """Tests whether the descriptor output is invariant to permutation of
    atom indexing.
    """
    system = molecule("H2O")
    features = create(system)
    is_perm_sym = True

    for permutation in ([0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1]):
        i_system = system[permutation]
        i_features = create(i_system)
        deviation = np.max(np.abs(features - i_features))
        if deviation > 1e-7:
            is_perm_sym = False

    assert is_perm_sym


def check_derivatives_include(descriptor, methods):
    H2O = molecule("H2O")
    CO2 = molecule("CO2")
    for method in methods:
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


def check_derivatives_exclude():
    pass


def check_derivatives_numerical():
    pass
