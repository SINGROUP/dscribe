import math
import abc
import numpy as np
import unittest
from ase import Atoms

ABC = abc.ABCMeta('ABC', (object,), {})

finite_system = Atoms(
    cell=[
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
    ],
    positions=[
        [0, 0, 0],
        [0.95, 0, 0],
        [0.95*(1+math.cos(76/180*math.pi)), 0.95*math.sin(76/180*math.pi), 0.0]
    ],
    symbols=["H", "O", "H"],
    pbc=False,
)


class TestBaseClass(unittest.TestCase, ABC):
    """This is a base class that contains common tests and utilities shared by
    different descriptors. It also defines a set of common basic tests that
    should be implemented for every descriptor.
    """
    @abc.abstractmethod
    def test_constructor(self):
        """
        """

    @abc.abstractmethod
    def test_number_of_features(self):
        """
        """

    @abc.abstractmethod
    def test_flatten(self):
        """
        """

    @abc.abstractmethod
    def test_symmetries(self):
        """
        """

    def is_rotationally_symmetric(self, desc, **kwargs):
        """Tests whether the descriptor output is invariant to rotations of the
        original system.
        """
        features = desc.create(finite_system, **kwargs)
        is_rot_sym = True

        # Rotational Check
        for rotation in ['x', 'y', 'z']:
            i_system = finite_system.copy()
            i_system.rotate(45, rotation, rotate_cell=True)
            i_features = desc.create(i_system, **kwargs)
            deviation = np.max(np.abs(features - i_features))
            if deviation > 10e-7:
                is_rot_sym = False
        return is_rot_sym

    def is_translationally_symmetric(self, desc, **kwargs):
        """Tests whether the descriptor output is invariant to translations of
        the original system.
        """
        features = desc.create(finite_system, **kwargs)
        is_trans_sym = True

        # Rotational Check
        for translation in [[1.0, 1.0, 1.0], [-5.0, 5.0, -5.0], [1.0, 1.0, -10.0]]:
            i_system = finite_system.copy()
            i_system.translate(translation)
            i_features = desc.create(i_system, **kwargs)
            deviation = np.max(np.abs(features - i_features))
            if deviation > 10e-9:
                is_trans_sym = False

        return is_trans_sym

    def is_permutation_symmetric(self, desc, **kwargs):
        """Tests whether the descriptor output is invariant to permutation of
        atom indexing.
        """
        features = desc.create(finite_system, **kwargs)
        is_perm_sym = True

        for permutation in ([0, 2, 1], [1, 0, 2], [1, 0, 2], [2, 1, 0], [2, 0, 1]):
            i_system = finite_system[permutation]
            i_features = desc.create(i_system, **kwargs)
            deviation = np.max(np.abs(features - i_features))
            if deviation > 10e-9:
                is_perm_sym = False

        return is_perm_sym
