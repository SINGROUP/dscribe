from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import math
import numpy as np
import unittest

from describe.descriptors import LMBTR

from ase import Atoms


H2O = Atoms(
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
)

H2O_2 = Atoms(
    cell=[[5.0, 0.0, 0], [0, 5, 0], [0, 0, 5.0]],
    positions=[[0.95, 0, 0], [0, 0, 0], [0.95*(1+math.cos(76/180*math.pi)), 0.95*math.sin(76/180*math.pi), 0.0]],
    symbols=["O", "H", "H"],
)

HHe = Atoms(
    cell=[
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
        ],
    positions=[
        [0, 0, 0],
        [0.71, 0, 0],
    ],
    symbols=["H", "He"],
)


class LMBTRTests(unittest.TestCase):

    def test_constructor(self):
        """LMBTR: Tests different valid and invalid constructor values.
        """
        with self.assertRaises(ValueError):
            LMBTR(
                atomic_numbers=[1],
                k=0,
                periodic=False,
            )

        with self.assertRaises(ValueError):
            LMBTR(
                atomic_numbers=[1],
                k=[-1, 2],
                periodic=False,
            )

        with self.assertRaises(ValueError):
            LMBTR(
                atomic_numbers=[1],
                k=1,
                periodic=False,
            )

        with self.assertRaises(ValueError):
            LMBTR(
                atomic_numbers=[1],
                k={1, 4},
                periodic=False,
            )

    def test_create(self):
        """
        LMBTR: Test all fail-safe in create method
        """
        decay_factor = 0.5
        lmbtr = LMBTR(
            atomic_numbers=[1, 8],
            k=[1, 2],
            periodic=True,
            grid={
                "k1": {
                    "min": 10,
                    "max": 18,
                    "sigma": 0.1,
                    "n": 200,
                },
                "k2": {
                    "min": 0,
                    "max": 0.7,
                    "sigma": 0.01,
                    "n": 200,
                },
                "k3": {
                    "min": -1.0,
                    "max": 1.0,
                    "sigma": 0.05,
                    "n": 200,
                }
            },
            weighting={
                "k2": {
                    "function": lambda x: np.exp(-decay_factor*x),
                    "threshold": 1e-3
                },
                "k3": {
                    "function": lambda x: np.exp(-decay_factor*x),
                    "threshold": 1e-3
                },
            },
            flatten=False
        )

        with self.assertRaises(ValueError):
            desc = lmbtr.create(H2O, list_atom_indices=[3])

        with self.assertRaises(ValueError):
            desc = lmbtr.create(H2O)

        with self.assertRaises(ValueError):
            desc = lmbtr.create(H2O, list_positions=[[0, 0, 0]])

        with self.assertRaises(ValueError):
            H = Atoms(
                positions=[[0, 0, 0]],
                symbols=["H"],
            )

            desc = lmbtr.create(
                H,
                list_positions=[[0, 0, 1]],
                scaled_positions=True
            )

    def test_number_of_features(self):
        """LMBTR: Tests that the reported number of features is correct.
        """
        # K = 1
        n = 100
        atomic_numbers = [1, 8]
        n_elem = len(atomic_numbers)
        lmbtr = LMBTR(
            atomic_numbers=atomic_numbers,
            k=[1],
            grid={
                "k1": {
                    "min": 1,
                    "max": 8,
                    "sigma": 0.1,
                    "n": 100,
                }
            },
            periodic=False,
            flatten=True
        )
        n_features = lmbtr.get_number_of_features()
        expected = n
        self.assertEqual(n_features, expected)

        # K = 2
        lmbtr = LMBTR(
            atomic_numbers=atomic_numbers,
            k={1, 2},
            grid={
                "k1": {
                    "min": 1,
                    "max": 8,
                    "sigma": 0.1,
                    "n": 100,
                },
                "k2": {
                    "min": 0,
                    "max": 1/0.7,
                    "sigma": 0.1,
                    "n": n,
                }
            },
            periodic=False,
            flatten=True
        )
        n_features = lmbtr.get_number_of_features()
        expected = n + n_elem*n
        self.assertEqual(n_features, expected)

        # K = 3
        lmbtr = LMBTR(
            atomic_numbers=atomic_numbers,
            k={3},
            grid={
                "k1": {
                    "min": 1,
                    "max": 8,
                    "sigma": 0.1,
                    "n": 100,
                },
                "k2": {
                    "min": 0,
                    "max": 1/0.7,
                    "sigma": 0.1,
                    "n": n,
                },
                "k3": {
                    "min": -1,
                    "max": 1,
                    "sigma": 0.1,
                    "n": n,
                }
            },
            periodic=False,
            flatten=True
        )
        n_features = lmbtr.get_number_of_features()
        expected = 1/2*(n_elem)*(n_elem+1)*n
        self.assertEqual(n_features, expected)

    # def test_flatten(self):
        # """LMBTR: Tests the flattening.
        # """
        # # Unflattened
        # desc = LMBTR(n_atoms_max=5, permutation="none", flatten=False)
        # cm = desc.create(H2O)
        # self.assertEqual(cm.shape, (5, 5))

        # # Flattened
        # desc = LMBTR(n_atoms_max=5, permutation="none", flatten=True)
        # cm = desc.create(H2O)
        # self.assertEqual(cm.shape, (25,))

    def test_periodic(self):
        """
        LMBTR: Test periodic flag
        """
        test_sys = Atoms(
            cell=[[5.0, 0.0, 0.0], [0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            positions=[[0, 0, 1], [0, 0, 3]],
            symbols=["H", "H"],
        )
        test_sys_ = Atoms(
            cell=[[5.0, 0.0, 0.0], [0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            positions=[[0, 0, 1], [0, 0, 4]],
            symbols=["H", "H"],
        )

        decay_factor = 0.5
        lmbtr = LMBTR(
            atomic_numbers=[1],
            k=[2],
            periodic=True,
            grid={
                "k2": {
                    "min": 1/5,
                    "max": 1,
                    "sigma": 0.001,
                    "n": 200,
                },
            },
            weighting={
                "k2": {
                    "function": lambda x: np.exp(-decay_factor*x),
                    "threshold": 1e-3
                },
            },
            flatten=False
        )

        desc = lmbtr.create(test_sys, list_atom_indices=[0])
        desc_ = lmbtr.create(test_sys_, list_atom_indices=[0])

        self.assertTrue(np.linalg.norm(desc_[0][0] - desc[0][0]) < 1e-6)

    def test_inverse_distances(self):
        """
        LMBTR: Test inverse distances
        """
        lmbtr = LMBTR([1, 8], k=[2], periodic=False)
        lmbtr.create(H2O, list_atom_indices=[1])
        inv_dist = lmbtr._inverse_distances

        # Test against the assumed values
        pos = H2O.get_positions()
        assumed = {
            2: [np.inf],
            1: 2*[1/np.linalg.norm(pos[0] - pos[1])]
        }
        self.assertEqual(assumed, inv_dist)

        # Test against system with different indexing
        lmbtr = LMBTR([1, 8], k=[2], periodic=False)
        lmbtr.create(H2O_2, list_atom_indices=[0])
        inv_dist_2 = lmbtr._inverse_distances
        self.assertEqual(inv_dist, inv_dist_2)

    def test_cosines(self):
        """
        LMBTR: Test cosines
        """
        lmbtr = LMBTR([1, 8], k=[3], periodic=False)
        lmbtr.create(H2O, list_atom_indices=[1])
        angles = lmbtr._angles

        # Test against the assumed values.
        assumed = {
            1: {
                1: 2*[np.cos(104/180*np.pi)]
            }
        }
        self.assertTrue(
            np.abs(angles[1][1][0]
            - np.cos(104/180*np.pi))
            < 1e-6
        )

        # Test against system with different indexing
        lmbtr = LMBTR([1, 8], k=[3], periodic=False)
        lmbtr.create(H2O_2, list_atom_indices=[0])
        angles2 = lmbtr._angles
        self.assertEqual(angles, angles2)

    def test_symmetries(self):
        """LMBTR: Tests translational and rotational symmetries for a finite system.
        """
        desc = LMBTR(
            atomic_numbers=[1, 8],
            k=[1, 2, 3],
            periodic=False,
            grid={
                "k1": {
                    "min": 10,
                    "max": 18,
                    "sigma": 0.1,
                    "n": 100,
                },
                "k2": {
                    "min": 0,
                    "max": 0.7,
                    "sigma": 0.01,
                    "n": 100,
                },
                "k3": {
                    "min": -1.0,
                    "max": 1.0,
                    "sigma": 0.05,
                    "n": 100,
                }
            },
            weighting={
                "k2": {
                    "function": lambda x: np.exp(-0.5*x),
                    "threshold": 1e-3
                },
                "k3": {
                    "function": lambda x: np.exp(-0.5*x),
                    "threshold": 1e-3
                },
            },
            flatten=True
        )

        # Rotational check
        molecule = H2O.copy()
        features = desc.create(molecule, list_atom_indices=[0])[0].toarray()

        for rotation in ['x', 'y', 'z']:
            molecule.rotate(45, rotation)
            rot_features = desc.create(molecule, list_atom_indices=[0])[0].toarray()
            deviation = np.max(np.abs(features - rot_features))
            self.assertTrue(deviation < 1e-6)

        # Translation check
        for translation in [[1.0, 1.0, 1.0], [-5.0, 5.0, -5.0], [1.0, 1.0, -10.0]]:
            molecule.translate(translation)
            trans_features = desc.create(molecule, list_atom_indices=[0])[0].toarray()
            deviation = np.max(np.abs(features - trans_features))
            self.assertTrue(deviation < 1e-6)


if __name__ == "__main__":
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(LMBTRTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
