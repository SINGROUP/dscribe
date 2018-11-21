from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import math
import unittest

import numpy as np

import scipy.sparse

from dscribe.descriptors import LMBTR

from ase import Atoms
from ase.visualize import view

from testbaseclass import TestBaseClass


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

default_grid = {
    "k1": {
        "min": 1,
        "max": 90,
        "sigma": 0.1,
        "n": 50,
    },
    "k2": {
        "min": 0,
        "max": 1/0.7,
        "sigma": 0.1,
        "n": 50,
    },
    "k3": {
        "min": -1,
        "max": 1,
        "sigma": 0.1,
        "n": 50,
    }
}


class LMBTRTests(TestBaseClass, unittest.TestCase):

    def test_constructor(self):
        """LMBTR: Tests different valid and invalid constructor values.
        """
        with self.assertRaises(ValueError):
            LMBTR(
                atomic_numbers=[1],
                k=0,
                grid=default_grid,
                virtual_positions=False,
                periodic=False,
            )

        with self.assertRaises(ValueError):
            LMBTR(
                atomic_numbers=[1],
                k=[-1, 2],
                grid=default_grid,
                virtual_positions=False,
                periodic=False,
            )

        with self.assertRaises(ValueError):
            LMBTR(
                atomic_numbers=[1],
                k={1, 4},
                grid=default_grid,
                virtual_positions=False,
                periodic=False,
            )

    def test_positions(self):
        """Tests that the position argument is handled correctly. The position
        can be a list of integers or a list of 3D positions.
        """
        decay_factor = 0.5
        lmbtr = LMBTR(
            atomic_numbers=[1, 8],
            k=[1, 2],
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
                    "function": "exponential",
                    "scale": decay_factor,
                    "cutoff": 1e-3
                },
                "k3": {
                    "function": "exponential",
                    "scale": decay_factor,
                    "cutoff": 1e-3
                },
            },
            periodic=True,
            virtual_positions=True,
            flatten=False,
            sparse=False
        )

        # Position as a cartesian coordinate in list
        lmbtr.create(H2O, positions=[[0, 1, 0]])

        # Position as a cartesian coordinate in numpy array
        lmbtr.create(H2O, positions=np.array([[0, 1, 0]]))

        # Position as a scaled coordinate in list
        lmbtr.create(H2O, positions=[[0, 0, 0.5]], scaled_positions=True)

        # Position as a scaled coordinate in numpy array
        lmbtr.create(H2O, positions=np.array([[0, 0, 0.5]]), scaled_positions=True)

        # Positions as lists of vectors
        positions = [[0, 1, 2], [0, 0, 0]]
        desc = lmbtr.create(H2O, positions)

        # Position outside range
        with self.assertRaises(ValueError):
            lmbtr.create(H2O, positions=[3])

        # Invalid data type
        with self.assertRaises(ValueError):
            lmbtr.create(H2O, positions=['a'])

        # Cannot use scaled positions without cell information
        with self.assertRaises(ValueError):
            H = Atoms(
                positions=[[0, 0, 0]],
                symbols=["H"],
            )

            lmbtr.create(
                H,
                positions=[[0, 0, 1]],
                scaled_positions=True
            )

        # Non-virtual positions
        lmbtr = LMBTR(
            [1, 8],
            k=[3],
            grid=default_grid,
            virtual_positions=False,
            periodic=False,
            flatten=True
        )

        # Positions as a list of integers pointing to atom indices
        positions = [0, 1, 2]
        desc = lmbtr.create(H2O, positions)

    def test_number_of_features(self):
        """LMBTR: Tests that the reported number of features is correct.
        """
        # K = 1
        n = 100
        atomic_numbers = [1, 8]
        n_elem = len(atomic_numbers) + 1  # Including ghost atom
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
            virtual_positions=False,
            periodic=False,
            flatten=True
        )
        n_features = lmbtr.get_number_of_features()
        expected = n_elem*n
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
            virtual_positions=False,
            periodic=False,
            flatten=True
        )
        n_features = lmbtr.get_number_of_features()
        expected = n_elem*n + 1/2*(n_elem)*(n_elem+1)*n
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
            virtual_positions=False,
            periodic=False,
            flatten=True
        )
        n_features = lmbtr.get_number_of_features()
        expected = n_elem*1/2*(n_elem)*(n_elem+1)*n
        self.assertEqual(n_features, expected)

    def test_flatten(self):
        system = H2O
        n = 10
        n_elem = len(set(system.get_atomic_numbers())) + 1

        # K2 unflattened
        desc = LMBTR(
            [1, 8],
            k=[2],
            grid={"k2": {"n": n, "min": 0, "max": 2, "sigma": 0.1}},
            virtual_positions=False,
            periodic=False,
            flatten=False,
            sparse=False
        )
        feat = desc.create(system, positions=[0])[0]["k2"]
        self.assertEqual(feat.shape, (n_elem, n_elem, n))

        # K2 flattened. The sparse matrix only supports 2D matrices, so the first
        # dimension is always present, even if it is of length 1.
        desc = LMBTR(
            [1, 8],
            k=[2],
            grid={"k2": {"n": n, "min": 0, "max": 2, "sigma": 0.1}},
            virtual_positions=False,
            periodic=False,
            flatten=True,
            sparse=False
        )
        feat = desc.create(system, positions=[0])[0]
        self.assertEqual(feat.shape, (1, (1/2*(n_elem)*(n_elem+1)*n)))

    def test_sparse(self):
        """Tests the sparse matrix creation.
        """
        # Dense
        desc = LMBTR(
            [1, 8],
            k=[1],
            grid=default_grid,
            virtual_positions=False,
            periodic=False,
            flatten=True,
            sparse=False
        )
        vec = desc.create(H2O, positions=[0])[0]
        self.assertTrue(type(vec) == np.ndarray)

        # Sparse
        desc = LMBTR(
            [1, 8],
            k=[1],
            grid=default_grid,
            virtual_positions=False,
            periodic=False,
            flatten=True,
            sparse=True
        )
        vec = desc.create(H2O, positions=[0])[0]
        self.assertTrue(type(vec) == scipy.sparse.coo_matrix)

    def test_periodic(self):
        """LMBTR: Test periodic flag
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
                    "function": "exponential",
                    "scale": decay_factor,
                    "cutoff": 1e-3
                },
            },
            virtual_positions=False,
            flatten=False,
            sparse=False
        )

        desc = lmbtr.create(test_sys, positions=[0])
        desc_ = lmbtr.create(test_sys_, positions=[0])

        self.assertTrue(np.linalg.norm(desc_[0]["k2"] - desc[0]["k2"]) < 1e-6)

    def test_k2_weights_and_geoms_finite(self):
        """Tests that the values of the weight and geometry functions are
        correct for the k=2 term.
        """
        lmbtr = LMBTR([1, 8], k=[2], grid=default_grid, virtual_positions=False, periodic=False)
        lmbtr.create(H2O, positions=[1])
        geoms = lmbtr._k2_geoms
        weights = lmbtr._k2_weights

        # Test against the assumed geom values
        pos = H2O.get_positions()
        assumed_geoms = {
            (0, 1): 2*[1/np.linalg.norm(pos[0] - pos[1])],
        }
        self.dict_comparison(assumed_geoms, geoms)

        # Test against the assumed weights
        assumed_weights = {
            (0, 1): 2*[1],
        }
        self.dict_comparison(assumed_weights, weights)

        # Test against system with different indexing
        lmbtr = LMBTR([1, 8], k=[2], grid=default_grid, virtual_positions=False, periodic=False)
        lmbtr.create(H2O_2, positions=[0])
        geoms2 = lmbtr._k2_geoms
        self.dict_comparison(geoms, geoms2)

    def test_k3_weights_and_geoms_finite(self):
        """Tests that all the correct angles are present in finite systems.
        There should be n*(n-1)*(n-2)/2 unique angles where the division by two
        gets rid of duplicate angles.
        """
        system = Atoms(
            scaled_positions=[
                [0, 0, 0],
                [0.5, 0, 0],
                [0, 0.5, 0],
                [0.5, 0.5, 0],
            ],
            symbols=["H", "H", "H", "O"],
            cell=[10, 10, 10],
            pbc=True,
        )

        lmbtr = LMBTR([1, 8], k=[3], grid=default_grid, virtual_positions=False, periodic=False)
        lmbtr.create(system, positions=[0])
        geoms = lmbtr._k3_geoms
        weights = lmbtr._k3_weights

        # Test against the assumed geom values.
        assumed_geoms = {
            (0, 1, 1): 2*[math.cos(45/180*math.pi)],
            (0, 2, 1): 2*[math.cos(45/180*math.pi)],
            (0, 1, 2): 2*[math.cos(90/180*math.pi)],
            (1, 0, 2): 2*[math.cos(45/180*math.pi)],
            (1, 0, 1): 1*[math.cos(90/180*math.pi)],
        }
        self.dict_comparison(geoms, assumed_geoms)

        # Test against the assumed weight values.
        assumed_weights = {
            (0, 1, 1): 2*[1],
            (0, 2, 1): 2*[1],
            (0, 1, 2): 2*[1],
            (1, 0, 2): 2*[1],
            (1, 0, 1): 1*[1],
        }
        self.dict_comparison(weights, assumed_weights)

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
                    "function": "exponential",
                    "scale": 0.5,
                    "cutoff": 1e-3
                },
                "k3": {
                    "function": "exponential",
                    "scale": 0.5,
                    "cutoff": 1e-3
                },
            },
            virtual_positions=False,
            flatten=True
        )

        def create_1(system):
            """This function uses atom indices so rotation and translation
            should not affect it.
            """
            return desc.create(system, positions=[0])[0]

        def create_2(system):
            """This function uses scaled positions so atom permutation should
            not affect it.
            """
            desc.virtual_positions = True
            return desc.create(system, positions=[[0, 1, 0]], scaled_positions=True)[0]

        # Rotational check
        self.assertTrue(self.is_rotationally_symmetric(create_1))

        # Translational
        self.assertTrue(self.is_translationally_symmetric(create_1))

        # Permutational
        self.assertTrue(self.is_permutation_symmetric(create_2))

if __name__ == "__main__":
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(LMBTRTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
