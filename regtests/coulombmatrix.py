import math
import unittest

import numpy as np

import scipy.stats

from ase import Atoms
from ase.build import molecule

from dscribe.descriptors import CoulombMatrix

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


class CoulombMatrixTests(TestBaseClass, unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """
        with self.assertRaises(ValueError):
            CoulombMatrix(n_atoms_max=5, permutation="unknown")
        with self.assertRaises(ValueError):
            CoulombMatrix(n_atoms_max=-1)

    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=False)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 25)

    def test_periodicity(self):
        """Tests that periodicity is not taken into account in Coulomb matrix
        even if the system is set as periodic.
        """
        system = Atoms(
            cell=[5, 5, 5],
            scaled_positions=[
                [0.1, 0, 0],
                [0.9, 0, 0],
            ],
            symbols=["H", "H"],
            pbc=True
        )
        desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=False)
        cm = desc.create(system)

        pos = system.get_positions()
        assumed = 1*1/np.linalg.norm((pos[0] - pos[1]))
        self.assertEqual(cm[0, 1], assumed)

    def test_flatten(self):
        """Tests the flattening.
        """
        # Unflattened
        desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=False)
        cm = desc.create(H2O)
        self.assertEqual(cm.shape, (5, 5))

        # Flattened
        desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=True)
        cm = desc.create(H2O)
        self.assertEqual(cm.shape, (1, 25))

    def test_sparse(self):
        """Tests the sparse matrix creation.
        """
        # Dense
        desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=False, sparse=False)
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == np.ndarray)

        # Sparse
        desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=True, sparse=True)
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == scipy.sparse.coo_matrix)

    def test_parallel_dense(self):
        """Tests creating dense output parallelly.
        """
        samples = [molecule("CO"), molecule("N2O")]
        desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=True, sparse=False)
        n_features = desc.get_number_of_features()

        # Test multiple systems, serial job
        output = desc.create(
            system=samples,
            n_jobs=1,
        )
        assumed = np.empty((2, n_features))
        assumed[0, :] = desc.create(samples[0])
        assumed[1, :] = desc.create(samples[1])
        self.assertTrue(np.allclose(output, assumed))

        # Test multiple systems, parallel job
        output = desc.create(
            system=samples,
            n_jobs=2,
        )
        assumed = np.empty((2, n_features))
        assumed[0, :] = desc.create(samples[0])
        assumed[1, :] = desc.create(samples[1])
        self.assertTrue(np.allclose(output, assumed))

        # Non-flattened output
        desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=False, sparse=False)
        output = desc.create(
            system=samples,
            n_jobs=2,
        )
        assumed = np.empty((2, 5, 5))
        assumed[0] = desc.create(samples[0])
        assumed[1] = desc.create(samples[1])
        self.assertTrue(np.allclose(np.array(output), assumed))

    def test_parallel_sparse(self):
        """Tests creating sparse output parallelly.
        """
        # Test indices
        samples = [molecule("CO"), molecule("N2O")]
        desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=True, sparse=True)
        n_features = desc.get_number_of_features()

        # Test multiple systems, serial job
        output = desc.create(
            system=samples,
            n_jobs=1,
        ).toarray()
        assumed = np.empty((2, n_features))
        assumed[0, :] = desc.create(samples[0]).toarray()
        assumed[1, :] = desc.create(samples[1]).toarray()
        self.assertTrue(np.allclose(output, assumed))

        # Test multiple systems, parallel job
        output = desc.create(
            system=samples,
            n_jobs=2,
        ).toarray()
        assumed = np.empty((2, n_features))
        assumed[0, :] = desc.create(samples[0]).toarray()
        assumed[1, :] = desc.create(samples[1]).toarray()
        self.assertTrue(np.allclose(output, assumed))

        # Non-flattened output
        desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=False, sparse=True)
        output = [x.toarray() for x in desc.create(
            system=samples,
            n_jobs=2,
        )]
        assumed = np.empty((2, 5, 5))
        assumed[0] = desc.create(samples[0]).toarray()
        assumed[1] = desc.create(samples[1]).toarray()
        self.assertTrue(np.allclose(np.array(output), assumed))

    def test_features(self):
        """Tests that the correct features are present in the desciptor.
        """
        desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=False)
        cm = desc.create(H2O)

        # Test against assumed values
        q = H2O.get_atomic_numbers()
        p = H2O.get_positions()
        norm = np.linalg.norm
        assumed = np.array(
            [
                [0.5*q[0]**2.4,              q[0]*q[1]/(norm(p[0]-p[1])),  q[0]*q[2]/(norm(p[0]-p[2]))],
                [q[1]*q[0]/(norm(p[1]-p[0])), 0.5*q[1]**2.4,               q[1]*q[2]/(norm(p[1]-p[2]))],
                [q[2]*q[0]/(norm(p[2]-p[0])), q[2]*q[1]/(norm(p[2]-p[1])), 0.5*q[2]**2.4],
            ]
        )
        zeros = np.zeros((5, 5))
        zeros[:3, :3] = assumed
        assumed = zeros

        self.assertTrue(np.array_equal(cm, assumed))

    def test_symmetries(self):
        """Tests the symmetries of the descriptor.
        """
        def create(system):
            desc = CoulombMatrix(n_atoms_max=3, permutation="none", flatten=True)
            return desc.create(system)

        # Rotational
        self.assertTrue(self.is_rotationally_symmetric(create))

        # Translational
        self.assertTrue(self.is_translationally_symmetric(create))

        # Permutational
        self.assertFalse(self.is_permutation_symmetric(create))


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(CoulombMatrixTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
