"""
Defines a set of regressions tests that should be run succesfully after all
major modification to the code.
"""
import math
import numpy as np
import unittest

from describe.descriptors import CoulombMatrix
from describe.descriptors import SortedCoulombMatrix
from describe.descriptors import CoulombMatrixEigenSpectrum

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


class CoulombMatrixTests(unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """

    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        desc = CoulombMatrix(n_atoms_max=5, flatten=False)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 25)

    def test_flatten(self):
        """Tests the flattening.
        """
        # Unflattened
        desc = CoulombMatrix(n_atoms_max=5, flatten=False)
        cm = desc.create(H2O)
        self.assertEqual(cm.shape, (5, 5))

        # Flattened
        desc = CoulombMatrix(n_atoms_max=5, flatten=True)
        cm = desc.create(H2O)
        self.assertEqual(cm.shape, (25,))

    def test_features(self):
        """Tests that the correct features are present in the desciptor.
        """
        desc = CoulombMatrix(n_atoms_max=5, flatten=False)
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


class SortedCoulombMatrixTests(unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """

    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        desc = SortedCoulombMatrix(n_atoms_max=5, flatten=False)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 25)

    def test_flatten(self):
        """Tests the flattening.
        """
        # Unflattened
        desc = SortedCoulombMatrix(n_atoms_max=5, flatten=False)
        cm = desc.create(H2O)
        self.assertEqual(cm.shape, (5, 5))

        # Flattened
        desc = SortedCoulombMatrix(n_atoms_max=5, flatten=True)
        cm = desc.create(H2O)
        self.assertEqual(cm.shape, (25,))

    def test_features(self):
        """Tests that the correct features are present in the desciptor.
        """
        desc = SortedCoulombMatrix(n_atoms_max=5, flatten=False)
        cm = desc.create(H2O)

        lens = np.linalg.norm(cm, axis=0)
        old_len = lens[0]
        for length in lens[1:]:
            self.assertTrue(length <= old_len)
            old_len = length


class CoulombMatrixEigenSpectrumTests(unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """

    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        desc = CoulombMatrixEigenSpectrum(n_atoms_max=5)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 5)

    def test_features(self):
        """Tests that the correct features are present in the desciptor.
        """
        desc = CoulombMatrixEigenSpectrum(n_atoms_max=5)
        cm = desc.create(H2O)

        # Test that eigenvalues are in decreasing order when looking at absolute value
        prev_eig = float("Inf")
        for eigenvalue in cm[:len(H2O)]:
            self.assertTrue(abs(eigenvalue) <= abs(prev_eig))
            prev_eig = eigenvalue

        # Test that array is zero-padded
        self.assertTrue(np.array_equal(cm[len(H2O):], [0, 0]))


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(CoulombMatrixTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SortedCoulombMatrixTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(CoulombMatrixEigenSpectrumTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
