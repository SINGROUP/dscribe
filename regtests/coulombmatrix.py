from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import math
import numpy as np
import unittest
import scipy.stats

from describe.descriptors import CoulombMatrix

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


class CoulombMatrixTests(unittest.TestCase):

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
        self.assertEqual(cm.shape, (25,))

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
        """Tests translational and rotational symmetries
        """
        desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=False)
        #Rotational Check
        molecule = H2O.copy()
        features = desc.create(molecule)

        for rotation in ['x', 'y', 'z']:
            molecule.rotate(45, rotation)
            rot_features =  desc.create(molecule)
            deviation = np.max(np.abs(features- rot_features))
            self.assertTrue(deviation < 10e-9)


        #Translation check
        for translation in [[1.0, 1.0, 1.0], [-5.0, 5.0, -5.0], [1.0, 1.0, -10.0],]:
            molecule.translate(translation)
            trans_features =   desc.create(molecule)
            deviation = np.max(np.abs(features- trans_features))
            self.assertTrue(deviation < 10e-9)




class SortedCoulombMatrixTests(unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """

    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        desc = CoulombMatrix(n_atoms_max=5, permutation="sorted_l2", flatten=False)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 25)

    def test_flatten(self):
        """Tests the flattening.
        """
        # Unflattened
        desc = CoulombMatrix(n_atoms_max=5, permutation="sorted_l2", flatten=False)
        cm = desc.create(H2O)
        self.assertEqual(cm.shape, (5, 5))

        # Flattened
        desc = CoulombMatrix(n_atoms_max=5, permutation="sorted_l2", flatten=True)
        cm = desc.create(H2O)
        self.assertEqual(cm.shape, (25,))

    def test_features(self):
        """Tests that the correct features are present in the desciptor.
        """
        desc = CoulombMatrix(n_atoms_max=5, permutation="sorted_l2", flatten=False)
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
        desc = CoulombMatrix(n_atoms_max=5, permutation="eigenspectrum")
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 5)

    def test_features(self):
        """Tests that the correct features are present in the desciptor.
        """
        desc = CoulombMatrix(n_atoms_max=5, permutation="eigenspectrum")
        cm = desc.create(H2O)

        self.assertEqual(len(cm), 5)

        # Test that eigenvalues are in decreasing order when looking at absolute value
        prev_eig = float("Inf")
        for eigenvalue in cm[:len(H2O)]:
            self.assertTrue(abs(eigenvalue) <= abs(prev_eig))
            prev_eig = eigenvalue

        # Test that array is zero-padded
        self.assertTrue(np.array_equal(cm[len(H2O):], [0, 0]))


class RandomCoulombMatrixTests(unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """
        with self.assertRaises(ValueError):
            CoulombMatrix(n_atoms_max=5, permutation="random", sigma=None)
        with self.assertRaises(ValueError):
            CoulombMatrix(n_atoms_max=5, permutation="sorted_l2", sigma=3)
        with self.assertRaises(ValueError):
            CoulombMatrix(n_atoms_max=5, permutation="none", sigma=3)
        with self.assertRaises(ValueError):
            CoulombMatrix(n_atoms_max=5, permutation="eigenspectrum", sigma=3)

    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        desc = CoulombMatrix(n_atoms_max=5, permutation="random", sigma=100)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 25)

    def test_norm_vector(self):
        """Tests if the attribute norm_vector is written and used correctly
        """
        desc = CoulombMatrix(n_atoms_max=5, permutation="random", sigma=100, flatten=False)
        cm = desc.create(H2O)
        self.assertEqual(len(cm), 5)

        # The norm_vector is not zero padded in this implementation. All zero-padding
        # is done at the end after randomly sorting
        self.assertEqual(len(desc.norm_vector), 3)
        cm = desc.create(H2O)
        self.assertEqual(len(cm), 5)

    def test_distribution(self):
        """Tests if the random sorting obeys a gaussian distribution. Can
        rarely fail when everything is OK.
        """
        # Get the mean value to compare to
        sigma = 5
        desc = CoulombMatrix(n_atoms_max=5, permutation="sorted_l2", flatten=False)
        cm = desc.create(HHe)
        means = sorted(np.linalg.norm(cm, axis=1))
        means = np.linalg.norm(cm, axis=1)
        mu2 = means[0]
        mu1 = means[1]

        # Measures how many times the two rows with biggest norm exchange place
        # when random noise is added. This should correspond to the probability
        # P(X > Y), where X = N(\mu_1, \sigma^2), Y = N(\mu_2, \sigma^2). This
        # probability can be reduced to P(X > Y) = P(X-Y > 0) = P(N(\mu_1 -
        # \mu_2, \sigma^2 + sigma^2) > 0). See e.g.
        # https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables
        desc = CoulombMatrix(n_atoms_max=5, permutation="random", sigma=sigma, flatten=False)
        count = 0
        rand_instances = 10000
        for i in range(0, rand_instances):
            cm = desc.create(HHe)
            if np.linalg.norm(cm[0]) < np.linalg.norm(cm[1]):
                count += 1

        # The expected probability is calculated from the cumulative
        # distribution function.
        expected = 1 - scipy.stats.norm.cdf(0, mu1 - mu2, np.sqrt(sigma**2 + sigma**2))
        observed = count/rand_instances

        self.assertTrue(abs(expected - observed) <= 1e-2)

    def test_match_with_sorted(self):
        """Tests if sorting the random coulomb matrix results in the same as
        the sorted coulomb matrix
        """
        desc = CoulombMatrix(n_atoms_max=5, permutation="random", sigma=100, flatten=False)
        rcm = desc.create(H2O)

        srcm = desc.sort(rcm)

        desc = CoulombMatrix(n_atoms_max=5, permutation="sorted_l2", flatten=False)

        scm = desc.create(H2O)

        self.assertTrue(np.array_equal(scm, srcm))


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(CoulombMatrixTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SortedCoulombMatrixTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(CoulombMatrixEigenSpectrumTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(RandomCoulombMatrixTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
