import math
import unittest

import numpy as np

import scipy.stats

from ase import Atoms

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


class SortedMatrixTests(TestBaseClass, unittest.TestCase):
    """Tests that getting rid of permutational symmetry by sorting rows and
    columns by L2-norm works properly. Uses Coulomb matrix as an example, but
    the functionality is the same for all other descriptors that are subclasses
    of MatrixDescriptor.
    """
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
        self.assertEqual(cm.shape, (1, 25))

    def test_sparse(self):
        """Tests the sparse matrix creation.
        """
        # Dense
        desc = CoulombMatrix(n_atoms_max=5, permutation="sorted_l2", flatten=False, sparse=False)
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == np.ndarray)

        # Sparse
        desc = CoulombMatrix(n_atoms_max=5, permutation="sorted_l2", flatten=True, sparse=True)
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == scipy.sparse.coo_matrix)

    def test_features(self):
        """Tests that the correct features are present in the desciptor.
        """
        desc = CoulombMatrix(n_atoms_max=5, permutation="sorted_l2", flatten=False)
        cm = desc.create(H2O)

        lens = np.linalg.norm(cm, axis=1)
        old_len = lens[0]
        for length in lens[1:]:
            self.assertTrue(length <= old_len)
            old_len = length

    def test_symmetries(self):
        """Tests the symmetries of the descriptor.
        """
        def create(system):
            desc = CoulombMatrix(n_atoms_max=3, permutation="sorted_l2", flatten=True)
            return desc.create(system)

        # Rotational
        self.assertTrue(self.is_rotationally_symmetric(create))

        # Translational
        self.assertTrue(self.is_translationally_symmetric(create))

        # Permutational
        self.assertTrue(self.is_permutation_symmetric(create))


class EigenSpectrumTests(TestBaseClass, unittest.TestCase):
    """Tests that getting rid of permutational symmetry with matrix
    eigenspectrum works properly. Uses Coulomb matrix as an example, but the
    functionality is the same for all other descriptors that are subclasses of
    MatrixDescriptor.
    """
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

        self.assertEqual(cm.shape, (1, 5))

        # Test that eigenvalues are in decreasing order when looking at absolute value
        prev_eig = float("Inf")
        for eigenvalue in cm[0, :len(H2O)]:
            self.assertTrue(abs(eigenvalue) <= abs(prev_eig))
            prev_eig = eigenvalue

        # Test that array is zero-padded
        self.assertTrue(np.array_equal(cm[0, len(H2O):], [0, 0]))

    def test_flatten(self):
        """Tests the flattening.
        """
        # Unflattened
        desc = CoulombMatrix(n_atoms_max=5, permutation="eigenspectrum", flatten=False)
        cm = desc.create(H2O)
        # print(cm)
        self.assertEqual(cm.shape, (1, 5))

        # Flattened
        desc = CoulombMatrix(n_atoms_max=5, permutation="eigenspectrum", flatten=True)
        cm = desc.create(H2O)
        self.assertEqual(cm.shape, (1, 5))

    def test_sparse(self):
        """Tests the sparse matrix creation.
        """
        # Dense
        desc = CoulombMatrix(n_atoms_max=5, permutation="eigenspectrum", flatten=False, sparse=False)
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == np.ndarray)

        # Sparse
        desc = CoulombMatrix(n_atoms_max=5, permutation="eigenspectrum", flatten=True, sparse=True)
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == scipy.sparse.coo_matrix)

    def test_symmetries(self):
        """Tests the symmetries of the descriptor.
        """
        def create(system):
            desc = CoulombMatrix(n_atoms_max=3, permutation="eigenspectrum", flatten=True)
            return desc.create(system)

        # Rotational
        self.assertTrue(self.is_rotationally_symmetric(create))

        # Translational
        self.assertTrue(self.is_translationally_symmetric(create))

        # Permutational
        self.assertTrue(self.is_permutation_symmetric(create))


class RandomMatrixTests(TestBaseClass, unittest.TestCase):
    """Tests that the sorting of matrix columns by row norm with added random
    noise works properly. Uses Coulomb matrix as an example, but the
    functionality is the same for all other descriptors that are subclasses of
    MatrixDescriptor.
    """
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

    def test_flatten(self):
        """Tests the flattening.
        """
        # Unflattened
        desc = CoulombMatrix(n_atoms_max=5, permutation="random", sigma=100, flatten=False)
        cm = desc.create(H2O)
        self.assertEqual(cm.shape, (5, 5))

        # Flattened
        desc = CoulombMatrix(n_atoms_max=5, permutation="random", sigma=100, flatten=True)
        cm = desc.create(H2O)
        self.assertEqual(cm.shape, (1, 25))

    def test_sparse(self):
        """Tests the sparse matrix creation.
        """
        # Dense
        desc = CoulombMatrix(n_atoms_max=5, permutation="random", sigma=100, flatten=False, sparse=False)
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == np.ndarray)

        # Sparse
        desc = CoulombMatrix(n_atoms_max=5, permutation="random", sigma=100, flatten=True, sparse=True)
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == scipy.sparse.coo_matrix)

    def test_norm_vector(self):
        """Tests if the attribute _norm_vector is written and used correctly
        """
        desc = CoulombMatrix(n_atoms_max=5, permutation="random", sigma=100, flatten=False)
        cm = desc.create(H2O)
        self.assertEqual(len(cm), 5)

        # The norm_vector is not zero padded in this implementation. All zero-padding
        # is done at the end after randomly sorting
        self.assertEqual(len(desc._norm_vector), 3)
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
        rand_instances = 20000
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

    def test_symmetries(self):
        """Tests the symmetries of the descriptor.
        """
        # The symmetries should be present when sigma is set to very low
        # values. With higer sigma values this descriptor is no longer
        # symmetric.
        def create(system):
            desc = CoulombMatrix(n_atoms_max=3, permutation="random", sigma=0.000001, flatten=True)
            return desc.create(system)

        # Rotational
        self.assertTrue(self.is_rotationally_symmetric(create))

        # Translational
        self.assertTrue(self.is_translationally_symmetric(create))

        # Permutational
        self.assertTrue(self.is_permutation_symmetric(create))


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SortedMatrixTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(EigenSpectrumTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(RandomMatrixTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
