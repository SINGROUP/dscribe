from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
import unittest

import numpy as np

from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel
from dscribe.kernels import AverageKernel

from ase.build import molecule


class AverageKernelTests(unittest.TestCase):

    def test_difference(self):
        """Tests that the similarity is correct.
        """
        # Create SOAP features for a system
        desc = SOAP([1, 6, 7, 8], 5.0, 2, 2, sigma=0.2, periodic=False, crossover=True, sparse=False)

        # Calculate that identical molecules are identical.
        a = molecule("H2O")
        a_features = desc.create(a)
        re = AverageKernel(metric="linear")
        re_kernel = re.create([a_features, a_features])
        self.assertTrue(np.all(np.abs(re_kernel - 1) < 1e-3))

        # Check that completely different molecules are completely different
        a = molecule("N2")
        b = molecule("H2O")
        a_features = desc.create(a)
        b_features = desc.create(b)
        re_kernel = re.create([a_features, b_features])
        self.assertTrue(np.all(np.abs(re_kernel - np.eye(2)) < 1e-3))

        # Check that somewhat similar molecules are somewhat similar
        a = molecule("H2O")
        b = molecule("H2O2")
        a_features = desc.create(a)
        b_features = desc.create(b)
        re_kernel = re.create([a_features, b_features])
        self.assertTrue(re_kernel[0, 1] > 0.9)

    def test_metrics(self):
        """Tests that different metrics as defined by scikit-learn can be used.
        """
        # Create SOAP features for a system
        desc = SOAP([1, 8], 5.0, 2, 2, sigma=0.2, periodic=False, crossover=True, sparse=False)
        a = molecule('H2O')
        a_features = desc.create(a)

        # Linear dot-product kernel
        re = AverageKernel(metric="linear")
        re_kernel = re.create([a_features, a_features])

        # Gaussian kernel
        re = AverageKernel(metric="rbf", gamma=1)
        re_kernel = re.create([a_features, a_features])

        # Laplacian kernel
        re = AverageKernel(metric="laplacian", gamma=1)
        re_kernel = re.create([a_features, a_features])


class REMatchKernelTests(unittest.TestCase):

    def test_difference(self):
        """Tests that the similarity is correct.
        """
        # Create SOAP features for a system
        desc = SOAP([1, 6, 7, 8], 5.0, 2, 2, sigma=0.2, periodic=False, crossover=True, sparse=False)

        # Calculate that identical molecules are identical.
        a = molecule("H2O")
        a_features = desc.create(a)
        re = REMatchKernel(metric="linear", alpha=1, threshold=1e-6)
        re_kernel = re.create([a_features, a_features])
        self.assertTrue(np.all(np.abs(re_kernel - 1) < 1e-3))

        # Check that completely different molecules are completely different
        a = molecule("N2")
        b = molecule("H2O")
        a_features = desc.create(a)
        b_features = desc.create(b)
        re_kernel = re.create([a_features, b_features])
        self.assertTrue(np.all(np.abs(re_kernel - np.eye(2)) < 1e-3))

        # Check that somewhat similar molecules are somewhat similar
        a = molecule("H2O")
        b = molecule("H2O2")
        a_features = desc.create(a)
        b_features = desc.create(b)
        re_kernel = re.create([a_features, b_features])
        self.assertTrue(re_kernel[0, 1] > 0.9)

    def test_convergence_infinity(self):
        """Tests that the REMatch kernel correctly converges to the average
        kernel at the the limit of infinite alpha.
        """
        # Create SOAP features for a system
        desc = SOAP([1, 8], 5.0, 2, 2, sigma=0.2, periodic=False, crossover=True, sparse=False)
        a = molecule('H2O')
        b = molecule('H2O2')
        a_features = desc.create(a)
        b_features = desc.create(b)

        # REMatch kernel with very high alpha
        re = REMatchKernel(metric="linear", alpha=1e20, threshold=1e-6)
        re_kernel = re.create([a_features, b_features])

        # Average kernel
        ave = AverageKernel(metric="linear")
        ave_kernel = ave.create([a_features, b_features])

        # Test approximate equality
        self.assertTrue(np.allclose(re_kernel, ave_kernel))

    def test_metrics(self):
        """Tests that different metrics as defined by scikit-learn can be used.
        """
        # Create SOAP features for a system
        desc = SOAP([1, 8], 5.0, 2, 2, sigma=0.2, periodic=False, crossover=True, sparse=False)
        a = molecule('H2O')
        a_features = desc.create(a)

        # Linear dot-product kernel
        re = REMatchKernel(metric="linear", alpha=0.1, threshold=1e-6)
        re_kernel = re.create([a_features, a_features])

        # Gaussian kernel
        re = REMatchKernel(metric="rbf", gamma=1, alpha=0.1, threshold=1e-6)
        re_kernel = re.create([a_features, a_features])

        # Laplacian kernel
        re = REMatchKernel(metric="laplacian", gamma=1, alpha=0.1, threshold=1e-6)
        re_kernel = re.create([a_features, a_features])


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(AverageKernelTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(REMatchKernelTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
