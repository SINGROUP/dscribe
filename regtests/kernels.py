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
        desc = SOAP(
            species=[1, 6, 7, 8],
            rcut=5.0,
            nmax=2,
            lmax=2,
            sigma=0.2,
            periodic=False,
            crossover=True,
            sparse=False
        )

        # Calculate that identical molecules are identical.
        a = molecule("H2O")
        a_features = desc.create(a)
        kernel = AverageKernel(metric="linear")
        K = kernel.create([a_features, a_features])
        self.assertTrue(np.all(np.abs(K - 1) < 1e-3))

        # Check that completely different molecules are completely different
        a = molecule("N2")
        b = molecule("H2O")
        a_features = desc.create(a)
        b_features = desc.create(b)
        K = kernel.create([a_features, b_features])
        self.assertTrue(np.all(np.abs(K - np.eye(2)) < 1e-3))

        # Check that somewhat similar molecules are somewhat similar
        a = molecule("H2O")
        b = molecule("H2O2")
        a_features = desc.create(a)
        b_features = desc.create(b)
        K = kernel.create([a_features, b_features])
        self.assertTrue(K[0, 1] > 0.9)

    def test_metrics(self):
        """Tests that different metrics as defined by scikit-learn can be used.
        """
        # Create SOAP features for a system
        desc = SOAP(
            species=[1, 8],
            rcut=5.0,
            nmax=2,
            lmax=2,
            sigma=0.2,
            periodic=False,
            crossover=True,
            sparse=False
        )
        a = molecule('H2O')
        a_features = desc.create(a)

        # Linear dot-product kernel
        kernel = AverageKernel(metric="linear")
        K = kernel.create([a_features, a_features])

        # Gaussian kernel
        kernel = AverageKernel(metric="rbf", gamma=1)
        K = kernel.create([a_features, a_features])

        # Laplacian kernel
        kernel = AverageKernel(metric="laplacian", gamma=1)
        K = kernel.create([a_features, a_features])

    def test_xy(self):
        """Tests that the kernel can be also calculated between two different
        sets, which is necessary for making predictions with kernel-based
        methods.
        """
        # Create SOAP features for a system
        desc = SOAP(
            species=[1, 8],
            rcut=5.0,
            nmax=2,
            lmax=2,
            sigma=0.2,
            periodic=False,
            crossover=True,
            sparse=False
        )
        a = molecule('H2O')
        b = molecule('O2')
        c = molecule('H2O2')

        a_feat = desc.create(a)
        b_feat = desc.create(b)
        c_feat = desc.create(c)

        # Linear dot-product kernel
        kernel = AverageKernel(metric="linear")
        K = kernel.create([a_feat, b_feat], [c_feat])

        self.assertEqual(K.shape, (2, 1))

    def test_sparse(self):
        """Tests that sparse features may also be used to construct the kernels.
        """
        # Create SOAP features for a system
        desc = SOAP(
            species=[1, 8],
            rcut=5.0,
            nmax=2,
            lmax=2,
            sigma=0.2,
            periodic=False,
            crossover=True,
            sparse=True
        )
        a = molecule('H2O')
        a_feat = desc.create(a)
        kernel = AverageKernel(metric="linear")
        K = kernel.create([a_feat])


class REMatchKernelTests(unittest.TestCase):

    def test_difference(self):
        """Tests that the similarity is correct.
        """
        # Create SOAP features for a system
        desc = SOAP(
            species=[1, 6, 7, 8],
            rcut=5.0,
            nmax=2,
            lmax=2,
            sigma=0.2,
            periodic=False,
            crossover=True,
            sparse=False
        )

        # Calculate that identical molecules are identical.
        a = molecule("H2O")
        a_features = desc.create(a)
        kernel = REMatchKernel(metric="linear", alpha=1, threshold=1e-6)
        K = kernel.create([a_features, a_features])
        self.assertTrue(np.all(np.abs(K - 1) < 1e-3))

        # Check that completely different molecules are completely different
        a = molecule("N2")
        b = molecule("H2O")
        a_features = desc.create(a)
        b_features = desc.create(b)
        K = kernel.create([a_features, b_features])
        self.assertTrue(np.all(np.abs(K - np.eye(2)) < 1e-3))

        # Check that somewhat similar molecules are somewhat similar
        a = molecule("H2O")
        b = molecule("H2O2")
        a_features = desc.create(a)
        b_features = desc.create(b)
        K = kernel.create([a_features, b_features])
        self.assertTrue(K[0, 1] > 0.9)

    def test_convergence_infinity(self):
        """Tests that the REMatch kernel correctly converges to the average
        kernel at the the limit of infinite alpha.
        """
        # Create SOAP features for a system
        desc = SOAP(
            species=[1, 8],
            rcut=5.0,
            nmax=2,
            lmax=2,
            sigma=0.2,
            periodic=False,
            crossover=True,
            sparse=False
        )
        a = molecule('H2O')
        b = molecule('H2O2')
        a_features = desc.create(a)
        b_features = desc.create(b)

        # REMatch kernel with very high alpha
        kernel_re = REMatchKernel(metric="linear", alpha=1e20, threshold=1e-6)
        K_re = kernel_re.create([a_features, b_features])

        # Average kernel
        kernel_ave = AverageKernel(metric="linear")
        K_ave = kernel_ave.create([a_features, b_features])

        # Test approximate equality
        self.assertTrue(np.allclose(K_re, K_ave))

    def test_metrics(self):
        """Tests that different metrics as defined by scikit-learn can be used.
        """
        # Create SOAP features for a system
        desc = SOAP(
            species=[1, 8],
            rcut=5.0,
            nmax=2,
            lmax=2,
            sigma=0.2,
            periodic=False,
            crossover=True,
            sparse=False
        )
        a = molecule('H2O')
        a_features = desc.create(a)

        # Linear dot-product kernel
        kernel = REMatchKernel(metric="linear", alpha=0.1, threshold=1e-6)
        K = kernel.create([a_features, a_features])

        # Gaussian kernel
        kernel = REMatchKernel(metric="rbf", gamma=1, alpha=0.1, threshold=1e-6)
        K = kernel.create([a_features, a_features])

        # Laplacian kernel
        kernel = REMatchKernel(metric="laplacian", gamma=1, alpha=0.1, threshold=1e-6)
        K = kernel.create([a_features, a_features])

    def test_xy(self):
        """Tests that the kernel can be also calculated between two different
        sets, which is necessary for making predictions with kernel-based
        methods.
        """
        # Create SOAP features for a system
        desc = SOAP(
            species=[1, 8],
            rcut=5.0,
            nmax=2,
            lmax=2,
            sigma=0.2,
            periodic=False,
            crossover=True,
            sparse=False
        )
        a = molecule('H2O')
        b = molecule('O2')
        c = molecule('H2O2')

        a_feat = desc.create(a)
        b_feat = desc.create(b)
        c_feat = desc.create(c)

        # Linear dot-product kernel
        kernel = REMatchKernel(metric="linear", alpha=0.1, threshold=1e-6)
        K = kernel.create([a_feat, b_feat], [c_feat])

        self.assertEqual(K.shape, (2, 1))

    def test_sparse(self):
        """Tests that sparse features may also be used to construct the kernels.
        """
        # Create SOAP features for a system
        desc = SOAP(
            species=[1, 8],
            rcut=5.0,
            nmax=2,
            lmax=2,
            sigma=0.2,
            periodic=False,
            crossover=True,
            sparse=True
        )
        a = molecule('H2O')
        a_feat = desc.create(a)
        kernel = REMatchKernel(metric="linear", alpha=0.1, threshold=1e-6)
        K = kernel.create([a_feat])


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(AverageKernelTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(REMatchKernelTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
