import math
import unittest

import numpy as np

import scipy.sparse

from ase import Atoms
from ase.build import bulk

from dscribe.descriptors import SineMatrix
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


class SineMatrixTests(TestBaseClass, unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """
        with self.assertRaises(ValueError):
            SineMatrix(n_atoms_max=5, permutation="unknown")
        with self.assertRaises(ValueError):
            SineMatrix(n_atoms_max=-1)

    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        desc = SineMatrix(n_atoms_max=5, permutation="none", flatten=False)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 25)

    def test_flatten(self):
        """Tests the flattening.
        """
        # Unflattened
        desc = SineMatrix(n_atoms_max=5, permutation="none", flatten=False)
        cm = desc.create(H2O)
        self.assertEqual(cm.shape, (5, 5))

        # Flattened
        desc = SineMatrix(n_atoms_max=5, permutation="none", flatten=True)
        cm = desc.create(H2O)
        self.assertEqual(cm.shape, (1, 25))

    def test_sparse(self):
        """Tests the sparse matrix creation.
        """
        # Dense
        desc = SineMatrix(n_atoms_max=5, permutation="none", flatten=True, sparse=False)
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == np.ndarray)

        # Sparse
        desc = SineMatrix(n_atoms_max=5, permutation="none", flatten=True, sparse=True)
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == scipy.sparse.coo_matrix)

    def test_parallel_dense(self):
        """Tests creating dense output parallelly.
        """
        samples = [bulk("NaCl", "rocksalt", a=5.64), bulk('Cu', 'fcc', a=3.6)]
        desc = SineMatrix(n_atoms_max=5, permutation="none", flatten=True, sparse=False)
        n_features = desc.get_number_of_features()

        # Multiple systems, serial job
        output = desc.create(
            system=samples,
            n_jobs=1,
        )
        assumed = np.empty((2, n_features))
        assumed[0, :] = desc.create(samples[0])
        assumed[1, :] = desc.create(samples[1])
        self.assertTrue(np.allclose(output, assumed))

        # Multiple systems, parallel job
        output = desc.create(
            system=samples,
            n_jobs=2,
        )
        assumed = np.empty((2, n_features))
        assumed[0, :] = desc.create(samples[0])
        assumed[1, :] = desc.create(samples[1])
        self.assertTrue(np.allclose(output, assumed))

        # Non-flattened output
        desc = SineMatrix(n_atoms_max=5, permutation="none", flatten=False, sparse=False)
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
        samples = [bulk("NaCl", "rocksalt", a=5.64), bulk('Cu', 'fcc', a=3.6)]
        desc = SineMatrix(n_atoms_max=5, permutation="none", flatten=True, sparse=True)
        n_features = desc.get_number_of_features()

        # Multiple systems, serial job
        output = desc.create(
            system=samples,
            n_jobs=1,
        ).toarray()
        assumed = np.empty((2, n_features))
        assumed[0, :] = desc.create(samples[0]).toarray()
        assumed[1, :] = desc.create(samples[1]).toarray()
        self.assertTrue(np.allclose(output, assumed))

        # Multiple systems, parallel job
        output = desc.create(
            system=samples,
            n_jobs=2,
        ).toarray()
        assumed = np.empty((2, n_features))
        assumed[0, :] = desc.create(samples[0]).toarray()
        assumed[1, :] = desc.create(samples[1]).toarray()
        self.assertTrue(np.allclose(output, assumed))

        # Non-flattened output
        desc = SineMatrix(n_atoms_max=5, permutation="none", flatten=False, sparse=True)
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
        desc = SineMatrix(n_atoms_max=2, permutation="none", flatten=False)

        # Test that without cell the matrix cannot be calculated
        system = Atoms(
            positions=[[0, 0, 0], [1.0, 1.0, 1.0]],
            symbols=["H", "H"],
        )
        with self.assertRaises(ValueError):
            desc.create(system)

        # Test that periodic boundaries are considered by seeing that an atom
        # in the origin is replicated to the  corners
        system = Atoms(
            cell=[
                [10, 10, 0],
                [0, 10, 0],
                [0, 0, 10],
            ],
            scaled_positions=[[0, 0, 0], [1.0, 1.0, 1.0]],
            symbols=["H", "H"],
            pbc=True,
        )
        # from ase.visualize import view
        # view(system)
        matrix = desc.create(system)

        # The interaction between atoms 1 and 2 should be infinite due to
        # periodic boundaries.
        self.assertEqual(matrix[0, 1], float("Inf"))

        # The interaction of an atom with itself is always 0.5*Z**2.4
        atomic_numbers = system.get_atomic_numbers()
        for i, i_diag in enumerate(np.diag(matrix)):
            self.assertEqual(i_diag, 0.5*atomic_numbers[i]**2.4)

    def test_unit_cells(self):
        """Tests if arbitrary unit cells are accepted"""
        desc = SineMatrix(n_atoms_max=3, permutation="none", flatten=False)

        molecule = H2O.copy()

        molecule.set_cell([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        with self.assertRaises(ValueError):
            nocell = desc.create(molecule)

        molecule.set_pbc(True)
        molecule.set_cell([
            [20.0, 0.0, 0.0],
            [0.0, 30.0, 0.0],
            [0.0, 0.0, 40.0]
        ])

        largecell = desc.create(molecule)

        molecule.set_cell([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])

        cubic_cell = desc.create(molecule)

        molecule.set_cell([
            [0.0, 2.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 0.0]
        ])

        triclinic_smallcell = desc.create(molecule)

    def test_symmetries(self):
        """Tests the symmetries of the descriptor.
        """
        def create(system):
            desc = SineMatrix(n_atoms_max=3, permutation="sorted_l2", flatten=True)
            return desc.create(system)

        # Rotational
        self.assertTrue(self.is_rotationally_symmetric(create))

        # Translational
        self.assertTrue(self.is_translationally_symmetric(create))

        # Permutational
        self.assertTrue(self.is_permutation_symmetric(create))

    # def test_visual(self):
        # import matplotlib.pyplot as mpl
        # """Plot the
        # """
        # test_sys = Atoms(
            # cell=[[1, 0.0, 0.0], [1, 1, 0.0], [0.0, 0.0, 1.0]],
            # positions=[[0, 0, 0], [2, 1, 1]],
            # symbols=["H", "H"],
        # )
        # test_sys.charges = np.array([1, 1])

        # desc = SineMatrix(n_atoms_max=5, flatten=False)

        # # Create a graph of the interaction in a 2D slice
        # size = 100
        # x_min = 0.0
        # x_max = 3
        # y_min = 0.0
        # y_max = 3
        # x_axis = np.linspace(x_min, x_max, size)
        # y_axis = np.linspace(y_min, y_max, size)
        # interaction = np.empty((size, size))
        # for i, x in enumerate(x_axis):
            # for j, y in enumerate(y_axis):
                # temp_sys = Atoms(
                    # cell=[[1, 0.0, 0.0], [1, 1, 0.0], [0.0, 0.0, 1.0]],
                    # positions=[[0, 0, 0], [x, y, 0]],
                    # symbols=["H", "H"],
                # )
                # temp_sys.set_initial_charges(np.array([1, 1]))
                # value = desc.create(temp_sys)
                # interaction[i, j] = value[0, 1]

        # mpl.imshow(interaction, cmap='RdBu', vmin=0, vmax=100,
                # extent=[x_min, x_max, y_min, y_max],
                # interpolation='nearest', origin='lower')
        # mpl.colorbar()
        # mpl.show()

if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SineMatrixTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
