from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import math
import unittest

import numpy as np

import scipy.sparse

from dscribe.descriptors import SOAP
from testbaseclass import TestBaseClass

from ase import Atoms


H2O = Atoms(
    cell=[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ],
    positions=[
        [0, 0, 0],
        [0.95, 0, 0],
        [0.95*(1+math.cos(76/180*math.pi)), 0.95*math.sin(76/180*math.pi), 0.0]
    ],
    symbols=["H", "O", "H"],
)

H = Atoms(
    cell=[
        [15.0, 0.0, 0.0],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 15.0]
    ],
    positions=[
        [0, 0, 0],

    ],
    symbols=["H"],
)


class SoapTests(TestBaseClass, unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """
        # Invalid atomic numbers
        with self.assertRaises(ValueError):
            SOAP(atomic_numbers=[-1, 2], rcut=5, nmax=5, lmax=5, periodic=True)

    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        lmax = 5
        nmax = 5
        n_elems = 2
        desc = SOAP(atomic_numbers=[1, 8], rcut=5, nmax=nmax, lmax=lmax, periodic=True)

        # Test that the reported number of features matches the expected
        n_features = desc.get_number_of_features()
        n_blocks = n_elems*(n_elems+1)/2
        expected = int((lmax + 1) * nmax * (nmax + 1) / 2 * n_blocks)
        self.assertEqual(n_features, expected)

        # Test that the outputted number of features matches the reported
        n_features = desc.get_number_of_features()
        vec = desc.create(H2O)
        self.assertEqual(n_features, vec.shape[1])

    def test_flatten(self):
        """Tests the flattening.
        """

    def test_sparse(self):
        """Tests the sparse matrix creation.
        """
        # Dense
        desc = SOAP(atomic_numbers=[1, 8], rcut=5, nmax=5, lmax=5, periodic=True, sparse=False)
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == np.ndarray)

        # Sparse
        desc = SOAP(atomic_numbers=[1, 8], rcut=5, nmax=5, lmax=5, periodic=True, sparse=True)
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == scipy.sparse.coo_matrix)

    def test_positions(self):
        """Tests that different positions are handled correctly.
        """
        desc = SOAP([1, 6, 8], 10.0, 2, 0, periodic=False, crossover=True,)
        self.assertEqual(desc.get_number_of_features(), desc.create(H2O, positions=[[0, 0, 0]]).shape[1])
        self.assertEqual(desc.get_number_of_features(), desc.create(H2O, positions=[0]).shape[1])
        self.assertEqual(desc.get_number_of_features(), desc.create(H2O, positions=[]).shape[1])

        desc = SOAP([1, 6, 8], 10.0, 2, 0, periodic=True, crossover=True,)
        self.assertEqual(desc.get_number_of_features(), desc.create(H2O, positions=[[0, 0, 0]]).shape[1])
        self.assertEqual(desc.get_number_of_features(), desc.create(H2O, positions=[0]).shape[1])
        self.assertEqual(desc.get_number_of_features(), desc.create(H2O, positions=[]).shape[1])

        desc = SOAP([1, 6, 8], 10.0, 2, 0, periodic=True, crossover=False,)
        self.assertEqual(desc.get_number_of_features(), desc.create(H2O, positions=[[0, 0, 0]]).shape[1])
        self.assertEqual(desc.get_number_of_features(), desc.create(H2O, positions=[0]).shape[1])
        self.assertEqual(desc.get_number_of_features(), desc.create(H2O, positions=[]).shape[1])

        desc = SOAP([1, 6, 8], 10.0, 2, 0, periodic=False, crossover=False,)
        self.assertEqual(desc.get_number_of_features(), desc.create(H2O, positions=[[0, 0, 0]]).shape[1])
        self.assertEqual(desc.get_number_of_features(), desc.create(H2O, positions=[0]).shape[1])
        self.assertEqual(desc.get_number_of_features(), desc.create(H2O, positions=[]).shape[1])

        with self.assertRaises(ValueError):
            desc.create(H2O, positions=['a'])

    def test_unit_cells(self):
        """Tests if arbitrary unit cells are accepted"""
        desc = SOAP([1, 6, 8], 10.0, 2, 0, periodic=False, crossover=True,)

        molecule = H2O.copy()

        molecule.set_cell([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])

        nocell = desc.create(molecule, positions=[[0, 0, 0]])

        desc = SOAP([1, 6, 8], 10.0, 2, 0, periodic=True, crossover=True,)

        # Invalid unit cell
        molecule.set_cell([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        with self.assertRaises(ValueError):
            desc.create(molecule, positions=[[0, 0, 0]])

        molecule.set_pbc(True)
        molecule.set_cell([
        [20.0, 0.0, 0.0],
        [0.0, 30.0, 0.0],
        [0.0, 0.0, 40.0]
            ],
            )

        largecell = desc.create(molecule, positions=[[0, 0, 0]])

        molecule.set_cell([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])

        cubic_cell = desc.create(molecule, positions=[[0, 0, 0]])

        molecule.set_cell([
            [0.0, 2.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 0.0]
        ])

        triclinic_smallcell = desc.create(molecule, positions=[[0, 0, 0]])

    def test_is_periodic(self):
        """Tests whether periodic images are seen by the descriptor"""
        desc = SOAP([1, 6, 8], 10.0, 2, 0, periodic=False, crossover=True,)

        H2O.set_pbc(False)
        nocell = desc.create(H2O, positions=[[0, 0, 0]])

        H2O.set_pbc(True)
        H2O.set_cell([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        desc = SOAP([1, 6, 8], 10.0, 2, 0, periodic=True, crossover=True,)

        cubic_cell = desc.create(H2O, positions=[[0, 0, 0]])

        self.assertTrue(np.sum(cubic_cell) > 0)

    def test_periodic_images(self):
        """Tests the periodic images seen by the descriptor
        """
        desc = SOAP([1, 6, 8], 10.0, 2, 0, periodic=False, crossover=True,)

        molecule = H2O.copy()

        # non-periodic for comparison
        molecule.set_cell([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        nocell = desc.create(molecule, positions=[[0, 0, 0]]).toarray()

        # Make periodic
        desc = SOAP([1, 6, 8], 10.0, 2, 0, periodic=True, crossover=True,)
        molecule.set_pbc(True)

        # Cubic
        molecule.set_cell([
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0]
        ])
        cubic_cell = desc.create(molecule, positions=[[0, 0, 0]]).toarray()
        suce = molecule * (2, 1, 1)
        cubic_suce = desc.create(suce, positions=[[0, 0, 0]]).toarray()

        # Triclinic
        molecule.set_cell([
            [0.0, 2.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 0.0]
        ])
        triclinic_cell = desc.create(molecule, positions=[[0, 0, 0]]).toarray()
        suce = molecule * (2, 1, 1)
        triclinic_suce = desc.create(suce, positions=[[0, 0, 0]]).toarray()

        self.assertTrue(np.sum(np.abs((nocell[:3] - cubic_suce[:3]))) > 0.1)
        self.assertAlmostEqual(np.sum(cubic_cell[:3] - cubic_suce[:3]), 0)
        self.assertAlmostEqual(np.sum(triclinic_cell[:3] - triclinic_suce[:3]), 0)

    def test_symmetries(self):
        """Tests that the descriptor has the correct invariances.
        """
        def create(system):
            desc = SOAP(
                atomic_numbers=system.get_atomic_numbers(),
                rcut=8.0,
                lmax=5,
                nmax=5,
                periodic=False,
                crossover=True
            )
            return desc.create(system)

        # Rotational check
        self.assertTrue(self.is_rotationally_symmetric(create))

        # Translational
        self.assertTrue(self.is_translationally_symmetric(create))

        # Permutational
        self.assertTrue(self.is_permutation_symmetric(create))

    def test_atom_gaussian(self):
        """Tests that the gaussian width of the atoms is handled correctly.
        """
        # Test that it works for finite systems
        sys = Atoms(symbols=["H"], positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
        desc = SOAP(
            atomic_numbers=[1],
            rcut=5,
            nmax=2,
            lmax=5,
            periodic=False,
            crossover=False,
            sparse=False,
            alpha=1.001
        )
        output1 = desc.create(sys, positions=[0])

        desc = SOAP(
            atomic_numbers=[1],
            rcut=5,
            nmax=2,
            lmax=5,
            periodic=False,
            crossover=False,
            sparse=False,
            alpha=1.0
        )
        output2 = desc.create(sys, positions=[0])

        self.assertFalse(np.array_equal(output1, output2))
        self.assertTrue(np.allclose(output1, output2, rtol=0.01))

        # Test that it works for periodic systems
        sys = Atoms(symbols=["H"], positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
        desc = SOAP(
            atomic_numbers=[1],
            rcut=5,
            nmax=2,
            lmax=5,
            periodic=True,
            crossover=False,
            sparse=False,
            alpha=1.001
        )
        output1 = desc.create(sys, positions=[0])

        desc = SOAP(
            atomic_numbers=[1],
            rcut=5,
            nmax=2,
            lmax=5,
            periodic=True,
            crossover=False,
            sparse=False,
            alpha=1.0
        )
        output2 = desc.create(sys, positions=[0])

        self.assertFalse(np.array_equal(output1, output2))
        self.assertTrue(np.allclose(output1, output2, rtol=0.01))

    def test_average(self):
        """Tests that the average output is created correctly.
        """
        sys = Atoms(symbols=["H", "C"], positions=[[-1, 0, 0], [1, 0, 0]], cell=[2, 2, 2], pbc=True)

        # Create the average output
        desc = SOAP(
            atomic_numbers=[1, 6, 8],
            rcut=5,
            nmax=3,
            lmax=5,
            periodic=False,
            crossover=True,
            average=True,
            sparse=False
        )
        average = desc.create(sys)[0, :]

        # Create individual output for both atoms
        desc = SOAP(
            atomic_numbers=[1, 6, 8],
            rcut=5,
            nmax=3,
            lmax=5,
            periodic=False,
            crossover=True,
            average=False,
            sparse=False
        )
        first = desc.create(sys, positions=[0])[0, :]
        second = desc.create(sys, positions=[1])[0, :]

        # Check that the normalization is done correctly, by first normalizing
        # the outputs and then averaging them.
        first_normalized = first/np.linalg.norm(first, axis=0)
        second_normalized = second/np.linalg.norm(second, axis=0)
        assumed_average = (first_normalized+second_normalized)/2
        self.assertTrue(np.array_equal(average, assumed_average))

    def test_basis(self):
        """Tests that the output vectors behave correctly as a basis.
        """
        sys1 = Atoms(symbols=["H", "H"], positions=[[1, 0, 0], [0, 1, 0]], cell=[2, 2, 2], pbc=True)
        sys2 = Atoms(symbols=["O", "O"], positions=[[1, 0, 0], [0, 1, 0]], cell=[2, 2, 2], pbc=True)
        sys3 = Atoms(symbols=["C", "C"], positions=[[1, 0, 0], [0, 1, 0]], cell=[2, 2, 2], pbc=True)
        sys4 = Atoms(symbols=["H", "C"], positions=[[-1, 0, 0], [1, 0, 0]], cell=[2, 2, 2], pbc=True)
        sys5 = Atoms(symbols=["H", "C"], positions=[[1, 0, 0], [0, 1, 0]], cell=[2, 2, 2], pbc=True)
        sys6 = Atoms(symbols=["H", "O"], positions=[[1, 0, 0], [0, 1, 0]], cell=[2, 2, 2], pbc=True)
        sys7 = Atoms(symbols=["C", "O"], positions=[[1, 0, 0], [0, 1, 0]], cell=[2, 2, 2], pbc=True)

        desc = SOAP(
            atomic_numbers=[1, 6, 8],
            rcut=5,
            nmax=3,
            lmax=5,
            periodic=False,
            crossover=True,
            normalize=True,
            sparse=False
        )

        # Create normalized vectors for each system
        vec1 = desc.create(sys1, positions=[[0, 0, 0]])[0, :]
        vec2 = desc.create(sys2, positions=[[0, 0, 0]])[0, :]
        vec3 = desc.create(sys3, positions=[[0, 0, 0]])[0, :]
        vec4 = desc.create(sys4, positions=[[0, 0, 0]])[0, :]
        vec5 = desc.create(sys5, positions=[[0, 0, 0]])[0, :]
        vec6 = desc.create(sys6, positions=[[0, 0, 0]])[0, :]
        vec7 = desc.create(sys7, positions=[[0, 0, 0]])[0, :]

        # The dot-product should be zero when there are no overlapping elements
        dot = np.dot(vec1, vec2)
        self.assertEqual(dot, 0)
        dot = np.dot(vec2, vec3)
        self.assertEqual(dot, 0)

        # The dot-product should be non-zero when there are overlapping elements
        dot = np.dot(vec4, vec5)
        self.assertNotEqual(dot, 0)

        # Check that self-terms are in correct location
        n_elem_feat = desc.get_number_of_element_features()
        h_part1 = vec1[0:n_elem_feat]
        h_part2 = vec2[0:n_elem_feat]
        h_part4 = vec4[0:n_elem_feat]
        self.assertNotEqual(np.sum(h_part1), 0)
        self.assertEqual(np.sum(h_part2), 0)
        self.assertNotEqual(np.sum(h_part4), 0)

        # Check that cross terms are in correct location
        hc_part1 = vec1[1*n_elem_feat:2*n_elem_feat]
        hc_part4 = vec4[1*n_elem_feat:2*n_elem_feat]
        co_part6 = vec6[4*n_elem_feat:5*n_elem_feat]
        co_part7 = vec7[4*n_elem_feat:5*n_elem_feat]
        self.assertEqual(np.sum(hc_part1), 0)
        self.assertNotEqual(np.sum(hc_part4), 0)
        self.assertEqual(np.sum(co_part6), 0)
        self.assertNotEqual(np.sum(co_part7), 0)

if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SoapTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
