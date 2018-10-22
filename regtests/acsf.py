from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import math
import unittest

import numpy as np

import scipy.linalg

from describe.descriptors import ACSF
from testbaseclass import TestBaseClass

from ase import Atoms


H2O = Atoms(
    cell=[
        [15.0, 0.0, 0.0],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 15.0]
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

default_desc = ACSF(
    n_atoms_max=5,
    atomic_numbers=[1, 8],
    bond_params=[[1, 2], [4, 5]],
    bond_cos_params=[1, 2, 3, 4],
    ang4_params=[[1, 2, 3], [3, 1, 4], [4, 5, 6], [7, 8, 9]],
    ang5_params=[[1, 2, 3], [3, 1, 4], [4, 5, 6], [7, 8, 9]],
    flatten=True
)


class ACSFTests(TestBaseClass, unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """
        # Invalid n_atoms_max
        with self.assertRaises(ValueError):
            ACSF(n_atoms_max=-1, atomic_numbers=[1, 8, 6])

        # Invalid atomic_numbers
        with self.assertRaises(ValueError):
            ACSF(n_atoms_max=10, atomic_numbers=None)

        # Invalid bond_params
        with self.assertRaises(ValueError):
            ACSF(n_atoms_max=10, atomic_numbers=[1, 6, 8], bond_params=[1, 2, 3])

        # Invalid bond_cos_params
        with self.assertRaises(ValueError):
            ACSF(n_atoms_max=10, atomic_numbers=[1, 6, 8], bond_cos_params=[[1, 2], [3, 1]])

        # Invalid bond_cos_params
        with self.assertRaises(ValueError):
            ACSF(n_atoms_max=10, atomic_numbers=[1, 6, 8], bond_cos_params=[[1, 2, 4], [3, 1]])

        # Invalid ang4_params
        with self.assertRaises(ValueError):
            ACSF(n_atoms_max=10, atomic_numbers=[1, 6, 8], ang4_params=[[1, 2], [3, 1]])

        # Invalid ang5_params
        with self.assertRaises(ValueError):
            ACSF(n_atoms_max=10, atomic_numbers=[1, 6, 8], ang5_params=[[1, 2], [3, 1]])

    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        desc = ACSF(n_atoms_max=5, atomic_numbers=[1, 8], flatten=True)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 5 * 2)

        desc = ACSF(n_atoms_max=5, atomic_numbers=[1, 8], bond_params=[[1, 2], [4, 5]], flatten=True)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 5 * 2 * (2+1))

        desc = ACSF(n_atoms_max=5, atomic_numbers=[1, 8], bond_cos_params=[1, 2, 3, 4], flatten=True)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 5 * 2 * (4+1))

        desc = ACSF(n_atoms_max=5, atomic_numbers=[1, 8], ang4_params=[[1, 2, 3], [3, 1, 4], [4, 5, 6], [7, 8, 9]], flatten=True)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 5 * (2 + 4 * 3))

        desc = ACSF(n_atoms_max=5, atomic_numbers=[1, 8], bond_params=[[1, 2], [4, 5]], bond_cos_params=[1, 2, 3, 4],
            ang4_params=[[1, 2, 3], [3, 1, 4], [4, 5, 6], [7, 8, 9]], flatten=True)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 5 * ((2 * (1 + 2 + 4)) + 4 * 3))

    def test_flatten(self):
        """Tests the flattening.
        """
        # Unflattened
        default_desc.flatten = False
        x = default_desc.create(H2O)
        n_features = default_desc.get_number_of_features()
        self.assertEqual(x.shape, (5, n_features / 5))

        # Flattened
        default_desc.flatten = True
        x = default_desc.create(H2O)
        self.assertEqual(x.shape, (n_features,))

    def test_sparse(self):
        """Tests the sparse matrix creation.
        """
        # Sparse
        default_desc.sparse = True
        vec = default_desc.create(H2O)
        self.assertTrue(type(vec) == scipy.sparse.coo_matrix)

        # Dense
        default_desc.sparse = False
        vec = default_desc.create(H2O)
        self.assertTrue(type(vec) == np.ndarray)

    def test_features(self):
        """Tests that the correct features are present in the descriptor.
        """
        desc = ACSF(n_atoms_max=5, atomic_numbers=[1, 8], flatten=True)
        acsfg1 = desc.create(H2O)
        #print(acsfg1.shape)
        #print(acsfg1)

        desc = ACSF(n_atoms_max=3, atomic_numbers=[1, 8], bond_params=[[1, 0.5]], flatten=False)
        acsfg2 = desc.create(H2O)
        #print(acsfg2.shape)
        #print(acsfg2)

        desc = ACSF(n_atoms_max=3, atomic_numbers=[1, 8], bond_cos_params=[1], flatten=False)
        acsfg3 = desc.create(H2O)
        #print(acsfg3.shape)
        #print(acsfg3)

        desc = ACSF(n_atoms_max=3, atomic_numbers=[1, 8], ang4_params=[[1.0, 1.0, 1.0]], flatten=False)
        acsfg4 = desc.create(H2O)
        #print(acsfg4.shape)
        #print(acsfg4)

        desc = ACSF(n_atoms_max=3, atomic_numbers=[1, 8], ang5_params=[[1.0, 1.0, 1.0]], flatten=False)
        acsfg5 = desc.create(H2O)
        #print(acsfg5.shape)
        #print(acsfg5)

        # Test against assumed values
        dist_oh = H2O.get_distance(0, 1)
        dist_hh = H2O.get_distance(0, 2)
        ang_hoh = H2O.get_angle(0, 1, 2) * np.pi / 180.0
        ang_hho = H2O.get_angle(1, 0, 2) * np.pi / 180.0
        ang_ohh = - H2O.get_angle(2, 0, 1) * np.pi / 180.0

        rc = 5.0

        # G1
        g1_ho = 0.5 * (np.cos(np.pi*dist_oh / rc) + 1)
        g1_hh = 0.5 * (np.cos(np.pi*dist_hh / rc) + 1)
        g1_oh = 2 * 0.5 * (np.cos(np.pi*dist_oh / rc) + 1)
        #print(g1_hh, g1_ho, g1_oh)
        self.assertAlmostEqual(acsfg1[0], g1_hh)
        self.assertAlmostEqual(acsfg1[1], g1_ho)
        self.assertAlmostEqual(acsfg1[2], g1_oh)

        # G2
        eta = 1
        rs = 0.5
        g2_hh = np.exp(-eta * np.power((dist_hh - rs), 2)) * g1_hh
        g2_ho = np.exp(-eta * np.power((dist_oh - rs), 2)) * g1_ho
        g2_oh = np.exp(-eta * np.power((dist_oh - rs), 2)) * g1_oh
        #print(g2_hh, g2_ho, g2_oh)
        self.assertAlmostEqual(acsfg2[0, 1], g2_hh)
        self.assertAlmostEqual(acsfg2[0, 3], g2_ho)
        self.assertAlmostEqual(acsfg2[1, 1], g2_oh)

        # G3
        kappa = 1
        g3_hh = np.cos(dist_hh * kappa) * g1_hh
        g3_ho = np.cos(dist_oh * kappa) * g1_ho
        g3_oh = np.cos(dist_oh * kappa) * g1_oh
        #print(g3_hh, g3_ho, g3_oh)
        self.assertAlmostEqual(acsfg3[0, 1], g3_hh)
        self.assertAlmostEqual(acsfg3[0, 3], g3_ho)
        self.assertAlmostEqual(acsfg3[1, 1], g3_oh)

        # G4
        eta = 1
        lmbd = 1
        zeta = 1
        gauss = np.exp(-eta * (2 * dist_oh * dist_oh + dist_hh * dist_hh)) * g1_ho * g1_hh * g1_ho
        g4_h_ho = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_hho)), zeta) * gauss
        g4_h_oh = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_ohh)), zeta) * gauss
        g4_o_hh = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_hoh)), zeta) * gauss
        #print("G4")
        #print(g4_h_ho, g4_h_oh, g4_o_hh)
        self.assertAlmostEqual(acsfg4[0, 3], g4_h_ho)
        self.assertAlmostEqual(acsfg4[2, 3], g4_h_oh)
        self.assertAlmostEqual(acsfg4[1, 2], g4_o_hh)

        # G5
        gauss = np.exp(-eta * (dist_oh * dist_oh + dist_hh * dist_hh)) * g1_ho * g1_hh
        g5_h_ho = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_hho)), zeta) * gauss
        g5_h_oh = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_ohh)), zeta) * gauss
        g5_o_hh = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_hoh)), zeta) * np.exp(-eta * (2 * dist_oh * dist_oh)) * g1_ho * g1_ho
        #print("G5")
        #print(g5_h_ho, g5_h_oh, g5_o_hh)
        self.assertAlmostEqual(acsfg5[0, 3], g5_h_ho)
        self.assertAlmostEqual(acsfg5[2, 3], g5_h_oh)
        self.assertAlmostEqual(acsfg5[1, 2], g5_o_hh)

    def test_symmetries(self):
        """Tests translational and rotational symmetries
        """
        def create(system):
            acsf = default_desc.create(system)
            return acsf

        # Rotational check
        self.assertTrue(self.is_rotationally_symmetric(create))

        # Translational
        self.assertTrue(self.is_translationally_symmetric(create))

        # Permutational
        # self.assertTrue(self.is_permutation_symmetric(create))

    def test_unit_cells(self):
        """Tests if arbitrary unit cells are accepted.
        """
        # No cell
        molecule = H2O.copy()
        molecule.set_cell([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        nocell = default_desc.create(molecule)

        # Large cell
        molecule.set_pbc(True)
        molecule.set_cell([
            [20.0, 0.0, 0.0],
            [0.0, 30.0, 0.0],
            [0.0, 0.0, 40.0]
        ])
        largecell = default_desc.create(molecule)

        # Cubic cell
        molecule.set_cell([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        cubic_cell = default_desc.create(molecule)

        # Triclinic cell
        molecule.set_cell([
            [0.0, 2.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 0.0]
        ])
        triclinic_smallcell = default_desc.create(molecule)

    # def test_is_periodic(self):
        # """Tests whether periodic images are seen by the descriptor.
        # """
        # # Test that system without periodicity and one atom has zero output
        # H.set_pbc(False)
        # finite = default_desc.create(H)
        # self.assertTrue(np.sum(finite) == 0)

        # # Test that the same system with periodicity and small cell has
        # # non-zero output
        # H.set_pbc(True)
        # H.set_cell([
            # [2.0, 0.0, 0.0],
            # [0.0, 2.0, 0.0],
            # [0.0, 0.0, 2.0]
        # ])
        # periodic = default_desc.create(H)
        # self.assertTrue(np.sum(periodic) > 0)

    # def test_periodic_images(self):
        # """Tests the periodic images seen by the descriptor
        # """
        # molecule = H2O.copy()

        # # non-periodic for comparison
        # molecule.set_cell([
            # [0.0, 0.0, 0.0],
            # [0.0, 0.0, 0.0],
            # [0.0, 0.0, 0.0]
        # ])
        # nocell = default_desc.create(molecule)

        # # make periodic
        # molecule.set_pbc(True)

        # # cubic
        # molecule.set_cell([
            # [3.0, 0.0, 0.0],
            # [0.0, 3.0, 0.0],
            # [0.0, 0.0, 3.0]
        # ])

        # cubic_cell = default_desc.create(molecule)
        # suce = molecule * (2, 1, 1)
        # cubic_suce = default_desc.create(suce)

        # molecule.set_cell([
            # [20.0, 0.0, 0.0],
            # [0.0, 30.0, 0.0],
            # [0.0, 0.0, 40.0]
        # ])
        # largecell = default_desc.create(molecule)

        # # Triclinic
        # molecule.set_cell([
            # [0.0, 2.0, 2.0],
            # [2.0, 0.0, 2.0],
            # [2.0, 2.0, 0.0]
        # ])

        # triclinic_cell = default_desc.create(molecule)
        # suce = molecule * (2, 1, 1)
        # triclinic_suce = default_desc.create(suce)

        # self.assertTrue(np.sum(np.abs((nocell[:3] - cubic_suce[:3]))) > 0.1)
        # self.assertAlmostEqual(np.sum(cubic_cell[:3] - cubic_suce[:3]), 0)
        # self.assertAlmostEqual(np.sum(triclinic_cell[:3] - triclinic_suce[:3]), 0)


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ACSFTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
