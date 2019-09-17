# -*- coding: utf-8 -*-
"""Copyright 2019 DScribe developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import math
import unittest

import numpy as np

import scipy.linalg

from dscribe.descriptors import ACSF
from testbaseclass import TestBaseClass

from ase import Atoms
from ase.build import molecule
from ase.build import bulk


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
    rcut=6.0,
    species=[1, 8],
    g2_params=[[1, 2], [4, 5]],
    g3_params=[1, 2, 3, 4],
    g4_params=[[1, 2, 3], [3, 1, 4], [4, 5, 6], [7, 8, 9]],
    g5_params=[[1, 2, 3], [3, 1, 4], [4, 5, 6], [7, 8, 9]],
)


def cutoff(R, rcut):
    return 0.5 * (np.cos(np.pi*R / rcut) + 1)


class ACSFTests(TestBaseClass, unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """
        # Invalid species
        with self.assertRaises(ValueError):
            ACSF(rcut=6.0, species=None)

        # Invalid bond_params
        with self.assertRaises(ValueError):
            ACSF(rcut=6.0, species=[1, 6, 8], g2_params=[1, 2, 3])

        # Invalid bond_cos_params
        with self.assertRaises(ValueError):
            ACSF(rcut=6.0, species=[1, 6, 8], g3_params=[[1, 2], [3, 1]])

        # Invalid bond_cos_params
        with self.assertRaises(ValueError):
            ACSF(rcut=6.0, species=[1, 6, 8], g3_params=[[1, 2, 4], [3, 1]])

        # Invalid ang4_params
        with self.assertRaises(ValueError):
            ACSF(rcut=6.0, species=[1, 6, 8], g4_params=[[1, 2], [3, 1]])

        # Invalid ang5_params
        with self.assertRaises(ValueError):
            ACSF(rcut=6.0, species=[1, 6, 8], g5_params=[[1, 2], [3, 1]])

    def test_properties(self):
        """Used to test that changing the setup through properties works as
        intended.
        """
        # Test changing species
        a = ACSF(
            rcut=6.0,
            species=[1, 8],
            g2_params=[[1, 2]],
            sparse=False,
        )
        nfeat1 = a.get_number_of_features()
        vec1 = a.create(H2O)
        a.species = ["C", "H", "O"]
        nfeat2 = a.get_number_of_features()
        vec2 = a.create(molecule("CH3OH"))
        self.assertTrue(nfeat1 != nfeat2)
        self.assertTrue(vec1.shape[1] != vec2.shape[1])

    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        species = [1, 8]
        n_elem = len(species)

        desc = ACSF(rcut=6.0, species=species)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, n_elem)

        desc = ACSF(rcut=6.0, species=species, g2_params=[[1, 2], [4, 5]])
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, n_elem * (2+1))

        desc = ACSF(rcut=6.0, species=[1, 8], g3_params=[1, 2, 3, 4])
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, n_elem * (4+1))

        desc = ACSF(rcut=6.0, species=[1, 8], g4_params=[[1, 2, 3], [3, 1, 4], [4, 5, 6], [7, 8, 9]])
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, n_elem + 4 * 3)

        desc = ACSF(rcut=6.0, species=[1, 8], g2_params=[[1, 2], [4, 5]], g3_params=[1, 2, 3, 4],
            g4_params=[[1, 2, 3], [3, 1, 4], [4, 5, 6], [7, 8, 9]])
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, n_elem * (1 + 2 + 4) + 4 * 3)

    def test_sparse(self):
        """Tests the sparse matrix creation.
        """
        # Sparse
        default_desc._sparse = True
        vec = default_desc.create(H2O)
        self.assertTrue(type(vec) == scipy.sparse.coo_matrix)

        # Dense
        default_desc._sparse = False
        vec = default_desc.create(H2O)
        self.assertTrue(type(vec) == np.ndarray)

    def test_parallel_dense(self):
        """Tests creating dense output parallelly.
        """
        samples = [molecule("CO"), molecule("N2O")]
        desc = ACSF(
            rcut=6.0,
            species=[6, 7, 8],
            g2_params=[[1, 2], [4, 5]],
            g3_params=[1, 2, 3, 4],
            g4_params=[[1, 2, 3], [3, 1, 4], [4, 5, 6], [7, 8, 9]],
            g5_params=[[1, 2, 3], [3, 1, 4], [4, 5, 6], [7, 8, 9]],
        )
        n_features = desc.get_number_of_features()

        # Multiple systems, serial job
        output = desc.create(
            system=samples,
            positions=[[0], [0, 1]],
            n_jobs=1,
        )
        assumed = np.empty((3, n_features))
        assumed[0, :] = desc.create(samples[0], [0])
        assumed[1, :] = desc.create(samples[1], [0])
        assumed[2, :] = desc.create(samples[1], [1])
        self.assertTrue(np.allclose(output, assumed))

        # Test when position given as indices
        output = desc.create(
            system=samples,
            positions=[[0], [0, 1]],
            n_jobs=2,
        )
        assumed = np.empty((3, n_features))
        assumed[0, :] = desc.create(samples[0], [0])
        assumed[1, :] = desc.create(samples[1], [0])
        assumed[2, :] = desc.create(samples[1], [1])
        self.assertTrue(np.allclose(output, assumed))

        # Test with no positions specified
        output = desc.create(
            system=samples,
            positions=[None, None],
            n_jobs=2,
        )
        assumed = np.empty((2+3, n_features))
        assumed[0, :] = desc.create(samples[0], [0])
        assumed[1, :] = desc.create(samples[0], [1])
        assumed[2, :] = desc.create(samples[1], [0])
        assumed[3, :] = desc.create(samples[1], [1])
        assumed[4, :] = desc.create(samples[1], [2])
        self.assertTrue(np.allclose(output, assumed))

    def test_parallel_sparse(self):
        """Tests creating sparse output parallelly.
        """
        # Test indices
        samples = [molecule("CO"), molecule("N2O")]
        desc = ACSF(
            rcut=6.0,
            species=[6, 7, 8],
            g2_params=[[1, 2], [4, 5]],
            g3_params=[1, 2, 3, 4],
            g4_params=[[1, 2, 3], [3, 1, 4], [4, 5, 6], [7, 8, 9]],
            g5_params=[[1, 2, 3], [3, 1, 4], [4, 5, 6], [7, 8, 9]],
            sparse=True
        )
        n_features = desc.get_number_of_features()

        # Multiple systems, serial job
        output = desc.create(
            system=samples,
            positions=[[0], [0, 1]],
            n_jobs=1,
        ).toarray()
        assumed = np.empty((3, n_features))
        assumed[0, :] = desc.create(samples[0], [0]).toarray()
        assumed[1, :] = desc.create(samples[1], [0]).toarray()
        assumed[2, :] = desc.create(samples[1], [1]).toarray()
        self.assertTrue(np.allclose(output, assumed))

        # Test when position given as indices
        output = desc.create(
            system=samples,
            positions=[[0], [0, 1]],
            n_jobs=2,
        ).toarray()
        assumed = np.empty((3, n_features))
        assumed[0, :] = desc.create(samples[0], [0]).toarray()
        assumed[1, :] = desc.create(samples[1], [0]).toarray()
        assumed[2, :] = desc.create(samples[1], [1]).toarray()
        self.assertTrue(np.allclose(output, assumed))

        # Test with no positions specified
        output = desc.create(
            system=samples,
            positions=[None, None],
            n_jobs=2,
        ).toarray()

        assumed = np.empty((2+3, n_features))
        assumed[0, :] = desc.create(samples[0], [0]).toarray()
        assumed[1, :] = desc.create(samples[0], [1]).toarray()
        assumed[2, :] = desc.create(samples[1], [0]).toarray()
        assumed[3, :] = desc.create(samples[1], [1]).toarray()
        assumed[4, :] = desc.create(samples[1], [2]).toarray()
        self.assertTrue(np.allclose(output, assumed))

    def test_features(self):
        """Tests that the correct features are present in the descriptor.
        """
        rs = math.sqrt(2)
        kappa = math.sqrt(3)
        eta = math.sqrt(5)
        lmbd = 1
        zeta = math.sqrt(7)

        # Test against assumed values
        dist_oh = H2O.get_distance(0, 1)
        dist_hh = H2O.get_distance(0, 2)
        ang_hoh = H2O.get_angle(0, 1, 2) * np.pi / 180.0
        ang_hho = H2O.get_angle(1, 0, 2) * np.pi / 180.0
        ang_ohh = - H2O.get_angle(2, 0, 1) * np.pi / 180.0
        rc = 6.0

        # G1
        desc = ACSF(rcut=rc, species=[1, 8])
        acsfg1 = desc.create(H2O)
        g1_ho = cutoff(dist_oh, rc)
        g1_hh = cutoff(dist_hh, rc)
        g1_oh = 2 * cutoff(dist_oh, rc)
        self.assertAlmostEqual(acsfg1[0, 0], g1_hh, places=6)
        self.assertAlmostEqual(acsfg1[0, 1], g1_ho, places=6)
        self.assertAlmostEqual(acsfg1[1, 0], g1_oh, places=6)

        # G2
        desc = ACSF(rcut=6.0, species=[1, 8], g2_params=[[eta, rs]])
        acsfg2 = desc.create(H2O)
        g2_hh = np.exp(-eta * np.power((dist_hh - rs), 2)) * g1_hh
        g2_ho = np.exp(-eta * np.power((dist_oh - rs), 2)) * g1_ho
        g2_oh = np.exp(-eta * np.power((dist_oh - rs), 2)) * g1_oh
        self.assertAlmostEqual(acsfg2[0, 1], g2_hh, places=6)
        self.assertAlmostEqual(acsfg2[0, 3], g2_ho, places=6)
        self.assertAlmostEqual(acsfg2[1, 1], g2_oh, places=6)

        # G3
        desc = ACSF(rcut=6.0, species=[1, 8], g3_params=[kappa])
        acsfg3 = desc.create(H2O)
        g3_hh = np.cos(dist_hh * kappa) * g1_hh
        g3_ho = np.cos(dist_oh * kappa) * g1_ho
        g3_oh = np.cos(dist_oh * kappa) * g1_oh
        self.assertAlmostEqual(acsfg3[0, 1], g3_hh, places=6)
        self.assertAlmostEqual(acsfg3[0, 3], g3_ho, places=6)
        self.assertAlmostEqual(acsfg3[1, 1], g3_oh, places=6)

        # G4
        desc = ACSF(rcut=6.0, species=[1, 8], g4_params=[[eta, zeta, lmbd]])
        acsfg4 = desc.create(H2O)
        gauss = np.exp(-eta * (2 * dist_oh * dist_oh + dist_hh * dist_hh)) * g1_ho * g1_hh * g1_ho
        g4_h_ho = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_hho)), zeta) * gauss
        g4_h_oh = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_ohh)), zeta) * gauss
        g4_o_hh = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_hoh)), zeta) * gauss
        self.assertAlmostEqual(acsfg4[0, 3], g4_h_ho, places=6)
        self.assertAlmostEqual(acsfg4[2, 3], g4_h_oh, places=6)
        self.assertAlmostEqual(acsfg4[1, 2], g4_o_hh, places=6)

        # G5
        desc = ACSF(rcut=6.0, species=[1, 8], g5_params=[[eta, zeta, lmbd]])
        acsfg5 = desc.create(H2O)
        gauss = np.exp(-eta * (dist_oh * dist_oh + dist_hh * dist_hh)) * g1_ho * g1_hh
        g5_h_ho = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_hho)), zeta) * gauss
        g5_h_oh = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_ohh)), zeta) * gauss
        g5_o_hh = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_hoh)), zeta) * np.exp(-eta * (2 * dist_oh * dist_oh)) * g1_ho * g1_ho
        self.assertAlmostEqual(acsfg5[0, 3], g5_h_ho, places=6)
        self.assertAlmostEqual(acsfg5[2, 3], g5_h_oh, places=6)
        self.assertAlmostEqual(acsfg5[1, 2], g5_o_hh, places=6)

    def test_periodicity(self):
        """Test that periodic copies are correctly repeated and included in the
        output.
        """
        system = Atoms(
            symbols=["H"],
            positions=[[0, 0, 0]],
            cell=[2, 2, 2],
            pbc=False
        )
        rcut = 2.5

        # Non-periodic
        desc = ACSF(rcut=rcut, species=[1], periodic=False)
        feat = desc.create(system)
        self.assertTrue(feat.sum() == 0)

        # Periodic cubic: 6 neighbours at distance 2 Å
        desc = ACSF(rcut=rcut, species=[1], periodic=True)
        feat = desc.create(system)
        self.assertTrue(feat.sum() != 0)
        self.assertAlmostEqual(feat[0, 0], 6 * cutoff(2, rcut), places=6)

        # Periodic cubic: 6 neighbours at distance 2 Å
        # from ase.visualize import view
        rcut = 3
        system_nacl = bulk("NaCl", "rocksalt", a=4)
        eta, zeta, lambd = 0.01, 0.1, 1
        desc = ACSF(rcut=rcut, g4_params=[(eta, zeta, lambd)], species=["Na", "Cl"], periodic=True)
        feat = desc.create(system_nacl)

        # Cl-Cl: 12 triplets with 90 degree angle at 2 angstrom distance
        R_ij = 2
        R_ik = 2
        R_jk = np.sqrt(2)*2
        theta = np.pi/2
        g4_cl_cl = 2**(1-zeta)*12*(1+lambd*np.cos(theta))**zeta*np.e**(-eta*(R_ij**2+R_ik**2+R_jk**2))*cutoff(R_ij, rcut)*cutoff(R_ik, rcut)*cutoff(R_jk, rcut)
        self.assertTrue(np.allclose(feat[0, 4], g4_cl_cl, rtol=1e-6, atol=0))

        # Na-Cl: 24 triplets with 45 degree angle at sqrt(2)*2 angstrom distance
        R_ij = np.sqrt(2)*2
        R_ik = 2
        R_jk = 2
        theta = np.pi/4
        g4_na_cl = 2**(1-zeta)*24*(1+lambd*np.cos(theta))**zeta*np.e**(-eta*(R_ij**2+R_ik**2+R_jk**2))*cutoff(R_ij, rcut)*cutoff(R_ik, rcut)*cutoff(R_jk, rcut)
        self.assertTrue(np.allclose(feat[0, 3], g4_na_cl, rtol=1e-6, atol=0))

        # Periodic primitive FCC: 12 neighbours at distance sqrt(2)/2*5
        rcut = 4
        system_fcc = bulk("H", "fcc", a=5)
        desc = ACSF(rcut=rcut, species=[1], periodic=True)
        feat = desc.create(system_fcc)
        self.assertTrue(feat.sum() != 0)
        self.assertAlmostEqual(feat[0, 0], 12 * 0.5 * (np.cos(np.pi*np.sqrt(2)/2*5 / rcut) + 1), places=6)

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

    def test_basis(self):
        """Tests that the output vectors behave correctly as a basis.
        """
        sys1 = Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [1, 0, 0]], cell=[2, 2, 2], pbc=False)
        sys2 = Atoms(symbols=["H", "O"], positions=[[0, 0, 0], [1, 0, 0]], cell=[2, 2, 2], pbc=False)

        # Create vectors for each system
        vec1 = default_desc.create(sys1, positions=[0])[0, :]

        vec2 = default_desc.create(sys2, positions=[0])[0, :]

        vec3 = default_desc.create(sys1, positions=[1])[0, :]

        vec4 = default_desc.create(sys2, positions=[1])[0, :]

        # The dot-product should be zero when the environment does not have the
        # same elements
        dot = np.dot(vec1, vec2)
        self.assertTrue(abs(dot) < 1e-8)

        # The dot-product should be the same for two atoms with the same
        # environment, even if the central atom is different.
        dot1 = np.dot(vec3, vec3)
        dot2 = np.dot(vec3, vec4)
        self.assertTrue(abs(dot1-dot2) < 1e-8)

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


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ACSFTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
