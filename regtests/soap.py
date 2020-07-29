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
import itertools

import numpy as np

import scipy
import scipy.sparse
from scipy.integrate import tplquad
from scipy.linalg import sqrtm

from dscribe.descriptors import SOAP
from testbaseclass import TestBaseClass

from ase import Atoms
from ase.build import molecule


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
        # Invalid gaussian width
        with self.assertRaises(ValueError):
            SOAP(species=[-1, 2], rcut=5, sigma=0, nmax=5, lmax=5, periodic=True)
        with self.assertRaises(ValueError):
            SOAP(species=[-1, 2], rcut=5, sigma=-1, nmax=5, lmax=5, periodic=True)

        # Invalid rcut
        with self.assertRaises(ValueError):
            SOAP(species=[-1, 2], rcut=0.5, sigma=0, nmax=5, lmax=5, periodic=True)

        # Invalid lmax
        with self.assertRaises(ValueError):
            SOAP(species=[-1, 2], rcut=0.5, sigma=0, nmax=5, lmax=10, rbf="gto", periodic=True)

        # Invalid nmax
        with self.assertRaises(ValueError):
            SOAP(species=["H", "O"], rcut=4, sigma=1, nmax=0, lmax=8, rbf="gto", periodic=True)

        # Too high radial basis set density: poly
        with self.assertRaises(ValueError):
            a = SOAP(species=["H", "O"], rcut=10, sigma=0.5, nmax=12, lmax=8, rbf="polynomial", periodic=False)
            a.create(H2O)

        # Too high radial basis set density: gto
        with self.assertRaises(ValueError):
            a = SOAP(species=["H", "O"], rcut=10, sigma=0.5, nmax=20, lmax=8, rbf="gto", periodic=False)
            a.create(H2O)

    def test_properties(self):
        """Used to test that changing the setup through properties works as
        intended.
        """
        # Test changing species
        a = SOAP(
            species=[1, 8],
            rcut=3,
            nmax=3,
            lmax=3,
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
        lmax = 5
        nmax = 5
        n_elems = 2
        desc = SOAP(species=[1, 8], rcut=3, nmax=nmax, lmax=lmax, periodic=True)

        # Test that the reported number of features matches the expected
        n_features = desc.get_number_of_features()
        expected = int((lmax + 1) * (nmax*n_elems) * (nmax*n_elems + 1) / 2)
        self.assertEqual(n_features, expected)

        # Test that the outputted number of features matches the reported
        n_features = desc.get_number_of_features()
        vec = desc.create(H2O)
        self.assertEqual(n_features, vec.shape[1])

    def test_crossover(self):
        """Tests that disabling/enabling crossover works as expected.
        """
        pos = [[0.1, 0.1, 0.1]]
        species = [1, 8]
        nmax = 5
        lmax = 5

        # GTO
        desc = SOAP(species=species, rbf="gto", crossover=True, rcut=3, nmax=nmax, lmax=lmax, periodic=False)
        hh_loc_full = desc.get_location(("H", "H"))
        oo_loc_full = desc.get_location(("O", "O"))
        full_output = desc.create(H2O, positions=pos)
        desc.crossover = False
        hh_loc = desc.get_location(("H", "H"))
        oo_loc = desc.get_location(("O", "O"))
        partial_output = desc.create(H2O, positions=pos)
        self.assertTrue(oo_loc_full != oo_loc)
        self.assertTrue(np.array_equal(full_output[:, hh_loc_full], partial_output[:, hh_loc]))
        self.assertTrue(np.array_equal(full_output[:, oo_loc_full], partial_output[:, oo_loc]))

        # Polynomial
        desc = SOAP(species=species, rbf="polynomial", crossover=True, rcut=3, nmax=lmax, lmax=lmax, periodic=False)
        hh_loc_full = desc.get_location(("H", "H"))
        oo_loc_full = desc.get_location(("O", "O"))
        full_output = desc.create(H2O, pos)
        desc.crossover = False
        hh_loc = desc.get_location(("H", "H"))
        oo_loc = desc.get_location(("O", "O"))
        partial_output = desc.create(H2O, pos)
        self.assertTrue(oo_loc_full != oo_loc)
        self.assertTrue(np.array_equal(full_output[:, hh_loc_full], partial_output[:, hh_loc]))
        self.assertTrue(np.array_equal(full_output[:, oo_loc_full], partial_output[:, oo_loc]))

    def test_get_location_w_crossover(self):
        """Tests that disabling/enabling crossover works as expected.
        """
        # With crossover
        species = ["H", "O", "C"]
        desc = SOAP(species=species, rbf="gto", crossover=True, rcut=3, nmax=5, lmax=5, periodic=False)

        # Symbols
        loc_hh = desc.get_location(("H", "H"))
        loc_ho = desc.get_location(("H", "O"))
        loc_oh = desc.get_location(("O", "H"))
        loc_oo = desc.get_location(("O", "O"))
        loc_cc = desc.get_location(("C", "C"))
        loc_co = desc.get_location(("C", "O"))
        loc_ch = desc.get_location(("C", "H"))

        # Undefined elements
        with self.assertRaises(ValueError):
            desc.get_location((2, 1))
        with self.assertRaises(ValueError):
            desc.get_location(("He", "H"))

        # Check that slices in the output are correctly empty or filled
        co2 = molecule("CO2")
        h2o = molecule("H2O")
        co2_out = desc.create(co2)
        h2o_out = desc.create(h2o)

        # Check that slices with reversed atomic numbers are identical
        self.assertTrue(loc_ho == loc_oh)

        # H-H
        self.assertTrue(co2_out[:, loc_hh].sum() == 0)
        self.assertTrue(h2o_out[:, loc_hh].sum() != 0)

        # H-C
        self.assertTrue(co2_out[:, loc_ch].sum() == 0)
        self.assertTrue(h2o_out[:, loc_ch].sum() == 0)

        # H-O
        self.assertTrue(co2_out[:, loc_ho].sum() == 0)
        self.assertTrue(h2o_out[:, loc_ho].sum() != 0)

        # C-O
        self.assertTrue(co2_out[:, loc_co].sum() != 0)
        self.assertTrue(h2o_out[:, loc_co].sum() == 0)

        # C-C
        self.assertTrue(co2_out[:, loc_cc].sum() != 0)
        self.assertTrue(h2o_out[:, loc_cc].sum() == 0)

        # O-O
        self.assertTrue(co2_out[:, loc_oo].sum() != 0)
        self.assertTrue(h2o_out[:, loc_oo].sum() != 0)

    def test_get_location_wo_crossover(self):
        """Tests that disabling/enabling crossover works as expected.
        """
        # With crossover
        species = ["H", "O", "C"]
        desc = SOAP(species=species, rbf="gto", crossover=False, rcut=3, nmax=5, lmax=5, periodic=False)

        # Symbols
        loc_hh = desc.get_location(("H", "H"))
        loc_oo = desc.get_location(("O", "O"))
        loc_cc = desc.get_location(("C", "C"))

        # Undefined elements
        with self.assertRaises(ValueError):
            desc.get_location((2, 1))
        with self.assertRaises(ValueError):
            desc.get_location(("He", "H"))

        # Check that pairwise distances are not supported
        with self.assertRaises(ValueError):
            loc_oo = desc.get_location(("H", "O"))
            loc_oo = desc.get_location(("H", "C"))
            loc_oo = desc.get_location(("C", "H"))

        # Check that slices in the output are correctly empty or filled
        co2 = molecule("CO2")
        h2o = molecule("H2O")
        co2_out = desc.create(co2)
        h2o_out = desc.create(h2o)

        # H-H
        self.assertTrue(co2_out[:, loc_hh].sum() == 0)
        self.assertTrue(h2o_out[:, loc_hh].sum() != 0)

        # C-C
        self.assertTrue(co2_out[:, loc_cc].sum() != 0)
        self.assertTrue(h2o_out[:, loc_cc].sum() == 0)

        # O-O
        self.assertTrue(co2_out[:, loc_oo].sum() != 0)
        self.assertTrue(h2o_out[:, loc_oo].sum() != 0)

    def test_multiple_species(self):
        """Tests multiple species are handled correctly.
        """
        lmax = 5
        nmax = 5
        species = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        desc = SOAP(species=species, rcut=5, rbf="polynomial", nmax=nmax, lmax=lmax, periodic=False, sparse=False)

        pos = np.expand_dims(np.linspace(0, 8, 8), 1)
        pos = np.hstack((pos, pos, pos))
        sys = Atoms(
            symbols=species[0:8],
            positions=pos,
            pbc=False
        )
        vec1 = desc.create(sys)

        sys2 = Atoms(
            symbols=species[8:],
            positions=pos,
            pbc=False
        )
        vec2 = desc.create(sys2)

        sys3 = Atoms(
            symbols=species[4:12],
            positions=pos,
            pbc=False
        )
        vec3 = desc.create(sys3)

        dot1 = np.dot(vec1[6, :], vec2[6, :])
        dot2 = np.dot(vec1[3, :], vec3[3, :])
        dot3 = np.dot(vec2[3, :], vec3[3, :])

        # The dot product for systems without overlap in species should be zero
        self.assertTrue(abs(dot1) <= 1e-8)

        # The systems with overlap in the elements should have onerlap in the
        # dot product
        self.assertTrue(abs(dot2) > 1e-3)
        self.assertTrue(abs(dot3) > 1e-3)

    def test_flatten(self):
        """Tests the flattening.
        """

    def test_soap_structure(self):
        """Tests that when no positions are given, the SOAP for the full
        structure is calculated.
        """
        lmax = 5
        nmax = 5
        desc = SOAP(species=[1, 8], rcut=5, nmax=nmax, lmax=lmax, periodic=True)

        vec = desc.create(H2O)
        self.assertTrue(vec.shape[0] == 3)

    def test_sparse(self):
        """Tests the sparse matrix creation.
        """
        # Dense
        desc = SOAP(species=[1, 8], rcut=5, nmax=5, lmax=5, periodic=True, sparse=False)
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == np.ndarray)

        # Sparse
        desc = SOAP(species=[1, 8], rcut=5, nmax=5, lmax=5, periodic=True, sparse=True)
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == scipy.sparse.coo_matrix)

    def test_positions(self):
        """Tests that different positions are handled correctly.
        """
        desc = SOAP(species=[1, 6, 8], rcut=10.0, nmax=2, lmax=0, periodic=False, crossover=True)
        n_feat = desc.get_number_of_features()
        self.assertEqual((1, n_feat), desc.create(H2O, positions=np.array([[0, 0, 0]])).shape)
        self.assertEqual((1, n_feat), desc.create(H2O, positions=[[0, 0, 0]]).shape)
        self.assertEqual((3, n_feat), desc.create(H2O, positions=[0, 1, 2]).shape)
        self.assertEqual((3, n_feat), desc.create(H2O, positions=np.array([0, 1, 2])).shape)
        self.assertEqual((3, n_feat), desc.create(H2O).shape)

        desc = SOAP(species=[1, 6, 8], rcut=10.0, nmax=2, lmax=0, periodic=True, crossover=True,)
        n_feat = desc.get_number_of_features()
        self.assertEqual((1, n_feat), desc.create(H2O, positions=np.array([[0, 0, 0]])).shape)
        self.assertEqual((1, n_feat), desc.create(H2O, positions=[[0, 0, 0]]).shape)
        self.assertEqual((3, n_feat), desc.create(H2O, positions=[0, 1, 2]).shape)
        self.assertEqual((3, n_feat), desc.create(H2O, positions=np.array([0, 1, 2])).shape)
        self.assertEqual((3, n_feat), desc.create(H2O).shape)

        desc = SOAP(species=[1, 6, 8], rcut=10.0, nmax=2, lmax=0, periodic=True, crossover=False,)
        n_feat = desc.get_number_of_features()
        self.assertEqual((1, n_feat), desc.create(H2O, positions=np.array([[0, 0, 0]])).shape)
        self.assertEqual((1, n_feat), desc.create(H2O, positions=[[0, 0, 0]]).shape)
        self.assertEqual((3, n_feat), desc.create(H2O, positions=[0, 1, 2]).shape)
        self.assertEqual((3, n_feat), desc.create(H2O, positions=np.array([0, 1, 2])).shape)
        self.assertEqual((3, n_feat), desc.create(H2O).shape)

        desc = SOAP(species=[1, 6, 8], rcut=10.0, nmax=2, lmax=0, periodic=False, crossover=False,)
        n_feat = desc.get_number_of_features()
        self.assertEqual((1, n_feat), desc.create(H2O, positions=np.array([[0, 0, 0]])).shape)
        self.assertEqual((1, n_feat), desc.create(H2O, positions=[[0, 0, 0]]).shape)
        self.assertEqual((3, n_feat), desc.create(H2O, positions=[0, 1, 2]).shape)
        self.assertEqual((3, n_feat), desc.create(H2O, positions=np.array([0, 1, 2])).shape)
        self.assertEqual((3, n_feat), desc.create(H2O).shape)

        with self.assertRaises(ValueError):
            desc.create(H2O, positions=['a'])

    def test_parallel_dense(self):
        """Tests creating dense output parallelly.
        """
        samples = [molecule("CO"), molecule("N2O")]
        desc = SOAP(
            species=[6, 7, 8],
            rcut=5,
            nmax=3,
            lmax=3,
            sigma=1,
            periodic=False,
            crossover=True,
            average="off",
            sparse=False,
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

        # Test with cartesian positions
        output = desc.create(
            system=samples,
            positions=[[[0, 0, 0], [1, 2, 0]], [[1, 2, 0]]],
            n_jobs=2,
        )
        assumed = np.empty((2+1, n_features))
        assumed[0, :] = desc.create(samples[0], [[0, 0, 0]])
        assumed[1, :] = desc.create(samples[0], [[1, 2, 0]])
        assumed[2, :] = desc.create(samples[1], [[1, 2, 0]])
        self.assertTrue(np.allclose(output, assumed))

        # Test averaged output
        desc._average = "outer"
        output = desc.create(
            system=samples,
            positions=[[0], [0, 1]],
            n_jobs=2,
        )
        assumed = np.empty((2, n_features))
        assumed[0, :] = desc.create(samples[0], [0])
        assumed[1, :] = 1/2*(desc.create(samples[1], [0]) + desc.create(samples[1], [1]))
        self.assertTrue(np.allclose(output, assumed))

    def test_parallel_sparse(self):
        """Tests creating sparse output parallelly.
        """
        # Test indices
        samples = [molecule("CO"), molecule("N2O")]
        desc = SOAP(
            species=[6, 7, 8],
            rcut=5,
            nmax=3,
            lmax=3,
            sigma=1,
            periodic=False,
            crossover=True,
            average="off",
            sparse=True,
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

        # Test with cartesian positions
        output = desc.create(
            system=samples,
            positions=[[[0, 0, 0], [1, 2, 0]], [[1, 2, 0]]],
            n_jobs=2,
        ).toarray()
        assumed = np.empty((2+1, n_features))
        assumed[0, :] = desc.create(samples[0], [[0, 0, 0]]).toarray()
        assumed[1, :] = desc.create(samples[0], [[1, 2, 0]]).toarray()
        assumed[2, :] = desc.create(samples[1], [[1, 2, 0]]).toarray()
        self.assertTrue(np.allclose(output, assumed))

        # Test averaged output
        desc._average = "outer"
        output = desc.create(
            system=samples,
            positions=[[0], [0, 1]],
            n_jobs=2,
        ).toarray()
        assumed = np.empty((2, n_features))
        assumed[0, :] = desc.create(samples[0], [0]).toarray()
        assumed[1, :] = 1/2*(desc.create(samples[1], [0]).toarray() + desc.create(samples[1], [1]).toarray())
        self.assertTrue(np.allclose(output, assumed))

    def test_unit_cells(self):
        """Tests if arbitrary unit cells are accepted"""
        desc = SOAP(species=[1, 6, 8], rcut=10.0, nmax=2, lmax=0, periodic=False, crossover=True)

        molecule = H2O.copy()

        molecule.set_cell([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])

        nocell = desc.create(molecule, positions=[[0, 0, 0]])

        desc = SOAP(species=[1, 6, 8], rcut=10.0, nmax=2, lmax=0, periodic=True, crossover=True,)

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
            [0.0, 0.0, 40.0],
        ])

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
        system = H2O.copy()

        desc = SOAP(species=[1, 6, 8], rcut=10.0, nmax=2, lmax=0, periodic=False, crossover=True,)

        system.set_pbc(False)
        nocell = desc.create(system, positions=[[0, 0, 0]])

        system.set_pbc(True)
        system.set_cell([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        desc = SOAP(species=[1, 6, 8], rcut=10.0, nmax=2, lmax=0, periodic=True, crossover=True)

        cubic_cell = desc.create(system, positions=[[0, 0, 0]])

        self.assertTrue(np.sum(cubic_cell) > 0)

    def test_periodic_images(self):
        """Tests the periodic images seen by the descriptor
        """
        desc = SOAP(species=[1, 6, 8], rcut=10.0, nmax=2, lmax=0, periodic=False, crossover=True)

        molecule = H2O.copy()

        # Non-periodic for comparison
        molecule.set_cell([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        nocell = desc.create(molecule, positions=[[0, 0, 0]])

        # Make periodic
        desc = SOAP(species=[1, 6, 8], rcut=10.0, nmax=2, lmax=0, periodic=True, crossover=True)
        molecule.set_pbc(True)

        # Cubic
        molecule.set_cell([
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0]
        ])
        cubic_cell = desc.create(molecule, positions=[[0, 0, 0]])
        suce = molecule * (2, 1, 1)
        cubic_suce = desc.create(suce, positions=[[0, 0, 0]])

        # Triclinic
        molecule.set_cell([
            [0.0, 2.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 0.0]
        ])
        triclinic_cell = desc.create(molecule, positions=[[0, 0, 0]])
        suce = molecule * (2, 1, 1)
        triclinic_suce = desc.create(suce, positions=[[0, 0, 0]])

        self.assertTrue(np.sum(np.abs((nocell[:3] - cubic_suce[:3]))) > 0.1)
        self.assertAlmostEqual(np.sum(cubic_cell[:3] - cubic_suce[:3]), 0)
        self.assertAlmostEqual(np.sum(triclinic_cell[:3] - triclinic_suce[:3]), 0)

    def test_symmetries(self):
        """Tests that the descriptor has the correct invariances.
        """
        def create_gto(system):
            desc = SOAP(
                species=system.get_atomic_numbers(),
                rcut=8.0,
                lmax=5,
                nmax=5,
                rbf="gto",
                periodic=False,
                crossover=True
            )
            return desc.create(system)

        # Rotational check
        self.assertTrue(self.is_rotationally_symmetric(create_gto))

        # Translational
        self.assertTrue(self.is_translationally_symmetric(create_gto))

        def create_poly(system):
            desc = SOAP(
                species=system.get_atomic_numbers(),
                rcut=8.0,
                lmax=2,
                nmax=1,
                rbf="polynomial",
                periodic=False,
                crossover=True
            )
            return desc.create(system)

        # Rotational check
        self.assertTrue(self.is_rotationally_symmetric(create_poly))

        # Translational
        self.assertTrue(self.is_translationally_symmetric(create_poly))

    def test_average_outer(self):
        """Tests the outer averaging (averaging done after calculating power
        spectrum).
        """
        sys = Atoms(symbols=["H", "C"], positions=[[-1, 0, 0], [1, 0, 0]], cell=[2, 2, 2], pbc=True)

        # Create the average output
        for rbf in ["gto", "polynomial"]:
            desc = SOAP(
                species=[1, 6, 8],
                rcut=5,
                nmax=3,
                lmax=5,
                periodic=False,
                rbf=rbf,
                crossover=True,
                average="outer",
                sparse=False
            )
            average = desc.create(sys)[0, :]

            # Create individual output for both atoms
            desc = SOAP(
                species=[1, 6, 8],
                rcut=5,
                nmax=3,
                lmax=5,
                periodic=False,
                rbf=rbf,
                crossover=True,
                average="off",
                sparse=False
            )
            first = desc.create(sys, positions=[0])[0, :]
            second = desc.create(sys, positions=[1])[0, :]

            # Check that the averaging is done correctly
            assumed_average = (first+second)/2
            self.assertTrue(np.array_equal(average, assumed_average))

    def test_average_inner_gto(self):
        """Tests the inner averaging (averaging done before calculating power
        spectrum).
        """
        sigma = 0.55
        rcut = 2.0
        nmax = 2
        lmax = 2

        positions = np.array([[0.0, 0.0, 0.0], [-0.3, 0.5, 0.4]])
        symbols = np.array(["H", "C"])
        system = Atoms(positions=positions, symbols=symbols)
        soap_centers = [
            [0, 0, 0],
            [1/3, 1/3, 1/3],
            [2/3, 2/3, 2/3],
        ]

        species = system.get_atomic_numbers()
        elements = set(system.get_atomic_numbers())
        n_elems = len(elements)

        # Calculate the analytical power spectrum and the weights and decays of
        # the radial basis functions.
        soap = SOAP(
            species=species,
            lmax=lmax,
            nmax=nmax,
            sigma=sigma,
            rcut=rcut,
            rbf="gto",
            crossover=True,
            average="inner",
            sparse=False
        )
        analytical_inner = soap.create(system, positions=soap_centers)
        alphagrid = np.reshape(soap._alphas, [10, nmax])
        betagrid = np.reshape(soap._betas, [10, nmax, nmax])

        # Calculate numerical power spectrum
        coeffs = self.coefficients_gto(system, soap_centers, alphagrid, betagrid, nmax=nmax, lmax=lmax, rcut=rcut, sigma=sigma)
        numerical_inner = []
        for zi in range(n_elems):
            for zj in range(zi, n_elems):
                if zi == zj:
                    for l in range(lmax+1):
                        for ni in range(nmax):
                            for nj in range(ni, nmax):
                                # Average the m values over all atoms before doing the sum
                                value = np.dot(coeffs[:, zi, ni, l, :].mean(axis=0), coeffs[:, zj, nj, l, :].mean(axis=0))
                                prefactor = np.pi*np.sqrt(8/(2*l+1))
                                value *= prefactor
                                numerical_inner.append(value)
                else:
                    for l in range(lmax+1):
                        for ni in range(nmax):
                            for nj in range(nmax):
                                # Average the m values over all atoms before doing the sum
                                value = np.dot(coeffs[:, zi, ni, l, :].mean(axis=0), coeffs[:, zj, nj, l, :].mean(axis=0))
                                prefactor = np.pi*np.sqrt(8/(2*l+1))
                                value *= prefactor
                                numerical_inner.append(value)

        # print("Numerical: {}".format(numerical_inner))
        # print("Analytical: {}".format(analytical_inner))

        self.assertTrue(np.allclose(numerical_inner, analytical_inner, atol=1e-15, rtol=0.01))

    def test_average_inner_poly(self):
        """Tests the inner averaging (averaging done before calculating power
        spectrum).
        """
        sigma = 0.55
        rcut = 2.0
        nmax = 2
        lmax = 2

        positions = np.array([[0.0, 0.0, 0.0], [-0.3, 0.5, 0.4]])
        symbols = np.array(["H", "C"])
        system = Atoms(positions=positions, symbols=symbols)
        soap_centers = [
            [0, 0, 0],
            [1/3, 1/3, 1/3],
            [2/3, 2/3, 2/3],
        ]
        species = system.get_atomic_numbers()
        elements = set(system.get_atomic_numbers())
        n_elems = len(elements)

        # Calculate mostly analytical (radial part is integrated numerically)
        # power spectrum
        soap = SOAP(
            species=species,
            lmax=lmax,
            nmax=nmax,
            sigma=sigma,
            rcut=rcut,
            rbf="polynomial",
            crossover=True,
            average="inner",
            sparse=False
        )
        analytical_inner = soap.create(system, positions=soap_centers)

        # Calculate numerical power spectrum
        coeffs = self.coefficients_poly(system, soap_centers, nmax=nmax, lmax=lmax, rcut=rcut, sigma=sigma)
        numerical_inner = []
        for zi in range(n_elems):
            for zj in range(zi, n_elems):
                if zi == zj:
                    for l in range(lmax+1):
                        for ni in range(nmax):
                            for nj in range(ni, nmax):
                                # Average the m values over all atoms before doing the sum
                                value = np.dot(coeffs[:, zi, ni, l, :].mean(axis=0), coeffs[:, zj, nj, l, :].mean(axis=0))
                                prefactor = np.pi*np.sqrt(8/(2*l+1))
                                value *= prefactor
                                numerical_inner.append(value)
                else:
                    for l in range(lmax+1):
                        for ni in range(nmax):
                            for nj in range(nmax):
                                # Average the m values over all atoms before doing the sum
                                value = np.dot(coeffs[:, zi, ni, l, :].mean(axis=0), coeffs[:, zj, nj, l, :].mean(axis=0))
                                prefactor = np.pi*np.sqrt(8/(2*l+1))
                                value *= prefactor
                                numerical_inner.append(value)

        # print("Numerical: {}".format(numerical_inner))
        # print("Analytical poly: {}".format(analytical_inner))

        self.assertTrue(np.allclose(numerical_inner, analytical_inner, atol=1e-15, rtol=0.01))

    def test_basis(self):
        """Tests that the output vectors for both GTO and polynomial radial
        basis behave correctly.
        """
        sys1 = Atoms(symbols=["H", "H"], positions=[[1, 0, 0], [0, 1, 0]], cell=[2, 2, 2], pbc=True)
        sys2 = Atoms(symbols=["O", "O"], positions=[[1, 0, 0], [0, 1, 0]], cell=[2, 2, 2], pbc=True)
        sys3 = Atoms(symbols=["C", "C"], positions=[[1, 0, 0], [0, 1, 0]], cell=[2, 2, 2], pbc=True)
        sys4 = Atoms(symbols=["H", "C"], positions=[[-1, 0, 0], [1, 0, 0]], cell=[2, 2, 2], pbc=True)
        sys5 = Atoms(symbols=["H", "C"], positions=[[1, 0, 0], [0, 1, 0]], cell=[2, 2, 2], pbc=True)
        sys6 = Atoms(symbols=["H", "O"], positions=[[1, 0, 0], [0, 1, 0]], cell=[2, 2, 2], pbc=True)
        sys7 = Atoms(symbols=["C", "O"], positions=[[1, 0, 0], [0, 1, 0]], cell=[2, 2, 2], pbc=True)

        for rbf in ["gto", "polynomial"]:
            desc = SOAP(
                species=[1, 6, 8],
                rcut=5,
                nmax=1,
                lmax=1,
                rbf=rbf,
                periodic=False,
                crossover=True,
                sparse=False
            )

            # Create vectors for each system
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
            hh_loc = desc.get_location(("H", "H"))
            h_part1 = vec1[hh_loc]
            h_part2 = vec2[hh_loc]
            h_part4 = vec4[hh_loc]
            self.assertNotEqual(np.sum(h_part1), 0)
            self.assertEqual(np.sum(h_part2), 0)
            self.assertNotEqual(np.sum(h_part4), 0)

            # Check that cross terms are in correct location
            hc_loc = desc.get_location(("H", "C"))
            co_loc = desc.get_location(("C", "O"))
            hc_part1 = vec1[hc_loc]
            hc_part4 = vec4[hc_loc]
            co_part6 = vec6[co_loc]
            co_part7 = vec7[co_loc]
            self.assertEqual(np.sum(hc_part1), 0)
            self.assertNotEqual(np.sum(hc_part4), 0)
            self.assertEqual(np.sum(co_part6), 0)
            self.assertNotEqual(np.sum(co_part7), 0)

    def test_rbf_orthonormality(self):
        """Tests that the gto radial basis functions are orthonormal.
        """
        sigma = 0.15
        rcut = 2.0
        nmax = 2
        lmax = 3
        soap = SOAP(species=[1], lmax=lmax, nmax=nmax, sigma=sigma, rcut=rcut, crossover=True, sparse=False)
        alphas = np.reshape(soap._alphas, [10, nmax])
        betas = np.reshape(soap._betas, [10, nmax, nmax])

        nr = 10000
        n_basis = 0
        functions = np.zeros((nmax, lmax+1, nr))

        # Form the radial basis functions
        for n in range(nmax):
            for l in range(lmax+1):
                gto = np.zeros((nr))
                rspace = np.linspace(0, rcut+5, nr)
                for k in range(nmax):
                    gto += betas[l, n, k]*rspace**l*np.exp(-alphas[l, k]*rspace**2)
                n_basis += 1
                functions[n, l, :] = gto

        # Calculate the overlap integrals
        S = np.zeros((nmax, nmax))
        l = 0
        for l in range(lmax+1):
            for i in range(nmax):
                for j in range(nmax):
                    overlap = np.trapz(rspace**2*functions[i, l, :]*functions[j, l, :], dx=(rcut+5)/nr)
                    S[i, j] = overlap

            # Check that the basis functions for each l are orthonormal
            diff = S-np.eye(nmax)
            self.assertTrue(np.allclose(diff, np.zeros((nmax, nmax)), atol=1e-3))

    def test_gto_integration(self):
        """Tests that the completely analytical partial power spectrum with the
        GTO basis corresponds to the easier-to-code but less performant
        numerical integration done with python.
        """
        sigma = 0.55
        rcut = 2.0
        nmax = 2
        lmax = 2

        positions = np.array([[0.0, 0.0, 0.0], [-0.3, 0.5, 0.4]])
        symbols = np.array(["H", "C"])
        system = Atoms(positions=positions, symbols=symbols)

        soap_centers = [
            [0, 0, 0],
        ]
        species = system.get_atomic_numbers()
        elements = set(system.get_atomic_numbers())
        n_elems = len(elements)

        # Calculate the analytical power spectrum and the weights and decays of
        # the radial basis functions.
        soap = SOAP(species=species, lmax=lmax, nmax=nmax, sigma=sigma, rcut=rcut, crossover=True, sparse=False)
        analytical_power_spectrum = soap.create(system, positions=soap_centers)[0]
        alphagrid = np.reshape(soap._alphas, [10, nmax])
        betagrid = np.reshape(soap._betas, [10, nmax, nmax])

        # Calculate the numerical power spectrum
        coeffs = self.coefficients_gto(system, soap_centers, alphagrid, betagrid, nmax=nmax, lmax=lmax, rcut=rcut, sigma=sigma)
        numerical_power_spectrum = []
        for i in range(len(soap_centers)):
            i_spectrum = []
            for zi in range(n_elems):
                for zj in range(zi, n_elems):
                    if zi == zj:
                        for l in range(lmax+1):
                            for ni in range(nmax):
                                for nj in range(ni, nmax):
                                    value = np.dot(coeffs[i, zi, ni, l, :], coeffs[i, zj, nj, l, :])
                                    prefactor = np.pi*np.sqrt(8/(2*l+1))
                                    value *= prefactor
                                    i_spectrum.append(value)
                    else:
                        for l in range(lmax+1):
                            for ni in range(nmax):
                                for nj in range(nmax):
                                    value = np.dot(coeffs[i, zi, ni, l, :], coeffs[i, zj, nj, l, :])
                                    prefactor = np.pi*np.sqrt(8/(2*l+1))
                                    value *= prefactor
                                    i_spectrum.append(value)
            numerical_power_spectrum.append(i_spectrum)

        # print("Numerical: {}".format(numerical_power_spectrum))
        # print("Analytical: {}".format(analytical_power_spectrum))

        self.assertTrue(np.allclose(numerical_power_spectrum, analytical_power_spectrum, atol=1e-15, rtol=0.01))

    def test_poly_integration(self):
        """Tests that the partial power spectrum with the polynomial basis done
        with C corresponds to the easier-to-code but less performant
        integration done with python.
        """
        sigma = 0.55
        rcut = 2.0
        nmax = 2
        lmax = 2

        positions = np.array([[0.0, 0.0, 0.0], [-0.3, 0.5, 0.4]])
        symbols = np.array(["H", "C"])
        system = Atoms(positions=positions, symbols=symbols)

        soap_centers = [
            [0, 0, 0],
        ]
        species = system.get_atomic_numbers()
        elements = set(system.get_atomic_numbers())
        n_elems = len(elements)

        # Calculate mostly analytical (radial part is integrated numerically)
        # power spectrum
        soap = SOAP(species=species, lmax=lmax, nmax=nmax, sigma=sigma, rcut=rcut, rbf="polynomial", crossover=True, sparse=False)
        analytical_power_spectrum = soap.create(system, positions=soap_centers)

        # Calculate numerical power spectrum
        coeffs = self.coefficients_poly(system, soap_centers, nmax=nmax, lmax=lmax, rcut=rcut, sigma=sigma)
        numerical_power_spectrum = []
        for i in range(len(soap_centers)):
            i_spectrum = []
            for zi in range(n_elems):
                for zj in range(zi, n_elems):
                    if zi == zj:
                        for l in range(lmax+1):
                            for ni in range(nmax):
                                for nj in range(ni, nmax):
                                    value = np.dot(coeffs[i, zi, ni, l, :], coeffs[i, zj, nj, l, :])
                                    prefactor = np.pi*np.sqrt(8/(2*l+1))
                                    value *= prefactor
                                    i_spectrum.append(value)
                    else:
                        for l in range(lmax+1):
                            for ni in range(nmax):
                                for nj in range(nmax):
                                    value = np.dot(coeffs[i, zi, ni, l, :], coeffs[i, zj, nj, l, :])
                                    prefactor = np.pi*np.sqrt(8/(2*l+1))
                                    value *= prefactor
                                    i_spectrum.append(value)
            numerical_power_spectrum.append(i_spectrum)

        # print("Numerical: {}".format(numerical_power_spectrum))
        # print("Analytical: {}".format(analytical_power_spectrum))

        self.assertTrue(np.allclose(numerical_power_spectrum, analytical_power_spectrum, atol=1e-15, rtol=0.01))

    def test_padding(self):
        """Tests that the padding used in constructing extended systems is
        sufficient.
        """
        # Fix random seed for tests
        np.random.seed(7)

        # Loop over different cell sizes
        for ncells in range(1, 6):
            ncells = int(ncells)

            # Loop over different radial cutoffs
            for rcut in np.linspace(2, 10, 11):

                # Loop over different sigmas
                for sigma in np.linspace(0.5, 2, 4):

                    # Create descriptor generators
                    soap_generator = SOAP(
                        rcut=rcut, nmax=4, lmax=4, sigma=sigma, species=["Ni", "Ti"], periodic=True
                    )

                    # Define unit cell
                    a = 2.993
                    niti = Atoms(
                        "NiTi",
                        positions=[[0.0, 0.0, 0.0], [a / 2, a / 2, a / 2]],
                        cell=[a, a, a],
                        pbc=[1, 1, 1],
                    )

                    # Replicate system
                    niti = niti * ncells
                    a *= ncells

                    # Add some noise to positions
                    positions = niti.get_positions()
                    noise = np.random.normal(scale=0.5, size=positions.shape)
                    niti.set_positions(positions + noise)
                    niti.wrap()

                    # Evaluate descriptors for orthogonal unit cell
                    orthogonal_soaps = soap_generator.create(niti)

                    # Redefine the cubic unit cell as monoclinic
                    # with a 45-degree angle,
                    # this should not affect the descriptors
                    niti.set_cell([[a, 0, 0], [0, a, 0], [a, 0, a]])
                    niti.wrap()

                    # Evaluate descriptors for new, monoclinic unit cell
                    non_orthogonal_soaps = soap_generator.create(niti)

                    # Check that the relative or absolute error is small enough
                    self.assertTrue(np.allclose(orthogonal_soaps, non_orthogonal_soaps, atol=1e-8, rtol=1e-6))

    def coefficients_poly(self, system, soap_centers, nmax, lmax, rcut, sigma):
        """Used to numerically calculate the inner product coeffientes of SOAP
        with polynomial radial basis.
        """
        positions = system.get_positions()
        symbols = system.get_chemical_symbols()
        species = system.get_atomic_numbers()
        elements = set(system.get_atomic_numbers())
        n_elems = len(elements)

        # Integration limits for radius
        r1 = 0.
        r2 = rcut+5

        # Integration limits for theta
        t1 = 0
        t2 = np.pi

        # Integration limits for phi
        p1 = 0
        p2 = 2*np.pi

        # Calculate the overlap of the different polynomial functions in a
        # matrix S. These overlaps defined through the dot product over the
        # radial coordinate are analytically calculable: Integrate[(rc - r)^(a
        # + 2) (rc - r)^(b + 2) r^2, {r, 0, rc}]. Then the weights B that make
        # the basis orthonormal are given by B=S^{-1/2}
        S = np.zeros((nmax, nmax))
        for i in range(1, nmax+1):
            for j in range(1, nmax+1):
                S[i-1, j-1] = (2*(rcut)**(7+i+j))/((5+i+j)*(6+i+j)*(7+i+j))
        betas = sqrtm(np.linalg.inv(S))

        coeffs = np.zeros((len(soap_centers), n_elems, nmax, lmax+1, 2*lmax+1))
        for i, ipos in enumerate(soap_centers):
            for iZ, Z in enumerate(elements):
                indices = np.argwhere(species == Z)[0]
                elem_pos = positions[indices]
                # This centers the coordinate system at the soap center
                elem_pos -= ipos
                for n in range(nmax):
                    for l in range(lmax+1):
                        for im, m in enumerate(range(-l, l+1)):

                            # Calculate numerical coefficients
                            def soap_coeff(phi, theta, r):

                                # Regular spherical harmonic, notice the abs(m)
                                # needed for constructing the real form
                                ylm_comp = scipy.special.sph_harm(np.abs(m), l, phi, theta)  # NOTE: scipy swaps phi and theta

                                # Construct real (tesseral) spherical harmonics for
                                # easier integration without having to worry about
                                # the imaginary part. The real spherical harmonics
                                # span the same space, but are just computationally
                                # easier.
                                ylm_real = np.real(ylm_comp)
                                ylm_imag = np.imag(ylm_comp)
                                if m < 0:
                                    ylm = np.sqrt(2)*(-1)**m*ylm_imag
                                elif m == 0:
                                    ylm = ylm_comp
                                else:
                                    ylm = np.sqrt(2)*(-1)**m*ylm_real

                                # Polynomial basis
                                poly = 0
                                for k in range(1, nmax+1):
                                    poly += betas[n, k-1]*(rcut-np.clip(r, 0, rcut))**(k+2)

                                # Atomic density
                                rho = 0
                                for i_pos in elem_pos:
                                    ix = i_pos[0]
                                    iy = i_pos[1]
                                    iz = i_pos[2]
                                    ri_squared = ix**2+iy**2+iz**2
                                    rho += np.exp(-1/(2*sigma**2)*(r**2 + ri_squared - 2*r*(np.sin(theta)*np.cos(phi)*ix + np.sin(theta)*np.sin(phi)*iy + np.cos(theta)*iz)))

                                # Jacobian
                                jacobian = np.sin(theta)*r**2

                                return poly*ylm*rho*jacobian

                            cnlm = tplquad(
                                soap_coeff,
                                r1,
                                r2,
                                lambda r: t1,
                                lambda r: t2,
                                lambda r, theta: p1,
                                lambda r, theta: p2,
                                epsabs=0.0001,
                                epsrel=0.0001,
                            )
                            integral, error = cnlm
                            coeffs[i, iZ, n, l, im] = integral

        return coeffs

    def coefficients_gto(self, system, soap_centers, alphas, betas, nmax, lmax, rcut, sigma):
        """Used to numerically calculate the inner product coeffientes of SOAP
        with GTO radial basis.
        """
        positions = system.get_positions()
        symbols = system.get_chemical_symbols()
        species = system.get_atomic_numbers()
        elements = set(system.get_atomic_numbers())
        n_elems = len(elements)

        # Integration limits for radius
        r1 = 0.
        r2 = rcut+5

        # Integration limits for theta
        t1 = 0
        t2 = np.pi

        # Integration limits for phi
        p1 = 0
        p2 = 2*np.pi

        coeffs = np.zeros((len(soap_centers), n_elems, nmax, lmax+1, 2*lmax+1))
        for i, ipos in enumerate(soap_centers):
            for iZ, Z in enumerate(elements):
                indices = np.argwhere(species == Z)[0]
                elem_pos = positions[indices]
                # This centers the coordinate system at the soap center
                elem_pos -= ipos
                for n in range(nmax):
                    for l in range(lmax+1):
                        for im, m in enumerate(range(-l, l+1)):

                            # Calculate numerical coefficients
                            def soap_coeff(phi, theta, r):

                                # Regular spherical harmonic, notice the abs(m)
                                # needed for constructing the real form
                                ylm_comp = scipy.special.sph_harm(np.abs(m), l, phi, theta)  # NOTE: scipy swaps phi and theta

                                # Construct real (tesseral) spherical harmonics for
                                # easier integration without having to worry about
                                # the imaginary part. The real spherical harmonics
                                # span the same space, but are just computationally
                                # easier.
                                ylm_real = np.real(ylm_comp)
                                ylm_imag = np.imag(ylm_comp)
                                if m < 0:
                                    ylm = np.sqrt(2)*(-1)**m*ylm_imag
                                elif m == 0:
                                    ylm = ylm_comp
                                else:
                                    ylm = np.sqrt(2)*(-1)**m*ylm_real

                                # Spherical gaussian type orbital
                                gto = 0
                                for i in range(nmax):
                                    i_alpha = alphas[l, i]
                                    i_beta = betas[l, n, i]
                                    i_gto = i_beta*r**l*np.exp(-i_alpha*r**2)
                                    gto += i_gto

                                # Atomic density
                                rho = 0
                                for i_pos in elem_pos:
                                    ix = i_pos[0]
                                    iy = i_pos[1]
                                    iz = i_pos[2]
                                    ri_squared = ix**2+iy**2+iz**2
                                    rho += np.exp(-1/(2*sigma**2)*(r**2 + ri_squared - 2*r*(np.sin(theta)*np.cos(phi)*ix + np.sin(theta)*np.sin(phi)*iy + np.cos(theta)*iz)))

                                # Jacobian
                                jacobian = np.sin(theta)*r**2

                                return gto*ylm*rho*jacobian

                            cnlm = tplquad(
                                soap_coeff,
                                r1,
                                r2,
                                lambda r: t1,
                                lambda r: t2,
                                lambda r, theta: p1,
                                lambda r, theta: p2,
                                epsabs=0.001,
                                epsrel=0.001,
                            )
                            integral, error = cnlm
                            coeffs[i, iZ, n, l, im] = integral

        return coeffs

if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SoapTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
