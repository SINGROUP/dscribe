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

import sparse

from scipy.integrate import tplquad
from scipy.linalg import sqrtm

from dscribe.descriptors import SOAP
from testbaseclass import TestBaseClass

from ase import Atoms
from ase.build import molecule

from testutils import (
    get_soap_gto_l_max_setup,
    get_soap_polynomial_l_max_setup,
    get_soap_default_setup,
    load_gto_coefficients,
    load_polynomial_coefficients,
)


H2O = Atoms(
    cell=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    positions=[
        [0, 0, 0],
        [0.95, 0, 0],
        [
            0.95 * (1 + math.cos(76 / 180 * math.pi)),
            0.95 * math.sin(76 / 180 * math.pi),
            0.0,
        ],
    ],
    symbols=["H", "O", "H"],
)

H = Atoms(
    cell=[[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]],
    positions=[
        [0, 0, 0],
    ],
    symbols=["H"],
)


class SoapTests(TestBaseClass, unittest.TestCase):
    # def test_exceptions(self):
    #     """Tests different invalid parameters that should raise an
    #     exception.
    #     """
    #     # Invalid sigma width
    #     with self.assertRaises(ValueError):
    #         SOAP(species=["H", "O"], r_cut=5, sigma=0, n_max=5, l_max=5)
    #     with self.assertRaises(ValueError):
    #         SOAP(species=["H", "O"], r_cut=5, sigma=-1, n_max=5, l_max=5)

    #     # Invalid r_cut
    #     with self.assertRaises(ValueError):
    #         SOAP(species=["H", "O"], r_cut=0.5, sigma=0.5, n_max=5, l_max=5)

    #     # Invalid l_max
    #     with self.assertRaises(ValueError):
    #         SOAP(species=["H", "O"], r_cut=0.5, sigma=0.5, n_max=5, l_max=20, rbf="gto")
    #     with self.assertRaises(ValueError):
    #         SOAP(
    #             species=["H", "O"],
    #             r_cut=0.5,
    #             sigma=0.5,
    #             n_max=5,
    #             l_max=21,
    #             rbf="polynomial",
    #         )

    #     # Invalid n_max
    #     with self.assertRaises(ValueError):
    #         SOAP(species=["H", "O"], r_cut=0.5, sigma=0.5, n_max=0, l_max=21)

    #     # Too high radial basis set density: poly
    #     with self.assertRaises(ValueError):
    #         a = SOAP(
    #             species=["H", "O"],
    #             r_cut=10,
    #             sigma=0.5,
    #             n_max=15,
    #             l_max=8,
    #             rbf="polynomial",
    #             periodic=False,
    #         )
    #         a.create(H2O)

    #     # Too high radial basis set density: gto
    #     with self.assertRaises(ValueError):
    #         a = SOAP(
    #             species=["H", "O"],
    #             r_cut=10,
    #             sigma=0.5,
    #             n_max=20,
    #             l_max=8,
    #             rbf="gto",
    #             periodic=False,
    #         )
    #         a.create(H2O)

    #     # Invalid weighting
    #     args = {
    #         "r_cut": 2,
    #         "sigma": 1,
    #         "n_max": 5,
    #         "l_max": 5,
    #         "species": ["H", "O"],
    #     }
    #     with self.assertRaises(ValueError):
    #         args["weighting"] = {"function": "poly", "c": -1, "r0": 1}
    #         SOAP(**args)
    #     with self.assertRaises(ValueError):
    #         args["weighting"] = {"function": "poly", "c": 1, "r0": 0}
    #         SOAP(**args)
    #     with self.assertRaises(ValueError):
    #         args["weighting"] = {"function": "poly", "c": 1, "r0": 1, "w0": -1}
    #         SOAP(**args)
    #     with self.assertRaises(ValueError):
    #         args["weighting"] = {"function": "pow", "c": -1, "d": 1, "r0": 1, "m": 1}
    #         SOAP(**args)
    #     with self.assertRaises(ValueError):
    #         args["weighting"] = {"function": "pow", "c": 1, "d": 1, "r0": 0, "m": 1}
    #         SOAP(**args)
    #     with self.assertRaises(ValueError):
    #         args["weighting"] = {"function": "pow", "c": 1, "d": 1, "r0": 1, "w0": -1}
    #         SOAP(**args)
    #     with self.assertRaises(ValueError):
    #         args["weighting"] = {"function": "exp", "c": -1, "d": 1, "r0": 1}
    #         SOAP(**args)
    #     with self.assertRaises(ValueError):
    #         args["weighting"] = {"function": "exp", "c": 1, "d": 1, "r0": 0}
    #         SOAP(**args)
    #     with self.assertRaises(ValueError):
    #         args["weighting"] = {"function": "exp", "c": 1, "d": 1, "r0": 1, "w0": -1}
    #         SOAP(**args)
    #     with self.assertRaises(ValueError):
    #         args["weighting"] = {"function": "invalid", "c": 1, "d": 1, "r0": 1}
    #         SOAP(**args)

    def test_properties(self):
        """Used to test that changing the setup through properties works as
        intended.
        """
        # Test changing species
        a = SOAP(
            species=[1, 8],
            r_cut=3,
            n_max=3,
            l_max=3,
            sparse=False,
        )
        nfeat1 = a.get_number_of_features()
        vec1 = a.create(H2O)
        a.species = ["C", "H", "O"]
        nfeat2 = a.get_number_of_features()
        vec2 = a.create(molecule("CH3OH"))
        self.assertTrue(nfeat1 != nfeat2)
        self.assertTrue(vec1.shape[1] != vec2.shape[1])

    # def test_number_of_features(self):
    #     """Tests that the reported number of features is correct."""
    #     l_max = 5
    #     n_max = 5
    #     n_elems = 2
    #     desc = SOAP(species=[1, 8], r_cut=3, n_max=n_max, l_max=l_max, periodic=True)

    #     # Test that the reported number of features matches the expected
    #     n_features = desc.get_number_of_features()
    #     expected = int((l_max + 1) * (n_max * n_elems) * (n_max * n_elems + 1) / 2)
    #     self.assertEqual(n_features, expected)

    #     # Test that the outputted number of features matches the reported
    #     n_features = desc.get_number_of_features()
    #     vec = desc.create(H2O)
    #     self.assertEqual(n_features, vec.shape[1])

    # def test_dtype(self):
    #     """Tests that the the specified data type is respected."""
    #     # Dense, float32
    #     soap = SOAP(species=[1, 8], r_cut=3, n_max=1, l_max=1, dtype="float32")
    #     desc1 = soap.create(H2O)
    #     der, desc2 = soap.derivatives(H2O)
    #     self.assertTrue(desc1.dtype == np.float32)
    #     self.assertTrue(desc2.dtype == np.float32)
    #     self.assertTrue(der.dtype == np.float32)

    #     # Sparse, float32
    #     soap = SOAP(
    #         species=[1, 8], r_cut=3, n_max=1, l_max=1, sparse=True, dtype="float32"
    #     )
    #     desc1 = soap.create(H2O)
    #     der, desc2 = soap.derivatives(H2O)
    #     self.assertTrue(desc1.dtype == np.float32)
    #     self.assertTrue(desc2.dtype == np.float32)
    #     self.assertTrue(der.dtype == np.float32)

    #     # Dense, float64
    #     soap = SOAP(species=[1, 8], r_cut=3, n_max=1, l_max=1, dtype="float64")
    #     desc1 = soap.create(H2O)
    #     der, desc2 = soap.derivatives(H2O)
    #     self.assertTrue(desc1.dtype == np.float64)
    #     self.assertTrue(desc2.dtype == np.float64)
    #     self.assertTrue(der.dtype == np.float64)

    #     # Sparse, float64
    #     soap = SOAP(
    #         species=[1, 8], r_cut=3, n_max=1, l_max=1, sparse=True, dtype="float64"
    #     )
    #     desc1 = soap.create(H2O)
    #     der, desc2 = soap.derivatives(H2O)
    #     self.assertTrue(desc1.dtype == np.float64)
    #     self.assertTrue(desc2.dtype == np.float64)
    #     self.assertTrue(der.dtype == np.float64)

    # def test_infer_r_cut(self):
    #     """Tests that the r_cut is correctly inferred from the weighting
    #     function.
    #     """
    #     # poly
    #     weighting = {
    #         "function": "poly",
    #         "c": 2,
    #         "m": 3,
    #         "r0": 4,
    #     }
    #     soap = SOAP(
    #         n_max=1,
    #         l_max=1,
    #         weighting=weighting,
    #         species=[1, 8],
    #         sparse=True,
    #     )
    #     r_cut = weighting["r0"]
    #     self.assertAlmostEqual(soap._r_cut, r_cut)

    #     # pow
    #     weighting = {
    #         "function": "pow",
    #         "threshold": 1e-3,
    #         "c": 1,
    #         "d": 1,
    #         "m": 1,
    #         "r0": 1,
    #     }
    #     soap = SOAP(
    #         n_max=1,
    #         l_max=1,
    #         weighting=weighting,
    #         species=[1, 8],
    #         sparse=True,
    #     )
    #     r_cut = weighting["c"] * (1 / weighting["threshold"] - 1)
    #     self.assertAlmostEqual(soap._r_cut, r_cut)

    #     # exp
    #     weighting = {
    #         "c": 2,
    #         "d": 1,
    #         "r0": 2,
    #         "function": "exp",
    #         "threshold": 1e-3,
    #     }
    #     soap = SOAP(species=[1, 8], n_max=1, l_max=1, sparse=True, weighting=weighting)
    #     r_cut = weighting["r0"] * np.log(
    #         weighting["c"] / weighting["threshold"] - weighting["d"]
    #     )
    #     self.assertAlmostEqual(soap._r_cut, r_cut)

    def test_crossover(self):
        """Tests that disabling/enabling crossover works as expected."""
        pos = [[0.1, 0.1, 0.1]]
        species = [1, 8]
        n_max = 5
        l_max = 5

        # GTO
        desc = SOAP(
            species=species,
            rbf="gto",
            crossover=True,
            r_cut=3,
            n_max=n_max,
            l_max=l_max,
            periodic=False,
        )
        hh_loc_full = desc.get_location(("H", "H"))
        oo_loc_full = desc.get_location(("O", "O"))
        full_output = desc.create(H2O, positions=pos)
        desc.crossover = False
        hh_loc = desc.get_location(("H", "H"))
        oo_loc = desc.get_location(("O", "O"))
        partial_output = desc.create(H2O, positions=pos)
        self.assertTrue(oo_loc_full != oo_loc)
        self.assertTrue(
            np.array_equal(full_output[:, hh_loc_full], partial_output[:, hh_loc])
        )
        self.assertTrue(
            np.array_equal(full_output[:, oo_loc_full], partial_output[:, oo_loc])
        )

        # Polynomial
        desc = SOAP(
            species=species,
            rbf="polynomial",
            crossover=True,
            r_cut=3,
            n_max=l_max,
            l_max=l_max,
            periodic=False,
        )
        hh_loc_full = desc.get_location(("H", "H"))
        oo_loc_full = desc.get_location(("O", "O"))
        full_output = desc.create(H2O, pos)
        desc.crossover = False
        hh_loc = desc.get_location(("H", "H"))
        oo_loc = desc.get_location(("O", "O"))
        partial_output = desc.create(H2O, pos)
        self.assertTrue(oo_loc_full != oo_loc)
        self.assertTrue(
            np.array_equal(full_output[:, hh_loc_full], partial_output[:, hh_loc])
        )
        self.assertTrue(
            np.array_equal(full_output[:, oo_loc_full], partial_output[:, oo_loc])
        )

    def test_get_location_w_crossover(self):
        """Tests that disabling/enabling crossover works as expected."""
        # With crossover
        species = ["H", "O", "C"]
        desc = SOAP(
            species=species,
            rbf="gto",
            crossover=True,
            r_cut=3,
            n_max=5,
            l_max=5,
            periodic=False,
        )

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
        """Tests that disabling/enabling crossover works as expected."""
        # With crossover
        species = ["H", "O", "C"]
        desc = SOAP(
            species=species,
            rbf="gto",
            crossover=False,
            r_cut=3,
            n_max=5,
            l_max=5,
            periodic=False,
        )

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
        """Tests multiple species are handled correctly."""
        l_max = 5
        n_max = 5
        species = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        desc = SOAP(
            species=species,
            r_cut=5,
            rbf="polynomial",
            n_max=n_max,
            l_max=l_max,
            periodic=False,
            sparse=False,
        )

        pos = np.expand_dims(np.linspace(0, 8, 8), 1)
        pos = np.hstack((pos, pos, pos))
        sys = Atoms(symbols=species[0:8], positions=pos, pbc=False)
        vec1 = desc.create(sys)

        sys2 = Atoms(symbols=species[8:], positions=pos, pbc=False)
        vec2 = desc.create(sys2)

        sys3 = Atoms(symbols=species[4:12], positions=pos, pbc=False)
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

    # def test_flatten(self):
    #     """Tests the flattening."""

    # def test_soap_structure(self):
    #     """Tests that when no positions are given, the SOAP for the full
    #     structure is calculated.
    #     """
    #     l_max = 5
    #     n_max = 5
    #     desc = SOAP(species=[1, 8], r_cut=5, n_max=n_max, l_max=l_max, periodic=True)

    #     vec = desc.create(H2O)
    #     self.assertTrue(vec.shape[0] == 3)

    # def test_sparse(self):
    #     """Tests the sparse matrix creation."""
    #     # Dense
    #     desc = SOAP(
    #         species=[1, 8], r_cut=5, n_max=5, l_max=5, periodic=True, sparse=False
    #     )
    #     vec = desc.create(H2O)
    #     self.assertTrue(type(vec) == np.ndarray)

    #     # Sparse
    #     desc = SOAP(
    #         species=[1, 8], r_cut=5, n_max=5, l_max=5, periodic=True, sparse=True
    #     )
    #     vec = desc.create(H2O)
    #     self.assertTrue(type(vec) == sparse.COO)

    def test_positions(self):
        """Tests that different positions are handled correctly."""
        desc = SOAP(
            species=[1, 6, 8],
            r_cut=10.0,
            n_max=2,
            l_max=0,
            periodic=False,
            crossover=True,
        )
        n_feat = desc.get_number_of_features()
        self.assertEqual(
            (1, n_feat), desc.create(H2O, positions=np.array([[0, 0, 0]])).shape
        )
        self.assertEqual((1, n_feat), desc.create(H2O, positions=[[0, 0, 0]]).shape)
        self.assertEqual((3, n_feat), desc.create(H2O, positions=[0, 1, 2]).shape)
        self.assertEqual(
            (3, n_feat), desc.create(H2O, positions=np.array([0, 1, 2])).shape
        )
        self.assertEqual((3, n_feat), desc.create(H2O).shape)

        desc = SOAP(
            species=[1, 6, 8],
            r_cut=10.0,
            n_max=2,
            l_max=0,
            periodic=True,
            crossover=True,
        )
        n_feat = desc.get_number_of_features()
        self.assertEqual(
            (1, n_feat), desc.create(H2O, positions=np.array([[0, 0, 0]])).shape
        )
        self.assertEqual((1, n_feat), desc.create(H2O, positions=[[0, 0, 0]]).shape)
        self.assertEqual((3, n_feat), desc.create(H2O, positions=[0, 1, 2]).shape)
        self.assertEqual(
            (3, n_feat), desc.create(H2O, positions=np.array([0, 1, 2])).shape
        )
        self.assertEqual((3, n_feat), desc.create(H2O).shape)

        desc = SOAP(
            species=[1, 6, 8],
            r_cut=10.0,
            n_max=2,
            l_max=0,
            periodic=True,
            crossover=False,
        )
        n_feat = desc.get_number_of_features()
        self.assertEqual(
            (1, n_feat), desc.create(H2O, positions=np.array([[0, 0, 0]])).shape
        )
        self.assertEqual((1, n_feat), desc.create(H2O, positions=[[0, 0, 0]]).shape)
        self.assertEqual((3, n_feat), desc.create(H2O, positions=[0, 1, 2]).shape)
        self.assertEqual(
            (3, n_feat), desc.create(H2O, positions=np.array([0, 1, 2])).shape
        )
        self.assertEqual((3, n_feat), desc.create(H2O).shape)

        desc = SOAP(
            species=[1, 6, 8],
            r_cut=10.0,
            n_max=2,
            l_max=0,
            periodic=False,
            crossover=False,
        )
        n_feat = desc.get_number_of_features()
        self.assertEqual(
            (1, n_feat), desc.create(H2O, positions=np.array([[0, 0, 0]])).shape
        )
        self.assertEqual((1, n_feat), desc.create(H2O, positions=[[0, 0, 0]]).shape)
        self.assertEqual((3, n_feat), desc.create(H2O, positions=[0, 1, 2]).shape)
        self.assertEqual(
            (3, n_feat), desc.create(H2O, positions=np.array([0, 1, 2])).shape
        )
        self.assertEqual((3, n_feat), desc.create(H2O).shape)

        with self.assertRaises(ValueError):
            desc.create(H2O, positions=["a"])

    # def test_parallel_dense(self):
    #     """Tests creating dense output parallelly."""
    #     samples = [molecule("CO"), molecule("NO")]
    #     desc = SOAP(
    #         species=[6, 7, 8],
    #         r_cut=5,
    #         n_max=3,
    #         l_max=3,
    #         sigma=1,
    #         periodic=False,
    #         crossover=True,
    #         average="off",
    #         sparse=False,
    #     )
    #     n_features = desc.get_number_of_features()

    #     # Determining number of jobs based on the amount of CPUs
    #     desc.create(system=samples, n_jobs=-1, only_physical_cores=False)
    #     desc.create(system=samples, n_jobs=-1, only_physical_cores=True)

    #     # Multiple systems, serial job, indices, fixed size
    #     output = desc.create(
    #         system=samples,
    #         positions=[[0, 1], [0, 1]],
    #         n_jobs=1,
    #     )
    #     assumed = np.empty((2, 2, n_features))
    #     assumed[0, 0] = desc.create(samples[0], [0])
    #     assumed[0, 1] = desc.create(samples[0], [1])
    #     assumed[1, 0] = desc.create(samples[1], [0])
    #     assumed[1, 1] = desc.create(samples[1], [1])
    #     self.assertTrue(np.allclose(output, assumed))

    #     # Multiple systems, parallel job, indices, fixed size
    #     output = desc.create(
    #         system=samples,
    #         positions=[[0, 1], [0, 1]],
    #         n_jobs=2,
    #     )
    #     assumed = np.empty((2, 2, n_features))
    #     assumed[0, 0] = desc.create(samples[0], [0])
    #     assumed[0, 1] = desc.create(samples[0], [1])
    #     assumed[1, 0] = desc.create(samples[1], [0])
    #     assumed[1, 1] = desc.create(samples[1], [1])
    #     self.assertTrue(np.allclose(output, assumed))

    #     # Multiple systems, parallel job, all atoms, fixed size
    #     output = desc.create(
    #         system=samples,
    #         positions=[None, None],
    #         n_jobs=2,
    #     )
    #     assumed = np.empty((2, 2, n_features))
    #     assumed[0, 0] = desc.create(samples[0], [0])
    #     assumed[0, 1] = desc.create(samples[0], [1])
    #     assumed[1, 0] = desc.create(samples[1], [0])
    #     assumed[1, 1] = desc.create(samples[1], [1])
    #     self.assertTrue(np.allclose(output, assumed))

    #     # Multiple systems, parallel job, cartesian positions, fixed size
    #     output = desc.create(
    #         system=samples,
    #         positions=[[[0, 0, 0], [1, 2, 0]], [[0, 0, 0], [1, 2, 0]]],
    #         n_jobs=2,
    #     )
    #     assumed = np.empty((2, 2, n_features))
    #     assumed[0, 0] = desc.create(samples[0], [[0, 0, 0]])
    #     assumed[0, 1] = desc.create(samples[0], [[1, 2, 0]])
    #     assumed[1, 0] = desc.create(samples[1], [[0, 0, 0]])
    #     assumed[1, 1] = desc.create(samples[1], [[1, 2, 0]])
    #     self.assertTrue(np.allclose(output, assumed))

    #     # Multiple systems, parallel job, indices, variable size
    #     output = desc.create(
    #         system=samples,
    #         positions=[[0], [0, 1]],
    #         n_jobs=2,
    #     )
    #     self.assertTrue(np.allclose(output[0][0], desc.create(samples[0], [0])))
    #     self.assertTrue(np.allclose(output[1][0], desc.create(samples[1], [0])))
    #     self.assertTrue(np.allclose(output[1][1], desc.create(samples[1], [1])))

    #     # Test averaged output
    #     desc.average = "outer"
    #     output = desc.create(
    #         system=samples,
    #         positions=[[0], [0, 1]],
    #         n_jobs=2,
    #     )
    #     assumed = np.empty((2, n_features))
    #     assumed[0] = desc.create(samples[0], [0])
    #     assumed[1] = (
    #         1 / 2 * (desc.create(samples[1], [0]) + desc.create(samples[1], [1]))
    #     )
    #     self.assertTrue(np.allclose(output, assumed))

    # def test_parallel_sparse(self):
    #     """Tests creating sparse output parallelly."""
    #     # Test indices
    #     samples = [molecule("CO"), molecule("NO")]
    #     desc = SOAP(
    #         species=[6, 7, 8],
    #         r_cut=5,
    #         n_max=3,
    #         l_max=3,
    #         sigma=1,
    #         periodic=False,
    #         crossover=True,
    #         average="off",
    #         sparse=True,
    #     )
    #     n_features = desc.get_number_of_features()

    #     # Multiple systems, serial job, fixed size
    #     output = desc.create(
    #         system=samples,
    #         positions=[[0, 1], [0, 1]],
    #         n_jobs=1,
    #     ).todense()
    #     assumed = np.empty((2, 2, n_features))
    #     assumed[0, 0] = desc.create(samples[0], [0]).todense()
    #     assumed[0, 1] = desc.create(samples[0], [1]).todense()
    #     assumed[1, 0] = desc.create(samples[1], [0]).todense()
    #     assumed[1, 1] = desc.create(samples[1], [1]).todense()
    #     self.assertTrue(np.allclose(output, assumed))

    #     # Multiple systems, parallel job, fixed size
    #     output = desc.create(
    #         system=samples,
    #         positions=[[0, 1], [0, 1]],
    #         n_jobs=2,
    #     ).todense()
    #     assumed = np.empty((2, 2, n_features))
    #     assumed[0, 0] = desc.create(samples[0], [0]).todense()
    #     assumed[0, 1] = desc.create(samples[0], [1]).todense()
    #     assumed[1, 0] = desc.create(samples[1], [0]).todense()
    #     assumed[1, 1] = desc.create(samples[1], [1]).todense()
    #     self.assertTrue(np.allclose(output, assumed))

    #     # Multiple systems, parallel job, all atoms, fixed size
    #     output = desc.create(
    #         system=samples,
    #         positions=[None, None],
    #         n_jobs=2,
    #     ).todense()
    #     assumed = np.empty((2, 2, n_features))
    #     assumed[0, 0] = desc.create(samples[0], [0]).todense()
    #     assumed[0, 1] = desc.create(samples[0], [1]).todense()
    #     assumed[1, 0] = desc.create(samples[1], [0]).todense()
    #     assumed[1, 1] = desc.create(samples[1], [1]).todense()
    #     self.assertTrue(np.allclose(output, assumed))

    #     # Multiple systems, parallel job, cartesian positions, fixed size
    #     output = desc.create(
    #         system=samples,
    #         positions=[[[0, 0, 0]], [[1, 2, 0]]],
    #         n_jobs=2,
    #     ).todense()
    #     assumed = np.empty((2, 1, n_features))
    #     assumed[0, 0] = desc.create(samples[0], [[0, 0, 0]]).todense()
    #     assumed[1, 0] = desc.create(samples[1], [[1, 2, 0]]).todense()
    #     self.assertTrue(np.allclose(output, assumed))

    #     # Multiple systems, parallel job, indices, variable size
    #     output = desc.create(
    #         system=samples,
    #         positions=[[0], [0, 1]],
    #         n_jobs=2,
    #     )
    #     self.assertTrue(
    #         np.allclose(output[0][0].todense(), desc.create(samples[0], [0]).todense())
    #     )
    #     self.assertTrue(
    #         np.allclose(output[1][0].todense(), desc.create(samples[1], [0]).todense())
    #     )
    #     self.assertTrue(
    #         np.allclose(output[1][1].todense(), desc.create(samples[1], [1]).todense())
    #     )

    #     # Test averaged output
    #     desc.average = "outer"
    #     output = desc.create(
    #         system=samples,
    #         positions=[[0], [0, 1]],
    #         n_jobs=2,
    #     ).todense()
    #     assumed = np.empty((2, n_features))
    #     assumed[0] = desc.create(samples[0], [0]).todense()
    #     assumed[1] = (
    #         1
    #         / 2
    #         * (
    #             desc.create(samples[1], [0]).todense()
    #             + desc.create(samples[1], [1]).todense()
    #         )
    #     )
    #     self.assertTrue(np.allclose(output, assumed))

    # def test_unit_cells(self):
    #     """Tests if arbitrary unit cells are accepted"""
    #     desc = SOAP(
    #         species=[1, 6, 8],
    #         r_cut=10.0,
    #         n_max=2,
    #         l_max=0,
    #         periodic=False,
    #         crossover=True,
    #     )

    #     molecule = H2O.copy()

    #     molecule.set_cell([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    #     nocell = desc.create(molecule, positions=[[0, 0, 0]])

    #     desc = SOAP(
    #         species=[1, 6, 8],
    #         r_cut=10.0,
    #         n_max=2,
    #         l_max=0,
    #         periodic=True,
    #         crossover=True,
    #     )

    #     # Invalid unit cell
    #     molecule.set_cell([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    #     with self.assertRaises(ValueError):
    #         desc.create(molecule, positions=[[0, 0, 0]])

    #     molecule.set_pbc(True)
    #     molecule.set_cell(
    #         [
    #             [20.0, 0.0, 0.0],
    #             [0.0, 30.0, 0.0],
    #             [0.0, 0.0, 40.0],
    #         ]
    #     )

    #     largecell = desc.create(molecule, positions=[[0, 0, 0]])

    #     molecule.set_cell([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])

    #     cubic_cell = desc.create(molecule, positions=[[0, 0, 0]])

    #     molecule.set_cell([[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]])

    #     triclinic_smallcell = desc.create(molecule, positions=[[0, 0, 0]])

    def test_is_periodic(self):
        """Tests whether periodic images are seen by the descriptor"""
        system = H2O.copy()

        desc = SOAP(
            species=[1, 6, 8],
            r_cut=10.0,
            n_max=2,
            l_max=0,
            periodic=False,
            crossover=True,
        )

        system.set_pbc(False)
        nocell = desc.create(system, positions=[[0, 0, 0]])

        system.set_pbc(True)
        system.set_cell([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        desc = SOAP(
            species=[1, 6, 8],
            r_cut=10.0,
            n_max=2,
            l_max=0,
            periodic=True,
            crossover=True,
        )

        cubic_cell = desc.create(system, positions=[[0, 0, 0]])

        self.assertTrue(np.sum(cubic_cell) > 0)

    def test_periodic_images(self):
        """Tests the periodic images seen by the descriptor"""
        desc = SOAP(
            species=[1, 6, 8],
            r_cut=10.0,
            n_max=2,
            l_max=0,
            periodic=False,
            crossover=True,
        )

        molecule = H2O.copy()

        # Non-periodic for comparison
        molecule.set_cell([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        nocell = desc.create(molecule, positions=[[0, 0, 0]])

        # Make periodic
        desc = SOAP(
            species=[1, 6, 8],
            r_cut=10.0,
            n_max=2,
            l_max=0,
            periodic=True,
            crossover=True,
        )
        molecule.set_pbc(True)

        # Cubic
        molecule.set_cell([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
        cubic_cell = desc.create(molecule, positions=[[0, 0, 0]])
        suce = molecule * (2, 1, 1)
        cubic_suce = desc.create(suce, positions=[[0, 0, 0]])

        # Triclinic
        molecule.set_cell([[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]])
        triclinic_cell = desc.create(molecule, positions=[[0, 0, 0]])
        suce = molecule * (2, 1, 1)
        triclinic_suce = desc.create(suce, positions=[[0, 0, 0]])

        self.assertTrue(np.sum(np.abs((nocell[:3] - cubic_suce[:3]))) > 0.1)
        self.assertAlmostEqual(np.sum(cubic_cell[:3] - cubic_suce[:3]), 0)
        self.assertAlmostEqual(np.sum(triclinic_cell[:3] - triclinic_suce[:3]), 0)

    # def test_symmetries(self):
    #     """Tests that the descriptor has the correct invariances."""

    #     def create_gto(system):
    #         desc = SOAP(
    #             species=system.get_atomic_numbers(),
    #             r_cut=8.0,
    #             l_max=5,
    #             n_max=5,
    #             rbf="gto",
    #             periodic=False,
    #             crossover=True,
    #         )
    #         return desc.create(system)

    #     # Rotational check
    #     self.assertTrue(self.is_rotationally_symmetric(create_gto))

    #     # Translational
    #     self.assertTrue(self.is_translationally_symmetric(create_gto))

    #     def create_poly(system):
    #         desc = SOAP(
    #             species=system.get_atomic_numbers(),
    #             r_cut=8.0,
    #             l_max=2,
    #             n_max=1,
    #             rbf="polynomial",
    #             periodic=False,
    #             crossover=True,
    #         )
    #         return desc.create(system)

    #     # Rotational check
    #     self.assertTrue(self.is_rotationally_symmetric(create_poly))

    #     # Translational
    #     self.assertTrue(self.is_translationally_symmetric(create_poly))

    # def test_basis(self):
    #     """Tests that the output vectors for both GTO and polynomial radial
    #     basis behave correctly.
    #     """
    #     sys1 = Atoms(
    #         symbols=["H", "H"],
    #         positions=[[1, 0, 0], [0, 1, 0]],
    #         cell=[2, 2, 2],
    #         pbc=True,
    #     )
    #     sys2 = Atoms(
    #         symbols=["O", "O"],
    #         positions=[[1, 0, 0], [0, 1, 0]],
    #         cell=[2, 2, 2],
    #         pbc=True,
    #     )
    #     sys3 = Atoms(
    #         symbols=["C", "C"],
    #         positions=[[1, 0, 0], [0, 1, 0]],
    #         cell=[2, 2, 2],
    #         pbc=True,
    #     )
    #     sys4 = Atoms(
    #         symbols=["H", "C"],
    #         positions=[[-1, 0, 0], [1, 0, 0]],
    #         cell=[2, 2, 2],
    #         pbc=True,
    #     )
    #     sys5 = Atoms(
    #         symbols=["H", "C"],
    #         positions=[[1, 0, 0], [0, 1, 0]],
    #         cell=[2, 2, 2],
    #         pbc=True,
    #     )
    #     sys6 = Atoms(
    #         symbols=["H", "O"],
    #         positions=[[1, 0, 0], [0, 1, 0]],
    #         cell=[2, 2, 2],
    #         pbc=True,
    #     )
    #     sys7 = Atoms(
    #         symbols=["C", "O"],
    #         positions=[[1, 0, 0], [0, 1, 0]],
    #         cell=[2, 2, 2],
    #         pbc=True,
    #     )

    #     for rbf in ["gto", "polynomial"]:
    #         desc = SOAP(
    #             species=[1, 6, 8],
    #             r_cut=5,
    #             n_max=1,
    #             l_max=1,
    #             rbf=rbf,
    #             periodic=False,
    #             crossover=True,
    #             sparse=False,
    #         )

    #         # Create vectors for each system
    #         vec1 = desc.create(sys1, positions=[[0, 0, 0]])[0, :]
    #         vec2 = desc.create(sys2, positions=[[0, 0, 0]])[0, :]
    #         vec3 = desc.create(sys3, positions=[[0, 0, 0]])[0, :]
    #         vec4 = desc.create(sys4, positions=[[0, 0, 0]])[0, :]
    #         vec5 = desc.create(sys5, positions=[[0, 0, 0]])[0, :]
    #         vec6 = desc.create(sys6, positions=[[0, 0, 0]])[0, :]
    #         vec7 = desc.create(sys7, positions=[[0, 0, 0]])[0, :]

    #         # The dot-product should be zero when there are no overlapping elements
    #         dot = np.dot(vec1, vec2)
    #         self.assertEqual(dot, 0)
    #         dot = np.dot(vec2, vec3)
    #         self.assertEqual(dot, 0)

    #         # The dot-product should be non-zero when there are overlapping elements
    #         dot = np.dot(vec4, vec5)
    #         self.assertNotEqual(dot, 0)

    #         # Check that self-terms are in correct location
    #         hh_loc = desc.get_location(("H", "H"))
    #         h_part1 = vec1[hh_loc]
    #         h_part2 = vec2[hh_loc]
    #         h_part4 = vec4[hh_loc]
    #         self.assertNotEqual(np.sum(h_part1), 0)
    #         self.assertEqual(np.sum(h_part2), 0)
    #         self.assertNotEqual(np.sum(h_part4), 0)

    #         # Check that cross terms are in correct location
    #         hc_loc = desc.get_location(("H", "C"))
    #         co_loc = desc.get_location(("C", "O"))
    #         hc_part1 = vec1[hc_loc]
    #         hc_part4 = vec4[hc_loc]
    #         co_part6 = vec6[co_loc]
    #         co_part7 = vec7[co_loc]
    #         self.assertEqual(np.sum(hc_part1), 0)
    #         self.assertNotEqual(np.sum(hc_part4), 0)
    #         self.assertEqual(np.sum(co_part6), 0)
    #         self.assertNotEqual(np.sum(co_part7), 0)

    # def test_rbf_orthonormality(self):
    #     """Tests that the gto radial basis functions are orthonormal."""
    #     sigma = 0.15
    #     r_cut = 2.0
    #     n_max = 2
    #     l_max = 20
    #     soap = SOAP(
    #         species=[1],
    #         l_max=l_max,
    #         n_max=n_max,
    #         sigma=sigma,
    #         r_cut=r_cut,
    #         crossover=True,
    #         sparse=False,
    #     )
    #     alphas = np.reshape(soap._alphas, [l_max + 1, n_max])
    #     betas = np.reshape(soap._betas, [l_max + 1, n_max, n_max])

    #     nr = 10000
    #     n_basis = 0
    #     functions = np.zeros((n_max, l_max + 1, nr))

    #     # Form the radial basis functions
    #     for n in range(n_max):
    #         for l in range(l_max + 1):
    #             gto = np.zeros((nr))
    #             rspace = np.linspace(0, r_cut + 5, nr)
    #             for k in range(n_max):
    #                 gto += (
    #                     betas[l, n, k]
    #                     * rspace**l
    #                     * np.exp(-alphas[l, k] * rspace**2)
    #                 )
    #             n_basis += 1
    #             functions[n, l, :] = gto

    #     # Calculate the overlap integrals
    #     S = np.zeros((n_max, n_max))
    #     for l in range(l_max + 1):
    #         for i in range(n_max):
    #             for j in range(n_max):
    #                 overlap = np.trapz(
    #                     rspace**2 * functions[i, l, :] * functions[j, l, :],
    #                     dx=(r_cut + 5) / nr,
    #                 )
    #                 S[i, j] = overlap

    #         # Check that the basis functions for each l are orthonormal
    #         diff = S - np.eye(n_max)
    #         self.assertTrue(np.allclose(diff, np.zeros((n_max, n_max)), atol=1e-3))

    # def test_average_outer(self):
    #     """Tests the outer averaging (averaging done after calculating power
    #     spectrum).
    #     """
    #     system, centers, args = get_soap_default_setup()

    #     # Create the average output
    #     for rbf in ["gto", "polynomial"]:
    #         desc = SOAP(**args, rbf=rbf, average="outer")
    #         average = desc.create(system, centers[0:2])

    #         # Create individual output for both atoms
    #         desc = SOAP(**args, rbf=rbf, average="off")
    #         first = desc.create(system, [centers[0]])[0, :]
    #         second = desc.create(system, [centers[1]])[0, :]

    #         # Check that the averaging is done correctly
    #         assumed_average = (first + second) / 2
    #         self.assertTrue(np.allclose(average, assumed_average))

    # def test_average_inner(self):
    #     """Tests the inner averaging (averaging done before calculating power
    #     spectrum).
    #     """
    #     for rbf in ["gto", "polynomial"]:
    #         system, centers, args = globals()["get_soap_{}_l_max_setup".format(rbf)]()
    #         # Calculate the analytical power spectrum
    #         soap = SOAP(**args, rbf=rbf, average="inner")
    #         analytical_inner = soap.create(system, positions=centers)

    #         # Calculate the numerical power spectrum
    #         coeffs = globals()["load_{}_coefficients".format(rbf)](args)
    #         numerical_inner = self.get_power_spectrum(
    #             coeffs, crossover=args["crossover"], average="inner"
    #         )

    #         # print("Numerical: {}".format(numerical_inner))
    #         # print("Analytical: {}".format(analytical_inner))
    #         self.assertTrue(
    #             np.allclose(numerical_inner, analytical_inner, atol=1e-15, rtol=0.01)
    #         )

    # def test_gto_integration(self):
    #     """Tests that the completely analytical partial power spectrum with the
    #     GTO basis corresponds to the easier-to-code but less performant
    #     numerical integration done with python.
    #     """
    #     # Calculate the analytical power spectrum
    #     system, centers, args = get_soap_gto_l_max_setup()
    #     soap = SOAP(**args, rbf="gto", dtype="float64")
    #     analytical_power_spectrum = soap.create(system, positions=centers)

    #     # Fetch the precalculated numerical power spectrum
    #     coeffs = load_gto_coefficients(args)
    #     numerical_power_spectrum = self.get_power_spectrum(
    #         coeffs, crossover=args["crossover"]
    #     )

    #     self.assertTrue(
    #         np.allclose(
    #             numerical_power_spectrum,
    #             analytical_power_spectrum,
    #             atol=1e-15,
    #             rtol=0.01,
    #         )
    #     )

    # def test_poly_integration(self):
    #     """Tests that the partial power spectrum with the polynomial basis done
    #     with C corresponds to the easier-to-code but less performant
    #     integration done with python.
    #     """
    #     # Calculate mostly analytical (radial part is integrated numerically)
    #     # power spectrum
    #     system, centers, args = get_soap_polynomial_l_max_setup()
    #     soap = SOAP(**args, rbf="polynomial", dtype="float64")
    #     analytical_power_spectrum = soap.create(system, positions=centers)

    #     # Calculate numerical power spectrum
    #     coeffs = load_polynomial_coefficients(args)
    #     numerical_power_spectrum = self.get_power_spectrum(
    #         coeffs, crossover=args["crossover"]
    #     )

    #     # print("Numerical: {}".format(numerical_power_spectrum))
    #     # print("Analytical: {}".format(analytical_power_spectrum))
    #     # print(analytical_power_spectrum.dtype)
    #     self.assertTrue(
    #         np.allclose(
    #             numerical_power_spectrum,
    #             analytical_power_spectrum,
    #             atol=1e-15,
    #             rtol=0.01,
    #         )
    #     )

    # def test_padding(self):
    #     """Tests that the padding used in constructing extended systems is
    #     sufficient.
    #     """
    #     # Fix random seed for tests
    #     np.random.seed(7)

    #     # Loop over different cell sizes
    #     for ncells in range(1, 6):
    #         ncells = int(ncells)

    #         # Loop over different radial cutoffs
    #         for r_cut in np.linspace(2, 10, 11):

    #             # Loop over different sigmas
    #             for sigma in np.linspace(0.5, 2, 4):

    #                 # Create descriptor generators
    #                 soap_generator = SOAP(
    #                     r_cut=r_cut,
    #                     n_max=4,
    #                     l_max=4,
    #                     sigma=sigma,
    #                     species=["Ni", "Ti"],
    #                     periodic=True,
    #                 )

    #                 # Define unit cell
    #                 a = 2.993
    #                 niti = Atoms(
    #                     "NiTi",
    #                     positions=[[0.0, 0.0, 0.0], [a / 2, a / 2, a / 2]],
    #                     cell=[a, a, a],
    #                     pbc=[1, 1, 1],
    #                 )

    #                 # Replicate system
    #                 niti = niti * ncells
    #                 a *= ncells

    #                 # Add some noise to positions
    #                 positions = niti.get_positions()
    #                 noise = np.random.normal(scale=0.5, size=positions.shape)
    #                 niti.set_positions(positions + noise)
    #                 niti.wrap()

    #                 # Evaluate descriptors for orthogonal unit cell
    #                 orthogonal_soaps = soap_generator.create(niti)

    #                 # Redefine the cubic unit cell as monoclinic
    #                 # with a 45-degree angle,
    #                 # this should not affect the descriptors
    #                 niti.set_cell([[a, 0, 0], [0, a, 0], [a, 0, a]])
    #                 niti.wrap()

    #                 # Evaluate descriptors for new, monoclinic unit cell
    #                 non_orthogonal_soaps = soap_generator.create(niti)

    #                 # Check that the relative or absolute error is small enough
    #                 self.assertTrue(
    #                     np.allclose(
    #                         orthogonal_soaps, non_orthogonal_soaps, atol=1e-8, rtol=1e-6
    #                     )
    #                 )

    # def test_weighting(self):
    #     """Tests that the weighting done with C corresponds to the
    #     easier-to-code but less performant python version.
    #     """
    #     system, centers, args = get_soap_default_setup()

    #     for rbf in ["gto", "polynomial"]:
    #         for weighting in [
    #             {"function": "poly", "r0": 2, "c": 3, "m": 4},
    #             {"function": "pow", "r0": 2, "c": 3, "d": 4, "m": 5},
    #             {"function": "exp", "r0": 2, "c": 3, "d": 4},
    #         ]:

    #             # Calculate the analytical power spectrum
    #             soap = SOAP(**args, rbf=rbf, weighting=weighting)
    #             analytical_power_spectrum = soap.create(system, positions=centers)

    #             # Calculate and save the numerical power spectrum to disk
    #             filename = "{rbf}_coefficients_{n_max}_{l_max}_{r_cut}_{sigma}_{func}.npy".format(
    #                 **args, rbf=rbf, func=weighting["function"]
    #             )
    #             # coeffs = getattr(self, "coefficients_{}".format(rbf))(
    #             # system_num,
    #             # soap_centers_num,
    #             # n_max_num,
    #             # l_max_num,
    #             # r_cut_num,
    #             # sigma_num,
    #             # weighting,
    #             # )
    #             # np.save(filename, coeffs)

    #             # Load coefficients from disk
    #             coeffs = np.load(filename)
    #             numerical_power_spectrum = self.get_power_spectrum(
    #                 coeffs, crossover=args["crossover"]
    #             )

    #             # print("Numerical: {}".format(numerical_power_spectrum))
    #             # print("Analytical: {}".format(analytical_power_spectrum))
    #             self.assertTrue(
    #                 np.allclose(
    #                     numerical_power_spectrum,
    #                     analytical_power_spectrum,
    #                     atol=1e-15,
    #                     rtol=0.01,
    #                 )
    #             )

    # def get_power_spectrum(self, coeffs, crossover=True, average="off"):
    #     """Given the expansion coefficients, returns the power spectrum."""
    #     numerical_power_spectrum = []
    #     shape = coeffs.shape
    #     n_centers = 1 if average != "off" else shape[0]
    #     n_species = shape[1]
    #     n_max = shape[2]
    #     l_max = shape[3] - 1
    #     for i in range(n_centers):
    #         i_spectrum = []
    #         for zi in range(n_species):
    #             for zj in range(zi, n_species if crossover else zi + 1):
    #                 if zi == zj:
    #                     for l in range(l_max + 1):
    #                         for ni in range(n_max):
    #                             for nj in range(ni, n_max):
    #                                 if average == "inner":
    #                                     value = np.dot(
    #                                         coeffs[:, zi, ni, l, :].mean(axis=0),
    #                                         coeffs[:, zj, nj, l, :].mean(axis=0),
    #                                     )
    #                                 else:
    #                                     value = np.dot(
    #                                         coeffs[i, zi, ni, l, :],
    #                                         coeffs[i, zj, nj, l, :],
    #                                     )
    #                                 prefactor = np.pi * np.sqrt(8 / (2 * l + 1))
    #                                 value *= prefactor
    #                                 i_spectrum.append(value)
    #                 else:
    #                     for l in range(l_max + 1):
    #                         for ni in range(n_max):
    #                             for nj in range(n_max):
    #                                 if average == "inner":
    #                                     value = np.dot(
    #                                         coeffs[:, zi, ni, l, :].mean(axis=0),
    #                                         coeffs[:, zj, nj, l, :].mean(axis=0),
    #                                     )
    #                                 else:
    #                                     value = np.dot(
    #                                         coeffs[i, zi, ni, l, :],
    #                                         coeffs[i, zj, nj, l, :],
    #                                     )
    #                                 prefactor = np.pi * np.sqrt(8 / (2 * l + 1))
    #                                 value *= prefactor
    #                                 i_spectrum.append(value)
    #         numerical_power_spectrum.append(i_spectrum)
    #     return np.array(numerical_power_spectrum)


if __name__ == "__main__":
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SoapTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
