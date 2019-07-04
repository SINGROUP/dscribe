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
from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import math
import unittest
import copy

import numpy as np

import scipy.sparse

from dscribe.descriptors import LMBTR

from ase import Atoms
from ase.build import molecule, bulk

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

H2O_2 = Atoms(
    cell=[[5.0, 0.0, 0], [0, 5, 0], [0, 0, 5.0]],
    positions=[[0.95, 0, 0], [0, 0, 0], [0.95*(1+math.cos(76/180*math.pi)), 0.95*math.sin(76/180*math.pi), 0.0]],
    symbols=["O", "H", "H"],
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

nk2 = 50
default_k2 = {
    "geometry": {"function": "inverse_distance"},
    "grid": {"min": 0, "max": 1/0.7, "sigma": 0.1, "n": nk2},
    "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-2},
}

default_k3 = {
    "geometry": {"function": "angle"},
    "grid": {"min": 0, "max": 180, "sigma": 2, "n": 50},
    "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-2},
}

default_desc_k2 = LMBTR(
    species=[1, 8],
    k2=default_k2,
    periodic=False,
    flatten=True,
    sparse=False,
)

default_desc_k3 = LMBTR(
    species=[1, 8],
    k3=default_k3,
    periodic=False,
    flatten=True,
    sparse=False,
)

default_desc_k2_k3 = LMBTR(
    species=[1, 8],
    k2=default_k2,
    k3=default_k3,
    periodic=False,
    flatten=True,
    sparse=False,
)


class LMBTRTests(TestBaseClass, unittest.TestCase):
    def test_constructor(self):
        """Tests different valid and invalid constructor values. As LMBTR is
        subclassed from MBTr, many of the tests are already performed in the
        regression tests for MBTR.
        """
        # Cannot make center periodic if the whole system is not
        with self.assertRaises(ValueError):
            LMBTR(
                species=[1],
                k2=default_k2,
                periodic=False,
                is_center_periodic=True,
            )

    def test_positions(self):
        """Tests that the position argument is handled correctly. The position
        can be a list of integers or a list of 3D positions.
        """
        lmbtr = copy.deepcopy(default_desc_k2_k3)

        # Position as a cartesian coordinate in list
        lmbtr.create(H2O, positions=[[0, 1, 0]])

        # Position as a cartesian coordinate in numpy array
        lmbtr.create(H2O, positions=np.array([[0, 1, 0]]))

        # Positions as lists of vectors
        positions = [[0, 1, 2], [0, 0, 0]]
        lmbtr.create(H2O, positions)

        # Position outside range
        with self.assertRaises(ValueError):
            lmbtr.create(H2O, positions=[3])

        # Invalid data type
        with self.assertRaises(ValueError):
            lmbtr.create(H2O, positions=['a'])

        # Positions as a list of integers pointing to atom indices
        positions = [0, 1, 2]
        lmbtr.create(H2O, positions)

    def test_normalization(self):
        """Tests that each normalization method works correctly.
        """
        n = 100
        desc = copy.deepcopy(default_desc_k2_k3)
        desc.species = ("H", "O")
        desc.normalization = "none"
        desc.flatten = False
        desc.sparse = False

        # Calculate the norms
        feat1 = desc.create(H2O, positions=[0])
        k2 = feat1[0]["k2"]
        k3 = feat1[0]["k3"]
        k2_norm = np.linalg.norm(k2.ravel())
        k3_norm = np.linalg.norm(k3.ravel())

        # Test normalization of non-flat dense output with l2_each
        desc.normalization = "l2_each"
        feat2 = desc.create(H2O, [0])
        k2_each = feat2[0]["k2"]
        k3_each = feat2[0]["k3"]
        self.assertTrue(np.array_equal(k2/k2_norm, k2_each))
        self.assertTrue(np.array_equal(k3/k3_norm, k3_each))

        # Check that the n_atoms normalization is not allowed
        with self.assertRaises(ValueError):
            desc.normalization = "n_atoms"

    def test_number_of_features(self):
        """LMBTR: Tests that the reported number of features is correct.
        """
        # k=2
        n = 50
        species = [1, 8]
        n_elem = len(species) + 1  # Including ghost atom
        lmbtr = copy.deepcopy(default_desc_k2)
        n_features = lmbtr.get_number_of_features()
        expected = n_elem*n
        real = lmbtr.create(H2O, positions=[0]).shape[1]
        self.assertEqual(n_features, expected)
        self.assertEqual(n_features, real)

        # k=3
        lmbtr = copy.deepcopy(default_desc_k3)
        n_features = lmbtr.get_number_of_features()
        expected = (n_elem*n_elem+(n_elem-1)*(n_elem)/2)*n
        real = lmbtr.create(H2O, positions=[0]).shape[1]
        self.assertEqual(n_features, expected)
        self.assertEqual(n_features, real)

    def test_locations_k2(self):
        """Tests that the function used to query combination locations for k=2
        in the output works.
        """

        CO2 = molecule("CO2")
        H2O = molecule("H2O")
        descriptors = [
            copy.deepcopy(default_desc_k2),
            copy.deepcopy(default_desc_k2_k3)
        ]

        for desc in descriptors:
            desc.periodic = False
            desc.species = ["H", "O", "C"]

            CO2 = molecule("CO2")
            H2O = molecule("H2O")

            co2_out = desc.create(CO2, positions=[1])[0, :]
            h2o_out = desc.create(H2O, positions=[1])[0, :]

            loc_xh = desc.get_location(("X", "H"))
            loc_xc = desc.get_location(("X", "C"))
            loc_xo = desc.get_location(("X", "O"))

            # X-H
            self.assertTrue(co2_out[loc_xh].sum() == 0)
            self.assertTrue(h2o_out[loc_xh].sum() != 0)

            # X-C
            self.assertTrue(co2_out[loc_xc].sum() != 0)
            self.assertTrue(h2o_out[loc_xc].sum() == 0)

            # X-O
            self.assertTrue(co2_out[loc_xo].sum() != 0)
            self.assertTrue(h2o_out[loc_xo].sum() != 0)

    def test_locations_k3(self):
        """Tests that the function used to query combination locations for k=2
        in the output works.
        """

        CO2 = molecule("CO2")
        H2O = molecule("H2O")
        descriptors = [
            copy.deepcopy(default_desc_k3),
            copy.deepcopy(default_desc_k2_k3)
        ]

        for desc in descriptors:
            desc.periodic = False
            desc.species = ["H", "O", "C"]
            co2_out = desc.create(CO2, positions=[1])[0, :]
            h2o_out = desc.create(H2O, positions=[1])[0, :]

            loc_xhh = desc.get_location(("X", "H", "H"))
            loc_xho = desc.get_location(("X", "H", "O"))
            loc_xhc = desc.get_location(("X", "H", "C"))
            loc_xoh = desc.get_location(("X", "O", "H"))
            loc_xoo = desc.get_location(("X", "O", "O"))
            loc_xoc = desc.get_location(("X", "O", "C"))
            loc_xch = desc.get_location(("X", "C", "H"))
            loc_xco = desc.get_location(("X", "C", "O"))
            loc_xcc = desc.get_location(("X", "C", "C"))
            loc_hxh = desc.get_location(("H", "X", "H"))
            loc_hxo = desc.get_location(("H", "X", "O"))
            loc_hxc = desc.get_location(("H", "X", "C"))
            loc_cxo = desc.get_location(("C", "X", "O"))
            loc_oxo = desc.get_location(("O", "X", "O"))
            loc_cxc = desc.get_location(("C", "X", "C"))

            # X-H-H
            self.assertTrue(co2_out[loc_xhh].sum() == 0)
            self.assertTrue(h2o_out[loc_xhh].sum() == 0)

            # X-H-O
            self.assertTrue(co2_out[loc_xho].sum() == 0)
            self.assertTrue(h2o_out[loc_xho].sum() != 0)

            # X-H-C
            self.assertTrue(co2_out[loc_xhc].sum() == 0)
            self.assertTrue(h2o_out[loc_xhc].sum() == 0)

            # X-O-H
            self.assertTrue(co2_out[loc_xoh].sum() == 0)
            self.assertTrue(h2o_out[loc_xoh].sum() != 0)

            # X-O-O
            self.assertTrue(co2_out[loc_xoo].sum() == 0)
            self.assertTrue(h2o_out[loc_xoo].sum() == 0)

            # X-O-C
            self.assertTrue(co2_out[loc_xoc].sum() != 0)
            self.assertTrue(h2o_out[loc_xoc].sum() == 0)

            # X-C-H
            self.assertTrue(co2_out[loc_xch].sum() == 0)
            self.assertTrue(h2o_out[loc_xch].sum() == 0)

            # X-C-O
            self.assertTrue(co2_out[loc_xco].sum() != 0)
            self.assertTrue(h2o_out[loc_xco].sum() == 0)

            # X-C-C
            self.assertTrue(co2_out[loc_xcc].sum() == 0)
            self.assertTrue(h2o_out[loc_xcc].sum() == 0)

            # H-X-H
            self.assertTrue(co2_out[loc_hxh].sum() == 0)
            self.assertTrue(h2o_out[loc_hxh].sum() == 0)

            # H-X-O
            self.assertTrue(co2_out[loc_hxo].sum() == 0)
            self.assertTrue(h2o_out[loc_hxo].sum() != 0)

            # H-X-C
            self.assertTrue(co2_out[loc_hxc].sum() == 0)
            self.assertTrue(h2o_out[loc_hxc].sum() == 0)

            # C-X-O
            self.assertTrue(co2_out[loc_cxo].sum() != 0)
            self.assertTrue(h2o_out[loc_cxo].sum() == 0)

            # O-X-O
            self.assertTrue(co2_out[loc_oxo].sum() == 0)
            self.assertTrue(h2o_out[loc_oxo].sum() == 0)

            # C-X-C
            self.assertTrue(co2_out[loc_cxc].sum() == 0)
            self.assertTrue(h2o_out[loc_cxc].sum() == 0)

    def test_center_periodicity(self):
        """Tests that the flag that controls whether the central atoms is
        repeated is working corrrectly.
        """
        system = bulk("Si", "diamond", a=5)

        # k=2
        lmbtr = copy.deepcopy(default_desc_k2)
        lmbtr.species = ["Si"]
        lmbtr.is_center_periodic = True
        lmbtr.periodic = True
        out = lmbtr.create(system, positions=[0])
        xx = lmbtr.get_location(("X", "X"))
        out_xx = out[0, xx]

        # Test that the output contains some as the central atom is
        # repeated
        self.assertTrue(out_xx.sum() > 0)

        lmbtr = copy.deepcopy(default_desc_k2)
        lmbtr.species = ["Si"]
        lmbtr.is_center_periodic = False
        lmbtr.periodic = True
        out = lmbtr.create(system, positions=[0])
        xx = lmbtr.get_location(("X", "X"))
        out_xx = out[0, xx]

        # Test that the output contains no features as the central atom is not
        # repeated
        self.assertTrue(out_xx.sum() == 0)

    def test_flatten(self):
        system = H2O
        n_elem = len(set(system.get_atomic_numbers())) + 1

        # K2 unflattened
        desc = copy.deepcopy(default_desc_k2)
        desc.flatten = False
        feat = desc.create(system, positions=[0])[0]["k2"]
        self.assertEqual(feat.shape, (n_elem, nk2))

        # K2 flattened. The sparse matrix only supports 2D matrices, so the first
        # dimension is always present, even if it is of length 1.
        desc = copy.deepcopy(default_desc_k2)
        desc.flatten = True
        feat = desc.create(system, positions=[0])
        self.assertEqual(feat.shape, (1, n_elem*nk2))

    def test_sparse(self):
        """Tests the sparse matrix creation.
        """
        # Dense
        desc = copy.deepcopy(default_desc_k2_k3)
        desc.sparse = False
        vec = desc.create(H2O, positions=[0])
        self.assertTrue(type(vec) == np.ndarray)

        # Sparse
        desc = copy.deepcopy(default_desc_k2_k3)
        desc.sparse = True
        vec = desc.create(H2O, positions=[0])
        self.assertTrue(type(vec) == scipy.sparse.coo_matrix)

    def test_parallel_dense(self):
        """Tests creating dense output parallelly.
        """
        samples = [molecule("CO"), molecule("N2O")]
        desc = copy.deepcopy(default_desc_k2)
        desc.species = ["C", "O", "N"]
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

        # Test with cartesian positions.
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

    def test_parallel_sparse(self):
        """Tests creating sparse output parallelly.
        """
        # Test indices
        samples = [molecule("CO"), molecule("N2O")]
        desc = copy.deepcopy(default_desc_k2)
        desc.species = ["C", "O", "N"]
        desc.sparse = True
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

        # Test with cartesian positions. In this case virtual positions have to
        # be enabled
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

    def test_periodic(self):
        """Test that the periodic-flag is working.
        """
        test_sys = Atoms(
            cell=[[2.0, 0.0, 0.0], [0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            positions=[[0, 0, 0], [0, 0, 1]],
            symbols=["H", "H"],
        )
        test_sys_shifted = Atoms(
            cell=[[2.0, 0.0, 0.0], [0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            positions=[[1, 1, 0], [1, 1, 1]],
            symbols=["H", "H"],
        )

        desc = copy.deepcopy(default_desc_k2_k3)
        desc.species = ["H"]
        desc.k3["weighting"]["scale"] = 0.3
        desc.periodic = True

        out = desc.create(test_sys, positions=[0])[0, :]
        out_shifted = desc.create(test_sys_shifted, positions=[0])[0, :]

        # Check that the output from two periodic systems, that just shifted
        # with respect to each other, are identical.
        self.assertTrue(np.linalg.norm(out - out_shifted) < 1e-6)

        # Check that the output changes as the periodicity flag is disabled
        desc.periodic = False
        out_finite = desc.create(test_sys, positions=[0])[0, :]
        self.assertTrue(np.linalg.norm(out - out_finite) > 1)

    def test_k2_weights_and_geoms_finite(self):
        """Tests that the values of the weight and geometry functions are
        correct for the k=2 term.
        """
        desc = copy.deepcopy(default_desc_k2)
        desc.k2["weighting"] = {"function": "unity"}
        desc.create(H2O, positions=[1])
        geoms = desc._k2_geoms
        weights = desc._k2_weights

        # Test against the assumed geom values
        pos = H2O.get_positions()
        assumed_geoms = {
            (0, 1): 2*[1/np.linalg.norm(pos[0] - pos[1])],
        }
        self.dict_comparison(assumed_geoms, geoms)

        # Test against the assumed weights
        assumed_weights = {
            (0, 1): 2*[1],
        }
        self.dict_comparison(assumed_weights, weights)

        # Test against system with different indexing
        desc.create(H2O_2, positions=[0])
        geoms2 = desc._k2_geoms
        self.dict_comparison(geoms, geoms2)

    def test_k3_weights_and_geoms_finite(self):
        """Tests that all the correct angles are present in finite systems.
        There should be n*(n-1)*(n-2)/2 unique angles where the division by two
        gets rid of duplicate angles.
        """
        system = Atoms(
            scaled_positions=[
                [0, 0, 0],
                [0.5, 0, 0],
                [0, 0.5, 0],
                [0.5, 0.5, 0],
            ],
            symbols=["H", "H", "H", "O"],
            cell=[10, 10, 10],
            pbc=True,
        )

        # lmbtr = LMBTR(species=[1, 8], k=[3], grid=default_grid, virtual_positions=False, periodic=False)
        desc = copy.deepcopy(default_desc_k3)
        desc.k3["weighting"] = {"function": "unity"}
        desc.create(system, positions=[0])
        geoms = desc._k3_geoms
        weights = desc._k3_weights

        # Test against the assumed geom values.
        assumed_geoms = {
            (0, 1, 1): 2*[45],
            (0, 2, 1): 2*[45],
            (0, 1, 2): 2*[90],
            (1, 0, 2): 2*[45],
            (1, 0, 1): 1*[90],
        }
        self.dict_comparison(geoms, assumed_geoms)

        # Test against the assumed weight values.
        assumed_weights = {
            (0, 1, 1): 2*[1],
            (0, 2, 1): 2*[1],
            (0, 1, 2): 2*[1],
            (1, 0, 2): 2*[1],
            (1, 0, 1): 1*[1],
        }
        self.dict_comparison(weights, assumed_weights)

    def test_symmetries(self):
        """Tests translational and rotational symmetries for a finite system.
        """
        desc = copy.deepcopy(default_desc_k2_k3)

        def create_1(system):
            """This function uses atom indices so rotation and translation
            should not affect it.
            """
            return desc.create(system, positions=[0])

        # Rotational check
        self.assertTrue(self.is_rotationally_symmetric(create_1))

        # Translational
        self.assertTrue(self.is_translationally_symmetric(create_1))

if __name__ == "__main__":
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(LMBTRTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
