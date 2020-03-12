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
import copy

import numpy as np

import scipy.sparse
from scipy.signal import find_peaks

from dscribe.descriptors import LMBTR

from ase import Atoms
from ase.build import molecule

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
        # Cannot use n_atoms normalization
        with self.assertRaises(ValueError):
            LMBTR(
                species=[1],
                k2=default_k2,
                periodic=False,
                normalization="n_atoms",
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

    def test_flatten(self):
        system = H2O
        n_elem = len(set(system.get_atomic_numbers())) + 1

        # K2 unflattened
        desc = copy.deepcopy(default_desc_k2)
        desc.flatten = False
        feat = desc.create(system)
        self.assertEqual(feat[0]["k2"].shape, (n_elem, nk2))

        # K2 flattened. The sparse matrix only supports 2D matrices, so the first
        # dimension is always present, even if it is of length 1.
        desc = copy.deepcopy(default_desc_k2)
        desc.flatten = True
        feat_flat = desc.create(system)
        self.assertEqual(feat_flat.shape, (3, n_elem*nk2))

        # Check that the elements in flattened and unflattened match
        sorted_species = sorted(desc._atomic_numbers+[0])
        for i_pos in range(len(system)):
            for i_species in range(len(sorted_species)):
                i_z = sorted_species[i_species]
                slc = desc.get_location((0, i_z))
                i_flat = feat_flat[i_pos, slc]
                i_unflat = feat[i_pos]["k2"][i_species]
                self.assertTrue(np.array_equal(i_flat, i_unflat))

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

    def test_k2_peaks_finite(self):
        """Tests the correct peak locations and intensities are found for the
        k=2 term in finite systems.
        """
        desc = LMBTR(
            species=["H", "O"],
            k2={
                "geometry": {"function": "distance"},
                "grid": {"min": -1, "max": 3, "sigma": 0.5, "n": 1000},
                "weighting": {"function": "unity"},
            },
            normalize_gaussians=False,
            periodic=False,
            flatten=True,
            sparse=False
        )
        features = desc.create(H2O, [0])[0, :]
        pos = H2O.get_positions()
        x = desc.get_k2_axis()

        # Check the X-H peaks
        xh_feat = features[desc.get_location(("X", "H"))]
        xh_peak_indices = find_peaks(xh_feat, prominence=0.5)[0]
        xh_peak_locs = x[xh_peak_indices]
        xh_peak_ints = xh_feat[xh_peak_indices]
        self.assertTrue(np.allclose(xh_peak_locs, [np.linalg.norm(pos[0] - pos[2])], rtol=0, atol=1e-2))
        self.assertTrue(np.allclose(xh_peak_ints, [1], rtol=0, atol=1e-2))

        # Check the X-O peaks
        xo_feat = features[desc.get_location(("X", "O"))]
        xo_peak_indices = find_peaks(xo_feat, prominence=0.5)[0]
        xo_peak_locs = x[xo_peak_indices]
        xo_peak_ints = xo_feat[xo_peak_indices]
        self.assertTrue(np.allclose(xo_peak_locs, np.linalg.norm(pos[0] - pos[1]), rtol=0, atol=1e-2))
        self.assertTrue(np.allclose(xo_peak_ints, [1], rtol=0, atol=1e-2))

        # Check that everything else is zero
        features[desc.get_location(("X", "H"))] = 0
        features[desc.get_location(("X", "O"))] = 0
        self.assertEqual(features.sum(), 0)

    def test_k2_peaks_periodic(self):
        """Tests the correct peak locations and intensities are found for the
        k=2 term in periodic systems.
        """
        atoms = Atoms(
            cell=[
                [10, 0, 0],
                [10, 10, 0],
                [10, 0, 10],
            ],
            symbols=["H", "C"],
            scaled_positions=[
                [0.1, 0.5, 0.5],
                [0.9, 0.5, 0.5],
            ]
        )

        desc = LMBTR(
            species=["H", "C"],
            k2={
                "geometry": {"function": "distance"},
                "grid": {"min": 0, "max": 10, "sigma": 0.5, "n": 1000},
                "weighting": {"function": "exp", "scale": 0.8, "cutoff": 1e-3},
            },
            normalize_gaussians=False,
            periodic=True,
            flatten=True,
            sparse=False
        )
        features = desc.create(atoms, [0])[0, :]
        x = desc.get_k2_axis()

        # Calculate assumed locations and intensities.
        assumed_locs = np.array([2, 8])
        assumed_ints = np.exp(-0.8*np.array([2, 8]))

        # Check the X-C peaks
        xc_feat = features[desc.get_location(("X", "C"))]
        xc_peak_indices = find_peaks(xc_feat, prominence=0.001)[0]
        xc_peak_locs = x[xc_peak_indices]
        xc_peak_ints = xc_feat[xc_peak_indices]
        self.assertTrue(np.allclose(xc_peak_locs, assumed_locs, rtol=0, atol=1e-2))
        self.assertTrue(np.allclose(xc_peak_ints, assumed_ints, rtol=0, atol=1e-2))

        # Check that everything else is zero
        features[desc.get_location(("X", "C"))] = 0
        self.assertEqual(features.sum(), 0)

    def test_k3_peaks_finite(self):
        """Tests the correct peak locations and intensities are found for the
        k=3 term in finite systems.
        """
        desc = LMBTR(
            species=["H", "O"],
            k3={
                "geometry": {"function": "angle"},
                "grid": {"min": -10, "max": 180, "sigma": 5, "n": 2000},
                "weighting": {"function": "unity"},
            },
            normalize_gaussians=False,
            periodic=False,
            flatten=True,
            sparse=False
        )
        features = desc.create(H2O, [0])[0, :]
        x = desc.get_k3_axis()

        # Check the X-H-O peaks
        xho_assumed_locs = np.array([38])
        xho_assumed_ints = np.array([1])
        xho_feat = features[desc.get_location(("X", "H", "O"))]
        xho_peak_indices = find_peaks(xho_feat, prominence=0.5)[0]
        xho_peak_locs = x[xho_peak_indices]
        xho_peak_ints = xho_feat[xho_peak_indices]
        self.assertTrue(np.allclose(xho_peak_locs, xho_assumed_locs, rtol=0, atol=5e-2))
        self.assertTrue(np.allclose(xho_peak_ints, xho_assumed_ints, rtol=0, atol=5e-2))

        # Check the X-O-H peaks
        xoh_assumed_locs = np.array([104])
        xoh_assumed_ints = np.array([1])
        xoh_feat = features[desc.get_location(("X", "O", "H"))]
        xoh_peak_indices = find_peaks(xoh_feat, prominence=0.5)[0]
        xoh_peak_locs = x[xoh_peak_indices]
        xoh_peak_ints = xoh_feat[xoh_peak_indices]
        self.assertTrue(np.allclose(xoh_peak_locs, xoh_assumed_locs, rtol=0, atol=5e-2))
        self.assertTrue(np.allclose(xoh_peak_ints, xoh_assumed_ints, rtol=0, atol=5e-2))

        # Check the H-X-O peaks
        hxo_assumed_locs = np.array([38])
        hxo_assumed_ints = np.array([1])
        hxo_feat = features[desc.get_location(("H", "X", "O"))]
        hxo_peak_indices = find_peaks(hxo_feat, prominence=0.5)[0]
        hxo_peak_locs = x[hxo_peak_indices]
        hxo_peak_ints = hxo_feat[hxo_peak_indices]
        self.assertTrue(np.allclose(hxo_peak_locs, hxo_assumed_locs, rtol=0, atol=5e-2))
        self.assertTrue(np.allclose(hxo_peak_ints, hxo_assumed_ints, rtol=0, atol=5e-2))

        # Check that everything else is zero
        features[desc.get_location(("X", "H", "O"))] = 0
        features[desc.get_location(("X", "O", "H"))] = 0
        features[desc.get_location(("H", "X", "O"))] = 0
        self.assertEqual(features.sum(), 0)

    def test_k3_peaks_periodic(self):
        """Tests the correct peak locations and intensities are found for the
        k=3 term in periodic systems.
        """
        scale = 0.85
        desc = LMBTR(
            species=["H"],
            k3={
                "geometry": {"function": "angle"},
                "grid": {"min": 0, "max": 180, "sigma": 5, "n": 2000},
                "weighting": {"function": "exp", "scale": scale, "cutoff": 1e-3},
            },
            normalize_gaussians=False,
            periodic=True,
            flatten=True,
            sparse=False
        )

        atoms = Atoms(
            cell=[
                [10, 0, 0],
                [0, 10, 0],
                [0, 0, 10],
            ],
            symbols=3*["H"],
            scaled_positions=[
                [0.05, 0.40, 0.5],
                [0.05, 0.60, 0.5],
                [0.95, 0.5, 0.5],
            ],
            pbc=True
        )
        features = desc.create(atoms, [0])[0, :]
        x = desc.get_k3_axis()

        # Calculate assumed locations and intensities.
        assumed_locs = np.array([45, 90])
        dist = 2+2*np.sqrt(2)  # The total distance around the three atoms
        weight = np.exp(-scale*dist)
        assumed_ints = np.array([weight, weight])

        # Check the X-H-H peaks
        xhh_feat = features[desc.get_location(("X", "H", "H"))]
        xhh_peak_indices = find_peaks(xhh_feat, prominence=0.01)[0]
        xhh_peak_locs = x[xhh_peak_indices]
        xhh_peak_ints = xhh_feat[xhh_peak_indices]
        self.assertTrue(np.allclose(xhh_peak_locs, assumed_locs, rtol=0, atol=1e-1))
        self.assertTrue(np.allclose(xhh_peak_ints, assumed_ints, rtol=0, atol=1e-1))

        # Calculate assumed locations and intensities.
        assumed_locs = np.array([45])
        dist = 2+2*np.sqrt(2)  # The total distance around the three atoms
        weight = np.exp(-scale*dist)
        assumed_ints = np.array([weight])

        # Check the H-X-H peaks
        hxh_feat = features[desc.get_location(("H", "X", "H"))]
        hxh_peak_indices = find_peaks(hxh_feat, prominence=0.01)[0]
        hxh_peak_locs = x[hxh_peak_indices]
        hxh_peak_ints = hxh_feat[hxh_peak_indices]
        self.assertTrue(np.allclose(hxh_peak_locs, assumed_locs, rtol=0, atol=1e-1))
        self.assertTrue(np.allclose(hxh_peak_ints, assumed_ints, rtol=0, atol=1e-1))

        # Check that everything else is zero
        features[desc.get_location(("X", "H", "H"))] = 0
        features[desc.get_location(("H", "X", "H"))] = 0
        self.assertEqual(features.sum(), 0)

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
