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

import matplotlib.pyplot as mpl


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
        # Cannot make center periodic if the whole sytem is not
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

    def test_locations(self):
        """Tests that the function used to query combination locations in the
        output works.
        """
        desc = copy.deepcopy(default_desc_k2_k3)
        desc.species = ["H", "O", "C", "N"]

        CO2 = molecule("CO2")
        N2O = molecule("N2O")
        H2O = molecule("H2O")

        co2_out = desc.create(CO2, positions=[1])[0, :]
        n2o_out = desc.create(N2O, positions=[0])[0, :]
        h2o_out = desc.create(H2O, positions=[1])[0, :]

        loc_o = desc.get_location(("X", "O"))
        loc_h = desc.get_location(("X", "H"))
        loc_n = desc.get_location(("X", "N"))
        loc_c = desc.get_location(("X", "C"))

        loc_oc = desc.get_location(("X", "O", "C"))
        loc_on = desc.get_location(("X", "O", "N"))
        loc_oh = desc.get_location(("X", "O", "H"))

        # Test that only CO has X-C pairs
        co2_c = co2_out[loc_c]
        n2o_c = n2o_out[loc_c]
        h2o_c = h2o_out[loc_c]
        self.assertTrue(n2o_c.sum() == 0)
        self.assertTrue(h2o_c.sum() == 0)
        self.assertTrue(co2_c.sum() != 0)

        # Test that only N2O has X-N pairs
        co2_n = co2_out[loc_n]
        n2o_n = n2o_out[loc_n]
        h2o_n = h2o_out[loc_n]
        self.assertTrue(n2o_n.sum() != 0)
        self.assertTrue(h2o_n.sum() == 0)
        self.assertTrue(co2_n.sum() == 0)

        # Test that only H2O has X-H pairs
        co2_h = co2_out[loc_h]
        n2o_h = n2o_out[loc_h]
        h2o_h = h2o_out[loc_h]
        self.assertTrue(n2o_h.sum() == 0)
        self.assertTrue(h2o_h.sum() != 0)
        self.assertTrue(co2_h.sum() == 0)

        # Test that all have X-O pairs
        co2_o = co2_out[loc_o]
        n2o_o = n2o_out[loc_o]
        h2o_o = h2o_out[loc_o]
        self.assertTrue(n2o_o.sum() != 0)
        self.assertTrue(h2o_o.sum() != 0)
        self.assertTrue(co2_o.sum() != 0)

        # Test that only CO2 has X-O-C
        co2_oc = co2_out[loc_oc]
        n2o_oc = n2o_out[loc_oc]
        h2o_oc = h2o_out[loc_oc]
        self.assertTrue(n2o_oc.sum() == 0)
        self.assertTrue(h2o_oc.sum() == 0)
        self.assertTrue(co2_oc.sum() != 0)

        # Test that only N2O has X-O-N
        co2_on = co2_out[loc_on]
        n2o_on = n2o_out[loc_on]
        h2o_on = h2o_out[loc_on]
        self.assertTrue(n2o_on.sum() != 0)
        self.assertTrue(h2o_on.sum() == 0)
        self.assertTrue(co2_on.sum() == 0)

        # Test that only H2O has X-O-H
        co2_oh = co2_out[loc_oh]
        n2o_oh = n2o_out[loc_oh]
        h2o_oh = h2o_out[loc_oh]
        self.assertTrue(n2o_oh.sum() == 0)
        self.assertTrue(h2o_oh.sum() != 0)
        self.assertTrue(co2_oh.sum() == 0)

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
