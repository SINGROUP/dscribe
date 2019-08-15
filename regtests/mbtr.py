import math
import copy
import numpy as np
import unittest

import scipy.sparse
from scipy.signal import find_peaks_cwt, find_peaks

from dscribe.descriptors import MBTR

from ase.build import bulk
from ase.build import molecule
from ase import Atoms
import ase.geometry

from testbaseclass import TestBaseClass

default_k1 = {
    "geometry": {"function": "atomic_number"},
    "grid": {"min": 1, "max": 8, "sigma": 0.1, "n": 50}
}

default_k2 = {
    "geometry": {"function": "inverse_distance"},
    "grid": {"min": 0, "max": 1/0.7, "sigma": 0.1, "n": 50},
    "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-2},
}

default_k3 = {
    "geometry": {"function": "angle"},
    "grid": {"min": 0, "max": 180, "sigma": 2, "n": 50},
    "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-2},
}

default_desc_k1 = MBTR(
    species=[1, 8],
    k1=default_k1,
    periodic=False,
    flatten=True,
    sparse=False
)

default_desc_k2 = MBTR(
    species=[1, 8],
    k2=default_k2,
    periodic=False,
    flatten=True,
    sparse=False
)

default_desc_k3 = MBTR(
    species=[1, 8],
    k3=default_k3,
    periodic=False,
    flatten=True,
    sparse=False
)

default_desc_k1_k2_k3 = MBTR(
    species=[1, 8],
    periodic=True,
    k1={
        "geometry": {"function": "atomic_number"},
        "grid": {"min": 0, "max": 9, "sigma": 0.1, "n": 100},
    },
    k2={
        "geometry": {"function": "inverse_distance"},
        "grid": {"min": 0, "max": 1.0, "sigma": 0.02, "n": 100},
        "weighting": {"function": "exponential", "scale": 1, "cutoff": 1e-3},
    },
    k3={
        "geometry": {"function": "cosine"},
        "grid": {"min": -1.0, "max": 1.0, "sigma": 0.02, "n": 100},
        "weighting": {"function": "exponential", "scale": 1, "cutoff": 1e-3},
    },
    flatten=True,
    sparse=False,
    normalization="l2_each",
)

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

H = Atoms(
    cell=[
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
    ],
    positions=[
        [0, 0, 0],
    ],
    symbols=["H"],
)


class MBTRTests(TestBaseClass, unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """
        # Cannot create a sparse and non-flattened output.
        with self.assertRaises(ValueError):
            MBTR(
                species=["H"],
                k1=default_k1,
                periodic=False,
                flatten=False,
                sparse=True,
            )

        # Weighting needs to be provided for periodic system and terms k>1
        with self.assertRaises(ValueError):
            MBTR(
                species=["H"],
                k2={"geometry": default_k2["geometry"],
                    "grid": default_k2["grid"]
                },
                periodic=True,
            )
            MBTR(
                species=["H"],
                k2={"geometry": default_k2["geometry"],
                    "grid": default_k2["grid"],
                    "weighting": {"function": "unity"}
                },
                periodic=True,
            )

        with self.assertRaises(ValueError):
            MBTR(
                species=["H"],
                k3={"geometry": default_k3["geometry"],
                    "grid": default_k3["grid"]},
                periodic=True,
            )
            MBTR(
                species=["H"],
                k3={"geometry": default_k3["geometry"],
                    "grid": default_k3["grid"],
                    "weighting": {"function": "unity"}},
                periodic=True,
            )

        # Invalid weighting function
        with self.assertRaises(ValueError):
            MBTR(
                species=[1],
                k1={"geometry": default_k1["geometry"],
                    "grid": default_k1["grid"],
                    "weighting": {"function": "none"}
                },
                periodic=True
            )

        with self.assertRaises(ValueError):
            MBTR(
                species=[1],
                k2={"geometry": default_k2["geometry"],
                    "grid": default_k2["grid"],
                    "weighting": {"function": "none"}
                },
                periodic=True,
            )
        with self.assertRaises(ValueError):
            MBTR(
                species=[1],
                k3={"geometry": default_k3["geometry"],
                    "grid": default_k3["grid"],
                    "weighting": {"function": "none"}
                },
                periodic=True,
            )

        # Invalid geometry function
        with self.assertRaises(ValueError):
            MBTR(
                species=[1],
                k1={"geometry": {"function": "none"},
                    "grid": {"min": 0, "max": 1, "n": 10, "sigma": 0.1}
                },
                periodic=False,
            )
        with self.assertRaises(ValueError):
            MBTR(
                species=[1],
                k2={"geometry": {"function": "none"},
                    "grid": {"min": 0, "max": 1, "n": 10, "sigma": 0.1}
                },
                periodic=False,
            )
        with self.assertRaises(ValueError):
            MBTR(
                species=[1],
                k3={"geometry": {"function": "none"},
                    "grid": {"min": 0, "max": 1, "n": 10, "sigma": 0.1}
                },
                periodic=False,
            )

        # Missing cutoff
        with self.assertRaises(ValueError):
            setup = copy.deepcopy(default_k2)
            del setup["weighting"]["cutoff"]
            MBTR(
                species=[1],
                k2=setup,
                periodic=True,
            )

        # Missing scale
        with self.assertRaises(ValueError):
            setup = copy.deepcopy(default_k2)
            del setup["weighting"]["scale"]
            MBTR(
                species=[1],
                k2=setup,
                periodic=True,
            )

    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        # K=1
        n = 100
        atomic_numbers = [1, 8]
        n_elem = len(atomic_numbers)
        mbtr = MBTR(
            species=atomic_numbers,
            k1={
                "geometry": {"function": "atomic_number"},
                "grid": {"min": 1, "max": 8, "sigma": 0.1, "n": 100}
            },
            periodic=False,
            flatten=True
        )
        n_features = mbtr.get_number_of_features()
        expected = n_elem*n
        self.assertEqual(n_features, expected)

        # K=2
        mbtr = MBTR(
            species=atomic_numbers,
            k1={
                "geometry": {"function": "atomic_number"},
                "grid": {"min": 1, "max": 8, "sigma": 0.1, "n": 100},
            },
            k2={
                "geometry": {"function": "inverse_distance"},
                "grid": {"min": 0, "max": 1/0.7, "sigma": 0.1, "n": n},
                "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-2},
            },
            periodic=False,
            flatten=True
        )
        n_features = mbtr.get_number_of_features()
        expected = n_elem*n + 1/2*(n_elem)*(n_elem+1)*n
        self.assertEqual(n_features, expected)

        # K=3
        mbtr = MBTR(
            species=atomic_numbers,
            k1={
                "geometry": {"function": "atomic_number"},
                "grid": {"min": 1, "max": 8, "sigma": 0.1, "n": 100},
            },
            k2={
                "geometry": {"function": "inverse_distance"},
                "grid": {"min": 0, "max": 1/0.7, "sigma": 0.1, "n": n},
                "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-2},
            },
            k3={
                "geometry": {"function": "cosine"},
                "grid": {"min": -1, "max": 1, "sigma": 0.1, "n": n},
                "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-2},
            },
            periodic=False,
            flatten=True
        )
        n_features = mbtr.get_number_of_features()
        expected = n_elem*n + 1/2*(n_elem)*(n_elem+1)*n + n_elem*1/2*(n_elem)*(n_elem+1)*n
        self.assertEqual(n_features, expected)

    def test_locations_k1(self):
        """Tests that the function used to query combination locations for k=1
        in the output works.
        """
        CO2 = molecule("CO2")
        H2O = molecule("H2O")
        descriptors = [
            copy.deepcopy(default_desc_k1),
            copy.deepcopy(default_desc_k1_k2_k3)
        ]

        for desc in descriptors:
            desc.periodic = False
            desc.species = ["H", "O", "C"]

            co2_out = desc.create(CO2)[0, :]
            h2o_out = desc.create(H2O)[0, :]

            loc_h = desc.get_location(("H"))
            loc_c = desc.get_location(("C"))
            loc_o = desc.get_location(("O"))

            # H
            self.assertTrue(co2_out[loc_h].sum() == 0)
            self.assertTrue(h2o_out[loc_h].sum() != 0)

            # C
            self.assertTrue(co2_out[loc_c].sum() != 0)
            self.assertTrue(h2o_out[loc_c].sum() == 0)

            # O
            self.assertTrue(co2_out[loc_o].sum() != 0)
            self.assertTrue(h2o_out[loc_o].sum() != 0)

    def test_locations_k2(self):
        """Tests that the function used to query combination locations for k=2
        in the output works.
        """

        CO2 = molecule("CO2")
        H2O = molecule("H2O")
        descriptors = [
            copy.deepcopy(default_desc_k2),
            copy.deepcopy(default_desc_k1_k2_k3)
        ]

        for desc in descriptors:
            desc.periodic = False
            desc.species = ["H", "O", "C"]

            CO2 = molecule("CO2")
            H2O = molecule("H2O")

            co2_out = desc.create(CO2)[0, :]
            h2o_out = desc.create(H2O)[0, :]

            loc_hh = desc.get_location(("H", "H"))
            loc_hc = desc.get_location(("H", "C"))
            loc_ho = desc.get_location(("H", "O"))
            loc_co = desc.get_location(("C", "O"))
            loc_cc = desc.get_location(("C", "C"))
            loc_oo = desc.get_location(("O", "O"))

            # H-H
            self.assertTrue(co2_out[loc_hh].sum() == 0)
            self.assertTrue(h2o_out[loc_hh].sum() != 0)

            # H-C
            self.assertTrue(co2_out[loc_hc].sum() == 0)
            self.assertTrue(h2o_out[loc_hc].sum() == 0)

            # H-O
            self.assertTrue(co2_out[loc_ho].sum() == 0)
            self.assertTrue(h2o_out[loc_ho].sum() != 0)

            # C-O
            self.assertTrue(co2_out[loc_co].sum() != 0)
            self.assertTrue(h2o_out[loc_co].sum() == 0)

            # C-C
            self.assertTrue(co2_out[loc_cc].sum() == 0)
            self.assertTrue(h2o_out[loc_cc].sum() == 0)

            # O-O
            self.assertTrue(co2_out[loc_oo].sum() != 0)
            self.assertTrue(h2o_out[loc_oo].sum() == 0)

    def test_locations_k3(self):
        """Tests that the function used to query combination locations for k=2
        in the output works.
        """
        CO2 = molecule("CO2")
        H2O = molecule("H2O")
        descriptors = [
            copy.deepcopy(default_desc_k3),
            copy.deepcopy(default_desc_k1_k2_k3)
        ]

        for desc in descriptors:
            desc.periodic = False
            desc.species = ["H", "O", "C"]
            co2_out = desc.create(CO2)[0, :]
            h2o_out = desc.create(H2O)[0, :]

            loc_hhh = desc.get_location(("H", "H", "H"))
            loc_hho = desc.get_location(("H", "H", "O"))
            loc_hoo = desc.get_location(("H", "O", "O"))
            loc_hoh = desc.get_location(("H", "O", "H"))
            loc_ooo = desc.get_location(("O", "O", "O"))
            loc_ooh = desc.get_location(("O", "O", "H"))
            loc_oho = desc.get_location(("O", "H", "O"))
            loc_ohh = desc.get_location(("O", "H", "H"))

            # H-H-H
            self.assertTrue(co2_out[loc_hhh].sum() == 0)
            self.assertTrue(h2o_out[loc_hhh].sum() == 0)

            # H-H-O
            self.assertTrue(co2_out[loc_hho].sum() == 0)
            self.assertTrue(h2o_out[loc_hho].sum() != 0)

            # H-O-O
            self.assertTrue(co2_out[loc_hoo].sum() == 0)
            self.assertTrue(h2o_out[loc_hoo].sum() == 0)

            # H-O-H
            self.assertTrue(co2_out[loc_hoh].sum() == 0)
            self.assertTrue(h2o_out[loc_hoh].sum() != 0)

            # O-O-O
            self.assertTrue(co2_out[loc_ooo].sum() == 0)
            self.assertTrue(h2o_out[loc_ooo].sum() == 0)

            # O-O-H
            self.assertTrue(co2_out[loc_ooh].sum() == 0)
            self.assertTrue(h2o_out[loc_ooh].sum() == 0)

            # O-H-O
            self.assertTrue(co2_out[loc_oho].sum() == 0)
            self.assertTrue(h2o_out[loc_oho].sum() == 0)

            # O-H-H
            self.assertTrue(co2_out[loc_ohh].sum() == 0)
            self.assertTrue(h2o_out[loc_ohh].sum() != 0)

    def test_sparse(self):
        """Tests the sparse matrix creation.
        """
        # Dense
        desc = copy.deepcopy(default_desc_k1)
        desc.sparse = False
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == np.ndarray)

        # Sparse
        desc = copy.deepcopy(default_desc_k1)
        desc.sparse = True
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == scipy.sparse.coo_matrix)

    def test_properties(self):
        """Used to test that changing the setup through properties works as
        intended.
        """
        # Test changing species
        a = MBTR(
            k1=default_k1,
            k2=default_k2,
            k3=default_k3,
            periodic=False,
            species=[1, 8],
            sparse=False,
            flatten=True,
        )
        nfeat1 = a.get_number_of_features()
        vec1 = a.create(H2O)
        a.species = ["C", "H", "O"]
        nfeat2 = a.get_number_of_features()
        vec2 = a.create(molecule("CH3OH"))
        self.assertTrue(nfeat1 != nfeat2)
        self.assertTrue(vec1.shape[1] != vec2.shape[1])

        # Test changing geometry function and grid setup
        a.k1 = {
            "geometry": {"function": "atomic_number"},
            "grid": {"min": 5, "max": 6, "sigma": 0.1, "n": 50},
        }
        vec3 = a.create(H2O)
        self.assertTrue(not np.allclose(vec2, vec3))

        a.k2 = {
            "geometry": {"function": "distance"},
            "grid": {"min": 0, "max": 10, "sigma": 0.1, "n": 50},
            "weighting": {"function": "exponential", "scale": 0.6, "cutoff": 1e-2},
        }
        vec4 = a.create(H2O)
        self.assertTrue(not np.allclose(vec3, vec4))

        a.k3 = {
            "geometry": {"function": "angle"},
            "grid": {"min": 0, "max": 180, "sigma": 5, "n": 50},
            "weighting": {"function": "exponential", "scale": 0.6, "cutoff": 1e-2},
        }
        vec5 = a.create(H2O)
        self.assertTrue(not np.allclose(vec4, vec5))

    def test_flatten(self):
        """Tests that flattened, and non-flattened output works correctly.
        """
        system = H2O
        n = 10
        n_species = len(set(system.get_atomic_numbers()))

        # K1 unflattened
        desc = MBTR(
            species=[1, 8],
            k1={
                "grid": {"n": n, "min": 1, "max": 8, "sigma": 0.1},
                "geometry": {"function": "atomic_number"}
            },
            periodic=False,
            flatten=False,
            sparse=False
        )
        feat = desc.create(system)["k1"]
        self.assertEqual(feat.shape, (n_species, n))

        # K1 flattened. The sparse matrix only supports 2D matrices, so the first
        # dimension is always present, even if it is of length 1.
        desc.flatten = True
        feat = desc.create(system)
        self.assertEqual(feat.shape, (1, n_species*n))

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
        desc._flatten = False
        output = desc.create(
            system=samples,
            n_jobs=2,
        )
        assumed = []
        assumed.append(desc.create(samples[0]))
        assumed.append(desc.create(samples[1]))
        for i, val in enumerate(output):
            for key in val.keys():
                i_tensor = val[key]
                j_tensor = assumed[i][key]
                self.assertTrue(np.allclose(i_tensor, j_tensor))

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

    def test_periodic_supercell_similarity(self):
        """Tests that the output spectrum of various supercells of the same
        crystal is identical after it is normalized.
        """
        decay = 1
        desc = MBTR(
            species=["H"],
            periodic=True,
            k1={
                "geometry": {"function": "atomic_number"},
                "grid": {"min": 0, "max": 2, "sigma": 0.1, "n": 100},
            },
            k2={
                "geometry": {"function": "inverse_distance"},
                "grid": {"min": 0, "max": 1.0, "sigma": 0.02, "n": 200},
                "weighting": {"function": "exponential", "scale": decay, "cutoff": 1e-3},
            },
            k3={
                "geometry": {"function": "cosine"},
                "grid": {"min": -1.0, "max": 1.0, "sigma": 0.02, "n": 200},
                "weighting": {"function": "exponential", "scale": decay, "cutoff": 1e-3},
            },
            flatten=True,
            sparse=False,
            normalization="l2_each",
        )

        # Create various supercells for the FCC structure
        a1 = bulk('H', 'fcc', a=2.0)                     # Primitive
        a2 = a1*[2, 2, 2]                                # Supercell
        a3 = bulk('H', 'fcc', a=2.0, orthorhombic=True)  # Orthorhombic
        a4 = bulk('H', 'fcc', a=2.0, cubic=True)         # Conventional cubic

        output = desc.create([a1, a2, a3, a4])

        # Test for equality
        self.assertTrue(np.allclose(output[0, :], output[0, :], atol=1e-5, rtol=0))
        self.assertTrue(np.allclose(output[0, :], output[1, :], atol=1e-5, rtol=0))
        self.assertTrue(np.allclose(output[0, :], output[2, :], atol=1e-5, rtol=0))
        self.assertTrue(np.allclose(output[0, :], output[3, :], atol=1e-5, rtol=0))

    def test_normalization(self):
        """Tests that each normalization method works correctly.
        """
        n = 100
        desc = copy.deepcopy(default_desc_k1_k2_k3)
        desc.species = ("H", "O")
        desc.normalization = "none"
        desc.flatten = False
        desc.sparse = False

        # Calculate the norms
        feat1 = desc.create(H2O)
        k1 = feat1["k1"]
        k2 = feat1["k2"]
        k3 = feat1["k3"]
        k1_norm = np.linalg.norm(k1.ravel())
        k2_norm = np.linalg.norm(k2.ravel())
        k3_norm = np.linalg.norm(k3.ravel())

        # Test normalization of non-flat dense output with l2_each
        desc.normalization = "l2_each"
        feat2 = desc.create(H2O)
        k1_each = feat2["k1"]
        k2_each = feat2["k2"]
        k3_each = feat2["k3"]
        self.assertTrue(np.array_equal(k1/k1_norm, k1_each))
        self.assertTrue(np.array_equal(k2/k2_norm, k2_each))
        self.assertTrue(np.array_equal(k3/k3_norm, k3_each))

        # Flattened dense output
        desc.flatten = True
        desc.normalization = "none"
        feat_flat = desc.create(H2O)

        # Test normalization of flat dense output with l2_each
        desc.sparse = False
        desc.normalization = "l2_each"
        n_elem = len(desc.species)
        feat = desc.create(H2O)
        n1 = int(n*n_elem)
        n2 = int((n_elem*(n_elem+1)/2)*n)
        a1 = feat_flat[0, 0:n1]/k1_norm
        a2 = feat_flat[0, n1:n1+n2]/k2_norm
        a3 = feat_flat[0, n1+n2:]/k3_norm
        feat_flat_manual_norm_each = np.hstack((a1, a2, a3))

        self.assertTrue(np.allclose(feat[0, :], feat_flat_manual_norm_each, atol=1e-7, rtol=0))

        # Test normalization of flat sparse output with l2_each
        desc.sparse = True
        desc.normalization = "l2_each"
        feat = desc.create(H2O).toarray()
        self.assertTrue(np.allclose(feat[0, :], feat_flat_manual_norm_each, atol=1e-7, rtol=0))

        # Test normalization of flat dense output with n_atoms
        desc.sparse = False
        desc.normalization = "n_atoms"
        n_atoms = len(H2O)
        n_elem = len(desc.species)
        feat = desc.create(H2O)
        self.assertTrue(np.allclose(feat[0, :], feat_flat/n_atoms, atol=1e-7, rtol=0))

        # Test normalization of flat sparse output with n_atoms
        desc.sparse = True
        desc.normalization = "n_atoms"
        feat = desc.create(H2O).toarray()
        self.assertTrue(np.allclose(feat[0, :], feat_flat/n_atoms, atol=1e-7, rtol=0))

    def test_k1_peaks_finite(self):
        """Tests the correct peak locations and intensities are found for the
        k=1 term.
        """
        desc = MBTR(
            species=[1, 8],
            k1={
                "geometry": {"function": "atomic_number"},
                "grid": {"min": 0, "max": 9, "sigma": 0.5, "n": 1000}
            },
            normalize_gaussians=False,
            periodic=False,
            flatten=True,
            sparse=False
        )
        features = desc.create(H2O)[0, :]
        x = desc.get_k1_axis()

        # Check the H peaks
        h_feat = features[desc.get_location(("H"))]
        h_peak_indices = find_peaks(h_feat, prominence=1)[0]
        h_peak_locs = x[h_peak_indices]
        h_peak_ints = h_feat[h_peak_indices]
        self.assertTrue(np.allclose(h_peak_locs, [1], rtol=0, atol=1e-2))
        self.assertTrue(np.allclose(h_peak_ints, [2], rtol=0, atol=1e-2))

        # Check the O peaks
        o_feat = features[desc.get_location(("O"))]
        o_peak_indices = find_peaks(o_feat, prominence=1)[0]
        o_peak_locs = x[o_peak_indices]
        o_peak_ints = o_feat[o_peak_indices]
        self.assertTrue(np.allclose(o_peak_locs, [8], rtol=0, atol=1e-2))
        self.assertTrue(np.allclose(o_peak_ints, [1], rtol=0, atol=1e-2))

        # Check that everything else is zero
        features[desc.get_location(("H"))] = 0
        features[desc.get_location(("O"))] = 0
        self.assertEqual(features.sum(), 0)

    def test_k2_peaks_finite(self):
        """Tests the correct peak locations and intensities are found for the
        k=2 term in finite systems.
        """
        desc = MBTR(
            species=[1, 8],
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
        features = desc.create(H2O)[0, :]
        pos = H2O.get_positions()
        x = desc.get_k2_axis()

        # Check the H-H peaks
        hh_feat = features[desc.get_location(("H", "H"))]
        hh_peak_indices = find_peaks(hh_feat, prominence=0.5)[0]
        hh_peak_locs = x[hh_peak_indices]
        hh_peak_ints = hh_feat[hh_peak_indices]
        self.assertTrue(np.allclose(hh_peak_locs, [np.linalg.norm(pos[0] - pos[2])], rtol=0, atol=1e-2))
        self.assertTrue(np.allclose(hh_peak_ints, [1], rtol=0, atol=1e-2))

        # Check the O-H peaks
        ho_feat = features[desc.get_location(("H", "O"))]
        ho_peak_indices = find_peaks(ho_feat, prominence=0.5)[0]
        ho_peak_locs = x[ho_peak_indices]
        ho_peak_ints = ho_feat[ho_peak_indices]
        self.assertTrue(np.allclose(ho_peak_locs, np.linalg.norm(pos[0] - pos[1]), rtol=0, atol=1e-2))
        self.assertTrue(np.allclose(ho_peak_ints, [2], rtol=0, atol=1e-2))

        # Check that everything else is zero
        features[desc.get_location(("H", "H"))] = 0
        features[desc.get_location(("H", "O"))] = 0
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

        desc = MBTR(
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
        features = desc.create(atoms)[0, :]
        x = desc.get_k2_axis()

        # Calculate assumed locations and intensities.
        assumed_locs = np.array([2, 8])
        assumed_ints = np.exp(-0.8*np.array([2, 8]))
        assumed_ints[0] *= 2  # There are two periodic distances at 2Å
        assumed_ints[0] /= 2  # The periodic distances ar halved because they belong to different cells

        # Check the H-C peaks
        hc_feat = features[desc.get_location(("H", "C"))]
        hc_peak_indices = find_peaks(hc_feat, prominence=0.001)[0]
        hc_peak_locs = x[hc_peak_indices]
        hc_peak_ints = hc_feat[hc_peak_indices]
        self.assertTrue(np.allclose(hc_peak_locs, assumed_locs, rtol=0, atol=1e-2))
        self.assertTrue(np.allclose(hc_peak_ints, assumed_ints, rtol=0, atol=1e-2))

        # Check that everything else is zero
        features[desc.get_location(("H", "C"))] = 0
        self.assertEqual(features.sum(), 0)

    def test_k2_periodic_cell_translation(self):
        """Tests that the final spectra does not change when translating atoms
        in a periodic cell. This is not trivially true unless the weight of
        distances between periodic neighbours are not halfed. Notice that the
        values of the geometry and weight functions are not equal before
        summing them up in the final graph.
        """
        # Original system with atoms separated by a cell wall
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
            ],
            pbc=True
        )

        # Translated system with atoms next to each other
        atoms2 = atoms.copy()
        atoms2.translate([5, 0, 0])
        atoms2.wrap()

        desc = copy.deepcopy(default_desc_k2)
        desc.species = ["H", "C"]
        desc.periodic = True
        desc.k2["weighting"] = {"function": "exp", "scale": 0.8, "cutoff": 1e-3}

        # The resulting spectra should be indentical
        spectra1 = desc.create(atoms)[0, :]
        spectra2 = desc.create(atoms2)[0, :]
        self.assertTrue(np.allclose(spectra1, spectra2, rtol=0, atol=1e-7))

    def test_k3_peaks_finite(self):
        """Tests that all the correct angles are present in finite systems.
        There should be n*(n-1)*(n-2)/2 unique angles where the division by two
        gets rid of duplicate angles.
        """
        desc = MBTR(
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
        features = desc.create(H2O)[0, :]
        x = desc.get_k3_axis()

        # Check the H-H-O peaks
        hho_assumed_locs = np.array([38])
        hho_assumed_ints = np.array([2])
        hho_feat = features[desc.get_location(("H", "H", "O"))]
        hho_peak_indices = find_peaks(hho_feat, prominence=0.5)[0]
        hho_peak_locs = x[hho_peak_indices]
        hho_peak_ints = hho_feat[hho_peak_indices]
        self.assertTrue(np.allclose(hho_peak_locs, hho_assumed_locs, rtol=0, atol=5e-2))
        self.assertTrue(np.allclose(hho_peak_ints, hho_assumed_ints, rtol=0, atol=5e-2))

        # Check the H-O-H peaks
        hoh_assumed_locs = np.array([104])
        hoh_assumed_ints = np.array([1])
        hoh_feat = features[desc.get_location(("H", "O", "H"))]
        hoh_peak_indices = find_peaks(hoh_feat, prominence=0.5)[0]
        hoh_peak_locs = x[hoh_peak_indices]
        hoh_peak_ints = hoh_feat[hoh_peak_indices]
        self.assertTrue(np.allclose(hoh_peak_locs, hoh_assumed_locs, rtol=0, atol=5e-2))
        self.assertTrue(np.allclose(hoh_peak_ints, hoh_assumed_ints, rtol=0, atol=5e-2))

        # Check that everything else is zero
        features[desc.get_location(("H", "H", "O"))] = 0
        features[desc.get_location(("H", "O", "H"))] = 0
        self.assertEqual(features.sum(), 0)

    def test_k3_peaks_periodic(self):
        """Tests that the final spectra does not change when translating atoms
        in a periodic cell. This is not trivially true unless the weight of
        angles is weighted according to the cell indices of the involved three
        atoms. Notice that the values of the geometry and weight functions are
        not equal before summing them up in the final graph.
        """
        scale = 0.85
        desc = MBTR(
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
        features = desc.create(atoms)[0, :]
        x = desc.get_k3_axis()

        # Calculate assumed locations and intensities.
        assumed_locs = np.array([45, 90])
        dist = 2+2*np.sqrt(2)  # The total distance around the three atoms
        weight = np.exp(-scale*dist)
        assumed_ints = np.array([4*weight, 2*weight])
        assumed_ints /= 2  # The periodic distances ar halved because they belong to different cells

        # Check the H-H-H peaks
        hhh_feat = features[desc.get_location(("H", "H", "H"))]
        hhh_peak_indices = find_peaks(hhh_feat, prominence=0.01)[0]
        hhh_peak_locs = x[hhh_peak_indices]
        hhh_peak_ints = hhh_feat[hhh_peak_indices]
        self.assertTrue(np.allclose(hhh_peak_locs, assumed_locs, rtol=0, atol=1e-1))
        self.assertTrue(np.allclose(hhh_peak_ints, assumed_ints, rtol=0, atol=1e-1))

        # Check that everything else is zero
        features[desc.get_location(("H", "H", "H"))] = 0
        self.assertEqual(features.sum(), 0)

    def test_k3_periodic_cell_translation(self):
        """Tests that the final spectra does not change when translating atoms
        in a periodic cell. This is not trivially true unless the weight of
        distances between periodic neighbours are not halfed. Notice that the
        values of the geometry and weight functions are not equal before
        summing them up in the final graph.
        """
        # Original system with atoms separated by a cell wall
        atoms = Atoms(
            cell=[
                [10, 0, 0],
                [0, 10, 0],
                [0, 0, 10],
            ],
            symbols=["H", "H", "H", "H"],
            scaled_positions=[
                [0.1, 0.50, 0.5],
                [0.1, 0.60, 0.5],
                [0.9, 0.50, 0.5],
                [0.9, 0.60, 0.5],
            ],
            pbc=True
        )

        # Translated system with atoms next to each other
        atoms2 = atoms.copy()
        atoms2.translate([5, 0, 0])
        atoms2.wrap()

        desc = copy.deepcopy(default_desc_k3)
        desc.k3["weighting"] = {"function": "exp", "scale": 1, "cutoff": 1e-3}
        desc.periodic = True

        # The resulting spectra should be indentical
        spectra1 = desc.create(atoms)[0, :]
        spectra2 = desc.create(atoms2)[0, :]
        self.assertTrue(np.allclose(spectra1, spectra2, rtol=0, atol=1e-8))

    def test_gaussian_distribution(self):
        """Check that the broadening follows gaussian distribution.
        """
        # Check with normalization
        std = 1
        start = -3
        stop = 11
        n = 500
        desc = copy.deepcopy(default_desc_k1)
        desc.flatten = False
        desc.normalize_gaussians = True
        desc.k1["grid"] = {"min": start, "max": stop, "sigma": std, "n": n}
        y = desc.create(H2O)["k1"]
        k1_axis = desc.get_k1_axis()

        # Find the location of the peaks
        peak1_x = np.searchsorted(k1_axis, 1)
        peak1_y = y[0, peak1_x]
        peak2_x = np.searchsorted(k1_axis, 8)
        peak2_y = y[1, peak2_x]

        # Check against the analytical value
        gaussian = lambda x, mean, sigma: 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*sigma**2))
        self.assertTrue(np.allclose(peak1_y, 2*gaussian(1, 1, std), rtol=0, atol=0.001))
        self.assertTrue(np.allclose(peak2_y, gaussian(8, 8, std), rtol=0, atol=0.001))

        # Check the integral
        pdf = y[0, :]
        dx = (stop-start)/(n-1)
        sum_cum = np.sum(0.5*dx*(pdf[:-1]+pdf[1:]))
        exp = 2
        self.assertTrue(np.allclose(sum_cum, exp, rtol=0, atol=0.001))

        # Check without normalization
        std = 1
        start = -3
        stop = 11
        n = 500
        desc.normalize_gaussians = False
        desc.k1["grid"] = {"min": start, "max": stop, "sigma": std, "n": n}
        y = desc.create(H2O)["k1"]
        k1_axis = desc.get_k1_axis()

        # Find the location of the peaks
        peak1_x = np.searchsorted(k1_axis, 1)
        peak1_y = y[0, peak1_x]
        peak2_x = np.searchsorted(k1_axis, 8)
        peak2_y = y[1, peak2_x]

        # Check against the analytical value
        gaussian = lambda x, mean, sigma: np.exp(-(x-mean)**2/(2*sigma**2))
        self.assertTrue(np.allclose(peak1_y, 2*gaussian(1, 1, std), rtol=0, atol=0.001))
        self.assertTrue(np.allclose(peak2_y, gaussian(8, 8, std), rtol=0, atol=0.001))

        # Check the integral
        pdf = y[0, :]
        dx = (stop-start)/(n-1)
        sum_cum = np.sum(0.5*dx*(pdf[:-1]+pdf[1:]))
        exp = 2/(1/math.sqrt(2*math.pi*std**2))
        self.assertTrue(np.allclose(sum_cum, exp, rtol=0, atol=0.001))

    def test_symmetries(self):

        def create(system):
            desc = copy.deepcopy(default_desc_k1_k2_k3)
            desc.species = ["H", "O"]
            return desc.create(system)

        # Rotational check
        self.assertTrue(self.is_rotationally_symmetric(create))

        # Translational
        self.assertTrue(self.is_translationally_symmetric(create))

        # Permutational
        self.assertTrue(self.is_permutation_symmetric(create))

    def test_unit_cells(self):
        """Tests that arbitrary unit cells are accepted.
        """
        desc = copy.deepcopy(default_desc_k1_k2_k3)
        desc.periodic = False
        desc.species = ["H", "O"]
        molecule = H2O.copy()

        # No cell needed for finite systems
        molecule.set_cell([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        nocell = desc.create(molecule)

        # Different periodic cells
        desc.periodic = True
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

    def test_periodic_images(self):
        """Tests that periodic images are handled correctly.
        """
        decay = 1
        desc = MBTR(
            species=[1],
            periodic=True,
            k1={
                "geometry": {"function": "atomic_number"},
                "grid": {"min": 0, "max": 2, "sigma": 0.1, "n": 21}
            },
            k2={
                "geometry": {"function": "inverse_distance"},
                "grid": {"min": 0, "max": 1.0, "sigma": 0.02, "n": 21},
                "weighting": {"function": "exp", "scale": decay, "cutoff": 1e-4}
            },
            k3={
                "geometry": {"function": "cosine"},
                "grid": {"min": -1.0, "max": 1.0, "sigma": 0.02, "n": 21},
                "weighting": {"function": "exp", "scale": decay, "cutoff": 1e-4},
            },
            normalization="l2_each",  # This normalizes the spectrum
            flatten=True
        )

        # Tests that a system has the same spectrum as the supercell of
        # the same system.
        molecule = H.copy()
        a = 1.5
        molecule.set_cell([
            [a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, 0.0, a]
        ])
        cubic_cell = desc.create(molecule)
        suce = molecule * (2, 1, 1)
        cubic_suce = desc.create(suce)

        diff = abs(np.sum(cubic_cell[0, :] - cubic_suce[0, :]))
        cubic_sum = abs(np.sum(cubic_cell[0, :]))
        self.assertTrue(diff/cubic_sum < 0.05)  # A 5% error is tolerated

        # Same test but for triclinic cell
        molecule.set_cell([
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 2.0, 0.0]
        ])

        triclinic_cell = desc.create(molecule)
        suce = molecule * (2, 1, 1)
        triclinic_suce = desc.create(suce)

        diff = abs(np.sum(triclinic_cell[0, :] - triclinic_suce[0, :]))
        tricl_sum = abs(np.sum(triclinic_cell[0, :]))
        self.assertTrue(diff/tricl_sum < 0.05)

        # Testing that the same crystal, but different unit cells will have a
        # similar spectrum when they are normalized. There will be small
        # differences in the shape (due to not double counting distances)
        a1 = bulk('H', 'fcc', a=2.0)
        a2 = bulk('H', 'fcc', a=2.0, orthorhombic=True)
        a3 = bulk('H', 'fcc', a=2.0, cubic=True)

        triclinic_cell = desc.create(a1)
        orthorhombic_cell = desc.create(a2)
        cubic_cell = desc.create(a3)

        diff1 = abs(np.sum(triclinic_cell[0, :] - orthorhombic_cell[0, :]))
        diff2 = abs(np.sum(triclinic_cell[0, :] - cubic_cell[0, :]))
        tricl_sum = abs(np.sum(triclinic_cell[0, :]))
        self.assertTrue(diff1/tricl_sum < 0.05)
        self.assertTrue(diff2/tricl_sum < 0.05)

        # Tests that the correct peak locations are present in a cubic periodic
        desc = MBTR(
            species=["H"],
            periodic=True,
            k3={
                "geometry": {"function": "cosine"},
                "grid": {"min": -1.1, "max": 1.1, "sigma": 0.010, "n": 600},
                "weighting": {"function": "exp", "scale": decay, "cutoff": 1e-4}
            },
            normalization="l2_each",  # This normalizes the spectrum
            flatten=True
        )
        a = 2.2
        system = Atoms(
            cell=[
                [a, 0.0, 0.0],
                [0.0, a, 0.0],
                [0.0, 0.0, a]
            ],
            positions=[
                [0, 0, 0],
            ],
            symbols=["H"],
        )
        cubic_spectrum = desc.create(system)[0, :]
        x3 = desc.get_k3_axis()

        peak_ids = find_peaks_cwt(cubic_spectrum, [2])
        peak_locs = x3[peak_ids]

        assumed_peaks = np.cos(np.array(
            [
                180,
                90,
                np.arctan(np.sqrt(2))*180/np.pi,
                45,
                np.arctan(np.sqrt(2)/2)*180/np.pi,
                0
            ])*np.pi/180
        )
        self.assertTrue(np.allclose(peak_locs, assumed_peaks, rtol=0, atol=5*np.pi/180))

        # Tests that the correct peak locations are present in a system with a
        # non-cubic basis
        desc = MBTR(
            species=["H"],
            periodic=True,
            k3={
                "geometry": {"function": "cosine"},
                "grid": {"min": -1.0, "max": 1.0, "sigma": 0.030, "n": 200},
                "weighting": {"function": "exp", "scale": 1.5, "cutoff": 1e-4}
            },
            normalization="l2_each",  # This normalizes the spectrum
            flatten=True,
            sparse=False
        )
        a = 2.2
        system = Atoms(
            cell=[
                [a, 0.0, 0.0],
                [0.0, a, 0.0],
                [0.0, 0.0, a]
            ],
            positions=[
                [0, 0, 0],
            ],
            symbols=["H"],
        )
        angle = 30
        system = Atoms(
            cell=ase.geometry.cellpar_to_cell([3*a, a, a, angle, 90, 90]),
            positions=[
                [0, 0, 0],
            ],
            symbols=["H"],
        )
        tricl_spectrum = desc.create(system)
        x3 = desc.get_k3_axis()

        peak_ids = find_peaks_cwt(tricl_spectrum[0, :], [3])
        peak_locs = x3[peak_ids]

        angle = (6)/(np.sqrt(5)*np.sqrt(8))
        assumed_peaks = np.cos(np.array([180, 105, 75, 51.2, 30, 0])*np.pi/180)
        self.assertTrue(np.allclose(peak_locs, assumed_peaks, rtol=0, atol=5*np.pi/180))

    def test_basis(self):
        """Tests that the output vectors behave correctly as a basis.
        """
        sys1 = Atoms(symbols=["H"], positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
        sys2 = Atoms(symbols=["O"], positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
        sys3 = sys2*[2, 2, 2]

        desc = copy.deepcopy(default_desc_k1_k2_k3)
        desc.sparse = False

        # Create normalized vectors for each system
        vec1 = desc.create(sys1)[0, :]
        vec1 /= np.linalg.norm(vec1)

        vec2 = desc.create(sys2)[0, :]
        vec2 /= np.linalg.norm(vec2)

        vec3 = desc.create(sys3)[0, :]
        vec3 /= np.linalg.norm(vec3)

        # The dot-product should be zero when there are no overlapping elements
        dot = np.dot(vec1, vec2)
        self.assertEqual(dot, 0)

        # The dot-product should be rougly one for a primitive cell and a supercell
        dot = np.dot(vec2, vec3)
        self.assertTrue(abs(dot-1) < 1e-3)


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(MBTRTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
