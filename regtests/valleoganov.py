from dscribe.descriptors import ValleOganov, MBTR
from testbaseclass import TestBaseClass
import unittest
import numpy as np
import copy
import sparse
import math
from ase import Atoms

default_k2 = {"sigma": 10 ** (-0.625), "n": 100, "r_cut": 10}

default_k3 = {"sigma": 10 ** (-0.625), "n": 100, "r_cut": 10}

H2O = Atoms(
    cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
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


class ValleOganovTests(TestBaseClass, unittest.TestCase):
    def test_exceptions(self):
        """Tests different invalid parameters that should raise an
        exception.
        """
        # Cannot create a sparse and non-flattened output.
        with self.assertRaises(ValueError):
            ValleOganov(
                species=["H"],
                k2=default_k2,
                flatten=False,
                sparse=True,
            )

        # Missing r_cut
        with self.assertRaises(ValueError):
            ValleOganov(
                species=[1],
                k2={
                    "sigma": 10 ** (-0.625),
                    "n": 100,
                },
            )
        with self.assertRaises(ValueError):
            ValleOganov(
                species=[1],
                k3={
                    "sigma": 10 ** (-0.625),
                    "n": 100,
                },
            )

        # Missing n
        with self.assertRaises(ValueError):
            ValleOganov(
                species=[1],
                k2={"sigma": 10 ** (-0.625), "r_cut": 10},
            )
        with self.assertRaises(ValueError):
            ValleOganov(
                species=[1],
                k3={"sigma": 10 ** (-0.625), "r_cut": 10},
            )

        # Missing sigma
        with self.assertRaises(ValueError):
            ValleOganov(
                species=[1],
                k2={"n": 100, "r_cut": 10},
            )
        with self.assertRaises(ValueError):
            ValleOganov(
                species=[1],
                k3={"n": 100, "r_cut": 10},
            )

    def test_number_of_features(self):
        """Tests that the reported number of features is correct."""
        n = 100
        atomic_numbers = [1, 8]
        n_elem = len(atomic_numbers)

        # K=2
        vo = ValleOganov(
            species=atomic_numbers,
            k2=default_k2,
            flatten=True,
        )
        n_features = vo.get_number_of_features()
        expected = 1 / 2 * (n_elem) * (n_elem + 1) * n
        self.assertEqual(n_features, expected)

        # K=3
        vo = ValleOganov(
            species=atomic_numbers, k2=default_k2, k3=default_k3, flatten=True
        )
        n_features = vo.get_number_of_features()
        expected = (
            1 / 2 * (n_elem) * (n_elem + 1) * n
            + n_elem * 1 / 2 * (n_elem) * (n_elem + 1) * n
        )
        self.assertEqual(n_features, expected)

    def test_sparse(self):
        """Tests the sparse matrix creation."""
        vo = ValleOganov(species=[1, 8], k2=default_k2, flatten=True, sparse=False)
        # Dense
        desc = copy.deepcopy(vo)
        desc.sparse = False
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == np.ndarray)

        # Sparse
        desc = copy.deepcopy(vo)
        desc.sparse = True
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == sparse.COO)

    def test_symmetries(self):
        def create(system):
            desc = ValleOganov(
                k2=default_k2,
                k3=default_k3,
                flatten=True,
                sparse=False,
                species=["H", "O"],
            )
            return desc.create(system)

        # Rotational check
        self.assertTrue(self.is_rotationally_symmetric(create))

        # Translational
        self.assertTrue(self.is_translationally_symmetric(create))

        # Permutational
        self.assertTrue(self.is_permutation_symmetric(create))

    def test_flatten(self):
        """Tests that flattened, and non-flattened output works correctly."""
        system = H2O
        n = 10
        n_species = len(set(system.get_atomic_numbers()))

        # K2 unflattened
        desc = ValleOganov(
            species=[1, 8],
            k2={"sigma": 0.1, "n": n, "r_cut": 5},
            flatten=False,
            sparse=False,
        )
        feat = desc.create(system)["k2"]
        self.assertEqual(feat.shape, (n_species, n_species, n))

        # K2 flattened.
        n_features = desc.get_number_of_features()
        desc.flatten = True
        feat = desc.create(system)
        self.assertEqual(feat.shape, (n_features,))

    def test_subclass(self):
        """Tests that the ValleOganov subclass gives the same output as MBTR
        with the corresponding parameters.
        """
        desc = ValleOganov(
            species=[1, 8],
            k2={"sigma": 0.1, "n": 20, "r_cut": 5},
            k3={"sigma": 0.1, "n": 20, "r_cut": 5},
        )
        feat = desc.create(H2O)

        desc2 = MBTR(
            species=[1, 8],
            periodic=True,
            k2={
                "geometry": {"function": "distance"},
                "grid": {"min": 0, "max": 5, "sigma": 0.1, "n": 20},
                "weighting": {"function": "inverse_square", "r_cut": 5},
            },
            k3={
                "geometry": {"function": "angle"},
                "grid": {"min": 0, "max": 180, "sigma": 0.1, "n": 20},
                "weighting": {"function": "smooth_cutoff", "r_cut": 5},
            },
            normalization="valle_oganov",
            flatten=True,
            sparse=False,
        )
        feat2 = desc2.create(H2O)

        self.assertTrue(np.array_equal(feat, feat2))


if __name__ == "__main__":
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ValleOganovTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
