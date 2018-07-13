from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import math
import numpy as np
import unittest
from scipy.signal import argrelextrema

from describe.descriptors import MBTR
from describe.data.element_data import numbers_to_symbols

from ase.build import bulk
from ase import Atoms


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


class MBTRTests(unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """
        with self.assertRaises(ValueError):
            MBTR(
                atomic_numbers=[1],
                k=0,
                periodic=False,
            )

        with self.assertRaises(ValueError):
            MBTR(
                atomic_numbers=[1],
                k=[-1, 2],
                periodic=False,
            )

        with self.assertRaises(ValueError):
            MBTR(
                atomic_numbers=[1],
                k=1,
                periodic=False,
            )

        with self.assertRaises(ValueError):
            MBTR(
                atomic_numbers=[1],
                k={1, 4},
                periodic=False,
            )

    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        # K = 1
        n = 100
        atomic_numbers = [1, 8]
        n_elem = len(atomic_numbers)
        mbtr = MBTR(
            atomic_numbers=atomic_numbers,
            k=[1],
            grid={
                "k1": {
                    "min": 1,
                    "max": 8,
                    "sigma": 0.1,
                    "n": 100,
                }
            },
            periodic=False,
            flatten=True
        )
        n_features = mbtr.get_number_of_features()
        expected = n_elem*n
        self.assertEqual(n_features, expected)

        # K = 2
        mbtr = MBTR(
            atomic_numbers=atomic_numbers,
            k={1, 2},
            grid={
                "k1": {
                    "min": 1,
                    "max": 8,
                    "sigma": 0.1,
                    "n": 100,
                },
                "k2": {
                    "min": 0,
                    "max": 1/0.7,
                    "sigma": 0.1,
                    "n": n,
                }
            },
            periodic=False,
            flatten=True
        )
        n_features = mbtr.get_number_of_features()
        expected = n_elem*n + 1/2*(n_elem)*(n_elem+1)*n
        self.assertEqual(n_features, expected)

        # K = 3
        mbtr = MBTR(
            atomic_numbers=atomic_numbers,
            k={1, 2, 3},
            grid={
                "k1": {
                    "min": 1,
                    "max": 8,
                    "sigma": 0.1,
                    "n": 100,
                },
                "k2": {
                    "min": 0,
                    "max": 1/0.7,
                    "sigma": 0.1,
                    "n": n,
                },
                "k3": {
                    "min": -1,
                    "max": 1,
                    "sigma": 0.1,
                    "n": n,
                }
            },
            periodic=False,
            flatten=True
        )
        n_features = mbtr.get_number_of_features()
        expected = n_elem*n + 1/2*(n_elem)*(n_elem+1)*n + n_elem*1/2*(n_elem)*(n_elem+1)*n
        self.assertEqual(n_features, expected)

    # def test_flatten(self):
        # """Tests the flattening.
        # """
        # # Unflattened
        # desc = MBTR(n_atoms_max=5, permutation="none", flatten=False)
        # cm = desc.create(H2O)
        # self.assertEqual(cm.shape, (5, 5))

        # # Flattened
        # desc = MBTR(n_atoms_max=5, permutation="none", flatten=True)
        # cm = desc.create(H2O)
        # self.assertEqual(cm.shape, (25,))

    def test_counts(self):
        mbtr = MBTR([1, 8], k=[1], periodic=False)
        mbtr.create(H2O)
        counts = mbtr._counts

        # Test against the assumed values
        self.assertTrue(np.array_equal(counts, np.array([2, 1])))

        # Test against system with different indexing
        mbtr = MBTR([1, 8], k=[1], periodic=False)
        mbtr.create(H2O_2)
        counts2 = mbtr._counts
        self.assertTrue(np.array_equal(counts, counts2))

    def test_periodic(self):
        test_sys = Atoms(
            cell=[[5.0, 0.0, 0.0], [0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            positions=[[0, 0, 0]],
            symbols=["H"],
        )
        mbtr = MBTR([1], k=[2], weighting="exponential", periodic=True)
        desc = mbtr.create(test_sys)

    def test_inverse_distances(self):
        mbtr = MBTR([1, 8], k=[2], periodic=False)
        mbtr.create(H2O)
        inv_dist = mbtr._inverse_distances

        # Test against the assumed values
        pos = H2O.get_positions()
        assumed = {
            0: {
                0: [1/np.linalg.norm(pos[0] - pos[2])],
                1: 2*[1/np.linalg.norm(pos[0] - pos[1])]
            }
        }
        self.assertEqual(assumed, inv_dist)

        # Test against system with different indexing
        mbtr = MBTR([1, 8], k=[2], periodic=False)
        mbtr.create(H2O_2)
        inv_dist_2 = mbtr._inverse_distances
        self.assertEqual(inv_dist, inv_dist_2)

    def test_cosines(self):
        mbtr = MBTR([1, 8], k=[3], periodic=False)
        mbtr.create(H2O)
        angles = mbtr._angles

        # Test against the assumed values.
        assumed = {
            0: {
                1: {
                    0: 2*[math.cos(104/180*math.pi)]
                },
                0: {
                    1: 2*[math.cos(38/180*math.pi)]
                },
            }
        }

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    try:
                        assumed_elem = assumed[i][j][k]
                    except KeyError:
                        assumed_elem = None
                    try:
                        true_elem = angles[i][j][k]
                    except KeyError:
                        true_elem = None
                    if assumed_elem is None:
                        self.assertIsNone(true_elem)
                    else:
                        self.assertEqual(len(assumed_elem), len(true_elem))
                        for i_elem, val_assumed in enumerate(assumed_elem):
                            val_true = true_elem[i_elem]
                            self.assertAlmostEqual(val_assumed, val_true, places=6)

        # Test against system with different indexing
        mbtr = MBTR([1, 8], k=[3], periodic=False)
        mbtr.create(H2O_2)
        angles2 = mbtr._angles
        self.assertEqual(angles, angles2)

    def test_gaussian_distribution(self):
        """Check that the broadening follows gaussian distribution.
        """
        std = 1
        start = -3
        stop = 11
        n = 500
        mbtr = MBTR(
            [1, 8],
            k=[1],
            grid={
                "k1": {
                    "min": start,
                    "max": stop,
                    "sigma": std,
                    "n": n
                }
            },
            periodic=False,
            flatten=False)
        y = mbtr.create(H2O)
        k1_axis = mbtr._axis_k1

        # Find the location of the peaks
        peak1_x = np.searchsorted(k1_axis, 1)
        peak1_y = y[0][0, peak1_x]
        peak2_x = np.searchsorted(k1_axis, 8)
        peak2_y = y[0][1, peak2_x]

        # Check against the analytical value
        gaussian = lambda x, mean, sigma: np.exp(-(x-mean)**2/(2*sigma**2))
        self.assertTrue(np.allclose(peak1_y, 2*gaussian(1, 1, std), rtol=0, atol=0.001))
        self.assertTrue(np.allclose(peak2_y, gaussian(8, 8, std), rtol=0, atol=0.001))

        # Check the integral
        pdf = y[0][0, :]
        # mpl.plot(pdf)
        # mpl.show()
        dx = (stop-start)/(n-1)
        sum_cum = np.sum(0.5*dx*(pdf[:-1]+pdf[1:]))
        exp = 2/(1/math.sqrt(2*math.pi*std**2))
        self.assertTrue(np.allclose(sum_cum, exp, rtol=0, atol=0.001))

    # def test_k1(self):
        # mbtr = MBTR([1, 8], k=[1], periodic=False, flatten=False)
        # desc = mbtr.create(H2O)
        # x1 = mbtr._axis_k1

        # imap = mbtr.index_to_atomic_number
        # smap = {}
        # for index, number in imap.items():
            # smap[index] = numbers_to_symbols(number)

        # Visually check the contents
        # mpl.plot(y)
        # mpl.ylim(0, y.max())
        # mpl.show()

        # mpl.plot(x1, desc[0][0, :], label="{}".format(smap[0]))
        # mpl.plot(x1, desc[0][1, :], linestyle=":", linewidth=3, label="{}".format(smap[1]))
        # mpl.ylabel("$\phi$ (arbitrary units)", size=20)
        # mpl.xlabel("Inverse distance (1/angstrom)", size=20)
        # mpl.legend()
        # mpl.show()

    # def test_k2(self):
        # mbtr = MBTR([1, 8], k=[2], periodic=False, flatten=False)
        # desc = mbtr.create(H2O)

        # x2 = mbtr._axis_k2
        # imap = mbtr.index_to_atomic_number
        # smap = {}
        # for index, number in imap.items():
            # smap[index] = numbers_to_symbols(number)

        # Visually check the contents
        # mpl.plot(x2, desc[1][0, 1, :], label="{}-{}".format(smap[0], smap[1]))
        # mpl.plot(x2, desc[1][1, 0, :], linestyle=":", linewidth=3, label="{}-{}".format(smap[1], smap[0]))
        # mpl.plot(x2, desc[1][1, 1, :], label="{}-{}".format(smap[1], smap[1]))
        # mpl.plot(x2, desc[1][0, 0, :], label="{}-{}".format(smap[0], smap[0]))
        # mpl.ylabel("$\phi$ (arbitrary units)", size=20)
        # mpl.xlabel("Inverse distance (1/angstrom)", size=20)
        # mpl.legend()
        # mpl.show()

        # mbtr = MBTR([1, 8], k=2, periodic=False, flatten=True)
        # desc = mbtr.create(H2O)
        # y = desc.todense().T
        # mpl.plot(y)
        # mpl.show()

    # def test_k3(self):
        # mbtr = MBTR([1, 8], k=3, periodic=False)
        # desc = mbtr.create(H2O)
        # y = desc.todense().T

        # # Visually check the contents
        # mpl.plot(y)
        # mpl.show()

    def test_counts_duplicate(self):
        mbtr = MBTR([1, 8], k=[1], periodic=False)
        mbtr.create(H2O)

        # Check that there are correct number of counts. The counts are
        # calculated only from the original cell that is assumed to be
        # primitive
        self.assertTrue(np.array_equal(mbtr._counts, [2, 1]))

    # def test_distances_duplicate(self):
        # mbtr = MBTR([1, 8], k=[2], periodic=False)
        # mbtr.create(H2O)

        # # Check that there are correct number of inverse distances
        # n_atoms = len(H2O)
        # n_ext_atoms = (1+2*1)**3*n_atoms
        # n_inv_dist_analytic = sum([n_ext_atoms-i for i in range(1, n_atoms+1)])
        # inv_dist = mbtr._inverse_distances

        # n_inv_dist = 0
        # for dict1 in inv_dist.values():
            # for val in dict1.values():
                # n_inv_dist += len(val)

        # self.assertEqual(n_inv_dist_analytic, n_inv_dist)

    # def test_angles_duplicate(self):
        # mbtr = MBTR([1, 8], n_atoms_max=2, k=3, periodic=False)
        # mbtr.create(H2O)

        # Check that there are correct number of angles
        # n_atoms = len(H2O)
        # n_ext_atoms = (1+2*n_copies)**3*n_atoms
        # n_angles_analytic = ?  # Did not have the energy to figure out the correct analytic formula... :)
        # angles = mbtr._angles

        # n_angles = 0
        # for dict1 in angles.values():
            # for dict2 in dict1.values():
                # for val in dict2.values():
                    # n_angles += len(val)

        # self.assertEqual(n_angles_analytic, n_angles)

    def test_symmetries(self):
        """Tests translational and rotational symmetries for a finite system.
        """
        desc = MBTR(
            atomic_numbers=[1, 8],
            k=[1, 2, 3],
            periodic=False,
            grid={
                "k1": {
                    "min": 10,
                    "max": 18,
                    "sigma": 0.1,
                    "n": 100,
                },
                "k2": {
                    "min": 0,
                    "max": 0.7,
                    "sigma": 0.01,
                    "n": 100,
                },
                "k3": {
                    "min": -1.0,
                    "max": 1.0,
                    "sigma": 0.05,
                    "n": 100,
                }
            },
            weighting={
                "k2": {
                    "function": lambda x: np.exp(-0.5*x),
                    "threshold": 1e-3
                },
                "k3": {
                    "function": lambda x: np.exp(-0.5*x),
                    "threshold": 1e-3
                },
            },
            flatten=True
        )

        # Rotational check
        molecule = H2O.copy()
        features = desc.create(molecule)

        for rotation in ['x', 'y', 'z']:
            molecule.rotate(45, rotation)
            rot_features = desc.create(molecule)
            deviation = np.max(np.abs(features - rot_features))
            self.assertTrue(deviation < 1e-6)

        # Translation check
        for translation in [[1.0, 1.0, 1.0], [-5.0, 5.0, -5.0], [1.0, 1.0, -10.0]]:
            molecule.translate(translation)
            trans_features = desc.create(molecule)
            deviation = np.max(np.abs(features - trans_features))
            self.assertTrue(deviation < 1e-6)



    def test_unit_cells(self):
        """Tests if arbitrary unit cells are accepted"""
        desc = MBTR(
            atomic_numbers=[1, 8],
            k=[1, 2, 3],
            periodic=False,
            grid={
                "k1": {
                    "min": 10,
                    "max": 18,
                    "sigma": 0.1,
                    "n": 100,
                },
                "k2": {
                    "min": 0,
                    "max": 0.7,
                    "sigma": 0.01,
                    "n": 100,
                },
                "k3": {
                    "min": -1.0,
                    "max": 1.0,
                    "sigma": 0.05,
                    "n": 100,
                }
            },
            weighting={
                "k2": {
                    "function": lambda x: np.exp(-0.5*x),
                    "threshold": 1e-3
                },
                "k3": {
                    "function": lambda x: np.exp(-0.5*x),
                    "threshold": 1e-3
                },
            },
            flatten=True
        )

        molecule = H2O.copy()

        molecule.set_cell([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
            ],
            )

        nocell = desc.create(molecule)

        molecule.set_pbc(True)
        molecule.set_cell([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0]
            ],
            )

        cubic_cell = desc.create(molecule)

        molecule.set_cell([
        [0.0, 2.0, 2.0],
        [2.0, 0.0, 2.0],
        [2.0, 2.0, 0.0]
            ],
            )

        triclinic_smallcell = desc.create(molecule)

    def test_is_periodic(self):
        """Tests whether periodic images are seen by the descriptor""" 
        desc = MBTR(
            atomic_numbers=[1],
            k=[1, 2, 3],
            periodic=False,
            grid={
                "k1": {
                    "min": 10,
                    "max": 18,
                    "sigma": 0.1,
                    "n": 10,
                },
                "k2": {
                    "min": 0,
                    "max": 0.7,
                    "sigma": 0.01,
                    "n": 10,
                },
                "k3": {
                    "min": -1.0,
                    "max": 1.0,
                    "sigma": 0.05,
                    "n": 10,
                }
            },
            weighting={
                "k2": {
                    "function": lambda x: np.exp(-0.5*x),
                    "threshold": 1e-3
                },
                "k3": {
                    "function": lambda x: np.exp(-0.5*x),
                    "threshold": 1e-3
                },
            },
            flatten=True
        )

        H.set_pbc(False)
        nocell = desc.create(H).toarray()

        H.set_pbc(True)
        H.set_cell([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0]
            ],
            )

        desc = MBTR(
            atomic_numbers=[1],
            k=[1, 2, 3],
            periodic=True,
            grid={
                "k1": {
                    "min": 10,
                    "max": 18,
                    "sigma": 0.1,
                    "n": 10,
                },
                "k2": {
                    "min": 0,
                    "max": 0.7,
                    "sigma": 0.01,
                    "n": 10,
                },
                "k3": {
                    "min": -1.0,
                    "max": 1.0,
                    "sigma": 0.05,
                    "n": 10,
                }
            },
            weighting={
                "k2": {
                    "function": lambda x: np.exp(-0.5*x),
                    "threshold": 1e-3
                },
                "k3": {
                    "function": lambda x: np.exp(-0.5*x),
                    "threshold": 1e-3
                },
            },
            flatten=True
        )

        cubic_cell = desc.create(H).toarray()

        self.assertTrue(np.sum(np.abs(cubic_cell - nocell)) > 0.1)




    def test_periodic_images(self):
        """Tests the periodic images seen by the descriptor
        """
        desc = MBTR(
            atomic_numbers=[1],
            k=[2],
            periodic=True,
            grid={
                "k1": {
                    "min": 10,
                    "max": 18,
                    "sigma": 0.1,
                    "n": 10,
                },
                "k2": {
                    "min": 0,
                    "max": 1.0,
                    "sigma": 0.02,
                    "n": 21,
                },
                "k3": {
                    "min": -1.0,
                    "max": 1.0,
                    "sigma": 0.05,
                    "n": 10,
                }
            },
            weighting={
                "k2": {
                    "function": lambda x: np.exp(-0.7*x),
                    "threshold": 1e-2
                },
                "k3": {
                    "function": lambda x: np.exp(-0.5*x),
                    "threshold": 1e-3
                },
            },
            flatten=True
        )


        #molecule = H2O.copy()
        molecule = H.copy()

        # make periodic
        molecule.set_pbc(True)
        desc.periodic = True

        # cubic
        molecule.set_cell([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0]
            ],
            )

        cubic_cell = desc.create(molecule).toarray()
        suce = molecule * (2,1,1)
        cubic_suce = desc.create(suce).toarray()

        # triclinic
        molecule.set_cell([
        [0.0, 2.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 2.0, 0.0]
            ],
            )

        triclinic_cell = desc.create(molecule).toarray()
        suce = molecule * (2,1,1)
        triclinic_suce = desc.create(suce).toarray()


        #print(cubic_cell[:3] - cubic_suce[:3] / 2.0)
        #print(triclinic_cell[:3] - triclinic_suce[:3] / 2.0)
        #self.assertAlmostEqual(np.sum(cubic_cell[:3] -cubic_suce[:3] / 2.0), 0)
        #self.assertAlmostEqual(np.sum(triclinic_cell[:3] - triclinic_suce[:3] / 2.0), 0)
        

        # bulk structure test, cubic vs. orthorombic vs triclinic
        a1 = bulk('H', 'fcc', a=4.0)
        a2 = bulk('H', 'fcc', a=4.0, orthorhombic=True)
        a3 = bulk('H', 'fcc', a=4.0, cubic=True)


        triclinic_cell = desc.create(a1).toarray() / len(a1)
        orthorhombic_cell = desc.create(a2).toarray() / len(a2)
        cubic_cell = desc.create(a3).toarray() / len(a3)

        #print("bulk cells")
        #print(triclinic_cell)
        #print(orthorhombic_cell)
        #print(cubic_cell)

        #print("maximum deviation ortho", np.max(np.abs(triclinic_cell - cubic_cell)))
        #print("maximum deviation triclinic", np.max(np.abs(orthorhombic_cell - cubic_cell)))

        ids = argrelextrema(orthorhombic_cell.flatten(), np.greater)
        ids = ids[0]
        peak = 7
        self.assertTrue(peak in ids)

        ids = argrelextrema(cubic_cell.flatten(), np.greater)
        ids = ids[0]
        self.assertTrue(peak in ids)

        ids = argrelextrema(triclinic_cell.flatten(), np.greater)
        ids = ids[0]
        self.assertTrue(peak in ids)


        # simple two-peak test
        HH = Atoms(
            cell=[
                [6.0, 0.0, 0.0],
                [0.0, 20.0, 0.0],
                [0.0, 0.0, 20.0]
            ],
            positions=[
                [0.5, 5, 0],
                [4.5, 5, 0],
            ],
            symbols=["H", "H"],
        )

        orthorhombic_cell = desc.create(HH).toarray()

        orthorhombic_cell = orthorhombic_cell.flatten()
        # for local maxima
        ids = argrelextrema(orthorhombic_cell, np.greater)
        ids = ids[0]
        peak1, peak2 = 5 , 10
        self.assertTrue(peak1 in ids)
        self.assertTrue(peak2 in ids)



if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(MBTRTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
