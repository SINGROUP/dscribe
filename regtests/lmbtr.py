import math
import numpy as np
import unittest

from describe.descriptors import LMBTR
from describe.data.element_data import numbers_to_symbols

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


class LMBTRTests(unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """
        with self.assertRaises(ValueError):
            LMBTR(
                atomic_numbers=[1],
                k=0,
                periodic=False,
            )

        with self.assertRaises(ValueError):
            LMBTR(
                atomic_numbers=[1],
                k=[-1, 2],
                periodic=False,
            )

        with self.assertRaises(ValueError):
            LMBTR(
                atomic_numbers=[1],
                k=1,
                periodic=False,
            )

        with self.assertRaises(ValueError):
            LMBTR(
                atomic_numbers=[1],
                k={1, 4},
                periodic=False,
            )
    
    def test_create(self):
        decay_factor = 0.5
        mbtr = LMBTR(
        atomic_numbers=[1, 8],
        k=[1, 2],
        periodic=True,
        grid={
            "k1": {
                "min": 10,
                "max": 18,
                "sigma": 0.1,
                "n": 200,
            },
            "k2": {
                "min": 0,
                "max": 0.7,
                "sigma": 0.01,
                "n": 200,
            },
            "k3": {
                "min": -1.0,
                "max": 1.0,
                "sigma": 0.05,
                "n": 200,
            }
        },
        weighting={
            "k2": {
                "function": lambda x: np.exp(-decay_factor*x),
                "threshold": 1e-3
            },
            "k3": {
                "function": lambda x: np.exp(-decay_factor*x),
                "threshold": 1e-3
            },
        },
        flatten=False)
        
        with self.assertRaises(ValueError):
            desc = lmbtr.create(H2O, list_atom_indices=[3])

        with self.assertRaises(RuntimeError):
            desc = lmbtr.create(H2O)

        with self.assertRaises(RuntimeError):
            desc = lmbtr.create(H2O, list_positions=[[0,0,0]])

        with self.assertRaises(RuntimeError):
            H = Atoms(
                positions=[[0, 0, 0]],
                symbols=["H"],
            )

            desc = lmbtr.create(
                H,
                list_positions=[[0,0,1]],
                scaled_positions=True
            )


    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        # K = 1
        n = 100
        atomic_numbers = [1, 8]
        n_elem = len(atomic_numbers)
        lmbtr = LMBTR(
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
        n_features = lmbtr.get_number_of_features()
        expected = n
        self.assertEqual(n_features, expected)

        # K = 2
        lmbtr = LMBTR(
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
        n_features = lmbtr.get_number_of_features()
        expected = n + n_elem*n
        self.assertEqual(n_features, expected)

        # K = 3
        lmbtr = LMBTR(
            atomic_numbers=atomic_numbers,
            k={3},
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
        n_features = lmbtr.get_number_of_features()
        expected = 1/2*(n_elem)*(n_elem+1)*n
        self.assertEqual(n_features, expected)

    # def test_flatten(self):
        # """Tests the flattening.
        # """
        # # Unflattened
        # desc = LMBTR(n_atoms_max=5, permutation="none", flatten=False)
        # cm = desc.create(H2O)
        # self.assertEqual(cm.shape, (5, 5))

        # # Flattened
        # desc = LMBTR(n_atoms_max=5, permutation="none", flatten=True)
        # cm = desc.create(H2O)
        # self.assertEqual(cm.shape, (25,))

    def test_periodic(self):
        test_sys = Atoms(
            cell=[[5.0, 0.0, 0.0], [0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            positions=[[0, 0, 0]],
            symbols=["H"],
        )
        lmbtr = LMBTR([1], k=[1, 2, 3], weighting="exponential", periodic=True)
        desc = lmbtr.create(test_sys, list_atom_indices=[0])

    def test_inverse_distances(self):
        lmbtr = LMBTR([1, 8], k=[2], periodic=False)
        lmbtr.create(H2O, list_atom_indices=[1])
        inv_dist = lmbtr._inverse_distances

        # Test against the assumed values
        pos = H2O.get_positions()
        assumed = {
            2: [np.inf],
            1: 2*[1/np.linalg.norm(pos[0] - pos[1])]
        }
        self.assertEqual(assumed, inv_dist)

        # Test against system with different indexing
        lmbtr = LMBTR([1, 8], k=[2], periodic=False)
        lmbtr.create(H2O_2, list_atom_indices=[0])
        inv_dist_2 = lmbtr._inverse_distances
        self.assertEqual(inv_dist, inv_dist_2)

    def test_cosines(self):
        lmbtr = LMBTR([1, 8], k=[3], periodic=False)
        lmbtr.create(H2O)
        angles = lmbtr._angles

        # Test against the assumed values.
        assumed = {
            1: {
                1: 2*[math.cos(104/180*math.pi)]
            }
        }
        self.assertEqual(angles, assumed)

        # Test against system with different indexing
        lmbtr = LMBTR([1, 8], k=[3], periodic=False)
        lmbtr.create(H2O_2, list_atom_indices=[0])
        angles2 = lmbtr._angles
        self.assertEqual(angles, angles2)

#    def test_gaussian_distribution(self):
#        """Check that the broadening follows gaussian distribution.
#        """
#        std = 1
#        start = -3
#        stop = 11
#        n = 500
#        lmbtr = LMBTR(
#            [1, 8],
#            k=[1],
#            grid={
#                "k1": {
#                    "min": start,
#                    "max": stop,
#                    "sigma": std,
#                    "n": n
#                }
#            },
#            periodic=False,
#            flatten=False)
#        y = lmbtr.create(H2O)
#        k1_axis = lmbtr._axis_k1
#
#        # Find the location of the peaks
#        peak1_x = np.searchsorted(k1_axis, 1)
#        peak1_y = y[0][0, peak1_x]
#        peak2_x = np.searchsorted(k1_axis, 8)
#        peak2_y = y[0][1, peak2_x]
#
#        # Check against the analytical value
#        gaussian = lambda x, mean, sigma: np.exp(-(x-mean)**2/(2*sigma**2))
#        self.assertTrue(np.allclose(peak1_y, 2*gaussian(1, 1, std), rtol=0, atol=0.001))
#        self.assertTrue(np.allclose(peak2_y, gaussian(8, 8, std), rtol=0, atol=0.001))
#
#        # Check the integral
#        pdf = y[0][0, :]
#        # mpl.plot(pdf)
#        # mpl.show()
#        dx = (stop-start)/(n-1)
#        sum_cum = np.sum(0.5*dx*(pdf[:-1]+pdf[1:]))
#        exp = 2/(1/math.sqrt(2*math.pi*std**2))
#        self.assertTrue(np.allclose(sum_cum, exp, rtol=0, atol=0.001))

    # def test_k1(self):
        # lmbtr = LMBTR([1, 8], k=[1], periodic=False, flatten=False)
        # desc = lmbtr.create(H2O)
        # x1 = lmbtr._axis_k1

        # imap = lmbtr.index_to_atomic_number
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
        # lmbtr = LMBTR([1, 8], k=[2], periodic=False, flatten=False)
        # desc = lmbtr.create(H2O)

        # x2 = lmbtr._axis_k2
        # imap = lmbtr.index_to_atomic_number
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

        # lmbtr = LMBTR([1, 8], k=2, periodic=False, flatten=True)
        # desc = lmbtr.create(H2O)
        # y = desc.todense().T
        # mpl.plot(y)
        # mpl.show()

    # def test_k3(self):
        # lmbtr = LMBTR([1, 8], k=3, periodic=False)
        # desc = lmbtr.create(H2O)
        # y = desc.todense().T

        # # Visually check the contents
        # mpl.plot(y)
        # mpl.show()

    # def test_distances_duplicate(self):
        # lmbtr = LMBTR([1, 8], k=[2], periodic=False)
        # lmbtr.create(H2O)

        # # Check that there are correct number of inverse distances
        # n_atoms = len(H2O)
        # n_ext_atoms = (1+2*1)**3*n_atoms
        # n_inv_dist_analytic = sum([n_ext_atoms-i for i in range(1, n_atoms+1)])
        # inv_dist = lmbtr._inverse_distances

        # n_inv_dist = 0
        # for dict1 in inv_dist.values():
            # for val in dict1.values():
                # n_inv_dist += len(val)

        # self.assertEqual(n_inv_dist_analytic, n_inv_dist)

    # def test_angles_duplicate(self):
        # lmbtr = LMBTR([1, 8], n_atoms_max=2, k=3, periodic=False)
        # lmbtr.create(H2O)

        # Check that there are correct number of angles
        # n_atoms = len(H2O)
        # n_ext_atoms = (1+2*n_copies)**3*n_atoms
        # n_angles_analytic = ?  # Did not have the energy to figure out the correct analytic formula... :)
        # angles = lmbtr._angles

        # n_angles = 0
        # for dict1 in angles.values():
            # for dict2 in dict1.values():
                # for val in dict2.values():
                    # n_angles += len(val)

        # self.assertEqual(n_angles_analytic, n_angles)

    def test_symmetries(self):
        """Tests translational and rotational symmetries for a finite system.
        """
        desc = LMBTR(
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
        features = desc.create(molecule, list_atom_indices=[0])

        for rotation in ['x', 'y', 'z']:
            molecule.rotate(45, rotation)
            rot_features = desc.create(molecule, list_atom_indices=[0])
            deviation = np.max(np.abs(features - rot_features))
            self.assertTrue(deviation < 1e-6)

        # Translation check
        for translation in [[1.0, 1.0, 1.0], [-5.0, 5.0, -5.0], [1.0, 1.0, -10.0]]:
            molecule.translate(translation)
            trans_features = desc.create(molecule, list_atom_indices=[0])
            deviation = np.max(np.abs(features - trans_features))
            self.assertTrue(deviation < 1e-6)


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(LMBTRTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
