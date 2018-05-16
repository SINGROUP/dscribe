"""
Defines a set of regressions tests that should be run succesfully after all
major modification to the code.
"""
import sys
import math
import numpy as np
import unittest
import time

from describe.descriptors import MBTR
from describe.descriptors import CoulombMatrix
from describe.descriptors import SortedCoulombMatrix
from describe.descriptors import SineMatrix
from describe.descriptors import SortedSineMatrix
from describe.core import System
from describe.data.element_data import numbers_to_symbols

import matplotlib.pyplot as mpl

from ase import Atoms
from ase.lattice.cubic import SimpleCubicFactory

H2O = System(
    cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
    positions=[[0, 0, 0], [0.95, 0, 0], [0.95*(1+math.cos(76/180*math.pi)), 0.95*math.sin(76/180*math.pi), 0.0]],
    symbols=["H", "O", "H"],
)
H2O.set_initial_charges(H2O.numbers)

H2O_2 = System(
    cell=[[5.0, 0.0, 0], [0, 5, 0], [0, 0, 5.0]],
    positions=[[0.95, 0, 0], [0, 0, 0], [0.95*(1+math.cos(76/180*math.pi)), 0.95*math.sin(76/180*math.pi), 0.0]],
    symbols=["O", "H", "H"],
)

NaCl_prim = System(
    cell=[
        [
            0.0,
            2.8201,
            2.8201
        ],
        [
            2.8201,
            0.0,
            2.8201
        ],
        [
            2.8201,
            2.8201,
            0.0
        ]
    ],
    scaled_positions=[[0.5, 0.5, 0.5], [0, 0, 0]],
    symbols=["Na", "Cl"],
)

NaCl_conv = System(
    cell=[
        [
            5.6402,
            0.0,
            0.0
        ],
        [
            0.0,
            5.6402,
            0.0
        ],
        [
            0.0,
            0.0,
            5.6402
        ]
    ],
    scaled_positions=[
        [
            0.0,
            0.5,
            0.0
        ],
        [
            0.0,
            0.5,
            0.5
        ],
        [
            0.0,
            0.0,
            0.5
        ],
        [
            0.0,
            0.0,
            0.0
        ],
        [
            0.5,
            0.5,
            0.5
        ],
        [
            0.5,
            0.5,
            0.0
        ],
        [
            0.5,
            0.0,
            0.0
        ],
        [
            0.5,
            0.0,
            0.5
        ]],
    symbols=["Na", "Cl", "Na", "Cl", "Na", "Cl", "Na", "Cl"],
)


class GeometryTests(unittest.TestCase):

    def test_distances(self):
        """Tests that the periodicity is taken into account when calculating
        distances.
        """
        system = System(
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            symbols=["H", "H"],
            cell=[
                [5, 5, 0],
                [0, -5, -5],
                [5, 0, 5]
            ],
        )
        disp = system.get_displacement_tensor()

        # For a non-periodic system, periodicity is not taken into account even
        # if cell is defined.
        assumed = np.array([
            [[0.0, 0.0, 0.0], [-5, 0, 0]],
            [[5, 0, 0], [0.0, 0.0, 0.0]]])
        self.assertTrue(np.allclose(assumed, disp))

        # For a periodic system, the nearest copy should be considered when
        # comparing distances to neighbors or to self
        system.set_pbc([True, True, True])
        disp = system.get_displacement_tensor()
        assumed = np.array([
            [[5.0, 5.0, 0.0], [-5, 0, 0]],
            [[5, 0, 0], [5.0, 5.0, 0.0]]])
        self.assertTrue(np.allclose(assumed, disp))

    def test_transformations(self):
        """Test that coordinates are correctly transformed from scaled to
        cartesian and back again.
        """
        system = System(
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            symbols=["H", "H"],
            cell=[
                [5, 5, 0],
                [0, -5, -5],
                [5, 0, 5]
            ],
        )

        orig = np.array([[2, 1.45, -4.8]])
        scal = system.to_scaled(orig)
        cart = system.to_cartesian(scal)
        self.assertTrue(np.allclose(orig, cart))


class GaussianTests(unittest.TestCase):

    def test_cdf(self):
        """Test that the implementation of the gaussian value through the
        cumulative distribution function works as expected.
        """
        from scipy.stats import norm
        from scipy.special import erf

        start = -5
        stop = 5
        n_points = 9
        centers = np.array([0])
        sigma = 1

        area_cum = []
        area_pdf = []

        # Calculate errors for dfferent number of points
        for n_points in range(2, 10):

            axis = np.linspace(start, stop, n_points)

            # Calculate with cumulative function
            dx = (stop - start)/(n_points-1)
            x = np.linspace(start-dx/2, stop+dx/2, n_points+1)
            pos = x[np.newaxis, :] - centers[:, np.newaxis]
            y = 1/2*(1 + erf(pos/(sigma*np.sqrt(2))))
            f = np.sum(y, axis=0)
            f_rolled = np.roll(f, -1)
            pdf_cum = (f_rolled - f)[0:-1]/dx

            # Calculate with probability function
            dist2 = axis[np.newaxis, :] - centers[:, np.newaxis]
            dist2 *= dist2
            f = np.sum(np.exp(-dist2/(2*sigma**2)), axis=0)
            f *= 1/math.sqrt(2*sigma**2*math.pi)
            pdf_pdf = f

            true_axis = np.linspace(start, stop, 200)
            pdf_true = norm.pdf(true_axis, centers[0], sigma) # + norm.pdf(true_axis, centers[1], sigma)

            # Calculate differences
            sum_cum = np.sum(0.5*dx*(pdf_cum[:-1]+pdf_cum[1:]))
            sum_pdf = np.sum(0.5*dx*(pdf_pdf[:-1]+pdf_pdf[1:]))
            area_cum.append(sum_cum)
            area_pdf.append(sum_pdf)

            # mpl.plot(axis, pdf_pdf, linestyle=":", linewidth=3, color="r")
            # mpl.plot(axis, pdf_cum, linewidth=1, color="g")
            # mpl.plot(true_axis, pdf_true, linestyle="--", color="b")
            # mpl.show()

        mpl.plot(area_cum, linestyle=":", linewidth=3, color="r")
        mpl.plot(area_pdf, linewidth=1, color="g")
        # mpl.plot(true_axis, pdf_true, linestyle="--", color="b")
        mpl.show()


class ASETests(unittest.TestCase):

    def test_atoms_to_system(self):
        """Tests that an ASE Atoms is succesfully converted to a System object.
        """
        class NaClFactory(SimpleCubicFactory):
            "A factory for creating NaCl (B1, Rocksalt) lattices."

            bravais_basis = [[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0], [0, 0.5, 0.5],
                            [0.5, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 0],
                            [0.5, 0.5, 0.5]]
            element_basis = (0, 1, 1, 0, 1, 0, 0, 1)

        nacl = NaClFactory()(symbol=["Na", "Cl"], latticeconstant=5.6402)
        system = System.from_atoms(nacl)

        self.assertTrue(np.array_equal(nacl.get_positions(), system.get_positions()))
        self.assertTrue(np.array_equal(nacl.get_initial_charges(), system.get_initial_charges()))
        self.assertTrue(np.array_equal(nacl.get_atomic_numbers(), system.get_atomic_numbers()))
        self.assertTrue(np.array_equal(nacl.get_chemical_symbols(), system.get_chemical_symbols()))
        self.assertTrue(np.array_equal(nacl.get_cell(), system.get_cell()))
        self.assertTrue(np.array_equal(nacl.get_pbc(), system.get_pbc()))
        self.assertTrue(np.array_equal(nacl.get_scaled_positions(), system.get_scaled_positions()))


class CoulombMatrixTests(unittest.TestCase):

    def test_matrix(self):
        desc = CoulombMatrix(n_atoms_max=5, flatten=False)
        cm = desc.create(H2O)

        # Test against assumed values
        q = H2O.get_initial_charges()
        p = H2O.get_positions()
        norm = np.linalg.norm
        assumed = np.array(
            [
                [0.5*q[0]**2.4,              q[0]*q[1]/(norm(p[0]-p[1])),  q[0]*q[2]/(norm(p[0]-p[2]))],
                [q[1]*q[0]/(norm(p[1]-p[0])), 0.5*q[1]**2.4,               q[1]*q[2]/(norm(p[1]-p[2]))],
                [q[2]*q[0]/(norm(p[2]-p[0])), q[2]*q[1]/(norm(p[2]-p[1])), 0.5*q[2]**2.4],
            ]
        )
        zeros = np.zeros((5, 5))
        zeros[:3, :3] = assumed
        assumed = zeros

        self.assertTrue(np.array_equal(cm, assumed))


class SortedCoulombMatrixTests(unittest.TestCase):

    def test_matrix(self):
        desc = SortedCoulombMatrix(n_atoms_max=5, flatten=False)
        cm = desc.create(H2O)

        lens = np.linalg.norm(cm, axis=0)
        old_len = lens[0]
        for length in lens[1:]:
            self.assertTrue(length <= old_len)
            old_len = length


class SineMatrixTests(unittest.TestCase):

    def test_matrix(self):
        # Create simple toy system
        test_sys = System(
            cell=[[1, 0.0, 0.0], [1, 1, 0.0], [0.0, 0.0, 1.0]],
            positions=[[0, 0, 0], [2, 1, 1]],
            symbols=["H", "H"],
        )
        test_sys.charges = np.array([1, 1])

        desc = SineMatrix(n_atoms_max=5, flatten=False)

        # Create a graph of the interaction in a 2D slice
        size = 100
        x_min = 0.0
        x_max = 3
        y_min = 0.0
        y_max = 3
        x_axis = np.linspace(x_min, x_max, size)
        y_axis = np.linspace(y_min, y_max, size)
        interaction = np.empty((size, size))
        for i, x in enumerate(x_axis):
            for j, y in enumerate(y_axis):
                temp_sys = System(
                    cell=[[1, 0.0, 0.0], [1, 1, 0.0], [0.0, 0.0, 1.0]],
                    positions=[[0, 0, 0], [x, y, 0]],
                    symbols=["H", "H"],
                )
                temp_sys.set_initial_charges(np.array([1, 1]))
                value = desc.create(temp_sys)
                interaction[i, j] = value[0, 1]

        mpl.imshow(interaction, cmap='RdBu', vmin=0, vmax=5,
                extent=[x_min, x_max, y_min, y_max],
                interpolation='nearest', origin='lower')
        mpl.colorbar()
        mpl.show()

        # Test against assumed values
        q = test_sys.get_initial_charges()
        p = test_sys.get_positions()
        cell = test_sys.get_cell()
        cell_inv = test_sys.get_reciprocal_cell()
        sin = np.sin
        pi = np.pi
        dot = np.dot
        norm = np.linalg.norm
        assumed = np.array(

            [
                [0.5*q[0]**2.4, q[0]*q[1]/(norm(dot(cell, sin(pi*dot(p[0]-p[1], cell_inv))**2)))],
                [q[0]*q[1]/(norm(dot(cell, sin(pi*dot(p[1]-p[0], cell_inv))**2))), 0.5*q[1]**2.4],
            ]
        )
        zeros = np.zeros((5, 5))
        zeros[:2, :2] = assumed
        assumed = zeros

        sm = desc.create(test_sys)

        self.assertTrue(np.array_equal(sm, assumed))


class SortedSineMatrixTests(unittest.TestCase):

    def test_matrix(self):
        desc = SortedSineMatrix(n_atoms_max=5, flatten=False)
        cm = desc.create(H2O)

        lens = np.linalg.norm(cm, axis=0)
        old_len = lens[0]
        for length in lens[1:]:
            self.assertTrue(length <= old_len)
            old_len = length


class MBTRTests(unittest.TestCase):

    def test_invalid_parameters(self):
        """Test that invalid parameters raise the correct exception.
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

    def test_flattening(self):
        """Test that the flattened version equals the unflattened one.
        """

    def test_number_of_features(self):
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
        test_sys = System(
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
        # print(angles)
        # print(angles2)
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

    def test_k1(self):
        mbtr = MBTR([1, 8], k=[1], periodic=False, flatten=False)
        desc = mbtr.create(H2O)
        x1 = mbtr._axis_k1

        imap = mbtr.index_to_atomic_number
        smap = {}
        for index, number in imap.items():
            smap[index] = numbers_to_symbols(number)

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

    def test_k2(self):
        mbtr = MBTR([1, 8], k=[2], periodic=False, flatten=False)
        desc = mbtr.create(H2O)

        x2 = mbtr._axis_k2
        imap = mbtr.index_to_atomic_number
        smap = {}
        for index, number in imap.items():
            smap[index] = numbers_to_symbols(number)

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

    # def test_counts_duplicate(self):
        # mbtr = MBTR([1, 8], k=1, periodic=False)
        # mbtr.create(H2O)

        # Check that there are correct number of counts. The counts are
        # calculated only from the original cell that is assumed to be
        # primitive
        # self.assertTrue(np.array_equal(mbtr._counts, [2, 1]))

    # def test_distances_duplicate(self):
        # mbtr = MBTR([1, 8], k=2, periodic=False)
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

    # def test_haoyan_nacl(self):
        # # Test periodic NaCl crystal
        # cutoff_x = 12
        # cutoff_y = 0.01
        # rate = -math.log(cutoff_y)/cutoff_x
        # # rate = 0.5

        # mbtr = MBTR(
            # [11, 17],
            # n_atoms_max=8,
            # k=3,
            # periodic=True,
            # # grid={
                # # "k3": [0, np.pi, np.pi/200, 0.07]
            # # },
            # weighting={
                # "k2": {
                    # "function": lambda x: np.exp(-rate*x),
                    # "threshold": 1e-2
                # },
                # "k3": {
                    # "function": lambda x: np.exp(-rate*x),
                    # "threshold": 1e-2
                # },
            # },
            # flatten=False)

        # start = time.time()
        # # desc = mbtr.create(NaCl_prim)
        # desc = mbtr.create(NaCl_conv)
        # end = time.time()
        # print("DONE: {}".format(end-start))

        # x = mbtr._axis_k3/np.pi*180
        # imap = mbtr.index_to_atomic_number
        # smap = {}
        # for index, number in imap.items():
            # smap[index] = numbers_to_symbols(number)

        # mpl.rcParams['text.usetex'] = True
        # mpl.rcParams['font.family'] = 'serif'
        # mpl.rcParams['font.serif'] = ['cm']
        # mpl.rcParams['xtick.labelsize'] = 18
        # mpl.rcParams['ytick.labelsize'] = 18
        # mpl.rcParams['legend.fontsize'] = 18

        # print(desc[0].shape)
        # print(desc[1].shape)
        # print(desc[2].shape)

        # mpl.plot(x, desc[2][0, 0, 0, :], label="NaNaNa, ClClCl".format(smap[0], smap[0], smap[0]), color="blue")
        # mpl.plot(x, desc[2][0, 0, 1, :], label="NaNaCl, NaClCl".format(smap[0], smap[0], smap[1]), color="orange")
        # mpl.plot(x, desc[2][1, 0, 1, :], label="NaClNa, ClNaCl".format(smap[1], smap[0], smap[1]), color="green")
        # mpl.ylabel("$\phi$ (arbitrary units)", size=25)
        # mpl.xlabel("angle (degree)", size=25)
        # mpl.title("The exponentially weighted angle distribution in NaCl crystal.", size=30)
        # mpl.legend()
        # mpl.show()


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ASETests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(GeometryTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(GaussianTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(MBTRTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(CoulombMatrixTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SortedCoulombMatrixTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SineMatrixTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SortedSineMatrixTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
