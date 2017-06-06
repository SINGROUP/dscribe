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
from describe.descriptors import SineMatrix
from describe.core import System
from describe.data.element_data import numbers_to_symbols

import matplotlib.pyplot as mpl

from ase import Atoms
from ase.lattice.cubic import SimpleCubicFactory

H2O = System(
    lattice=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
    positions=[[0, 0, 0], [0.95, 0, 0], [0.95*(1+math.cos(76/180*math.pi)), 0.95*math.sin(76/180*math.pi), 0.0]],
    species=["H", "O", "H"],
    coords_are_cartesian=True
)
H2O.charges = H2O.numbers

H2O_2 = System(
    lattice=[[5.0, 0.0, 0], [0, 5, 0], [0, 0, 5.0]],
    positions=[[0.95, 0, 0], [0, 0, 0], [0.95*(1+math.cos(76/180*math.pi)), 0.95*math.sin(76/180*math.pi), 0.0]],
    species=["O", "H", "H"],
    coords_are_cartesian=True
)

NaCl_prim = System(
    lattice=[
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
    positions=[[0.5, 0.5, 0.5], [0, 0, 0]],
    species=["Na", "Cl"],
    coords_are_cartesian=False
)

NaCl_conv = System(
    lattice=[
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
    positions=[
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
    species=["Na", "Cl", "Na", "Cl", "Na", "Cl", "Na", "Cl"],
    coords_are_cartesian=False
)


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
        system = System.fromatoms(nacl)

        self.assertTrue(np.array_equal(nacl.get_positions(), system.cartesian_pos))
        self.assertTrue(np.array_equal(nacl.get_initial_charges(), system.charges))
        self.assertTrue(np.array_equal(nacl.get_atomic_numbers(), system.numbers))
        self.assertTrue(np.array_equal(nacl.get_chemical_symbols(), system.symbols))
        self.assertTrue(np.array_equal(nacl.get_cell(), system.lattice._matrix))
        self.assertTrue(np.array_equal(nacl.get_pbc(), system.periodicity))
        self.assertTrue(np.array_equal(nacl.get_scaled_positions(), system.relative_pos))


class CoulombMatrixTests(unittest.TestCase):

    def test_matrix(self):
        desc = CoulombMatrix(n_atoms_max=5, flatten=False)
        cm = desc.create(H2O)

        # Test against assumed values
        q = H2O.charges
        p = H2O.cartesian_pos
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


class SineMatrixTests(unittest.TestCase):

    def test_matrix(self):
        # Create simple toy system
        test_sys = System(
            lattice=[[1, 0.0, 0.0], [1, 1, 0.0], [0.0, 0.0, 1.0]],
            positions=[[0, 0, 0], [2, 1, 1]],
            species=["H", "H"],
            coords_are_cartesian=True
        )
        test_sys.charges = np.array([1, 1])

        desc = SineMatrix(n_atoms_max=5, flatten=False)

        # Create a graph of the interaction in a 2D slice
        # size = 100
        # x_min = 0.0
        # x_max = 3
        # y_min = 0.0
        # y_max = 3
        # x_axis = np.linspace(x_min, x_max, size)
        # y_axis = np.linspace(y_min, y_max, size)
        # interaction = np.empty((size, size))
        # for i, x in enumerate(x_axis):
            # for j, y in enumerate(y_axis):
                # temp_sys = System(
                    # lattice=[[1, 0.0, 0.0], [1, 1, 0.0], [0.0, 0.0, 1.0]],
                    # positions=[[0, 0, 0], [x, y, 0]],
                    # species=["H", "H"],
                    # coords_are_cartesian=True
                # )
                # temp_sys.charges = np.array([1, 1])
                # value = desc.create(temp_sys)
                # interaction[i, j] = value[0, 1]

        # mpl.imshow(interaction, cmap='RdBu', vmin=0, vmax=5,
                # extent=[x_min, x_max, y_min, y_max],
                # interpolation='nearest', origin='lower')
        # mpl.colorbar()
        # mpl.show()

        # Test against assumed values
        q = test_sys.charges
        p = test_sys.cartesian_pos
        cell = test_sys.lattice._matrix
        cell_inv = test_sys.lattice.inv_matrix
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


class MBTRTests(unittest.TestCase):

    def test_counts(self):
        mbtr = MBTR([1, 8], k=1, periodic=False)
        mbtr.create(H2O)
        counts = mbtr._counts

        # Test against the assumed values
        self.assertTrue(np.array_equal(counts, np.array([2, 1])))

        # Test against system with different indexing
        mbtr = MBTR([1, 8], k=1, periodic=False)
        mbtr.create(H2O_2)
        counts2 = mbtr._counts
        self.assertTrue(np.array_equal(counts, counts2))

    def test_inverse_distances(self):
        mbtr = MBTR([1, 8], k=2, periodic=False)
        mbtr.create(H2O)
        inv_dist = mbtr._inverse_distances

        # Test against the assumed values
        pos = H2O.cartesian_pos
        assumed = {
            0: {
                0: [1/np.linalg.norm(pos[0] - pos[2])],
                1: 2*[1/np.linalg.norm(pos[0] - pos[1])]
            }
        }
        self.assertEqual(assumed, inv_dist)

        # Test against system with different indexing
        mbtr = MBTR([1, 8], k=2, periodic=False)
        mbtr.create(H2O_2)
        inv_dist_2 = mbtr._inverse_distances
        self.assertEqual(inv_dist, inv_dist_2)

    def test_cosines(self):
        mbtr = MBTR([1, 8], k=3, periodic=False)
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
        mbtr = MBTR([1, 8], k=3, periodic=False)
        mbtr.create(H2O_2)
        angles2 = mbtr._angles
        # print(angles)
        # print(angles2)
        self.assertEqual(angles, angles2)

    def test_gaussian_distribution(self):
        """Check that the broadening follows gaussian distribution.
        """
        std = 1
        mbtr = MBTR(
            [1, 8],
            k=1,
            grid={
                "k1": [0, 9, 0.05, std]
            },
            periodic=False,
            flatten=False)
        y = mbtr.create(H2O)
        k1_axis = mbtr._axis_k1

        # Find the location of the peaks
        peak1_x = np.where(k1_axis == 1)
        peak1_y = y[0][0, peak1_x]
        peak2_x = np.where(k1_axis == 8)
        peak2_y = y[0][1, peak2_x]

        # Check against the analytical value
        gaussian = lambda x, mean, sigma: 1/math.sqrt(2*math.pi*sigma**2)*np.exp(-(x-mean)**2/(2*sigma**2))
        self.assertEqual(peak1_y, 2*gaussian(1, 1, std))
        self.assertEqual(peak2_y, gaussian(8, 8, std))

    def test_k1(self):
        mbtr = MBTR([1, 8], k=1, periodic=False)
        desc = mbtr.create(H2O)
        y = desc.todense().T

        # Visually check the contents
        # mpl.plot(y)
        # mpl.show()

    def test_k2(self):
        mbtr = MBTR([1, 8], k=2, periodic=False)
        desc = mbtr.create(H2O)
        y = desc.todense().T

        # Visually check the contents
        # mpl.plot(y)
        # mpl.show()

    def test_k3(self):
        mbtr = MBTR([1, 8], k=3, periodic=False)
        desc = mbtr.create(H2O)
        y = desc.todense().T

        # Visually check the contents
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

    # def test_sparse(self):
        # """Test the sparse matrix format.
        # """
        # cutoff_x = 12
        # cutoff_y = 0.01
        # mbtr = MBTR([11, 17], n_atoms_max=8, k=3, cutoff=(cutoff_x, cutoff_y), n_copies=1, flatten=True)
        # desc = mbtr.create(NaCl_prim)
        # print(desc.shape)


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ASETests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(MBTRTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(CoulombMatrixTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SineMatrixTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
