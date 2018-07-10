from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import math
import numpy as np
import unittest

from describe.descriptors import EwaldMatrix

from ase import Atoms

import scipy.constants

from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core.structure import Structure


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
rcut = 40
gcut = 20


class EwaldMatrixTests(unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """
        with self.assertRaises(ValueError):
            EwaldMatrix(n_atoms_max=5, permutation="unknown")
        with self.assertRaises(ValueError):
            EwaldMatrix(n_atoms_max=-1)

    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        desc = EwaldMatrix(n_atoms_max=5, permutation="none", flatten=False)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 25)

    def test_flatten(self):
        """Tests the flattening.
        """
        # Unflattened
        desc = EwaldMatrix(n_atoms_max=5, permutation="none", flatten=False)
        matrix = desc.create(H2O, rcut=rcut, gcut=gcut)
        self.assertEqual(matrix.shape, (5, 5))

        # Flattened
        desc = EwaldMatrix(n_atoms_max=5, permutation="none", flatten=True)
        matrix = desc.create(H2O, rcut=rcut, gcut=gcut)
        self.assertEqual(matrix.shape, (25,))

    def test_a_independence(self):
        """Tests that the matrix elements are independent of the screening
        parameter a used in the Ewald summation. Notice that the real space
        cutoff and reciprocal space cutoff have to be sufficiently large for
        this to be true, as a controls the width of the Gaussian charge
        distribution.
        """
        rcut = 40
        gcut = 30
        prev_array = None
        for i, a in enumerate([0.1, 0.5, 1, 2, 3]):
            desc = EwaldMatrix(n_atoms_max=5, permutation="none", flatten=False)
            matrix = desc.create(H2O, a=a, rcut=rcut, gcut=gcut)

            if i > 0:
                self.assertTrue(np.allclose(prev_array, matrix, atol=0.001, rtol=0))
            prev_array = matrix

    def test_electrostatics(self):
        """Tests that the results are consistent with the electrostatic
        interpretation. Each matrix [i, j] element should correspond to the
        Coulomb energy of a system consisting of the pair of atoms i, j.
        """
        system = H2O
        n_atoms = len(system)
        a = 0.5
        desc = EwaldMatrix(n_atoms_max=3, permutation="none", flatten=False)
        matrix = desc.create(system, a=a, rcut=rcut, gcut=gcut)

        # Converts unit of q*q/r into eV
        conversion = 1e10 * scipy.constants.e / (4 * math.pi * scipy.constants.epsilon_0)
        matrix *= conversion

        # The value in each matrix element should correspond to the Coulomb
        # energy of a system with with only those atoms. On the diagonal you
        # only have one atom in the system, on the off-diagonals you have two
        # different atoms and their combined Coulomb energy.
        positions = system.get_positions()
        atomic_num = system.get_atomic_numbers()
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    pos = [positions[i]]
                    sym = [atomic_num[i]]
                else:
                    pos = [positions[i], positions[j]]
                    sym = [atomic_num[i], atomic_num[j]]

                i_sys = Atoms(
                    cell=system.get_cell(),
                    positions=pos,
                    symbols=sym,
                    pbc=True,
                )

                structure = Structure(
                    lattice=i_sys.get_cell(),
                    species=i_sys.get_atomic_numbers(),
                    coords=i_sys.get_scaled_positions(),
                )
                structure.add_oxidation_state_by_site(i_sys.get_atomic_numbers())
                ewald = EwaldSummation(structure, eta=a, real_space_cut=rcut, recip_space_cut=gcut)
                energy = ewald.total_energy

                # Check that the energy given by the pymatgen implementation is
                # the same as given by the descriptor
                self.assertTrue(np.allclose(matrix[i, j], energy, atol=0.00001, rtol=0))

    # def test_features(self):
        # """Tests that the correct features are present in the desciptor.
        # """
        # desc = CoulombMatrix(n_atoms_max=5, permutation="none", flatten=False)
        # cm = desc.create(H2O)

        # # Test against assumed values
        # q = H2O.get_atomic_numbers()
        # p = H2O.get_positions()
        # norm = np.linalg.norm
        # assumed = np.array(
            # [
                # [0.5*q[0]**2.4,              q[0]*q[1]/(norm(p[0]-p[1])),  q[0]*q[2]/(norm(p[0]-p[2]))],
                # [q[1]*q[0]/(norm(p[1]-p[0])), 0.5*q[1]**2.4,               q[1]*q[2]/(norm(p[1]-p[2]))],
                # [q[2]*q[0]/(norm(p[2]-p[0])), q[2]*q[1]/(norm(p[2]-p[1])), 0.5*q[2]**2.4],
            # ]
        # )
        # zeros = np.zeros((5, 5))
        # zeros[:3, :3] = assumed
        # assumed = zeros

        # self.assertTrue(np.array_equal(cm, assumed))


# class SortedCoulombMatrixTests(unittest.TestCase):

    # def test_constructor(self):
        # """Tests different valid and invalid constructor values.
        # """

    # def test_number_of_features(self):
        # """Tests that the reported number of features is correct.
        # """
        # desc = CoulombMatrix(n_atoms_max=5, permutation="sorted_l2", flatten=False)
        # n_features = desc.get_number_of_features()
        # self.assertEqual(n_features, 25)

    # def test_flatten(self):
        # """Tests the flattening.
        # """
        # # Unflattened
        # desc = CoulombMatrix(n_atoms_max=5, permutation="sorted_l2", flatten=False)
        # cm = desc.create(H2O)
        # self.assertEqual(cm.shape, (5, 5))

        # # Flattened
        # desc = CoulombMatrix(n_atoms_max=5, permutation="sorted_l2", flatten=True)
        # cm = desc.create(H2O)
        # self.assertEqual(cm.shape, (25,))

    # def test_features(self):
        # """Tests that the correct features are present in the desciptor.
        # """
        # desc = CoulombMatrix(n_atoms_max=5, permutation="sorted_l2", flatten=False)
        # cm = desc.create(H2O)

        # lens = np.linalg.norm(cm, axis=0)
        # old_len = lens[0]
        # for length in lens[1:]:
            # self.assertTrue(length <= old_len)
            # old_len = length


# class CoulombMatrixEigenSpectrumTests(unittest.TestCase):

    # def test_constructor(self):
        # """Tests different valid and invalid constructor values.
        # """

    # def test_number_of_features(self):
        # """Tests that the reported number of features is correct.
        # """
        # desc = CoulombMatrix(n_atoms_max=5, permutation="eigenspectrum")
        # n_features = desc.get_number_of_features()
        # self.assertEqual(n_features, 5)

    # def test_features(self):
        # """Tests that the correct features are present in the desciptor.
        # """
        # desc = CoulombMatrix(n_atoms_max=5, permutation="eigenspectrum")
        # cm = desc.create(H2O)

        # self.assertEqual(len(cm), 5)

        # # Test that eigenvalues are in decreasing order when looking at absolute value
        # prev_eig = float("Inf")
        # for eigenvalue in cm[:len(H2O)]:
            # self.assertTrue(abs(eigenvalue) <= abs(prev_eig))
            # prev_eig = eigenvalue

        # # Test that array is zero-padded
        # self.assertTrue(np.array_equal(cm[len(H2O):], [0, 0]))


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(EwaldMatrixTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(SortedCoulombMatrixTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(CoulombMatrixEigenSpectrumTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
