# -*- coding: utf-8 -*-
import math
import numpy as np
import unittest

from dscribe.descriptors import EwaldSumMatrix

from ase import Atoms
from ase.build import bulk

import scipy.constants
import scipy.sparse

from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core.structure import Structure

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
rcut = 30
gcut = 20


class EwaldSumMatrixTests(TestBaseClass, unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """
        with self.assertRaises(ValueError):
            EwaldSumMatrix(n_atoms_max=5, permutation="unknown")
        with self.assertRaises(ValueError):
            EwaldSumMatrix(n_atoms_max=-1)

    def test_create(self):
        """Tests different valid and invalid create values.
        """
        with self.assertRaises(ValueError):
            desc = EwaldSumMatrix(n_atoms_max=5)
            desc.create(H2O, rcut=10)
        with self.assertRaises(ValueError):
            desc = EwaldSumMatrix(n_atoms_max=5)
            desc.create(H2O, gcut=10)

        # Providing a only is valid
        desc = EwaldSumMatrix(n_atoms_max=5)
        desc.create(H2O, a=0.5)

        # Providing no parameters is valid
        desc = EwaldSumMatrix(n_atoms_max=5)
        desc.create(H2O)

    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        desc = EwaldSumMatrix(n_atoms_max=5, permutation="none", flatten=False)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 25)

    def test_flatten(self):
        """Tests the flattening.
        """
        # Unflattened
        desc = EwaldSumMatrix(n_atoms_max=5, permutation="none", flatten=False)
        matrix = desc.create(H2O)
        self.assertEqual(matrix.shape, (5, 5))

        # Flattened
        desc = EwaldSumMatrix(n_atoms_max=5, permutation="none", flatten=True)
        matrix = desc.create(H2O)
        self.assertEqual(matrix.shape, (1, 25))

    def test_sparse(self):
        """Tests the sparse matrix creation.
        """
        # Dense
        desc = EwaldSumMatrix(n_atoms_max=5, permutation="none", flatten=True, sparse=False)
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == np.ndarray)

        # Sparse
        desc = EwaldSumMatrix(n_atoms_max=5, permutation="none", flatten=True, sparse=True)
        vec = desc.create(H2O)
        self.assertTrue(type(vec) == scipy.sparse.coo_matrix)

    def test_parallel_dense(self):
        """Tests creating dense output parallelly.
        """
        samples = [bulk("NaCl", "rocksalt", a=5.64), bulk('Cu', 'fcc', a=3.6)]
        desc = EwaldSumMatrix(n_atoms_max=5, permutation="none", flatten=True, sparse=False)
        n_features = desc.get_number_of_features()

        # Test multiple systems, serial job
        output = desc.create(
            system=samples,
            n_jobs=1,
        )
        assumed = np.empty((2, n_features))
        assumed[0, :] = desc.create(samples[0])
        assumed[1, :] = desc.create(samples[1])
        self.assertTrue(np.allclose(output, assumed))

        # Test multiple systems, parallel job
        output = desc.create(
            system=samples,
            n_jobs=2,
        )
        assumed = np.empty((2, n_features))
        assumed[0, :] = desc.create(samples[0])
        assumed[1, :] = desc.create(samples[1])
        self.assertTrue(np.allclose(output, assumed))

        # Non-flattened output
        desc = EwaldSumMatrix(n_atoms_max=5, permutation="none", flatten=False, sparse=False)
        output = desc.create(
            system=samples,
            n_jobs=2,
        )
        assumed = np.empty((2, 5, 5))
        assumed[0] = desc.create(samples[0])
        assumed[1] = desc.create(samples[1])
        self.assertTrue(np.allclose(np.array(output), assumed))

    def test_parallel_sparse(self):
        """Tests creating sparse output parallelly.
        """
        # Test indices
        samples = [bulk("NaCl", "rocksalt", a=5.64), bulk('Cu', 'fcc', a=3.6)]
        desc = EwaldSumMatrix(n_atoms_max=5, permutation="none", flatten=True, sparse=True)
        n_features = desc.get_number_of_features()

        # Test multiple systems, serial job
        output = desc.create(
            system=samples,
            n_jobs=1,
        ).toarray()
        assumed = np.empty((2, n_features))
        assumed[0, :] = desc.create(samples[0]).toarray()
        assumed[1, :] = desc.create(samples[1]).toarray()
        self.assertTrue(np.allclose(output, assumed))

        # Test multiple systems, parallel job
        output = desc.create(
            system=samples,
            n_jobs=2,
        ).toarray()
        assumed = np.empty((2, n_features))
        assumed[0, :] = desc.create(samples[0]).toarray()
        assumed[1, :] = desc.create(samples[1]).toarray()
        self.assertTrue(np.allclose(output, assumed))

        # Non-flattened output
        desc = EwaldSumMatrix(n_atoms_max=5, permutation="none", flatten=False, sparse=True)
        output = [x.toarray() for x in desc.create(
            system=samples,
            n_jobs=2,
        )]
        assumed = np.empty((2, 5, 5))
        assumed[0] = desc.create(samples[0]).toarray()
        assumed[1] = desc.create(samples[1]).toarray()
        self.assertTrue(np.allclose(np.array(output), assumed))

    def test_a_independence(self):
        """Tests that the matrix elements are independent of the screening
        parameter 'a' used in the Ewald summation. Notice that the real space
        cutoff and reciprocal space cutoff have to be sufficiently large for
        this to be true, as 'a' controls the width of the Gaussian charge
        distribution.
        """
        rcut = 40
        gcut = 30
        prev_array = None
        for i, a in enumerate([0.1, 0.5, 1, 2, 3]):
            desc = EwaldSumMatrix(n_atoms_max=5, permutation="none", flatten=False)
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
        desc = EwaldSumMatrix(n_atoms_max=3, permutation="none", flatten=False)

        # The Ewald matrix contains the electrostatic interaction between atoms
        # i and j. Here we construct the total electrostatic energy for a
        # system consisting of atoms i and j.
        matrix = desc.create(system, a=a, rcut=rcut, gcut=gcut)
        energy_matrix = np.zeros(matrix.shape)
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    energy_matrix[i, j] = matrix[i, j]
                else:
                    energy_matrix[i, j] = matrix[i, j] + matrix[i, i] + matrix[j, j]

        # Converts unit of q*q/r into eV
        conversion = 1e10 * scipy.constants.e / (4 * math.pi * scipy.constants.epsilon_0)
        energy_matrix *= conversion

        # The value in each matrix element should correspond to the Coulomb
        # energy of a system with with only those atoms. Here the energies from
        # the Ewald matrix are compared against the Ewald energy calculated
        # with pymatgen.
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
                self.assertTrue(np.allclose(energy_matrix[i, j], energy, atol=0.00001, rtol=0))

    def test_electrostatics_automatic(self):
        """Tests that the results are consistent with the electrostatic
        interpretation when using automatically determined parameters. Each
        matrix [i, j] element should correspond to the Coulomb energy of a
        system consisting of the pair of atoms i, j.
        """
        system = H2O
        n_atoms = len(system)
        desc = EwaldSumMatrix(n_atoms_max=3, permutation="none", flatten=False)

        # The Ewald matrix contains the electrostatic interaction between atoms i
        # and j. Here we construct the total electrostatic energy from this matrix.
        accuracy = 1e-6
        matrix = desc.create(system, accuracy=accuracy)
        energy_matrix = np.zeros(matrix.shape)
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    energy_matrix[i, j] = matrix[i, j]
                else:
                    energy_matrix[i, j] = matrix[i, j] + matrix[i, i] + matrix[j, j]

        # Converts unit of q*q/r into eV
        conversion = 1e10 * scipy.constants.e / (4 * math.pi * scipy.constants.epsilon_0)
        energy_matrix *= conversion

        # The value in each matrix element should correspond to the Coulomb
        # energy of a system with with only those atoms. Here the energies from
        # the Ewald matrix are compared against the Ewald energy calculated
        # with pymatgen.
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

                # Pymatgen uses a different definition for the accuracy: there
                # accuracy is determined as the number of significant digits.
                ewald = EwaldSummation(structure, acc_factor=-np.log(accuracy))
                energy = ewald.total_energy

                # Check that the energy given by the pymatgen implementation is
                # the same as given by the descriptor
                self.assertTrue(np.allclose(energy_matrix[i, j], energy, atol=0.00001, rtol=0))

    def test_unit_cells(self):
        """Tests if arbitrary unit cells are accepted
        """
        desc = EwaldSumMatrix(n_atoms_max=3, permutation="none", flatten=False)
        molecule = H2O.copy()

        # A system without cell should produce an error
        molecule.set_cell([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        with self.assertRaises(ValueError):
            nocell = desc.create(molecule,  a=0.5, rcut=rcut, gcut=gcut)

        # Large cell
        molecule.set_pbc(True)
        molecule.set_cell([
            [20.0, 0.0, 0.0],
            [0.0, 30.0, 0.0],
            [0.0, 0.0, 40.0]
        ])
        largecell = desc.create(molecule,  a=0.5, rcut=rcut, gcut=gcut)

        # Cubic cell
        molecule.set_cell([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        cubic_cell = desc.create(molecule,  a=0.5, rcut=rcut, gcut=gcut)

        # Triclinic cell
        molecule.set_cell([
            [0.0, 2.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 0.0]
        ])
        triclinic_smallcell = desc.create(molecule,  a=0.5, rcut=rcut, gcut=gcut)

    def test_symmetries(self):
        """Tests the symmetries of the descriptor.
        """
        def create(system):
            desc = EwaldSumMatrix(n_atoms_max=3, permutation="sorted_l2", flatten=True)
            return desc.create(system)

        # Rotational
        self.assertTrue(self.is_rotationally_symmetric(create))

        # # Translational
        self.assertTrue(self.is_translationally_symmetric(create))

        # # Permutational
        self.assertTrue(self.is_permutation_symmetric(create))


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(EwaldSumMatrixTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
