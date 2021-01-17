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
import itertools
from pprint import pprint as pp
import numpy as np

import scipy
import scipy.sparse
from scipy.integrate import tplquad
from scipy.linalg import sqrtm

from dscribe.descriptors import SOAP

from ase import Atoms
from ase.build import molecule
from ase.visualize import view
from ase.build import bulk

def _print_diff(num, ana):
    print("numerical")
    print(num)
    print("numerical derivatives shape")
    print(num.shape)
    print("analytical")
    print(ana)
    print("analytical der shape", ana.shape)
    print("num - ana")
    print(num - ana)
    return


H2 = Atoms(
    cell=[
        [15.0, 0.0, 0.0],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 15.0]
    ],
    positions=[
        [-0.5, 0, 0],
        [0.5, 0, 0],

    ],
    symbols=["H", "H"],
)
H = Atoms(
    positions=[[0.5, 0.5, 0.5]],
    symbols=["H"],
)
H2O = molecule("H2O")


class SoapDerivativeTests(unittest.TestCase):

    def test_interface(self):
        """Test the derivative interface.
        """
        soap = SOAP(
            species=[1, 8],
            rcut=3,
            nmax=2,
            lmax=0,
            sparse=False,
        )
        positions = [[0.0, 0.0, 0.0]]

        # Test that trying to do sparse output raises an exception
        soap.sparse = True
        with self.assertRaises(ValueError):
            soap.derivatives(H2, positions=positions, method="numerical")

        # Test that trying to get analytical derivatives with averaged output
        # raises an exception
        soap.sparse = False
        soap.average = "inner"
        with self.assertRaises(ValueError):
            soap.derivatives(H2, positions=positions, method="analytical")
        soap.average = "off"

        # Test that trying to get analytical derivatives with polynomial basis
        # raises an exception, but the numerical ones work.
        soap_poly = SOAP(
            species=[1, 8],
            rcut=3,
            nmax=2,
            lmax=0,
            rbf="polynomial",
            sparse=False,
        )
        with self.assertRaises(ValueError):
            soap_poly.derivatives(H2, positions=positions, method="analytical")
        soap_poly.derivatives(H2, positions=positions, method="numerical")

        # Test include
        with self.assertRaises(ValueError):
            soap.derivatives(H2O, positions=positions, include=[])
        with self.assertRaises(ValueError):
            soap.derivatives(H2O, positions=positions, include=[3])
        s = soap.derivatives(H2O, positions=positions, include=[2, 0, 0], return_descriptor=False)
        self.assertEqual(s.shape[1], 2)

        # Test exclude
        s = soap.derivatives(H2O, positions=positions, exclude=[], return_descriptor=False)
        self.assertEqual(s.shape[1], 3)
        with self.assertRaises(ValueError):
            soap.derivatives(H2O, positions=positions, exclude=[3])
        s = soap.derivatives(H2O, positions=positions, exclude=[0, 2, 2], return_descriptor=False)
        self.assertEqual(s.shape[1], 1)

    def test_parallel_dense(self):
        """Tests creating dense output parallelly.
        """
        desc = SOAP(
            species=[6, 7, 8],
            rcut=5,
            nmax=3,
            lmax=3,
            sigma=1,
            periodic=False,
            crossover=True,
            average="off",
            sparse=False,
        )
        n_features = desc.get_number_of_features()

        # Perhaps most common scenario: multiple systems with same atoms in
        # different locations, sames centers and indices, dense numpy output.
        samples = [molecule("CO"), molecule("CO")]
        centers = [[0], [0]]
        der, des = desc.derivatives(
            system=samples,
            positions=centers,
            n_jobs=2,
        )
        self.assertTrue(der.shape == (2, 1, 2, 3, n_features))
        self.assertTrue(des.shape == (2, 1, n_features))
        assumed_der = np.empty((2, 1, 2, 3, n_features))
        assumed_des = np.empty((2, 1, n_features))
        assumed_der[0, :], assumed_des[0, :]  = desc.derivatives(samples[0], centers[0], n_jobs=1)
        assumed_der[1, :], assumed_des[1, :]  = desc.derivatives(samples[1], centers[0], n_jobs=1)
        self.assertTrue(np.allclose(assumed_der, der))

        # Now with centers given in cartesian positions.
        centers = [[[0, 1, 2]], [[2, 1, 0]]]
        der, des = desc.derivatives(
            system=samples,
            positions=centers,
            n_jobs=2,
        )
        self.assertTrue(der.shape == (2, 1, 2, 3, n_features))
        self.assertTrue(des.shape == (2, 1, n_features))
        assumed_der = np.empty((2, 1, 2, 3, n_features))
        assumed_des = np.empty((2, 1, n_features))
        assumed_der[0, :], assumed_des[0, :]  = desc.derivatives(samples[0], centers[0], n_jobs=1)
        assumed_der[1, :], assumed_des[1, :]  = desc.derivatives(samples[1], centers[1], n_jobs=1)
        self.assertTrue(np.allclose(assumed_der, der))

        # Includes
        includes = [[0], [0]]
        der, des = desc.derivatives(
            system=samples,
            include=includes,
            n_jobs=2,
        )
        self.assertTrue(der.shape == (2, 2, 1, 3, n_features))
        self.assertTrue(des.shape == (2, 2, n_features))
        assumed_der = np.empty((2, 2, 1, 3, n_features))
        assumed_des = np.empty((2, 2, n_features))
        assumed_der[0, :], assumed_des[0, :]  = desc.derivatives(samples[0], include=includes[0], n_jobs=1)
        assumed_der[1, :], assumed_des[1, :]  = desc.derivatives(samples[1], include=includes[1], n_jobs=1)
        self.assertTrue(np.allclose(assumed_der, der))

        # Excludes
        excludes = [[0], [0]]
        der, des = desc.derivatives(
            system=samples,
            exclude=excludes,
            n_jobs=2,
        )
        self.assertTrue(der.shape == (2, 2, 1, 3, n_features))
        self.assertTrue(des.shape == (2, 2, n_features))
        assumed_der = np.empty((2, 2, 1, 3, n_features))
        assumed_des = np.empty((2, 2, n_features))
        assumed_der[0, :], assumed_des[0, :]  = desc.derivatives(samples[0], exclude=excludes[0], n_jobs=1)
        assumed_der[1, :], assumed_des[1, :]  = desc.derivatives(samples[1], exclude=excludes[1], n_jobs=1)
        self.assertTrue(np.allclose(assumed_der, der))

        # Test averaged output
        desc.average = "inner"
        positions=[[0], [0, 1]]
        der, des = desc.derivatives(
            system=samples,
            positions=positions,
            n_jobs=2,
        )
        self.assertTrue(der.shape == (2, 1, 2, 3, n_features))
        self.assertTrue(des.shape == (2, 1, n_features))
        assumed_der = np.empty((2, 1, 2, 3, n_features))
        assumed_des = np.empty((2, 1, n_features))
        assumed_der[0, :], assumed_des[0, :]  = desc.derivatives(samples[0], positions=positions[0], n_jobs=1)
        assumed_der[1, :], assumed_des[1, :]  = desc.derivatives(samples[1], positions=positions[1], n_jobs=1)
        self.assertTrue(np.allclose(assumed_der, der))

        # Variable size list output, as the systems have a different size
        desc.average = "off"
        samples = [molecule("CO"), molecule("NO2")]
        der, des = desc.derivatives(
            system=samples,
            n_jobs=2,
        )
        self.assertTrue(isinstance(der, list))
        self.assertTrue(der[0].shape == (2, 2, 3, n_features))
        self.assertTrue(der[1].shape == (3, 3, 3, n_features))
        assumed_der0, assumed_des0  = desc.derivatives(samples[0], n_jobs=1)
        assumed_der1, assumed_des1  = desc.derivatives(samples[1], n_jobs=1)
        self.assertTrue(np.allclose(assumed_der0, der[0]))
        self.assertTrue(np.allclose(assumed_der1, der[1]))

    def test_numerical(self):
        """Test numerical values against a naive python implementation.
        """
        # Elaborate test system with multiple species, non-cubic cell, and
        # close-by atoms.
        a = 1
        system = Atoms(
            symbols=["C", "H", "O"],
            cell=[
                [0, a, a],
                [a, 0, a],
                [a, a, 0]
            ],
            scaled_positions=[[0,0,0], [1/3,1/3,1/3], [2/3,2/3,2/3]],
            pbc=[True, True, True],
        )*(3, 3, 3)
        # view(system)

        # Two centers: one in the middle, one on the edge.
        centers = [np.sum(system.get_cell(), axis=0)/2, [0, 0, 0]]

        h = 0.0001
        n_atoms = len(system)
        n_comp = 3

        # The maximum erorr depends on how big the system is. With a small
        # system the error is smaller for non-periodic systems than the
        # corresponding error when periodicity is turned on. The errors become
        # equal (~1e-5) when the size of the system is increased.
        for periodic in [False, True]:
            # for rbf in ["gto"]:
            for rbf in ["gto", "polynomial"]:
                # for average in ["off"]:
                for average in ["off", "outer", "inner"]:
                    soap = SOAP(
                        species=[1, 8, 6],
                        rcut=3,
                        nmax=4,
                        lmax=4,
                        rbf=rbf,
                        sparse=False,
                        average=average,
                        crossover=True,
                        periodic=periodic,
                    )
                    n_features = soap.get_number_of_features()
                    n_centers = 1 if average != "off" else len(centers)
                    derivatives_python = np.zeros((n_centers, n_atoms, n_comp, n_features))
                    d0 = soap.create(system, centers)
                    coeffs = [-1.0/2.0, 1.0/2.0]
                    deltas = [-1.0, 1.0]
                    for i_atom in range(len(system)):
                        for i_center in range(n_centers):
                            for i_comp in range(3):
                                for i_stencil in range(2):
                                    if average == "off":
                                        i_cent = [centers[i_center]]
                                    else:
                                        i_cent = centers
                                    system_disturbed = system.copy()
                                    i_pos = system_disturbed.get_positions()
                                    i_pos[i_atom, i_comp] += h*deltas[i_stencil]
                                    system_disturbed.set_positions(i_pos)
                                    d1 = soap.create(system_disturbed, i_cent)
                                    derivatives_python[i_center, i_atom, i_comp, :] += coeffs[i_stencil]*d1[0, :]/h

                    # Calculate with central finite difference implemented in C++.
                    # Try both cartesian centers and indices.
                    for c in [centers]:
                    # for c in [centers, None]:
                        derivatives_cpp, d_cpp = soap.derivatives(system, positions=c, method="numerical")

                        # Test that descriptor values are correct
                        self.assertTrue(np.allclose(d0, d_cpp, atol=1e-6))

                        # Compare values
                        # print(np.abs(derivatives_python).max())
                        # print(derivatives_python[0,1,:,:])
                        # print(derivatives_cpp[0,0,:,:])
                        print(np.abs(derivatives_cpp - derivatives_python).max())
                        self.assertTrue(np.allclose(derivatives_python, derivatives_cpp, atol=1e-5))

    def test_periodic(self):
        """Tests that periodicity works correctly for both numerical and
        analytical code.
        """
        a = 1
        system = Atoms(
            symbols=["C"],
            cell=[
                [0, a, a],
                [a, 0, a],
                [a, a, 0]
            ],
            scaled_positions=[[0,0,0]],
            pbc=[True, True, True],
        )

        # Calculate as periodic first
        soap = SOAP(
            species=[6],
            rcut=3,
            nmax=1,
            lmax=1,
            rbf="gto",
            sparse=False,
            average="off",
            crossover=True,
            periodic=True,
        )
        a_normal = soap.create(system, positions=[0])
        a_der, a_des = soap.derivatives(system, positions=[0], include=[0], method="numerical")

        # Extend the system and calculate it as non-periodic
        n_copies = 13
        ext_system = system*(n_copies, n_copies, n_copies)
        ext_system.set_pbc(False)
        middle = ext_system.get_center_of_mass()
        positions = ext_system.get_positions()
        dist = np.linalg.norm(positions - middle, axis=1)
        idx = np.argmin(dist)
        soap.periodic = False
        b_normal = soap.create(ext_system, positions=[idx])
        b_der, b_des = soap.derivatives(ext_system, positions=[idx], include=[idx], method="numerical")

        # Assert that regularly calculated descriptors are the same
        # print(np.abs(a_normal - b_normal).max())
        self.assertTrue(np.allclose(a_normal, b_normal))

        # Assert that descriptors returned with the derivatives are the same
        # print(np.abs(a_des - b_des).max())
        self.assertTrue(np.allclose(a_des, b_des))

        # print(np.abs(a_der - b_der).max())
        self.assertTrue(np.allclose(a_der, b_der))


class SoapDerivativeComparisonTests(unittest.TestCase):

    def test_no_crossover(self):
        """Tests the analytical soap derivatives implementation without
        crossover against the numerical implementation.
        """
        soap = SOAP(
            species=["H", "O"],
            rcut=3,
            nmax=9,
            lmax=9,
            sparse=False,
            crossover=False,
        )
        positions = H2O.get_positions()
        derivatives_cpp, d_num = soap.derivatives(H2O, positions=positions, method="numerical")
        derivatives_anal, d_anal = soap.derivatives(H2O, positions=positions, method="analytical")
        self.assertTrue(np.allclose(derivatives_cpp, derivatives_anal, rtol=1e-6, atol=1e-6))

    def test_crossover(self):
        """Tests the analytical soap derivatives implementation with crossover
        against the numerical implementation.
        """
        soap = SOAP(
            species=["H", "O"],
            rcut=3,
            nmax=9,
            lmax=9,
            sparse=False,
            crossover=True,
        )
        positions = H2O.get_positions()
        derivatives_cpp, d_num = soap.derivatives(H2O, positions=positions, method="numerical")
        derivatives_anal, d_anal = soap.derivatives(H2O, positions=positions, method="analytical")
        self.assertTrue(np.allclose(derivatives_cpp, derivatives_anal, rtol=1e-6, atol=1e-6))

    def test_descriptor_output(self):
        """Tests that the descriptor is output correctly both for numerical and
        analytical version.
        """
        soap = SOAP(
            species=["H", "O"],
            rcut=3,
            nmax=3,
            lmax=3,
            crossover=True,
        )
        positions = H2O.get_positions()
        _, d_num = soap.derivatives(H2O, positions=positions, method="numerical")
        _, d_anal = soap.derivatives(H2O, positions=positions, method="analytical")
        d = soap.create(H2O, positions=positions)
        self.assertTrue(np.allclose(d, d_num))
        self.assertTrue(np.allclose(d, d_anal))


if __name__ == '__main__':
    SoapDerivativeTests().test_numerical()
    # SoapDerivativeTests().test_periodic()
    # suites = []
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(SoapDerivativeTests))
    # suites.append(unittest.TestLoader().loadTestsFromTestCase(SoapDerivativeComparisonTests))
    # alltests = unittest.TestSuite(suites)
    # result = unittest.TextTestRunner(verbosity=0).run(alltests)
