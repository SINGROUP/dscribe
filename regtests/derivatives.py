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
        der, des = desc.derivatives(
            system=samples,
            positions=[[0], [0]],
            n_jobs=1,
        )
        self.assertTrue(der.shape == (2, 1, 2, 3, n_features))
        self.assertTrue(des.shape == (2, 1, n_features))

        # Now with centers given in cartesian positions.
        der, des = desc.derivatives(
            system=samples,
            positions=[[[0, 1, 2]], [[0, 1, 2]]],
            n_jobs=1,
        )
        self.assertTrue(der.shape == (2, 1, 2, 3, n_features))
        self.assertTrue(des.shape == (2, 1, n_features))

        # Now with all atoms as centers, but derivatives with respect to only
        # one atom.
        der, des = desc.derivatives(
            system=samples,
            include=[[0], [0]],
            n_jobs=1,
        )
        self.assertTrue(der.shape == (2, 2, 1, 3, n_features))
        self.assertTrue(des.shape == (2, 2, n_features))

        # Multiple systems, serial job
        # der, des = desc.derivatives(
            # system=samples,
            # positions=[[0], [0, 1]],
            # n_jobs=1,
        # )
        # assumed = np.empty((3, n_features))
        # assumed[0, :] = desc.create(samples[0], [0])
        # assumed[1, :] = desc.create(samples[1], [0])
        # assumed[2, :] = desc.create(samples[1], [1])
        # self.assertTrue(np.allclose(output, assumed))

        # # Test when position given as indices
        # output = desc.create(
            # system=samples,
            # positions=[[0], [0, 1]],
            # n_jobs=2,
        # )
        # assumed = np.empty((3, n_features))
        # assumed[0, :] = desc.create(samples[0], [0])
        # assumed[1, :] = desc.create(samples[1], [0])
        # assumed[2, :] = desc.create(samples[1], [1])
        # self.assertTrue(np.allclose(output, assumed))

        # # Test with no positions specified
        # output = desc.create(
            # system=samples,
            # positions=[None, None],
            # n_jobs=2,
        # )
        # assumed = np.empty((2+3, n_features))
        # assumed[0, :] = desc.create(samples[0], [0])
        # assumed[1, :] = desc.create(samples[0], [1])
        # assumed[2, :] = desc.create(samples[1], [0])
        # assumed[3, :] = desc.create(samples[1], [1])
        # assumed[4, :] = desc.create(samples[1], [2])
        # self.assertTrue(np.allclose(output, assumed))

        # # Test with cartesian positions
        # output = desc.create(
            # system=samples,
            # positions=[[[0, 0, 0], [1, 2, 0]], [[1, 2, 0]]],
            # n_jobs=2,
        # )
        # assumed = np.empty((2+1, n_features))
        # assumed[0, :] = desc.create(samples[0], [[0, 0, 0]])
        # assumed[1, :] = desc.create(samples[0], [[1, 2, 0]])
        # assumed[2, :] = desc.create(samples[1], [[1, 2, 0]])
        # self.assertTrue(np.allclose(output, assumed))

        # # Test averaged output
        # desc.average = "outer"
        # output = desc.create(
            # system=samples,
            # positions=[[0], [0, 1]],
            # n_jobs=2,
        # )
        # assumed = np.empty((2, n_features))
        # assumed[0, :] = desc.create(samples[0], [0])
        # assumed[1, :] = 1/2*(desc.create(samples[1], [0]) + desc.create(samples[1], [1]))
        # self.assertTrue(np.allclose(output, assumed))

    def test_numerical(self):
        """Test numerical values.
        """
        # Compare against naive python implementation.
        soap = SOAP(
            species=[1, 8],
            rcut=3,
            nmax=2,
            lmax=0,
            sparse=False,
        )

        # Calculate with forward finite difference
        # positions = [[0.0, 0.0, 0.0]]
        positions = [[0.0, 0.0, 0.0], [0.12, -0.5, 0.3]]
        h = 0.0000001
        n_atoms = len(H2O)
        n_pos = len(positions)
        n_features = soap.get_number_of_features()
        n_comp = 3
        derivatives_python = np.zeros((n_pos, n_atoms, n_comp, n_features))
        for i_pos in range(len(positions)):
            center = positions[i_pos]
            d0 = soap.create(H2O, [center])
            for i_atom in range(len(H2O)):
                for i_comp in range(3):
                    H2O_disturbed = H2O.copy()
                    translation = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
                    translation[i_atom, i_comp] = h
                    H2O_disturbed.translate(translation)
                    d1 = soap.create(H2O_disturbed, [center])
                    ds = (-d0 + d1) / h
                    derivatives_python[i_pos, i_atom, i_comp, :] = ds[0, :]

        # Calculate with central finite difference implemented in C++
        derivatives_cpp, d1 = soap.derivatives(H2O, positions=positions, method="numerical")

        # Test that desriptor values are correct
        d2 = soap.create(H2O, positions=positions)
        self.assertTrue(np.allclose(d1, d2, atol=1e-5))

        # Compare values
        self.assertTrue(np.allclose(derivatives_python, derivatives_cpp, atol=1e-5))

        #print("derivatives numerical cpp")
        #pp(derivatives_cpp)
        #print("descriptor calculated together with derivatives")
        #print(d1)

    def test_analytical(self):
        """Tests if the analytical soap derivatives run
        """
        soap = SOAP(
            species=[1],
            rcut=3,
            nmax=2,
            lmax=0,
            sparse=False,
        )
        soap_arr = soap.create(H2, positions = [[0.0, 0.0, 0.0], ])

        #derivatives = soap.derivatives(H2, positions =[0 ] , method = "analytical", include=None, exclude=None)
        derivatives = soap.derivatives(H2, positions=[[0.0,0.0,0.0]], method="analytical")

        #print("analytical derivatives")
        #print(derivatives)
        #print(derivatives[0].shape)

        derivatives = soap.derivatives(H2, 
            positions =[[0.0, 0.0, 0.0], [-0.5, 0, 0], [0.5, 0, 0], ] , 
            method = "analytical", include=None, exclude=None)

        #print("3 centers analytical derivatives")
        #print(derivatives)
        #print(derivatives[0].shape)
        #pp(derivatives)

class SoapDerivativeComparisonTests(unittest.TestCase):
    def test_analytical_vs_numerical(self):
        """Tests the analytical soap derivatives implementation against the numerical cpp implementation
        """
        soap = SOAP(
            species=[1],
            rcut=3,
            nmax=2,
            lmax=0,
            sparse=False,
        )
        positions = [[0.0, 0.0, 0.0],]
        # Calculate with central finite difference implemented in C++
        derivatives_cpp, d_num = soap.derivatives(H2, positions=positions, method="numerical")
        derivatives_anal, d_anal = soap.derivatives(H2, positions=positions, method="analytical")
        diff = derivatives_cpp - derivatives_anal

        #_print_diff(derivatives_cpp, derivatives_anal)
        self.assertTrue(np.allclose(derivatives_cpp, derivatives_anal))


    def test_analytical_crossover(self):
        """Tests the analytical soap derivatives implementation with crossover against the numerical cpp implementation
        """
        soap = SOAP(
            species=[1],
            rcut=3,
            nmax=2,
            lmax=2,
            sparse=False,
            crossover = True
        )
        positions = [[0.0, 0.0, 0.0],]
        # Calculate with central finite difference implemented in C++
        derivatives_cpp, d_num = soap.derivatives(H2, positions=positions, method="numerical")
        derivatives_anal, d_anal = soap.derivatives(H2, positions=positions, method="analytical")

        diff = derivatives_cpp - derivatives_anal

        #_print_diff(derivatives_cpp, derivatives_anal)
        self.assertTrue(np.allclose(derivatives_cpp, derivatives_anal))


    def test_analytical_vs_numerical_L1(self):
        """Tests the analytical soap derivatives implementation against the numerical cpp implementation
        """
        soap = SOAP(
            species=[1],
            rcut=3,
            nmax=2,
            lmax=1,
            sparse=False,
            crossover=False,
        )
        positions = [[0.0, 0.0, 0.0],]
        # Calculate with central finite difference implemented in C++
        derivatives_cpp, d_num = soap.derivatives(H2, positions=positions, method="numerical")
        derivatives_anal, d_anal = soap.derivatives(H2, positions=positions, method="analytical")
        
        #_print_diff(derivatives_cpp, derivatives_anal)
        self.assertTrue(np.allclose(derivatives_cpp, derivatives_anal))

    def test_analytical_vs_numerical_L9(self):
        """Tests the analytical soap derivatives implementation against the numerical cpp implementation
        """
        soap = SOAP(
            species=[1],
            rcut=3,
            nmax=9,
            lmax=9,
            sparse=False,
            crossover=False,
        )
        positions = [[0.1, 0.2, 0.3],]
        # Calculate with central finite difference implemented in C++
        derivatives_cpp, d_num = soap.derivatives(H2, positions=positions, method="numerical")
        derivatives_anal, d_anal = soap.derivatives(H2, positions=positions, method="analytical")
        #_print_diff(derivatives_cpp, derivatives_anal)
        self.assertTrue(np.allclose(derivatives_cpp, derivatives_anal))

    def test_descriptor_output(self):
        """Tests the analytical soap descriptor implementation against the numerical cpp implementation
        """
        soap = SOAP(
            species=[1],
            rcut=3,
            nmax=3,
            lmax=19,
            sparse=False,
            crossover=False,
        )
        positions = [[0.1, 0.2, 0.3],]
        # Calculate with central finite difference implemented in C++
        derivatives_cpp, d_num = soap.derivatives(H2, positions=positions, method="numerical")
        derivatives_anal, d_anal = soap.derivatives(H2, positions=positions, method="analytical")
        print("compare descriptors")
        #_print_diff(d_num, d_anal)
        self.assertTrue(np.allclose(d_num, d_anal))

    def test_analytical_vs_numerical_multispecies(self):
        """Tests the analytical soap derivatives implementation against the numerical cpp implementation
        """
        soap = SOAP(
            species=[1, 8],
            rcut=3,
            nmax=2,
            lmax=0,
            sparse=False,
        )
        positions = [[0.0, 0.0, 0.0],]
        # Calculate with central finite difference implemented in C++
        derivatives_cpp, d_num = soap.derivatives(H2O, positions=positions, method="numerical")
        derivatives_anal, d_anal = soap.derivatives(H2O, positions=positions, method="analytical")
        diff = derivatives_cpp - derivatives_anal

        _print_diff(derivatives_cpp, derivatives_anal)
        self.assertTrue(np.allclose(derivatives_cpp, derivatives_anal))


        


if __name__ == '__main__':
    suites = []
    #suites.append(unittest.TestLoader().loadTestsFromTestCase(SoapDerivativeTests))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SoapDerivativeComparisonTests))
    #SoapDerivativeComparisonTests().test_analytical_vs_numerical_L9()
    #SoapDerivativeComparisonTests().test_analytical_crossover()
    #SoapDerivativeComparisonTests().test_analytical_vs_numerical()
    #SoapDerivativeComparisonTests().test_descriptor_output()
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
