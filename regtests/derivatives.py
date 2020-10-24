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

import numpy as np

import scipy
import scipy.sparse
from scipy.integrate import tplquad
from scipy.linalg import sqrtm

from dscribe.descriptors import SOAP

from ase import Atoms
from ase.build import molecule
from ase.visualize import view

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
            soap.derivatives_single(H2, positions=positions, method="numerical")

        # Test that trying to get analytical derivatives with averaged output
        # raises an exception
        soap.sparse = False
        soap.average = "inner"
        with self.assertRaises(ValueError):
            soap.derivatives_single(H2, positions=positions, method="analytical")
        soap.average = "off"

        # Test include
        with self.assertRaises(ValueError):
            soap.derivatives_single(H2O, positions=positions, include=[])
        with self.assertRaises(ValueError):
            soap.derivatives_single(H2O, positions=positions, include=[3])
        s = soap.derivatives_single(H2O, positions=positions, include=[2, 0, 0], return_descriptor=False)
        self.assertEqual(s.shape[1], 2)

        # Test exclude
        s = soap.derivatives_single(H2O, positions=positions, exclude=[], return_descriptor=False)
        self.assertEqual(s.shape[1], 3)
        with self.assertRaises(ValueError):
            soap.derivatives_single(H2O, positions=positions, exclude=[3])
        s = soap.derivatives_single(H2O, positions=positions, exclude=[0, 2, 2], return_descriptor=False)
        self.assertEqual(s.shape[1], 1)

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
        derivatives_cpp, d1 = soap.derivatives_single(H2O, positions=positions, method="numerical")

        # Test that desriptor values are correct
        d2 = soap.create(H2O, positions=positions)
        self.assertTrue(np.allclose(d1, d2, atol=1e-5))

        # Compare values
        self.assertTrue(np.allclose(derivatives_python, derivatives_cpp, atol=1e-5))

    # def test_analytical(self):
        # """Tests if the analytical soap derivatives run
        # """
        # soap = SOAP(
            # species=[1],
            # rcut=3,
            # nmax=2
            # lmax=0,
            # sparse=False,
        # )
        # # soap = a.create(H2, positions = [[0.0, 0.0, 0.0], ])
        # # soap_disturbed = a.create(H2_disturbed, positions = [[0.0, 0.0, 0.0], ])
        # # soap_diff = soap - soap_disturbed

        # # print("disturbed soap")
        # # print(soap_diff / 0.00001)

        # #derivatives = a.derivatives_single(H2, positions =[0 ] , method = "analytical", include=None, exclude=None)
        # derivatives = soap.derivatives_single(H2, positions=[[0.0,0.0,0.0]], method="analytical")
        # #derivatives = a.derivatives_single(H2, positions =[[0.0, 0.0, 0.0], [-0.5, 0, 0], [0.5, 0, 0], ] , method = "analytical", include=None, exclude=None)

        # print(derivatives)
        # print(derivatives.shape)
        # print(a._rcut)


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SoapDerivativeTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
