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

H = Atoms(
    cell=[
        [15.0, 0.0, 0.0],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 15.0]
    ],
    positions=[
        [0, 0, 0],

    ],
    symbols=["H"],
)

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

class SoapDerivativeTests(unittest.TestCase):

#    def test_derivatives(self):
#        """Used to test that changing the setup through properties works as
#        intended.
#        """
#        # Test changing species
#        soap = SOAP(
#            species=[1],
#            rcut=3,
#            nmax=3,
#            lmax=3,
#            sparse=False,
#        )
#        derivatives = soap.derivatives_single(H)
#
    def test_analytical(self):
        """Tests if the analytical soap derivatives run
        """
        a = SOAP(
            species=[1],
            rcut=3,
            nmax=2,
            lmax=0,
            sparse=False,
        )
        atoms = a.create(H2)

#        derivatives = a.derivatives(H2, positions =[[0.0, 0.0, 0.0], [-0.5, 0, 0], [0.5, 0, 0], ] , method = "analytical", include=None, exclude=None)
        derivatives = a.derivatives_single(H2, positions =[[0.0, 0.0, 0.0], ] , method = "analytical", include=None, exclude=None)

        print(derivatives)
        print(derivatives.shape)
        print(a._rcut)



if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SoapDerivativeTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

