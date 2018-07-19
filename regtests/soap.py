from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import math
import numpy as np
import unittest

from describe.descriptors import SOAP

from ase import Atoms


H2O = Atoms(
    cell=[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ],
    positions=[
        [0, 0, 0],
        [0.95, 0, 0],
        [0.95*(1+math.cos(76/180*math.pi)), 0.95*math.sin(76/180*math.pi), 0.0]
    ],
    symbols=["H", "O", "H"],
)

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


class SoapTests(unittest.TestCase):

#    def test_constructor(self):
#        """Tests different valid and invalid constructor values.
#        """
#        with self.assertRaises(ValueError):
#            SOAP(atomic_numbers, rcut,nmax, lmax,periodic, envPos = None, crossover = True)
#        with self.assertRaises(ValueError):
#            SOAP(atomic_numbers, rcut,nmax, lmax,periodic, envPos = [[0,0,0]], crossover = True)
#
#    def test_number_of_features(self):
#        """Tests that the reported number of features is correct.
#        """
#        desc = SOAP(n_atoms_max=5, permutation="none", flatten=False)
#        n_features = desc.get_number_of_features()
#        self.assertEqual(n_features, 25)
#
#    def test_flatten(self):
#        """Tests the flattening.
#        """
#        # Unflattened
#        desc = SOAP(n_atoms_max=5, permutation="none", flatten=False)
#        cm = desc.create(H2O)
#        self.assertEqual(cm.shape, (5, 5))
#
#        # Flattened
#        desc = SOAP(n_atoms_max=5, permutation="none", flatten=True)
#        cm = desc.create(H2O)
#        self.assertEqual(cm.shape, (25,))
#
    def test_features(self):
        """Tests that the correct features are present in the descriptor.
        """

        desc = SOAP([1,6,8], 10.0, 2, 0, periodic=False, envPos=[[0,0,0]],crossover=True,)
        desc2 = SOAP([1,6,8], 10.0, 2, 0, periodic=False, envPos=0, crossover=True, )
        desc3 = SOAP([1,6,8], 10.0, 2, 0, periodic=False, envPos=None,crossover=True,)

        desc4 = SOAP([1,6,8], 10.0, 2, 0, periodic=True, envPos=[[0,0,0]],crossover=True, )
        desc5 = SOAP([1,6,8], 10.0, 2, 0, periodic=True, envPos=0, crossover=True, )
        desc6 = SOAP([1,6,8], 10.0, 2, 0, periodic=True, envPos=None, crossover=True, )

        desc7 = SOAP([1,6,8], 10.0, 2, 0, periodic=True, envPos=[[0,0,0]],crossover=False, )
        desc8 = SOAP([1,6,8], 10.0, 2, 0, periodic=True, envPos=0, crossover=False, )
        desc9 = SOAP([1,6,8], 10.0, 2, 0, periodic=True, envPos=None, crossover=False, )

        desc10 = SOAP([1,6,8], 10.0, 2, 0, periodic=False, envPos=[[0,0,0]],crossover=False, )
        desc11 = SOAP([1,6,8], 10.0, 2, 0, periodic=False, envPos=0, crossover=False, )
        desc12 = SOAP([1,6,8], 10.0, 2, 0, periodic=False, envPos=None, crossover=False, )

        self.assertEqual(desc.get_number_of_features(), np.shape(desc.create(H2O).flatten())[0])
        self.assertEqual(desc2.get_number_of_features(), np.shape(desc2.create(H2O).flatten())[0])
        self.assertEqual(desc3.get_number_of_features(), np.shape(desc3.create(H2O).flatten())[0])
        self.assertEqual(desc4.get_number_of_features(), np.shape(desc4.create(H2O).flatten())[0])
        self.assertEqual(desc5.get_number_of_features(), np.shape(desc5.create(H2O).flatten())[0])
        self.assertEqual(desc6.get_number_of_features(), np.shape(desc6.create(H2O).flatten())[0])
        self.assertEqual(desc7.get_number_of_features(), np.shape(desc7.create(H2O).flatten())[0])
        self.assertEqual(desc8.get_number_of_features(), np.shape(desc8.create(H2O).flatten())[0])
        self.assertEqual(desc9.get_number_of_features(), np.shape(desc9.create(H2O).flatten())[0])
        self.assertEqual(desc10.get_number_of_features(), np.shape(desc10.create(H2O).flatten())[0])
        self.assertEqual(desc11.get_number_of_features(), np.shape(desc11.create(H2O).flatten())[0])
        self.assertEqual(desc12.get_number_of_features(), np.shape(desc12.create(H2O).flatten())[0])


    def test_unit_cells(self):
        """Tests if arbitrary unit cells are accepted"""
        desc = SOAP([1,6,8], 10.0, 2, 0, periodic=False, envPos=[[0,0,0]],crossover=True,)

        molecule = H2O.copy()

        molecule.set_cell([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
            ],
            )

        nocell = desc.create(molecule)

        desc = SOAP([1,6,8], 10.0, 2, 0, periodic=True, envPos=[[0,0,0]],crossover=True,)
        molecule.set_cell([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
            ],
            )
        with self.assertRaises(ValueError):
            nocell = desc.create(molecule)

        molecule.set_pbc(True)
        molecule.set_cell([
        [20.0, 0.0, 0.0],
        [0.0, 30.0, 0.0],
        [0.0, 0.0, 40.0]
            ],
            )

        largecell = desc.create(molecule)

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
        desc = SOAP([1,6,8], 10.0, 2, 0, periodic=False, envPos=[[0,0,0]],crossover=True,)


        H2O.set_pbc(False)
        nocell = desc.create(H2O)

        H2O.set_pbc(True)
        H2O.set_cell([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0]
            ],
            )
        desc = SOAP([1,6,8], 10.0, 2, 0, periodic=True, envPos=[[0,0,0]],crossover=True,)

        cubic_cell = desc.create(H2O)

        self.assertTrue(np.sum(cubic_cell) > 0)


    def test_periodic_images(self):
        """Tests the periodic images seen by the descriptor
        """
        desc = SOAP([1,6,8], 10.0, 2, 0, periodic=False, envPos=[[0,0,0]],crossover=True,)

        molecule = H2O.copy()

        # non-periodic for comparison
        molecule.set_cell([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
            ],
            )
        nocell = desc.create(molecule)

        # make periodic
        desc = SOAP([1,6,8], 10.0, 2, 0, periodic=True, envPos=[[0,0,0]],crossover=True,)
        molecule.set_pbc(True)

        # cubic
        molecule.set_cell([
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 3.0]
            ],
            )
        cubic_cell = desc.create(molecule)
        suce = molecule * (2,1,1)
        cubic_suce = desc.create(suce)
        
        # triclinic
        molecule.set_cell([
        [0.0, 2.0, 2.0],
        [2.0, 0.0, 2.0],
        [2.0, 2.0, 0.0]
            ],
            )
        triclinic_cell = desc.create(molecule)
        suce = molecule * (2,1,1)
        triclinic_suce = desc.create(suce)

        self.assertTrue(np.sum(np.abs((nocell[:3] - cubic_suce[:3]))) > 0.1)
        self.assertAlmostEqual(np.sum(cubic_cell[:3] -cubic_suce[:3]), 0)
        self.assertAlmostEqual(np.sum(triclinic_cell[:3] - triclinic_suce[:3]), 0)
        


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(SoapTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
