from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import math
import numpy as np
import unittest

from describe.descriptors import ACSF

from ase import Atoms


H2O = Atoms(
    cell=[
        [15.0, 0.0, 0.0],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 15.0]
    ],
    positions=[
        [0, 0, 0],
        [0.95, 0, 0],
        [0.95*(1+math.cos(76/180*math.pi)), 0.95*math.sin(76/180*math.pi), 0.0]
    ],
    symbols=["H", "O", "H"],
)


class ACSFTests(unittest.TestCase):

    def test_constructor(self):
        """Tests different valid and invalid constructor values.
        """
        with self.assertRaises(ValueError):
            ACSF(n_atoms_max=-1, types=[1,8,6])
        with self.assertRaises(ValueError):
            ACSF(n_atoms_max=10, types=None)

        with self.assertRaises(ValueError):
            ACSF(n_atoms_max=10, types=[1,6,8], bond_params=[1,2,3])

        with self.assertRaises(ValueError):
            ACSF(n_atoms_max=10, types=[1,6,8], bond_params=[1,2,3])
        with self.assertRaises(ValueError):
            ACSF(n_atoms_max=10, types=[1,6,8], bond_cos_params=[[1,2],[3,1]])

        with self.assertRaises(ValueError):
            ACSF(n_atoms_max=10, types=[1,6,8], ang_params=[[1,2],[3,1]])




    
    def test_number_of_features(self):
        """Tests that the reported number of features is correct.
        """
        desc = ACSF(n_atoms_max=5, types=[1,8], flatten=True)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 5 * 2)


        desc = ACSF(n_atoms_max=5, types=[1,8], bond_params=[[1,2,], [4,5,]], flatten=True)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 5 * 2 * (2+1))

        desc = ACSF(n_atoms_max=5, types=[1,8], bond_cos_params=[1,2,3,4], flatten=True)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 5 * 2 * (4+1))

        desc = ACSF(n_atoms_max=5, types=[1,8], ang_params=[[1,2,3],[3,1,4], [4,5,6], [7,8,9]], flatten=True)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 5 * (2 + 4 * 3))

        desc = ACSF(n_atoms_max=5, types=[1,8],bond_params=[[1,2,], [4,5,]], bond_cos_params=[1,2,3,4], 
            ang_params=[[1,2,3],[3,1,4], [4,5,6], [7,8,9]], flatten=True)
        n_features = desc.get_number_of_features()
        self.assertEqual(n_features, 5 * ((2 * (1 + 2 + 4)) + 4 * 3))

    def test_flatten(self):
        """Tests the flattening.
        """
        """print("Testing flattening now")
        # Unflattened
        #desc = ACSF(n_atoms_max=5, types=[1,8],bond_params=[[1,2,], [4,5,]], bond_cos_params=[1,2,3,4], 
        #    ang_params=[[1,2,3],[3,1,4], [4,5,6], [7,8,9]], flatten=False)
        #cm = desc.create(H2O)
        #self.assertEqual(cm.shape, (5, 5))
        # Flattened
        desc = ACSF(n_atoms_max=5, types=[1,8],bond_params=[[1,2,], [4,5,]], bond_cos_params=[1,2,3,4], 
            ang_params=[[1,2,3],[3,1,4], [4,5,6], [7,8,9]], flatten=True)
        cm = desc.create(H2O)
        self.assertEqual(cm.shape, (25,))
        """

    def test_features(self):
        """Tests that the correct features are present in the desciptor.
        """
        desc = ACSF(n_atoms_max=5, types=[1,8], flatten=True)
        acsf = desc.create(H2O)
        print(acsf.shape)
        print(acsf)

        desc = ACSF(n_atoms_max=5, types=[1,8], bond_params=[[1,2,], [4,5,]], flatten=True)
        acsf = desc.create(H2O)
        print(acsf.shape)

        desc = ACSF(n_atoms_max=5, types=[1,8], bond_cos_params=[1,2,3,4], flatten=True)
        acsf = desc.create(H2O)
        print(acsf.shape)

        #desc = ACSF(n_atoms_max=5, types=[1,8], ang_params=[[1,2,3],[3,1,4], [4,5,6], [7,8,9]], flatten=True)
        #desc = ACSF(n_atoms_max=5, types=[1,8], ang_params=[[1.0, 1.0, 1.0],], flatten=True)
        #acsf = desc.create(H2O)
        #print(acsf.shape)
        
        

        # Test against assumed values
        #G1  (Rij<qm->cutoff)? 0.5*(cos(Rij*PI/qm->cutoff)+1) : 0;
        #G2        Ga[g] += exp(-eta * (Rij - Rs)*(Rij - Rs)) * fc;
        #G3        val = cos(Rij*val)*fc;
        #G4                       Ga[g] += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;
        #   with                gauss  = exp(-eta*(Rij+Rik+Rjk)) * fc;
        #   and                 cos( theta_ijk ) = ( r_ij^2 + r_ik^2 - r_jk^2 ) / ( 2*r_ij*r_ik ),
        
        #G5

        #self.assertTrue(np.array_equal(cm, assumed))


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ACSFTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
