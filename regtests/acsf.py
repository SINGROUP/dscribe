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
        acsfg1 = desc.create(H2O)
        #print(acsfg1.shape)
        print(acsfg1)

        desc = ACSF(n_atoms_max=3, types=[1,8], bond_params=[[1,0.5,]], flatten=False)
        acsfg2 = desc.create(H2O)
        #print(acsfg2.shape)
        #print(acsfg2)

        desc = ACSF(n_atoms_max=3, types=[1,8], bond_cos_params=[1,], flatten=False)
        acsfg3 = desc.create(H2O)
        print(acsfg3.shape)
        print(acsfg3)
        

        desc = ACSF(n_atoms_max=5, types=[1,8], ang_params=[[1,2,3],[3,1,4], [4,5,6], [7,8,9]], flatten=True)
        desc = ACSF(n_atoms_max=3, types=[1,8], ang_params=[[1.0, 1.0, 1.0],], flatten=False)
        acsfg4 = desc.create(H2O)
        #print(acsfg4.shape)
        #print(acsfg4)
        

        # Test against assumed values
        dist_oh = H2O.get_distance(0,1)
        dist_hh = H2O.get_distance(0,2)
        ang_hoh = H2O.get_angle(0,1,2) * np.pi / 180.0        
        ang_hho = H2O.get_angle(1,0,2) * np.pi / 180.0
        ang_ohh = - H2O.get_angle(2,0,1) * np.pi / 180.0



        #print(dist_oh, dist_hh, ang_hoh)
        rc = 5.0




        #G1  (Rij<qm->cutoff)? 0.5*(cos(Rij*PI/qm->cutoff)+1) : 0;
        g1_ho = 0.5 * (np.cos(np.pi *dist_oh / rc) + 1)
        g1_hh = 0.5 * (np.cos(np.pi *dist_hh / rc) + 1)
        g1_oh = 2 * 0.5 * (np.cos(np.pi *dist_oh / rc) + 1)

        #print(g1_hh, g1_ho, g1_oh)
        self.assertAlmostEqual(acsfg1[0], g1_hh)
        self.assertAlmostEqual(acsfg1[1], g1_ho)
        self.assertAlmostEqual(acsfg1[2], g1_oh)

        #G2        Ga[g] += exp(-eta * (Rij - Rs)*(Rij - Rs)) * fc;
        eta = 1
        rs  = 0.5
        g2_hh = np.exp(-eta * np.power((dist_hh - rs),2)) * g1_hh
        g2_ho = np.exp(-eta * np.power((dist_oh - rs), 2)) * g1_ho
        g2_oh = np.exp(-eta * np.power((dist_oh - rs), 2)) * g1_oh

        #print(g2_hh, g2_ho, g2_oh)
        self.assertAlmostEqual(acsfg2[0,1], g2_hh)
        self.assertAlmostEqual(acsfg2[0,3], g2_ho)
        self.assertAlmostEqual(acsfg2[1,1], g2_oh)


        #G3        val = cos(Rij*val)*fc;
        kappa = 1
        g3_hh = np.cos(dist_hh * kappa) * g1_hh
        g3_ho = np.cos(dist_oh * kappa) * g1_ho
        g3_oh = np.cos(dist_oh * kappa) * g1_oh

        print(g3_hh, g3_ho, g3_oh)
        self.assertAlmostEqual(acsfg3[0,3], g3_hh)
        self.assertAlmostEqual(acsfg3[2,3], g3_ho)
        self.assertAlmostEqual(acsfg3[1,2], g3_oh)


        #G4                       Ga[g] += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;
        #   with                gauss  = exp(-eta*(Rij+Rik+Rjk)) * fc;
        #   and                 cos( theta_ijk ) = ( r_ij^2 + r_ik^2 - r_jk^2 ) / ( 2*r_ij*r_ik ),
        eta = 1
        lmbd = 1
        zeta = 1

        gauss = np.exp(-eta * (2 *dist_oh * dist_oh + dist_hh * dist_hh)) * g1_ho * g1_hh * g1_ho
        g4_h_ho = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_hho)), zeta) * gauss
        g4_h_oh = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_ohh)), zeta) * gauss
        g4_o_hh = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_hoh)), zeta) * gauss


        #print("G4")
        #print(g4_h_ho, g4_h_oh, g4_o_hh)
        self.assertAlmostEqual(acsfg4[0,3], g4_h_ho )
        self.assertAlmostEqual(acsfg4[2,3], g4_h_oh)
        self.assertAlmostEqual(acsfg4[1,2], g4_o_hh)

        #G5
        gauss = np.exp(-eta * (dist_oh * dist_oh + dist_hh * dist_hh)) * g1_ho * g1_hh
        g5_h_ho = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_hho)), zeta) * gauss
        g5_h_oh = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_ohh)), zeta)* np.exp(-eta * (2 *dist_oh * dist_oh)) * g1_ho * g1_ho
        g5_o_hh = np.power(2, 1 - zeta) * np.power((1 + lmbd*np.cos(ang_hoh)), zeta) * gauss
        
        print("G5")
        #print(g5_h_ho, g5_h_oh, g5_o_hh)

        #self.assertTrue(np.array_equal(cm, assumed))


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ACSFTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
