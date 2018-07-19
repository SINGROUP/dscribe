from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
from describe.descriptors import SOAP
import numpy as np
import unittest
from describe.utils import RematchKernel
from ase.build import molecule
from ase.collections import g2


class RematchKernelTests(unittest.TestCase):
    def test_is_same(self):
        """Tests the global similarity between two structures, 
        once different, once identical. 
        Also tests the global similarity computed from the unity 
        environment kernel"""
        desc = SOAP([1,8], 10.0, 2, 0, periodic=False,crossover=True)
        re = RematchKernel()
        A = molecule('H2O')
        B = molecule('H2O2')
        local_a = desc.create(A)
        local_b = desc.create(B)
        # different structures
        envkernel = re.compute_envkernel(local_a, local_b)
        glosim = re.rematch(envkernel, gamma = 0.01)
        self.assertTrue(0.99 > glosim)
        # same structures
        envkernel = re.compute_envkernel(local_a, local_a)
        glosim = re.rematch(envkernel, gamma =0.01)
        self.assertTrue(0.99 < glosim < 1.01)
        # unity environment kernel
        envkernel = np.ones((5,5)) * 1.0
        glosim = re.rematch(envkernel, gamma =0.01)
        self.assertAlmostEqual(glosim, 1.0)

    def test_glosim_molecules(self):
        is_pass = True
        # check if the same molecules give global similarity of around 1
        desc = SOAP([], 10.0, 2, 0, periodic=False,crossover=True)
        for molname in g2.names:
            atoms = molecule(molname)
            local_a = desc.create(atoms)
            re = RematchKernel()
            envkernel = re.compute_envkernel(local_a, local_a)
            glosim = re.rematch(envkernel, gamma = 0.01)

            if (0.99 < glosim < 1.01):
                pass
            else:
                is_pass = False

        all_atomtypes = [1,6,7,8]
        desc = SOAP([1,6,7,8], 10.0, 2, 0, periodic=False,crossover=True)
        # check randomly a few combinations of molecules, just for no errors
        for molname1 in g2.names:
            for molname2 in g2.names:
                if np.random.rand(1,1) < 0.99:
                    continue
                atoms1 = molecule(molname1)        
                atoms2 = molecule(molname2)

                local_a = desc.create(atoms1)
                local_b = desc.create(atoms2)
                if len(local_a) == 0 or len(local_b) == 0:
                    continue

                envkernel = re.compute_envkernel(local_a, local_b)
                glosim = re.rematch(envkernel, gamma = 0.01)
        self.assertTrue(is_pass)
        return is_pass

    def test_global_kernel(self):
        """Tests the global rematch kernel"""
        print("Start testing global kernel")
        desc = SOAP([1,6,7,8], 10.0, 2, 0, periodic=False,crossover=True)
        re = RematchKernel()
        A = molecule('CH3CHO')
        B = molecule('HCOOH')
        local_a = desc.create(A)
        local_b = desc.create(B)

        # identical structures
        desc_list = [local_a, local_a]
        envkernel_dict = re.get_all_envkernels(desc_list)

        global_matrix = re.get_global_kernel(envkernel_dict, 
            gamma = 0.01, threshold= 1e-6)

        self.assertTrue(np.sum(np.abs((global_matrix - 1.0))) < 0.05)

        # different structures
        desc_list = [local_a, local_b]
        envkernel_dict = re.get_all_envkernels(desc_list)

        global_matrix = re.get_global_kernel(envkernel_dict, 
            gamma = 0.01, threshold= 1e-6)

        self.assertTrue(np.sum(np.diag(np.abs((global_matrix - 1.0)))) < 0.02)
        self.assertTrue(global_matrix[0,1] < 0.99)
        return

if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(RematchKernelTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

