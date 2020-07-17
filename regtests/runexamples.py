import sys, os
#import math
#import numpy as np
import unittest

#from dscribe.core import System
#from dscribe.descriptors import ACSF
#from dscribe.utils.species import symbols_to_numbers
#from dscribe.ext import CellList

#from ase.lattice.cubic import SimpleCubicFactory
#from ase.build import bulk
#import ase.data
#from ase import Atoms
#from dscribe.descriptors import SOAP
import examples

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)



class ExampleTests(unittest.TestCase):

    def test_example_acsf(self):
        """Tests if the acsf.py example script runs without error.
        """
        from examples import acsf

    def test_example_mbtr(self):
        """Tests if the mbtr.py example script runs without error.
        """
        import examples.mbtr

    def test_example_lmbtr(self):
        """Tests if the lmbtr.py example script runs without error.
        """
        import examples.lmbtr

    def test_example_soap(self):
        """Tests if the soap.py example script runs without error.
        """
        import examples.soap

    def test_example_coulombmatrix(self):
        """Tests if the coulombmatrix.py example script runs without error.
        """
        import examples.coulombmatrix

    def test_example_sinematrix(self):
        """Tests if the sinematrix.py example script runs without error.
        """
        import examples.sinematrix
    
    def test_example_ewaldsummatrix(self):
        """Tests if the ewaldsummatrix.py example script runs without error.
        """
        import examples.ewaldsummatrix

    def test_example_aseatoms(self):
        """Tests if the aseatoms.py example script runs without error.
        """
        with cd("examples"):
            import examples.aseatoms


    def test_example_basics(self):
        """Tests if the basics.py example script runs without error.
        """
        with cd("examples"):
            import examples.basics

    def test_example_readme(self):
        """Tests if the readme.py example script runs without error.
        """
        with cd("examples"):
            import examples.readme



if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ExampleTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
