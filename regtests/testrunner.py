from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import sys

# Import the test modules
import coulombmatrix
import ewaldmatrix
import sinematrix
import acsf
import mbtr
import soap

# Initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Add tests to the test suite
suite.addTests(loader.loadTestsFromModule(coulombmatrix))
suite.addTests(loader.loadTestsFromModule(ewaldmatrix))
suite.addTests(loader.loadTestsFromModule(sinematrix))
suite.addTests(loader.loadTestsFromModule(acsf))
suite.addTests(loader.loadTestsFromModule(mbtr))
suite.addTests(loader.loadTestsFromModule(soap))

# Initialize a runner, pass it the suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)

# We need to return a non-zero exit code for the CI to detect errors
sys.exit(not result.wasSuccessful())
