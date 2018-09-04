from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import unittest
import sys

# Import the test modules
import generaltests
import coulombmatrix
import ewaldmatrix
import sinematrix
import acsf
import mbtr
import lmbtr
import soap
import elementaldistribution
import rematch_kernel

# Initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Add tests to the test suite
suite.addTests(loader.loadTestsFromModule(generaltests))
suite.addTests(loader.loadTestsFromModule(coulombmatrix))
suite.addTests(loader.loadTestsFromModule(ewaldmatrix))
suite.addTests(loader.loadTestsFromModule(sinematrix))
suite.addTests(loader.loadTestsFromModule(acsf))
suite.addTests(loader.loadTestsFromModule(mbtr))
suite.addTests(loader.loadTestsFromModule(lmbtr))
suite.addTests(loader.loadTestsFromModule(soap))
suite.addTests(loader.loadTestsFromModule(elementaldistribution))
suite.addTests(loader.loadTestsFromModule(rematch_kernel))

# Initialize a runner, pass it the suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)

# We need to return a non-zero exit code for the CI to detect errors
sys.exit(not result.wasSuccessful())
