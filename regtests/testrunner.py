from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import sys

# Import the test modules
import coulombmatrix

# Initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Add tests to the test suite
suite.addTests(loader.loadTestsFromModule(coulombmatrix))

# Initialize a runner, pass it the suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)

# We need to return a non-zero exit code for the CI to detect errors
sys.exit(not result.wasSuccessful())
