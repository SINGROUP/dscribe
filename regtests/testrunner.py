import unittest
import sys

# Import the test modules
import generaltests
import soap
import kernels
import derivatives
import runexamples

# Initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Add tests to the test suite
suite.addTests(loader.loadTestsFromModule(generaltests))
suite.addTests(loader.loadTestsFromModule(kernels))
suite.addTests(loader.loadTestsFromModule(derivatives))
suite.addTests(loader.loadTestsFromModule(runexamples))

# Initialize a runner, pass it the suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)

# We need to return a non-zero exit code for the CI to detect errors
sys.exit(not result.wasSuccessful())
