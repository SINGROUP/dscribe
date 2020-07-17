import unittest
import sys

# Import the test modules
import generaltests
import coulombmatrix
import ewaldsummatrix
import sinematrix
import matrixpermutation
import acsf
import mbtr
import lmbtr
import soap
import elementaldistribution
import kernels
import runexamples

# Initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Add tests to the test suite
suite.addTests(loader.loadTestsFromModule(generaltests))
suite.addTests(loader.loadTestsFromModule(coulombmatrix))
suite.addTests(loader.loadTestsFromModule(matrixpermutation))
suite.addTests(loader.loadTestsFromModule(ewaldsummatrix))
suite.addTests(loader.loadTestsFromModule(sinematrix))
suite.addTests(loader.loadTestsFromModule(acsf))
suite.addTests(loader.loadTestsFromModule(mbtr))
suite.addTests(loader.loadTestsFromModule(lmbtr))
suite.addTests(loader.loadTestsFromModule(soap))
suite.addTests(loader.loadTestsFromModule(elementaldistribution))
suite.addTests(loader.loadTestsFromModule(kernels))
suite.addTests(loader.loadTestsFromModule(runexamples))

# Initialize a runner, pass it the suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)

# We need to return a non-zero exit code for the CI to detect errors
sys.exit(not result.wasSuccessful())
