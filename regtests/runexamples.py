import sys, os
import unittest
from importlib import import_module


class ExampleTests(unittest.TestCase):

    def test_examples(self):
        # Walks through the examples folder and tests each example by
        # dynamically importing them which will also run their code.
        os.chdir("../examples")
        for root, dirs, files in os.walk('./'):
            for f in files:
                filename = os.path.join(root, f)
                if filename.endswith(".py"):
                    modulename = filename[2:-3].replace("/", ".")
                    import_module("examples."+modulename)

if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ExampleTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0, buffer=False).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
