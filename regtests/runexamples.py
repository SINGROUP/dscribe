import sys, os
import unittest
from importlib import import_module


class ExampleTests(unittest.TestCase):

    def test_examples(self):
        # Walks through the examples folder and tests each example by
        # dynamically importing them which will also run their code.
        old_cwd = os.getcwd()
        os.chdir("../examples")
        example_root = os.getcwd()
        paths = set()
        for root, dirs, files in os.walk('./'):
            for f in files:
                filename = os.path.join(root, f)
                if filename.endswith(".py"):
                    parts = filename[2:-3]
                    parts = parts.rsplit("/", 1)
                    if len(parts) != 1:
                        folder = parts[0]
                        cwd = "{}/{}".format(example_root, folder)
                    else:
                        cwd = example_root
                    os.chdir(cwd)
                    modulename = parts[-1]
                    if cwd not in paths:
                        sys.path.insert(0, cwd)
                        paths.add(cwd)
                    import_module(modulename)
        os.chdir(old_cwd)

if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ExampleTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0, buffer=True).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
