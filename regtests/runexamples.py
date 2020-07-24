import sys, os
import unittest
import examples


class ExampleTests(unittest.TestCase):

    savedPath = None

    @classmethod
    def setUpClass(cls):
        """Navigates to the examples folder before running the examples.
        """
        cls.savedPath = os.getcwd()
        os.chdir("examples")

    @classmethod
    def tearDownClass(cls):
        """Navigates back the the previous directory after running the
        examples.
        """
        os.chdir(cls.savedPath)

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
        import examples.aseatoms

    def test_example_basics(self):
        """Tests if the basics.py example script runs without error.
        """
        import examples.basics

    def test_example_readme(self):
        """Tests if the readme.py example script runs without error.
        """
        import examples.readme

if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ExampleTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0, buffer=True).run(alltests)

    # We need to return a non-zero exit code for the gitlab CI to detect errors
    sys.exit(not result.wasSuccessful())
