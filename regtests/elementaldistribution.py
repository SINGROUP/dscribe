import numpy as np
import unittest

import ase.build

from dscribe.descriptors import ElementalDistribution


class ElementalDistributionTests(unittest.TestCase):
    """Tests for the ElementalDistribution-descriptor.
    """
    def test_invalid_values(self):
        # Invalid distribution type
        with self.assertRaises(ValueError):
            ElementalDistribution(
                properties={
                    "first_property": {
                        "type": "unknown",
                        "min": 0,
                        "max": 2.5,
                        "std": 0.5,
                        "n": 50,
                        "values": {"H": 2}
                    }
                }
            )

        # Floating points in discrete distribution
        with self.assertRaises(ValueError):
            ElementalDistribution(
                properties={
                    "first_property": {
                        "type": "discrete",
                        "min": 0,
                        "max": 2.5,
                        "std": 0.5,
                        "n": 50,
                        "values": {"H": 2.0}
                    }
                }
            )

        # Out of range
        with self.assertRaises(ValueError):
            ElementalDistribution(
                properties={
                    "first_property": {
                        "type": "continuous",
                        "min": 0,
                        "max": 2.5,
                        "std": 0.5,
                        "n": 50,
                        "values": {"H": 5.0}
                    }
                }
            )

    def test_single_continuous_property(self):
        # Tested on a water molecule
        system = ase.build.molecule("H2O")

        # Descriptor setup
        std = 0.1
        elements = ["H", "O"]
        peaks = [0.3, 2.0]
        values = dict(zip(elements, peaks))
        elemdist = ElementalDistribution(
            properties={
                "first_property": {
                    "type": "continuous",
                    "min": 0,
                    "max": 2.5,
                    "std": std,
                    "n": 50,
                    "values": values
                }
            }
        )

        # Features
        y = elemdist.create(system)
        y = y.todense().A1
        x = elemdist.get_axis("first_property")

        # Test that the peak positions match
        from scipy.signal import find_peaks_cwt
        peak_indices = find_peaks_cwt(y, [std])
        peak_loc = x[peak_indices]

        # Test that the peak locations match within some tolerance
        self.assertTrue(np.allclose(peaks, peak_loc, rtol=0, atol=0.05))

        # Plot for visual inspection
        # mpl.plot(x, y)
        # mpl.show()

    def test_single_discrete_property(self):
        # Tested on a water molecule
        system = ase.build.molecule("H2O")

        # Descriptor setup
        elements = ["H", "O", "C", "Fe"]
        peaks = [0, 4, 18, 2]
        values = dict(zip(elements, peaks))
        elemdist = ElementalDistribution(
            properties={
                "first_property": {
                    "type": "discrete",
                    "values": values
                }
            }
        )

        # Features
        n_features = elemdist.get_number_of_features()
        self.assertEqual(n_features, 19)

        # Check that the axis is correct
        x = elemdist.get_axis("first_property")
        self.assertTrue(np.array_equal(x, np.arange(0, 18+1)))

        y = elemdist.create(system)
        y = y.todense().A1

        # Test that the peak positions match
        assumed = np.zeros((19))
        assumed[0] = 2
        assumed[4] = 1
        self.assertTrue(np.array_equal(y, assumed))

        # # Plot for visual inspection
        # mpl.plot(x, y)
        # mpl.show()


if __name__ == '__main__':
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(ElementalDistributionTests))
    alltests = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=0).run(alltests)
