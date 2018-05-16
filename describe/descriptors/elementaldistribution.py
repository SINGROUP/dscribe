from __future__ import absolute_import, division, print_function
import math
import numpy as np
from scipy.special import erf
from scipy.sparse import lil_matrix
from describe.descriptors import Descriptor


class ElementalDistribution(Descriptor):
    """Represents a generic N-dimensional smooth distribution on any given grid
    for any given elemental properties.
    """
    def __init__(
            self,
            properties,
            flatten=True,
            ):
        """
        Args:
            properties(dict): Contains a description of the elemental property
                for which a distribution is created. Should contain a dictionary of
                the following form:

                properties={
                    "property_name": {
                        "min": <Distribution minimum value>
                        "max": <Distribution maximum value>
                        "std": <Distribution standard deviation>
                        "n": <Number of discrete samples from distribution>
                        "values": {
                            "H": <Value for hydrogen>
                            ...
                        }
                    }
                }
            flatten(bool): Whether to flatten out the result.
        """

        # Check that the given properties are valid
        for prop_name, prop_grid in properties.items():
            i_min = prop_grid["min"]
            i_max = prop_grid["max"]
            std = prop_grid["std"]
            n = prop_grid["n"]
            prop_grid["values"]
            if i_min >= i_max:
                raise ValueError(
                    "Minimum value for '{}' cannot be larger than maximum "
                    "value.".format(prop_name)
                )
            if std <= 0:
                raise ValueError(
                    "The standard deviation must be a larger than zero."
                )
            if n <= 0:
                raise ValueError(
                    "The number of grid points must be a non-negative "
                    "integer."
                )
        self.properties = properties

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        n_features = 0
        for prop in self.properties.values():
            n_features += prop["n"]

        return n_features

    def describe(self, system):
        """
        Args:
            system (ase.Atoms): The system for which this descriptor is
                created.

        Returns:
            scipy.sparse.lil_matrix: The concatenated distributions of the
                specified properties in a sparse array.
        """
        occurrence = self.get_element_occurrence(system)
        weights = np.array(list(occurrence.values()))
        n_features = self.get_number_of_features()
        distribution = lil_matrix((1, n_features), dtype=np.float32)

        index = 0
        for prop in self.properties.values():
            n = prop["n"]
            minimum = prop["min"]
            maximum = prop["max"]
            std = prop["std"]
            values = prop["values"]
            centers = np.array([values[x] for x in occurrence.keys()])
            pdf = self.gaussian_sum(centers, weights, minimum, maximum, std, n)
            distribution[0, index:index+n] += pdf
            index += n

        return distribution

    def gaussian_sum(self, centers, weights, minimum, maximum, std, n):
        """Calculates a discrete version of a sum of Gaussian distributions.

        The calculation is done through the cumulative distribution function
        that is better at keeping the integral of the probability function
        constant with coarser grids.

        The values are normalized by dividing with the maximum value of a
        gaussian with the given standard deviation.

        Args:
            centers (1D np.ndarray): The means of the gaussians.
            weights (1D np.ndarray): The weights for the gaussians.
            minimum (float): The minimum grid value
            maximum (float): The maximum grid value
            std (float): Standard deviation of the gaussian
            n (int): Number of grid points
            settings (dict): The grid settings. A dictionary
                containing the following information:

        Returns:
            Value of the gaussian sums on the given grid.
        """
        max_val = 1/(std*math.sqrt(2*math.pi))

        dx = (maximum - minimum)/(n-1)
        x = np.linspace(minimum-dx/2, maximum+dx/2, n+1)
        pos = x[np.newaxis, :] - centers[:, np.newaxis]
        y = weights[:, np.newaxis]*1/2*(1 + erf(pos/(std*np.sqrt(2))))
        f = np.sum(y, axis=0)
        f /= max_val
        f_rolled = np.roll(f, -1)
        pdf = (f_rolled - f)[0:-1]/dx  # PDF is the derivative of CDF

        return pdf

    def get_element_occurrence(self, system):
        """Calculate the count of each atomic element in the given system.

        Args:
            system (ase.Atoms): The atomic system.

        Returns:
            1D ndarray: The counts for each element in a list where the index
            of atomic number x is self.atomic_number_to_index[x]
        """
        symbols = system.get_chemical_symbols()
        unique, counts = np.unique(symbols, return_counts=True)
        occurrence = dict(zip(unique, counts))

        return occurrence
