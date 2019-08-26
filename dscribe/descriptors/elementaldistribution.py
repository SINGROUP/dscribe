# -*- coding: utf-8 -*-
"""Copyright 2019 DScribe developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import math
import numpy as np
from scipy.special import erf
from scipy.sparse import lil_matrix
from dscribe.descriptors import Descriptor


class ElementalDistribution(Descriptor):
    """Represents a generic distribution on any given grid for any given
    properties. Can create both continuos and discrete distributions.

    Continuous distributions require a standard deviation and the number of
    sampling points. You can also specify the minimum and maximum values for
    the axis. If these are not specified, a limit is selected based
    automatically on the values with:

        min = values.min() - 3*std
        max = values.max() + 3*std

    Discrete distributions are assumed to be integer values, and you only need
    to specify the values.
    """
    def __init__(
            self,
            properties,
            flatten=True,
            sparse=True,
            ):
        """
        Args:
            properties(dict): Contains a description of the elemental property
                for which a distribution is created. Should contain a dictionary of
                the following form:

                properties={
                    "property_name": {
                        "type": "continuous"
                        "min": <Distribution minimum value>
                        "max": <Distribution maximum value>
                        "std": <Distribution standard deviation>
                        "n": <Number of discrete samples from distribution>
                        "values": {
                            "H": <Value for hydrogen>
                            ...
                        }
                    "property_name2": {
                        "type": "discrete"
                        "values": {
                            "H": <Value for hydrogen>
                            ...
                        }
                    }
                    }
                }
            flatten(bool): Whether to flatten out the result.
            sparse (bool): Whether the output should be a sparse matrix or a
                dense numpy array.
        """
        # Check that the given properties are valid
        for prop_name, prop_grid in properties.items():
            dist_type = prop_grid.get("type")
            valid_dist_types = set(["continuous", "discrete"])
            if dist_type not in valid_dist_types:
                raise ValueError(
                    "Please specify the distribution type. Valid options are: {}"
                    .format(valid_dist_types)
                )
            i_min = prop_grid.get("min")
            i_max = prop_grid.get("max")
            std = prop_grid.get("std")
            n = prop_grid.get("n")
            values = prop_grid.get("values")

            if values is None:
                raise ValueError(
                    "Please provide the property values, i.e. a dictionary that"
                    " maps an atomic element symbol to a property value."
                )

            values = np.array(list(values.values()))

            if dist_type == "continuous":
                true_min = values.min()
                true_max = values.max()
                if i_min is None:
                    i_min = true_min - 3*std
                    prop_grid["min"] = i_min
                if i_max is None:
                    i_max = true_max + 3*std
                    prop_grid["max"] = i_max
                if i_min >= i_max:
                    raise ValueError(
                        "Minimum value for '{}' cannot be larger or equal to maximum "
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
                if true_min < prop_grid["min"]:
                    raise ValueError(
                        "Property value is outside the specified minimum value."
                    )
                if true_max > prop_grid["max"]:
                    raise ValueError(
                        "Property value is outside the specified maximum value."
                    )
            elif dist_type == "discrete":

                # Check that all values are integer
                if not all(np.issubdtype(item, np.integer) for item in values):
                    raise ValueError(
                        "Not all the values given for property '{}' are integer "
                        "numbers.".format(prop_name)
                    )
                i_min = values.min()
                i_max = values.max()
                prop_grid["min"] = i_min
                prop_grid["max"] = i_max
                prop_grid["n"] = i_max - i_min + 1

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

    def get_axis(self, property_name):
        """Used to return the used x-axis for the given property.

        Args:
            property_name(str): The property name that was used in the
            constructor.

        Returns:
            np.ndarray: An array of x-axis values.
        """
        prop = self.properties[property_name]
        minimum = prop["min"]
        maximum = prop["max"]
        dist_type = prop["type"]
        if dist_type == "continuous":
            n = prop["n"]
            x = np.linspace(minimum, maximum, n)
        elif dist_type == "discrete":
            x = np.arange(minimum, maximum+1)
        return x

    def create(self, system):
        """
        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.

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
            dist_type = prop["type"]
            if dist_type == "continuous":
                n = prop["n"]
                minimum = prop["min"]
                maximum = prop["max"]
                std = prop["std"]
                values = prop["values"]
                centers = np.array([values[x] for x in occurrence.keys()])
                pdf = self.gaussian_sum(centers, weights, minimum, maximum, std, n)
                distribution[0, index:index+n] += pdf
                index += n
            elif dist_type == "discrete":
                n = prop["n"]
                values = prop["values"]
                hist = np.zeros((n))
                minimum = prop["min"]
                for element, occ in occurrence.items():
                    value = values[element]
                    hist_index = value - minimum
                    hist[hist_index] = occ
                distribution[0, index:index+n] += hist
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
