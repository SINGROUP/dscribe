from __future__ import absolute_import, division, print_function
from builtins import super
import math
import numpy as np
import itertools
from describe.core import System

from scipy.spatial.distance import squareform, pdist
from scipy.sparse import lil_matrix

from describe.descriptors import MBTR


class LMBTR(MBTR):
    """Implementation of local -- per chosen atom -- kind of the Many-body
    tensor representation up to K=3.

    You can use this descriptor for finite and periodic systems. When dealing
    with periodic systems, please always use a primitive cell. It does not
    matter which of the available primitive cell is used.
    """
    decay_factor = math.sqrt(2)*3

    def __init__(
            self,
            atomic_numbers,
            k,
            periodic,
            grid=None,
            weighting=None,
            flatten=True,
            ):
        """
        Args:
            atomic_numbers (iterable): A list of the atomic numbers that should
                be taken into account in the descriptor. Notice that this is
                not the atomic numbers that are present for an individual
                system, but should contain all the elements that are ever going
                to be encountered when creating the descriptors for a set of
                systems.  Keeping the number of handled elements as low as
                possible is preferable.
            k (set or list): The interaction terms to consider from 1 to 3. The
                size of the final output and the time taken in creating this
                descriptor is exponentially dependent on this value.
            periodic (bool): Boolean for if the system is periodic or none. If
                this is set to true, you should provide the primitive system as
                input and then the number of periodic copies is determined from the
                'cutoff'-values specified in the weighting argument.
            grid (dictionary): This dictionary can be used to precisely control
                the broadening width, grid spacing and grid length for all the
                different terms. If not provided, a set of sensible defaults
                will be used. Example:
                    grid = {
                        "k1": {
                            "min": 1,
                            "max": 10
                            "sigma": 0.1
                            "n": 100
                        },
                        "k2": {
                            "min": 0,
                            "max": 1/0.70,
                            "sigma": 0.01,
                            "n": 100
                        },
                        ...
                    }

                Here 'min' is the minimum value of the axis, 'max' is the
                maximum value of the axis, 'sigma' is the standard devation of
                the gaussian broadening and 'n' is the number of points sampled
                on the grid.
            weighting (dictionary or string): A dictionary of weighting functions and an
                optional threshold for each term. If None, weighting is not
                used. Weighting functions should be monotonically decreasing.
                The threshold is used to determine the minimum mount of
                periodic images to consider. If no explicit threshold is given,
                a reasonable default will be used.  The K1 term is
                0-dimensional, so weighting is not used. You can also use a
                string to indicate a certain preset. The available presets are:

                    'exponential':
                        weighting = {
                            "k2": {
                                "function": lambda x: np.exp(-0.5*x),
                                "threshold": 1e-3
                            },
                            "k3": {
                                "function": lambda x: np.exp(-0.5*x),
                                "threshold": 1e-3
                            }
                        }

                The meaning of x changes for different terms as follows:
                    K=1: x = 0
                    K=2: x = Distance between A->B
                    K=3: x = Distance from A->B->C->A.
            flatten (bool): Whether the output of create() should be flattened
                to a 1D array. If False, a list of the different tensors is
                provided.

        Raises:
            ValueError if the given k value is not supported, or the weighting
            is not specified for periodic systems.
        """
        atomic_numbers.append(0)  # Ghost
        super().__init__(
                    atomic_numbers,
                    k,
                    periodic,
                    grid,
                    weighting,
                    normalize=False,
                    flatten=flatten,
                    )

    def update(self):
        '''
        Updates relevant objects attached to LMBTR class, after changing
        one/many values
        '''
        self.atomic_numbers = np.unique(self.atomic_numbers + [0]).tolist()
        super().update()
        if 1 in self.k:
            print("Warning: K = 1 is deprecated for LMBTR")

    def describe(self,
                 system,
                 positions=[],
                 scaled_positions=False
                 ):
        """Return the local many-body tensor representation as a 1D array for the
        given system.

        Args:
            system (System): The system for which the descriptor is
                             created.
            positions (iterable): positions or atom index of points, from
                                  which local_mbtr is needed
            scaled_positions (boolean): if list of positions are scaled
                                        use only if system allows it

        Returns:
            1D ndarray: The local many-body tensor representations of given postions,
                        for k terms, as an array. These are ordered as given in
                        list_atom_indices, followed by list_positions
        """
        # ensuring self is updated
        self.update()
        
        system_new = system.copy()
        list_atoms = []
        list_positions = []
        len_sys = len(system)
        
        # Checking scaled position
        if scaled_positions:
            if np.linalg.norm(system.get_cell()) == 0:
                raise ValueError("System doesn't have cell to justify scaled positions.")
        
        for i in positions:
            if type(i) is list or type(i) is tuple:
                if scaled_positions:
                    pos = np.dot(i, system.get_cell())
                else:
                    pos = np.array(i)
                dist = np.linalg.norm(system.get_positions() - pos, axis=1)
                if np.sum( dist == 0):
                    list_atoms.append(np.where(dist == 0)[0][0])
                else:
                    list_positions.append(i)
                    list_atoms.append(len_sys)
                    len_sys += 1
            elif type(i) is int:
                if i >= len(system):
                    raise ValueError("Atom index: {}, larger than total number of atoms.".format(i))
                list_atoms.append(i)
            else:
                raise ValueError("create method requires the argument positions,"
                                 " a list of atom indices and/or positions")

        if len(list_positions):
            system_new += System(
                                'X{}'.format(len(list_positions)),
                                positions=list_positions
                          )

        desc = np.empty(len(list_atoms), dtype='object')

        for i, self.atom_index in enumerate(list_atoms):
            desc[i] = super().describe(system_new)

        return desc

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        n_features = 0
        n_elem = self.n_elements - 1  # Removing ghost

        if 1 in self.k:
            n_k1_grid = self.get_k1_settings()["n"]
            n_k1 = n_k1_grid
            n_features += n_k1
        if 2 in self.k:
            n_k2_grid = self.get_k2_settings()["n"]
            n_k2 = n_elem*n_k2_grid
            n_features += n_k2
        if 3 in self.k:
            n_k3_grid = self.get_k3_settings()["n"]
            n_k3 = (n_elem*(n_elem+1)/2)*n_k3_grid
            n_features += n_k3

        return int(n_features)

    def inverse_distances(self, system):
        """Calculates the inverse distances for the given atomic positions.

        Args:
            system (System): The atomic system.

        Returns:
            dict: Inverse distances in the form:
            {i: [list of distances] }.
        """
        if self._inverse_distances is None:
            inverse_dist = system.get_inverse_distance_matrix()

            numbers = system.numbers
            inv_dist_dict = {}
            for i_atom, i_element in enumerate(numbers):
                i_index = self.atomic_number_to_index[i_element]

                old_list = inv_dist_dict.get(i_index, [])
                inv_dist = inverse_dist[i_atom, self.atom_index]
                old_list.append(inv_dist)
                inv_dist_dict[i_index] = old_list

            self._inverse_distances = inv_dist_dict
        return self._inverse_distances

    def cosines_and_weights(self, system):
        """Calculates the cosine of the angles and their weights between unique
        three-body combinations.

        Args:
            system (System): The atomic system.

        Returns:
            tuple: (cosine, weights) Cosines of the angles (values between -1
            and 1) in the form {i: { j: [list of angles] }}. The weights
            corresponding to the angles are stored in a similar dictionary.

        """
        if self._angles is None or self._angle_weights is None:
            disp_tensor = system.get_displacement_tensor().astype(np.float32)
            distance_matrix = system.get_distance_matrix().astype(np.float32)
            numbers = system.numbers

            # Cosines between atoms i-self.atom_index-j can be found in the tensor:
            # cos_matrix[i, j] or equivalently cos_matrix[j, i] (symmetric)
            n_atoms = len(numbers)
            cos_matrix = np.empty(( n_atoms, n_atoms), dtype=np.float32)
            cos_matrix[:, :] = 1 - squareform(pdist(disp_tensor[self.atom_index, :, :], 'cosine'))

            # Remove the numerical noise from cosine values.
            np.clip(cos_matrix, -1, 1, cos_matrix)

            cos_dict = {}
            weight_dict = {}
            indices = range(len(numbers))

            # Determine the weighting function
            weighting_function = None
            if self.weighting is not None and self.weighting.get("k3") is not None:
                weighting_function = self.weighting["k3"]["function"]

            # Here we go through all the 3-permutations of the atoms in the system
            permutations = itertools.permutations(indices, 2)
            for i_atom, j_atom in permutations:

                i_element = numbers[i_atom]
                j_element = numbers[j_atom]

                i_index = self.atomic_number_to_index[i_element]
                j_index = self.atomic_number_to_index[j_element]

                # Save information in the part where j_index >= i_index
                if j_index < i_index or i_atom == self.atom_index or j_atom == self.atom_index:
                    continue

                # Save weights
                if weighting_function is not None:
                    dist1 = distance_matrix[i_atom, j_atom]
                    dist2 = distance_matrix[j_atom, self.atom_index]
                    dist3 = distance_matrix[self.atom_index, i_atom]
                    weight = weighting_function(dist1 + dist2 + dist3)
                else:
                    weight = 1

                old_dict_1 = weight_dict.get(i_index, {})
                old_list_2 = old_dict_1.get(j_index, [])

                old_list_2.append(weight)
                old_dict_1[j_index] = old_list_2
                weight_dict[i_index] = old_dict_1

                # Save cosines
                old_dict_1 = cos_dict.get(i_index, {})
                old_list_2 = old_dict_1.get(j_index, [])
                old_list_2.append(cos_matrix[i_atom, j_atom])
                old_dict_1[j_index] = old_list_2
                cos_dict[i_index] = old_dict_1

            self._angles = cos_dict
            self._angle_weights = weight_dict
        return self._angles, self._angle_weights

    def K1(self, system, settings):
        """Calculates the first order terms where the scalar mapping is the
        number of atoms of a certain type.

        Args:
            system (System): The atomic system.
            settings (dict): The grid settings

        Returns:
            1D ndarray: (flattened) K1 values.
        """
        start = settings["min"]
        stop = settings["max"]
        n = settings["n"]
        self._axis_k1 = np.linspace(start, stop, n)

        # Use sparse matrices for storing the result
        if self.flatten:
            k1 = lil_matrix((1, n), dtype=np.float32)
        else:
            k1 = np.zeros(n, dtype=np.float32)

        atomic_number = np.array([system.numbers[self.atom_index]])
        count = np.array([1.0])
        gaussian_sum = self.gaussian_sum(atomic_number, count, settings)

        if self.flatten:
            start = 0
            end = n
            k1[0, start:end] = gaussian_sum
        else:
            k1[:] = gaussian_sum

        return k1

    def K2(self, system, settings):
        """Calculates the second order terms where the scalar mapping is the
        inverse distance between atoms.

        Args:
            system (System): The atomic system.
            settings (dict): The grid settings

        Returns:
            1D ndarray: (flattened) K2 values.
        """
        start = settings["min"]
        stop = settings["max"]
        n = settings["n"]
        self._axis_k2 = np.linspace(start, stop, n)

        inv_dist_dict = self.inverse_distances(system)
        n_elem = self.n_elements - 1  # removing ghost

        if self.flatten:
            k2 = lil_matrix(
                (1, n_elem*n), dtype=np.float32)
        else:
            k2 = np.zeros((n_elem, n))

        # Determine the weighting function
        weighting_function = None
        if self.weighting is not None and self.weighting.get("k2") is not None:
            weighting_function = self.weighting["k2"]["function"]

        m = -1
        for i in range(1, n_elem+1):  # ignoring Ghost
                m += 1
                try:
                    inv_dist = np.array(inv_dist_dict[i])
                except KeyError:
                    continue

                # Calculate weights
                if weighting_function is not None:
                    weights = weighting_function(1/np.array(inv_dist))
                else:
                    weights = np.ones(len(inv_dist))

                # Broaden with a gaussian
                gaussian_sum = self.gaussian_sum(inv_dist, weights, settings)

                if self.flatten:
                    start = m*n
                    end = (m + 1)*n
                    k2[0, start:end] = gaussian_sum
                else:
                    k2[i-1, :] = gaussian_sum

        return k2

    def K3(self, system, settings):
        """Calculates the third order terms where the scalar mapping is the
        angle between 3 atoms.

        Args:
            system (System): The atomic system.
            settings (dict): The grid settings

        Returns:
            1D ndarray: (flattened) K3 values.
        """
        start = settings["min"]
        stop = settings["max"]
        n = settings["n"]
        self._axis_k3 = np.linspace(start, stop, n)

        cos_dict, cos_weight_dict = self.cosines_and_weights(system)

        n_elem = self.n_elements - 1  # removing ghost

        if self.flatten:
            k3 = lil_matrix(
                (1, int(n_elem*(n_elem+1)/2*n)), dtype=np.float32)
        else:
            k3 = np.zeros(( n_elem, n_elem, n))

        # Go through the angles, but leave out the duplicate cases by enforcing
        # k >= i. E.g. angles OHH are the same as HHO. This will half the size
        # of the K3 input.
        m = -1
        for i in range(1, n_elem+1):  # ignoring ghost
            for j in range(1, n_elem+1):  # ignoring ghost
                try:
                    cos_values = np.array(cos_dict[i][j])
                except KeyError:
                    continue
                try:
                    cos_weights = np.array(cos_weight_dict[i][j])
                except KeyError:
                    continue
                m += 1

                # Broaden with a gaussian
                gaussian_sum = self.gaussian_sum(cos_values, cos_weights, settings)

                if self.flatten:
                    start = m*n
                    end = (m+1)*n
                    k3[0, start:end] = gaussian_sum
                else:
                    k3[i-1, j-1, :] = gaussian_sum
                    k3[j-1, i-1, :] = gaussian_sum

        return k3
