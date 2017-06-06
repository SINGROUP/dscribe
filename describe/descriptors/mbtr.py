import math
import numpy as np
import itertools

from scipy.spatial.distance import squareform, pdist, cdist
from scipy.sparse import lil_matrix, coo_matrix

from describe.core import System
from describe.descriptors import Descriptor

import matplotlib.pyplot as mpl


class MBTR(Descriptor):
    """Implementation of the Many-body tensor representation up to K=3.

    You can use this descriptor for finite and periodic systems. When dealing
    with periodic systems, please always use a primitive cell. It does not
    matter which of the available primitive cell is used.
    """
    def __init__(
            self,
            atomic_numbers,
            k,
            periodic=True,
            grid=None,
            weighting=None,
            flatten=True
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
            k (int): The highest interaction term to consider. The size of the
                final output and the time taken in creating this descriptor is
                exponentially dependent on this value.
            periodic (bool): Boolean for if the system is periodic or none. If
                this is set to true, you should provide the primitive system as
                input and then the number of periodic copies is determined from the
                'cutoff'-values specified in the weighting argument.
            grid (dictionary): This dictionary can be used to precisely control
                the broadening width, grid spacing and grid length for all the
                different terms. If not provided, a set of sensible defaults
                will be used. Example:
                    grid = {
                        "k1": [0, 10, 0.1, 0.1]      # min, max, dx, sigma
                        "k2": [0, 1/0.70, 0.01, 0.01]  # min, max, dx, sigma
                        ...
                    }
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
            ValueError if the given k value is not supported.
        """
        super().__init__(flatten)

        # Check K value
        if k < 1 or k > 3:
            raise ValueError(
                "The given value of k={} is not supported.".format(k)
            )

        # Check the weighting information
        if weighting is not None:
            if weighting == "exponential":
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
            else:
                for i in range(k):
                    info = weighting.get("k{}".format(i+1))
                    if info is not None:
                        assert "function" in info, \
                            ("The weighting dictionary is missing 'function'.")
        elif periodic:
            raise ValueError(
                "Periodic systems will need to have a weighting function "
                "defined in the 'weighting' dictionary of the MBTR cconstructor."
            )

        # Check the given grid
        if grid is not None:
            for i in range(k):
                info = grid.get("k{}".format(i+1))
                if info is not None:
                    assert len(info) == 4, \
                        ("The parameters passed in 'grid' should have 4 values:"
                        " min, max, dx and sigma.")
                    assert info[0] < info[1], \
                        "The min value should be smaller than the max values"
        self.grid = grid

        self.k = k
        self.n_elements = None
        self.present_elements = None
        self.atomic_number_to_index = {}
        self.index_to_atomic_number = {}
        self.n_atoms_in_cell = None
        self.periodic = periodic
        self.n_copies_per_axis = None

        self.weighting = weighting

        # Sort the atomic numbers. This is not needed but makes things maybe a
        # bit easier to debug.
        atomic_numbers.sort()

        for i_atom, atomic_number in enumerate(atomic_numbers):
            self.atomic_number_to_index[atomic_number] = i_atom
            self.index_to_atomic_number[i_atom] = atomic_number
        self.n_elements = len(atomic_numbers)

        self.max_atomic_number = max(atomic_numbers)

        self._counts = None
        self._inverse_distances = None
        self._angles = None
        self._angle_weights = None
        self._axis_k1 = None
        self._axis_k2 = None
        self._axis_k3 = None

    def describe(self, system):
        """Return the many-body tensor representation as a 1D array for the
        given system.

        Args:
            system (System): The system for which the descriptor is created.

        Returns:
            1D ndarray: The many-body tensor representation up to the k:th term
            as a flattened array.
        """
        self.n_atoms_in_cell = len(system)
        present_element_numbers = set(system.numbers)
        self.present_indices = set()
        for number in present_element_numbers:
            index = self.atomic_number_to_index[number]
            self.present_indices.add(index)

        mbtr = []
        if self.k >= 1:

            grid_k1 = self.get_k1_axis()
            if self.grid is not None and self.grid.get("k1") is not None:
                sigma_k1 = self.grid["k1"][3]
            else:
                sigma_k1 = 1e-6

            # We will use the original system to calculate the counts, unlike
            # with the other terms that use the extended system
            k1 = self.K1(system, grid_k1, sigma_k1)
            mbtr.append(k1)

        if self.k >= 2:
            grid_k2 = self.get_k2_axis()
            if self.grid is not None and self.grid.get("k2") is not None:
                sigma_k2 = self.grid["k2"][3]
            else:
                sigma_k2 = 2**(-7)

            # If needed, create the extended system
            system_k2 = system
            if self.periodic:
                system_k2 = self.create_extended_system(system, 2)

            k2 = self.K2(system_k2, grid_k2, sigma_k2)

            # Free memory
            system_k2 = None

            mbtr.append(k2)

        if self.k >= 3:

            grid_k3 = self.get_k3_axis()
            if self.grid is not None and self.grid.get("k3") is not None:
                sigma_k3 = self.grid["k3"][3]
            else:
                sigma_k3 = 2**(-3.5)

            # If needed, create the extended system
            system_k3 = system
            if self.periodic:
                system_k3 = self.create_extended_system(system, 3)

            k3 = self.K3(system_k3, grid_k3, sigma_k3)

            # Free memory
            system_k3 = None

            mbtr.append(k3)

        if self.flatten:
            length = 0

            datas = []
            rows = []
            cols = []
            for tensor in mbtr:
                size = tensor.shape[1]
                length += size
                coo = tensor.tocoo()
                datas.append(coo.data)
                rows.append(coo.row)
                cols.append(coo.col)

            datas = np.concatenate(datas)
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            final_vector = coo_matrix((datas, (rows, cols)), shape=[1, length], dtype=np.float32)

            return final_vector
        else:
            return mbtr

    def get_k1_axis(self):
        if self.grid is not None and self.grid.get("k1") is not None:
            min_k1, max_k1, dx_k1, _ = self.grid["k1"]
        else:
            dx_k1 = 0.1
            min_k1 = 1.0
            max_k1 = self.max_atomic_number
        grid = np.arange(min_k1, max_k1, dx_k1)
        return grid

    def get_k2_axis(self):
        if self.grid is not None and self.grid.get("k2") is not None:
            min_k2, max_k2, dx_k2, _ = self.grid["k2"]
        else:
            min_k2 = 0.0
            max_k2 = 1/0.7
            dx_k2 = 1e-2
        grid = np.arange(min_k2, max_k2, dx_k2)
        return grid

    def get_k3_axis(self):
        if self.grid is not None and self.grid.get("k3") is not None:
            min_k3, max_k3, dx_k3, _ = self.grid["k3"]
        else:
            min_k3 = -1.0
            max_k3 = 1.0
            dx_k3 = 0.25*2**(-3.5)
        grid = np.arange(min_k3, max_k3, dx_k3)
        return grid

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        n_features = 0
        n_elem = self.n_elements

        if self.k >= 1:
            n_k1_grid = self.get_k1_axis().size
            n_k1 = n_elem*n_k1_grid
            n_features += n_k1
        if self.k >= 2:
            n_k2_grid = self.get_k2_axis().size
            n_k2 = (n_elem*(n_elem+1)/2)*n_k2_grid
            n_features += n_k2
        if self.k >= 3:
            n_k3_grid = self.get_k3_axis().size
            n_k3 = (n_elem*n_elem*(n_elem+1)/2)*n_k3_grid
            n_features += n_k3

        return n_features

    def create_extended_system(self, primitive_system, term_number):
        """Used to create a periodically extended system, that is as small as
        possible by rejecting atoms for which the given weighting will be below
        the given threshold.

        Args:
            primitive_system (System): The original primitive system to
                duplicate.
            term_number (int): The term number of the tensor. For k=2, the max
                distance is x, for k>2, the distance is given by 2*x.

        Returns:
            System: The new system that is extended so that each atom can at
            most have a weight that is larger or equivalent to the given
            threshold.
        """
        numbers = primitive_system.numbers
        relative_pos = primitive_system.relative_pos
        cartesian_pos = np.array(primitive_system.cartesian_pos)
        cell = primitive_system.lattice.matrix

        # Determine the upper limit of how many copies we need in each cell
        # vector direction. We take as many copies as needed for the
        # exponential weight to come down to the given threshold.
        cell_vector_lengths = np.linalg.norm(cell, axis=1)
        n_copies_axis = np.zeros(3, dtype=int)
        weighting_function = self.weighting["k{}".format(term_number)]["function"]
        threshold = self.weighting["k{}".format(term_number)].get("threshold", 1e-3)

        for i_axis, axis_length in enumerate(cell_vector_lengths):
            limit_found = False
            n_copies = -1
            while (not limit_found):
                n_copies += 1
                distance = n_copies*cell_vector_lengths[0]

                # For terms above k==2 we double the distances to take into
                # account the "loop" that is required.
                if term_number > 2:
                    distance = 2*distance

                weight = weighting_function(distance)
                if weight < threshold:
                    n_copies_axis[i_axis] = n_copies
                    limit_found = True

        # Create copies of the cell but keep track of the atoms in the
        # original cell
        num_extended = []
        pos_extended = []
        num_extended.append(numbers)
        pos_extended.append(cartesian_pos)
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        c = np.array([0, 0, 1])
        for i in range(-n_copies_axis[0], n_copies_axis[0]+1):
            for j in range(-n_copies_axis[1], n_copies_axis[1]+1):
                for k in range(-n_copies_axis[2], n_copies_axis[2]+1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    num_copy = np.array(numbers)

                    # Calculate the positions of the copied atoms and filter
                    # out the atoms that are farther away than the given
                    # cutoff.
                    pos_copy = np.array(relative_pos)-i*a-j*b-k*c
                    pos_copy_cartesian = np.dot(pos_copy, cell)
                    distances = cdist(pos_copy_cartesian, cartesian_pos)

                    # For terms above k==2 we double the distances to take into
                    # account the "loop" that is required.
                    if term_number > 2:
                        distances *= 2

                    weights = weighting_function(distances)
                    weight_mask = weights >= threshold

                    # Create a boolean mask that says if the atom is within the
                    # range from at least one atom in the original cell
                    valids_mask = np.any(weight_mask, axis=1)

                    valid_pos = pos_copy_cartesian[valids_mask]
                    valid_num = num_copy[valids_mask]

                    pos_extended.append(valid_pos)
                    num_extended.append(valid_num)

        pos_extended = np.concatenate(pos_extended)
        num_extended = np.concatenate(num_extended)

        extended_system = System(pos_extended, num_extended, cell,
            coords_are_cartesian=True)

        return extended_system

    def gaussian_sum(self, centers, weights, sigma, grid):
        """Creates function that represents a sum of normalized gaussians with
        an equal standard deviation.

        Args:
            centers (1D np.ndarray): The means of the gaussians.
            weights (1D np.ndarray): The weights for the gaussians.
            sigma (float): The standard deviation of all the gaussians.
            grid (1D np.ndarray): The grid on which to evaluate the gaussians.

        Returns:
            Value of the gaussian sums on the given grid.
        """
        dist2 = grid[np.newaxis, :] - centers[:, np.newaxis]
        dist2 *= dist2
        f = np.sum(weights[:, np.newaxis]*np.exp(-dist2/(2*sigma**2)), axis=0)
        f *= 1/math.sqrt(2*sigma**2*math.pi)

        return f

    def elements(self, system):
        """Calculate the atom count for each element.

        Args:
            system (System): The atomic system.

        Returns:
            1D ndarray: The counts for each element in a list where the index
            of atomic number x is self.atomic_number_to_index[x]
        """
        numbers = system.numbers
        unique, counts = np.unique(numbers, return_counts=True)
        counts_reindexed = np.zeros(self.n_elements)
        for atomic_number, count in zip(unique, counts):
            index = self.atomic_number_to_index[atomic_number]
            counts_reindexed[index] = count

        self._counts = counts_reindexed
        return counts_reindexed

    def inverse_distances(self, system):
        """Calculates the inverse distances for the given atomic positions.

        Args:
            system (System): The atomic system.

        Returns:
            dict: Inverse distances in the form:
            {i: { j: [list of angles] }}. The dictionaries are filled
            so that the entry for pair i and j is in the entry where j>=i.
        """
        inverse_dist = system.inverse_distance_matrix

        numbers = system.numbers
        inv_dist_dict = {}
        for i_atom, i_element in enumerate(numbers):
            for j_atom, j_element in enumerate(numbers):
                if j_atom > i_atom:
                    # Only consider pairs that have one atom in the original
                    # cell
                    if i_atom < self.n_atoms_in_cell or \
                       j_atom < self.n_atoms_in_cell:

                        i_index = self.atomic_number_to_index[i_element]
                        j_index = self.atomic_number_to_index[j_element]

                        # Make sure that j_index >= i_index so that we fill only
                        # the upper triangular part
                        i_index, j_index = sorted([i_index, j_index])

                        old_dict = inv_dist_dict.get(i_index)
                        if old_dict is None:
                            old_dict = {}
                        old_list = old_dict.get(j_index)
                        if old_list is None:
                            old_list = []
                        inv_dist = inverse_dist[i_atom, j_atom]
                        old_list.append(inv_dist)
                        old_dict[j_index] = old_list
                        inv_dist_dict[i_index] = old_dict

        self._inverse_distances = inv_dist_dict
        return inv_dist_dict

    def cosines_and_weights(self, system):
        """Calculates the cosine of the angles and their weights between unique
        three-body combinations.

        Args:
            system (System): The atomic system.

        Returns:
            tuple: (cosine, weights) Cosines of the angles (values between -1
            and 1) in the form {i: { j: {k: [list of angles] }}}. The weights
            corresponding to the angles are stored in a similar dictionary.

            #TODO:
            Some cosines are encountered twice, e.g. the angles for OHH would
            be the same as for HHO. These duplicate values are left out by only
            filling values where k>=i.
        """
        disp_tensor = system.displacement_tensor.astype(np.float32)
        distance_matrix = system.distance_matrix.astype(np.float32)
        numbers = system.numbers

        # Cosines between atoms i-j-k can be found in the tensor:
        # cos_tensor[i, j, k] or equivalently cos_tensor[k, j, i] (symmetric)
        n_atoms = len(numbers)
        cos_tensor = np.empty((n_atoms, n_atoms, n_atoms), dtype=np.float32)
        for i in range(disp_tensor.shape[0]):
            part = 1 - squareform(pdist(disp_tensor[i, :, :], 'cosine'))
            cos_tensor[:, i, :] = part

        # Remove the numerical noise from cosine values.
        np.clip(cos_tensor, -1, 1, cos_tensor)

        cos_dict = {}
        weight_dict = {}
        indices = range(len(numbers))

        # Determine the weighting function
        weighting_function = None
        if self.weighting is not None and self.weighting.get("k3") is not None:
            weighting_function = self.weighting["k3"]["function"]

        # Here we go through all the 3-permutations of the atoms in the system
        permutations = itertools.permutations(indices, 3)
        for i_atom, j_atom, k_atom in permutations:

            # if i_atom < k_atom:
                # continue

            # Only consider triplets that have one atom in the original
            # cell
            if i_atom < self.n_atoms_in_cell or \
               j_atom < self.n_atoms_in_cell or \
               k_atom < self.n_atoms_in_cell:

                i_element = numbers[i_atom]
                j_element = numbers[j_atom]
                k_element = numbers[k_atom]

                i_index = self.atomic_number_to_index[i_element]
                j_index = self.atomic_number_to_index[j_element]
                k_index = self.atomic_number_to_index[k_element]

                # Save information in the part where k_index >= i_index
                if k_index < i_index:
                    continue
                # if i_index > k_index:
                    # k_index, i_index = i_index, k_index

                # Save weights
                if weighting_function is not None:
                    dist1 = distance_matrix[i_atom, j_atom]
                    dist2 = distance_matrix[j_atom, k_atom]
                    dist3 = distance_matrix[k_atom, i_atom]
                    weight = weighting_function(dist1 + dist2 + dist3)
                else:
                    weight = 1

                old_dict_1 = weight_dict.get(i_index)
                if old_dict_1 is None:
                    old_dict_1 = {}
                old_dict_2 = old_dict_1.get(j_index)
                if old_dict_2 is None:
                    old_dict_2 = {}
                old_list_3 = old_dict_2.get(k_index)
                if old_list_3 is None:
                    old_list_3 = []

                old_list_3.append(weight)
                old_dict_2[k_index] = old_list_3
                old_dict_1[j_index] = old_dict_2
                weight_dict[i_index] = old_dict_1

                # Save cosines
                old_dict_1 = cos_dict.get(i_index)
                if old_dict_1 is None:
                    old_dict_1 = {}
                old_dict_2 = old_dict_1.get(j_index)
                if old_dict_2 is None:
                    old_dict_2 = {}
                old_list_3 = old_dict_2.get(k_index)
                if old_list_3 is None:
                    old_list_3 = []
                old_list_3.append(cos_tensor[i_atom, j_atom, k_atom])
                old_dict_2[k_index] = old_list_3
                old_dict_1[j_index] = old_dict_2
                cos_dict[i_index] = old_dict_1

        self._angles = cos_dict
        self._angle_weights = weight_dict
        return cos_dict, weight_dict

    def K1(self, system, space, sigma):
        """Calculates the first order terms where the scalar mapping is the
        number of atoms of a certain type.

        Args:
            system (System): The atomic system.
            start (float): Axis start value
            stop (float): Axis stop value
            dx (float): Axis spacing
            sigma: The smearing width

        Returns:
            1D ndarray: flattened K1 values.
        """
        self._axis_k1 = space
        n_elem = self.n_elements

        # Use sparse matrices for storing the result
        if self.flatten:
            k1 = lil_matrix((1, n_elem*space.size), dtype=np.float32)
        else:
            k1 = np.zeros((n_elem, space.size), dtype=np.float32)

        counts = self.elements(system)

        for i in range(n_elem):
            atomic_number = np.array([self.index_to_atomic_number[i]])
            count = np.array([counts[i]])
            gaussian_sum = self.gaussian_sum(atomic_number, count, sigma, space)

            if self.flatten:
                start = i*space.size
                end = (i+1)*space.size
                k1[0, start:end] = gaussian_sum
            else:
                k1[i, :] = gaussian_sum

            # For debugging
            # elem_i = self.index_to_atomic_number[i]
            # print("Count {} for: {}".format(count, elem_i))
            # mpl.plot(space, convolution)
            # mpl.show()

        return k1

    def K2(self, system, space, sigma):
        """Calculates the second order terms where the scalar mapping is the
        inverse distance between atoms.

        Args:
            system (System): The atomic system.
            start (float): Axis start value
            stop (float): Axis stop value
            dx (float): Axis spacing
            sigma: The smearing width

        Returns:
            1D ndarray: flattened K2 values.
        """
        self._axis_k2 = space
        inv_dist_dict = self.inverse_distances(system)
        # gaussian = self.gaussian(sigma, 4*sigma, dx)
        n_elem = self.n_elements

        if self.flatten:
            k2 = lil_matrix(
                (1, n_elem*(n_elem+1)/2*len(space)), dtype=np.float32)
        else:
            k2 = np.zeros((self.n_elements, self.n_elements, len(space)))

        # Determine the weighting function
        weighting_function = None
        if self.weighting is not None and self.weighting.get("k2") is not None:
            weighting_function = self.weighting["k2"]["function"]

        m = -1
        for i in range(n_elem):
            for j in range(n_elem):
                if j >= i:
                    m += 1
                    try:
                        inv_dist = np.array(inv_dist_dict[i][j])
                    except KeyError:
                        continue

                    # Calculate weights
                    if weighting_function is not None:
                        weights = weighting_function(1/np.array(inv_dist))
                    else:
                        weights = np.ones(len(inv_dist))

                    # Broaden with a gaussian
                    gaussian_sum = self.gaussian_sum(inv_dist, weights, sigma, space)

                    if self.flatten:
                        start = m*space.size
                        end = (m + 1)*space.size
                        k2[0, start:end] = gaussian_sum
                    else:
                        k2[i, j, :] = gaussian_sum

                    # For debugging
                    # elem_i = self.index_to_atomic_number[i]
                    # elem_j = self.index_to_atomic_number[j]
                    # print("Inverse distances {} for: {} {}"
                    #     .format(inv_dist, elem_i, elem_j))
                    # mpl.plot(space, gaussian_sum)
                    # mpl.show()

        return k2

    def K3(self, system, space, sigma):
        """Calculates the third order terms where the scalar mapping is the
        angle between 3 atoms.

        Args:
            system (System): The atomic system.
            start (float): Axis start value
            stop (float): Axis stop value
            dx (float): Axis spacing
            sigma: The smearing width

        Returns:
            1D ndarray: flattened K3 values.
        """
        self._axis_k3 = space
        cos_dict, cos_weight_dict = self.cosines_and_weights(system)
        n_elem = self.n_elements

        if self.flatten:
            k3 = lil_matrix(
                (1, n_elem*n_elem*(n_elem+1)/2*len(space)), dtype=np.float32)
        else:
            k3 = np.zeros((n_elem, n_elem, n_elem, len(space)))

        # Go through the angles, but leave out the duplicate cases by enforcing
        # k >= i. E.g. angles OHH are the same as HHO. This will half the size
        # of the K3 input.
        m = -1
        for i in range(n_elem):
            for j in range(n_elem):
                for k in range(n_elem):
                    if k >= i:
                        m += 1
                        try:
                            cos_values = np.array(cos_dict[i][j][k])
                            cos_weights = np.array(cos_weight_dict[i][j][k])
                        except KeyError:
                            continue

                        # Broaden with a gaussian
                        gaussian_sum = self.gaussian_sum(cos_values, cos_weights, sigma, space)

                        if self.flatten:
                            start = m*space.size
                            end = (m+1)*space.size
                            k3[0, start:end] = gaussian_sum
                        else:
                            k3[i, j, k, :] = gaussian_sum

                        # For debugging
                        # elem_i = self.index_to_atomic_number[i]
                        # elem_j = self.index_to_atomic_number[j]
                        # elem_k = self.index_to_atomic_number[k]
                        # print("Cosines {} for: {}{}{}".format(cos_values, elem_i, elem_j, elem_k))
                        # mpl.plot(space, gaussian_sum)
                        # mpl.show()

        return k3
