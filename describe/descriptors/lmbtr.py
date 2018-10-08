from __future__ import absolute_import, division, print_function
from builtins import super
import math
import numpy as np
from describe.core import System

from ase import Atoms
# from ase.visualize import view

from describe.descriptors import MBTR

from scipy.spatial.distance import cdist


class LMBTR(MBTR):
    """Implementation of local -- per chosen atom -- kind of the Many-body
    tensor representation up to k=3.

    This implementation provides the following geometry functions:

        -k=1: atomic number
        -k=2: inverse distances
        -k=3: cosines of angles

    and the following weighting functions:

        -k=1: unity(=no weighting)
        -k=2: unity(=no weighting), exponential (:math:`e^-(sx)`)
        -k=3: unity(=no weighting), exponential (:math:`e^-(sx)`)

    You can use this descriptor for finite and periodic systems. When dealing
    with periodic systems, it is advisable to use a primitive cell, or if
    supercells are included to use normalization e.g. by volume or by the norm
    of the final vector.

    If flatten=False, a list of dense np.ndarrays for each k in ascending order
    is returned. These arrays are of dimension (n_elements x n_elements x
    n_grid_points), where the elements are sorted in ascending order by their
    atomic number.

    If flatten=True, a scipy.sparse.coo_matrix is returned. This sparse matrix
    is of size (1, n_features), where n_features is given by
    get_number_of_features(). This vector is ordered so that the different
    k-terms are ordered in ascending order, and within each k-term the
    distributions at each entry (i, j, h) of the tensor are ordered in an
    ascending order by (i * n_elements) + (j * n_elements) + (h * n_elements).

    This implementation does not support the use of a non-identity correlation
    matrix.
    """
    decay_factor = math.sqrt(2)*3

    def __init__(
            self,
            atomic_numbers,
            k,
            periodic,
            grid=None,
            weighting=None,
            normalize_gaussians=True,
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
            weighting (dictionary or string): A dictionary of weighting
                function settings for each term. Example:

                    weighting = {
                        "k2": {
                            "function": "unity",
                        },
                        "k3": {
                            "function": "exponential",
                            "scale": 0.5,
                            "cutoff": 1e-3,
                        }
                    }

                Weighting functions should be monotonically decreasing.
                The threshold is used to determine the minimum mount of
                periodic images to consider. The variable 'cutoff' determines
                the value of the weighting function after which the rest of the
                terms will be ignored. The K1 term is 0-dimensional, so
                weighting is not used. Here are the available functions and a
                description for them:

                    "unity": Constant weighting of 1 for all samples.
                    "exponential": Weighting of the form :math:`e^-(sx)`. The
                        parameter :math:`s` is given in the attribute 'scale'.

                The meaning of x changes for different terms as follows:

                    k=1: x = 0
                    k=2: x = Distance between A->B
                    k=3: x = Distance from A->B->C->A.

            normalize_gaussians (bool): Determines whether the gaussians are
                normalized to an area of 1. If false, the normalization factor
                is dropped and the gaussians have the form.
                :math:`e^-(x-\mu)^2/2\sigma^2`
            flatten (bool): Whether the output of create() should be flattened
                to a 1D array. If False, a list of the different tensors is
                provided.

        Raises:
            ValueError if the given k value is not supported, or the weighting
            is not specified for periodic systems.
        """
        # First make sure that there are no atoms with atomic number 0 in the
        # original system, as it is reserved for the ghost atom.
        if 0 in atomic_numbers:
            raise ValueError(
                "Please do not use the atomic number 0 in local MBTR "
                ", as it is reserved for the ghost atom used by the "
                "implementation."
            )
        atomic_numbers = list(atomic_numbers)  # Create a copy so that the original is not modified
        atomic_numbers.append(0)  # The ghost atoms will have atomic number 0
        self.is_virtual = None
        super().__init__(
                    atomic_numbers,
                    k,
                    periodic,
                    grid,
                    weighting,
                    normalize_by_volume=False,
                    normalize_gaussians=normalize_gaussians,
                    flatten=flatten,
                    )

    def update(self):
        """Updates relevant objects attached to LMBTR class, after changing
        one/many values.
        """
        self.atomic_numbers = np.unique(self.atomic_numbers + [0]).tolist()
        super().update()

    def describe(self,
                 system,
                 positions,
                 is_virtual=False,
                 scaled_positions=False
                 ):
        """Return the local many-body tensor representation as a 1D array for the
        given system.

        Args:
            system (System): The system for which the descriptor is created.
            positions (iterable): Positions or atom index of points, from
                which local_mbtr is created. Can be a list of integer numbers
                or a list of xyz-coordinates.
            is_virtual (bool): Determines whether the given position is virtual
                or not. A virtual position does not correspond to any physical
                atom, and is thus not repeated in periodic systems. If set to
                False, the position corresponds to a physical atom which will
                be repeated in periodic systems and may interact with periodic
                copies of itself.
            scaled_positions (boolean): Controls whether the given positions
                are given as scaled to the unit cell basis or not. Scaled
                positions require that a cell is available for the system.

        Returns:
            1D ndarray: The local many-body tensor representations of given
                positions, for k terms, as an array. These are ordered as given
                in positions.
        """
        self.is_virtual = is_virtual

        # Ensure that the atomic number 0 is not present in the system
        if 0 in system.get_atomic_numbers():
            raise ValueError(
                "Please do not use the atomic number 0 in local MBTR "
                ", as it is reserved for the ghost atom used by the "
                "implementation."
            )

        # Ensuring self is updated
        self.update()

        # Checking scaled position
        if scaled_positions:
            if np.linalg.norm(system.get_cell()) == 0:
                raise ValueError(
                    "System doesn't have cell to justify scaled positions."
                )

        # Figure out the atom index or atom location from the given positions
        systems = []
        for i in positions:
            index = None
            new_location = None
            if isinstance(i, (list, tuple, np.ndarray)):
                if scaled_positions:
                    new_location = np.dot(i, system.get_cell())
                else:
                    new_location = np.array(i)
            elif type(i) is int:
                if i >= len(system):
                    raise ValueError(
                        "Atom index: {}, larger than total number of atoms."
                        .format(i)
                    )
                index = i
            else:
                raise ValueError(
                    "Create method requires the argument positions, a list of "
                    "atom indices and/or positions."
                )

            # If a position that does not match any existing atom is given,
            # create a ghost atom and add the old system to it.
            if new_location is not None:

                # Adding a dimension to the center atom location as ASE.Atoms
                # assumes a list of positions.
                new_location = np.expand_dims(new_location, axis=0)

                new_system = System(
                    'X1',
                    positions=new_location
                )
                new_system += system
            # If a specific atom is marked to be in the center, move it to be
            # at index 0 in the system.
            elif index is not None:
                system_copy = system.copy()
                center_atom = system[index]
                center_atom.symbol = "X"
                center_atom = System.from_atoms(Atoms() + center_atom)
                del system_copy[index]
                new_system = center_atom + system_copy

            # Set the periodicity and cell to match the original system, as
            # they are lost in the system concatenation
            new_system.set_cell(system.get_cell())
            new_system.set_pbc(system.get_pbc())

            systems.append(new_system)

        desc = np.empty(len(positions), dtype='object')
        for i, i_system in enumerate(systems):
            i_desc = super().describe(i_system)
            desc[i] = i_desc

        return desc

    def get_original_system_limit(self, system):
        """Used to return the limit of atoms considered to be part of the
        original system. In the local MBTR implementation only the first atom
        is considered, as it is the central ghost atom.
        """
        return 1

    def create_extended_system(self, primitive_system, term_number):
        """Used to create a periodically extended system, that is as small as
        possible by rejecting atoms for which the given weighting will be below
        the given threshold.

        Modified for the local MBTR to only consider distances from the central
        atom and to enable taking the

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
        relative_pos = primitive_system.get_scaled_positions()
        cartesian_pos = np.array(primitive_system.get_positions())
        cell = primitive_system.get_cell()

        # Determine the upper limit of how many copies we need in each cell
        # vector direction. We take as many copies as needed for the
        # exponential weight to come down to the given threshold.
        cell_vector_lengths = np.linalg.norm(cell, axis=1)
        n_copies_axis = np.zeros(3, dtype=int)
        weight_info = self.weighting["k{}".format(term_number)]
        weighting_function = weight_info["function"]
        cutoff = self.weighting["k{}".format(term_number)]["cutoff"]

        if weighting_function == "exponential":
            scale = weight_info["scale"]
            function = lambda x: np.exp(-scale*x)

        for i_axis, axis_length in enumerate(cell_vector_lengths):
            limit_found = False
            n_copies = -1
            while (not limit_found):
                n_copies += 1
                distance = n_copies*cell_vector_lengths[0]

                # For terms k>2 we double the distances to take into
                # account the "loop" that is required.
                if term_number > 2:
                    distance = 2*distance

                weight = function(distance)
                if weight < cutoff:
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

                    # Calculate the positions of the copied atoms and filter
                    # out the atoms that are farther away than the given
                    # cutoff.

                    # If the given position is virtual and does not correspond
                    # to a physical atom, the position is not repeated in the
                    # copies.
                    if self.is_virtual:
                        num_copy = np.array(numbers)[1:]
                        pos_copy = np.array(relative_pos)[1:]

                    # If the given position is not virtual and corresponds to
                    # an actual physical atom, the ghost atom is repeated in
                    # the extended system.
                    else:
                        num_copy = np.array(numbers)
                        pos_copy = np.array(relative_pos)

                    pos_shifted = pos_copy-i*a-j*b-k*c
                    pos_copy_cartesian = np.dot(pos_shifted, cell)

                    # In local MBTR only distances from the central atom to
                    # other atoms are considered.
                    positions_to_consider = np.expand_dims(cartesian_pos[0], axis=0)
                    distances = cdist(pos_copy_cartesian, positions_to_consider)

                    # For terms above k==2 we double the distances to take into
                    # account the "loop" that is required.
                    if term_number > 2:
                        distances *= 2

                    weights = function(distances)
                    weight_mask = weights >= cutoff

                    # Create a boolean mask that says if the atom is within the
                    # range from at least one atom in the original cell
                    valids_mask = np.any(weight_mask, axis=1)

                    valid_pos = pos_copy_cartesian[valids_mask]
                    valid_num = num_copy[valids_mask]

                    pos_extended.append(valid_pos)
                    num_extended.append(valid_num)

        pos_extended = np.concatenate(pos_extended)
        num_extended = np.concatenate(num_extended)

        extended_system = System(
            positions=pos_extended,
            numbers=num_extended,
            cell=cell,
            pbc=False
        )

        return extended_system
