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
from __future__ import absolute_import, division, print_function
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix

from ase import Atoms
import ase.data

from dscribe.core import System
from dscribe.descriptors import MBTR


class LMBTR(MBTR):
    """Implementation of local -- per chosen atom -- kind of the Many-body
    tensor representation up to k=3.

    Notice that the species of the central atom is not encoded in the output,
    but is instead represented by a chemical species X with atomic number 0.
    This allows LMBTR to be also used on general positions not corresponding to
    real atoms. The surrounding environment is encoded by the two- and
    three-body interactions with neighouring atoms. If there is a need to
    distinguish the central species, one can for example train a different
    model for each central species.

    You can choose which terms to include by providing a dictionary in the k2
    or k3 arguments. The k1 term is not used in the local version. This
    dictionary should contain information under three keys: "geometry", "grid"
    and "weighting". See the examples below for how to format these
    dictionaries.

    You can use this descriptor for finite and periodic systems. When dealing
    with periodic systems or when using machine learning models that use the
    Euclidean norm to measure distance between vectors, it is advisable to use
    some form of normalization.

    For the geometry functions the following choices are available:

    * :math:`k=2`:

       * "distance": Pairwise distance in angstroms.
       * "inverse_distance": Pairwise inverse distance in 1/angstrom.

    * :math:`k=3`:

       * "angle": Angle in degrees.
       * "cosine": Cosine of the angle.

    For the weighting the following functions are available:

    * :math:`k=2`:

       * "unity": No weighting.
       * "exp" or "exponential": Weighting of the form :math:`e^{-sx}`

    * :math:`k=3`:

       * "unity": No weighting.
       * "exp" or "exponential": Weighting of the form :math:`e^{-sx}`

    The exponential weighting is motivated by the exponential decay of screened
    Coulombic interactions in solids. In the exponential weighting the
    parameters **cutoff** determines the value of the weighting function after
    which the rest of the terms will be ignored and the parameter **scale**
    corresponds to :math:`s`. The meaning of :math:`x` changes for different
    terms as follows:

    * :math:`k=2`: :math:`x` = Distance between A->B
    * :math:`k=3`: :math:`x` = Distance from A->B->C->A.

    In the grid setup *min* is the minimum value of the axis, *max* is the
    maximum value of the axis, *sigma* is the standard deviation of the
    gaussian broadening and *n* is the number of points sampled on the
    grid.

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
    def __init__(
            self,
            species,
            periodic,
            k2=None,
            k3=None,
            is_center_periodic=None,
            normalize_gaussians=True,
            normalization="none",
            flatten=True,
            sparse=False,
            ):
        """
        Args:
            species (iterable): The chemical species as a list of atomic
                numbers or as a list of chemical symbols. Notice that this is not
                the atomic numbers that are present for an individual system, but
                should contain all the elements that are ever going to be
                encountered when creating the descriptors for a set of systems.
                Keeping the number of chemical speices as low as possible is
                preferable.
            periodic (bool): Determines whether the system is considered to be
                periodic.
            k2 (dict): Dictionary containing the setup for the k=2 term.
                Contains setup for the used geometry function, discretization and
                weighting function. For example::

                    k2 = {
                        "geometry": {"function": "inverse_distance"},
                        "grid": {"min": 0.1, "max": 2, "sigma": 0.1, "n": 50},
                        "weighting": {"function": "exp", "scale": 0.75, "cutoff": 1e-2}
                    }

            k3 (dict): Dictionary containing the setup for the k=3 term.
                Contains setup for the used geometry function, discretization and
                weighting function. For example::

                    k3 = {
                        "geometry": {"function": "angle"},
                        "grid": {"min": 0, "max": 180, "sigma": 5, "n": 50},
                        "weighting" = {"function": "exp", "scale": 0.5, "cutoff": 1e-3}
                    }

            is_center_periodic (bool): Determines whether the central positions
                are periodically repeated or not. If not specified, defaults to
                the value of the "periodic"-parameter. If False, the central
                position is not repeated in periodic systems. If set to True,
                the position will be repeated in periodic systems and may
                interact with periodic copies of itself. Typically set to False
                when studying non-physical positions in periodic systems,
                otherwise set to False.
            normalize_gaussians (bool): Determines whether the gaussians are
                normalized to an area of 1. Defaults to True. If False, the
                normalization factor is dropped and the gaussians have the form.
                :math:`e^{-(x-\mu)^2/2\sigma^2}`
            normalization (str): Determines the method for normalizing the
                output. The available options are:

                * "none": No normalization.
                * "l2_each": Normalize the Euclidean length of each k-term
                  individually to unity.

            flatten (bool): Whether the output should be flattened to a 1D
                array. If False, a dictionary of the different tensors is
                provided, containing the values under keys: "k1", "k2", and
                "k3":
            sparse (bool): Whether the output should be a sparse matrix or a
                dense numpy array.
        """
        super().__init__(
            k1=None,
            k2=k2,
            k3=k3,
            periodic=periodic,
            species=species,
            normalization=normalization,
            normalize_gaussians=normalize_gaussians,
            flatten=flatten,
            sparse=sparse,
        )
        # These attributes have to be set after the MBTR constructor to
        # override the defaults.
        self._is_local = True
        self._interaction_limit = 1

        # These attributes can be set whenever
        self.is_center_periodic = periodic if is_center_periodic is None else is_center_periodic

        # Check that center is not defined as periodic if the system is not
        if self.periodic is False and self.is_center_periodic is True:
            raise ValueError(
                "Cannot make the central atom periodic if the whole system is "
                "not periodic."
            )

        self.updated = True

    def create(self, system, positions=None, n_jobs=1, verbose=False):
        """Return the LMBTR output for the given systems and given positions.

        Args:
            system (:class:`ase.Atoms` or list of :class:`ase.Atoms`): One or
                many atomic structures.
            positions (list): Positions where to calculate LMBTR. Can be
                provided as cartesian positions or atomic indices. If no
                positions are defined, the LMBTR output will be created for all
                atoms in the system. When calculating LMBTR for multiple
                systems, provide the positions as a list for each system.
            n_jobs (int): Number of parallel jobs to instantiate. Parallellizes
                the calculation across samples. Defaults to serial calculation
                with n_jobs=1.
            verbose(bool): Controls whether to print the progress of each job
                into to the console.

        Returns:
            np.ndarray | scipy.sparse.csr_matrix: The LMBTR output for the given
            systems and positions. The return type depends on the
            'sparse'-attribute. The first dimension is determined by the amount
            of positions and systems and the second dimension is determined by
            the get_number_of_features()-function.
        """
        # If single system given, skip the parallelization
        if isinstance(system, (Atoms, System)):
            return self.create_single(system, positions)

        # Combine input arguments
        n_samples = len(system)
        inp = [(i_sys, i_pos) for i_sys, i_pos in zip(system, positions)]

        # Here we precalculate the size for each job to preallocate memory.
        if self._flatten:
            n_samples = len(system)
            k, m = divmod(n_samples, n_jobs)
            jobs = (inp[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_jobs))
            output_sizes = []
            for i_job in jobs:
                n_desc = 0
                if positions is None:
                    n_desc = 0
                    for job in i_job:
                        n_desc += len(job[0])
                else:
                    n_desc = 0
                    for i_sample, i_pos in i_job:
                        if i_pos is not None:
                            n_desc += len(i_pos)
                        else:
                            n_desc += len(i_sample)
                output_sizes.append(n_desc)
        else:
            output_sizes = None

        # Create in parallel
        output = self.create_parallel(inp, self.create_single, n_jobs, output_sizes, verbose=verbose)

        return output

    def create_single(
            self,
            system,
            positions,
            ):
        """Return the local many-body tensor representation for the given
        system and positions.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.
            positions (iterable): Positions or atom index of points, from
                which local_mbtr is created. Can be a list of integer numbers
                or a list of xyz-coordinates.

        Returns:
            1D ndarray: The local many-body tensor representations of given
            positions, for k terms, as an array. These are ordered as given in
            positions.
        """
        # Transform the input system into the internal System-object
        system = self.get_system(system)

        # Check that the system does not have elements that are not in the list
        # of atomic numbers
        atomic_number_set = set(system.get_atomic_numbers())
        self.check_atomic_numbers(atomic_number_set)

        # Ensure that the atomic number 0 is not present in the system
        if 0 in atomic_number_set:
            raise ValueError(
                "Please do not use the atomic number 0 in local MBTR"
                ", as it is reserved for the ghost atom used by the "
                "implementation."
            )

        # Figure out the atom index or atom location from the given positions
        systems = []

        # Positions specified, use them
        if positions is not None:

            # Check validity of position definitions and create final cartesian
            # position list
            list_positions = []
            if len(positions) == 0:
                raise ValueError(
                    "The argument 'positions' should contain a non-empty set of"
                    " atomic indices or cartesian coordinates with x, y and z "
                    "components."
                )
            for i in positions:
                if np.issubdtype(type(i), np.integer):
                    i_len = len(system)
                    if i >= i_len or i < 0:
                        raise ValueError(
                            "The provided index {} is not valid for the system "
                            "with {} atoms.".format(i, i_len)
                        )
                    list_positions.append(system.get_positions()[i])
                elif isinstance(i, (list, tuple, np.ndarray)):
                    if len(i) != 3:
                        raise ValueError(
                            "The argument 'positions' should contain a "
                            "non-empty set of atomic indices or cartesian "
                            "coordinates with x, y and z components."
                        )
                    list_positions.append(i)
                else:
                    raise ValueError(
                        "Create method requires the argument 'positions', a "
                        "list of atom indices and/or positions."
                    )

        for i_pos in positions:
            # Position designated as cartesian position, add a new atom at that
            # location with the chemical element X and place is as the first
            # atom in the system. The interaction limit makes sure that only
            # interactions of this first atom to every other atom are
            # considered.
            if isinstance(i_pos, (list, tuple, np.ndarray)):
                if len(i_pos) != 3:
                    raise ValueError(
                        "The argument 'positions' should contain a "
                        "non-empty set of atomic indices or cartesian "
                        "coordinates with x, y and z components."
                    )
                i_pos = np.array(i_pos)
                i_pos = np.expand_dims(i_pos, axis=0)
                new_system = System('X', positions=i_pos)
                new_system += system
            # Position designated as integer, use the atom at that index as
            # center. For the calculation this central atoms is shifted to be
            # the first atom in the system, and the interaction limit makes
            # sure that only interactions of this first atom to every other
            # atom are considered.
            elif np.issubdtype(type(i_pos), np.integer):
                new_system = Atoms()
                center_atom = system[i_pos]
                new_system += center_atom
                new_system.set_atomic_numbers([0])
                system_copy = system.copy()
                del system_copy[i_pos]
                new_system += system_copy
            else:
                raise ValueError(
                    "Create method requires the argument 'positions', a "
                    "list of atom indices and/or positions."
                )

            # Set the periodicity and cell to match the original system, as
            # they are lost in the system concatenation
            new_system.set_cell(system.get_cell())
            new_system.set_pbc(system.get_pbc())

            systems.append(new_system)

        # Request MBTR for each position. Return type depends on flattening and
        # whether a spares of dense result is requested.
        n_pos = len(positions)
        n_features = self.get_number_of_features()
        if self._flatten and self._sparse:
            data = []
            cols = []
            rows = []
            row_offset = 0
            for i, i_system in enumerate(systems):
                i_res = super().create_single(i_system)
                data.append(i_res.data)
                rows.append(i_res.row + row_offset)
                cols.append(i_res.col)

                # Increase the row offset
                row_offset += 1

            # Saves the descriptors as a sparse matrix
            data = np.concatenate(data)
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            desc = coo_matrix((data, (rows, cols)), shape=(n_pos, n_features), dtype=np.float32)
        else:
            if self._flatten and not self._sparse:
                desc = np.empty((n_pos, n_features), dtype=np.float32)
            else:
                desc = np.empty((n_pos), dtype='object')
            for i, i_system in enumerate(systems):
                i_desc = super().create_single(i_system)
                desc[i] = i_desc

        return desc

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        The number of features for the LMBTR is calculated as follows:

        For the pair term (k=2), only pairs where at least one of the atom is
        the central atom (in periodic systems the central atom may connect to
        itself) are considered. This means that there are only as many
        combinations as there are different elements to pair the central atom
        with (n_elem). This nmber of combinations is the multiplied by the
        discretization of the k=2 grid.

        For the three-body term (k=3), only triplets where at least one of the
        atoms is the central atom (in periodic systems the central atom may
        connect to itself) and the k >= i (symmetry) are considered. This means
        that as k runs from 0 to n-1, where n is the number of elements, there
        are n + k combinations that fill this rule. This sum becomes:
        :math:`\sum_{k=0}^{n-1} n + k = n^2+(n-1)*n/2`. This number of
        combinations is the multiplied by the discretization of the k=3 grid.

        Returns:
            int: Number of features for this descriptor.
        """
        n_features = 0
        n_elem = self.n_elements

        if self.k2 is not None:
            n_k2_grid = self.k2["grid"]["n"]
            n_k2 = (n_elem)*n_k2_grid
            n_features += n_k2
        if self.k3 is not None:
            n_k3_grid = self.k3["grid"]["n"]
            n_k3 = n_elem*(3*n_elem-1)*n_k3_grid/2  # = (n_elem*n_elem + (n_elem-1)*n_elem/2)*n_k3_grid
            n_features += n_k3

        return int(n_features)

    def get_k2_convolution(self, grid):
        """Calculates the second order terms where the scalar mapping is the
        inverse distance between atoms.

        Args:
            grid (dict): The grid settings

        Returns:
            1D ndarray: flattened K2 values.
        """
        start = grid["min"]
        stop = grid["max"]
        n = grid["n"]
        self._axis_k2 = np.linspace(start, stop, n)

        k2_geoms, k2_weights = self._k2_geoms, self._k2_weights
        n_elem = self.n_elements

        # Depending of flattening, use either a sparse matrix or a dense one.
        if self.flatten:
            k2 = lil_matrix(
                (1, n_elem*n), dtype=np.float32)

            for key in k2_geoms.keys():
                i = key[1]

                geoms = np.array(k2_geoms[key])
                weights = np.array(k2_weights[key])

                # Broaden with a gaussian
                gaussian_sum = self.gaussian_sum(geoms, weights, grid)

                m = i
                start = int(m*n)
                end = int((m+1)*n)
                k2[0, start:end] = gaussian_sum
        else:
            k2 = np.zeros((n_elem, n), dtype=np.float32)
            for key in k2_geoms.keys():
                i = key[0]

                geoms = np.array(k2_geoms[key])
                weights = np.array(k2_weights[key])

                # Broaden with a gaussian
                gaussian_sum = self.gaussian_sum(geoms, weights, grid)

                k2[i, :] = gaussian_sum

        return k2

    def get_k3_convolution(self, grid):
        """Calculates the third order terms where the scalar mapping is the
        angle between 3 atoms.

        Args:
            grid (dict): The grid settings

        Returns:
            1D ndarray: flattened K3 values.
        """
        start = grid["min"]
        stop = grid["max"]
        n = grid["n"]
        self._axis_k3 = np.linspace(start, stop, n)

        k3_geoms, k3_weights = self._k3_geoms, self._k3_weights
        n_elem = self.n_elements

        # Depending of flattening, use either a sparse matrix or a dense one.
        if self.flatten:
            k3 = lil_matrix(
                (1, int((n_elem*(3*n_elem-1)*n/2))), dtype=np.float32
            )
            for key in k3_geoms.keys():
                i = key[0]
                j = key[1]
                k = key[2]

                geoms = np.array(k3_geoms[key])
                weights = np.array(k3_weights[key])

                # Broaden with a gaussian
                gaussian_sum = self.gaussian_sum(geoms, weights, grid)

                # This is the index of the spectrum. It is given by enumerating the
                # elements of a three-dimensional array and only considering
                # elements for which k>=i and i || j == 0. The enumeration begins
                # from [0, 0, 0], and ends at [n_elem, n_elem, n_elem], looping the
                # elements in the order k, i, j.
                if j == 0:
                    m = k + i*n_elem - i*(i+1)/2
                else:
                    m = n_elem*(n_elem+1)/2+(j-1)*n_elem + k
                start = int(m*n)
                end = int((m+1)*n)

                k3[0, start:end] = gaussian_sum
        else:
            k3 = np.zeros((n_elem, n_elem, n_elem, n), dtype=np.float32)
            for key in k3_geoms.keys():
                i = key[0]
                j = key[1]
                k = key[2]

                geoms = np.array(k3_geoms[key])
                weights = np.array(k3_weights[key])

                # Broaden with a gaussian
                gaussian_sum = self.gaussian_sum(geoms, weights, grid)

                k3[i, j, k, :] = gaussian_sum

        return k3

    def get_location(self, species):
        """Can be used to query the location of a species combination in the
        the flattened output.

        Args:
            species(tuple): A tuple containing a species combination as
            chemical symbols or atomic numbers. The central atom is marked as
            species "X". The tuple can be for example ("X", "O") or ("X", "O",
            "H")

        Returns:
            slice: slice containing the location of the specified species
            combination. The location is given as a python slice-object, that
            can be directly used to target ranges in the output.

        Raises:
            ValueError: If the requested species combination is not in the
            output or if invalid species defined.
        """
        # Check that the corresponding part is calculated
        k = len(species)
        term = getattr(self, "k{}".format(k))
        if term is None:
            raise ValueError(
                "Cannot retrieve the location for {}, as the term {} has not "
                "been specifed.".format(species, term)
            )

        # Change chemical elements into atomic numbers
        numbers = []
        for specie in species:
            if isinstance(specie, str):
                try:
                    specie = ase.data.atomic_numbers[specie]
                except KeyError:
                    raise ValueError("Invalid chemical species")
            numbers.append(specie)

        # Change into internal indexing
        numbers = [self.atomic_number_to_index[x] for x in numbers]
        n_elem = self.n_elements

        # k=2
        if len(numbers) == 2:
            if numbers[0] > numbers[1]:
                numbers = list(reversed(numbers))

            n2 = self.k2["grid"]["n"]
            j = numbers[1]
            # if i != "X":
                # raise ValueError(
                    # "Local MBTR does not contain k=2 terms without the central "
                    # "species X."
                # )
            m = j
            start = int(m*n2)
            end = int((m+1)*n2)

        # k=3
        if len(numbers) == 3:
            if numbers[0] > numbers[2]:
                numbers = list(reversed(numbers))

            n3 = self.k3["grid"]["n"]
            i = numbers[0]
            j = numbers[1]
            k = numbers[2]
            # if i != "X" or j != "X":
                # raise ValueError(
                    # "Local MBTR does not contain k=3 terms without the central "
                    # "species X."
                # )

            # This is the index of the spectrum. It is given by enumerating the
            # elements of a three-dimensional array and only considering
            # elements for which k>=i and i || j == 0. The enumeration begins
            # from [0, 0, 0], and ends at [n_elem, n_elem, n_elem], looping the
            # elements in the order k, i, j.
            if j == 0:
                m = k + i*n_elem - i*(i+1)/2
            else:
                m = n_elem*(n_elem+1)/2+(j-1)*n_elem + k

            offset = 0
            if self.k2 is not None:
                n2 = self.k2["grid"]["n"]
                offset += n_elem*n2
            start = int(offset+m*n3)
            end = int(offset+(m+1)*n3)

        return slice(start, end)

        # Fix ordering
        # if len(numbers) == 2:
            # if numbers[0] > numbers[1]:
                # numbers = reversed(numbers)
        # if len(numbers) == 3:
            # if numbers[0] > numbers[2]:
                # numbers = reversed(numbers)

        # loc = self._locations
        # start, end = loc[tuple(numbers)]

        # return slice(start, end)

    # def calculate_locations(self):
        # """Used to calculate the locations of species combinations in the
        # flattened output.
        # """
        # k2updated = False
        # k3updated = False
        # if self.k2 is not None:
            # k2updated = self.k2["grid"].nupdated
        # if self.k3 is not None:
            # k3updated = self.k3["grid"].nupdated

        # if self.updated or k2updated or k3updated:
            # n_elem = self.n_elements
            # locations = {}
            # locations_each = {}

            # # k=2
            # offset = 0
            # if self.k2 is not None:
                # n2 = self.k2["grid"]["n"]
                # m = 0
                # for i in range(n_elem):
                    # start = m*n2
                    # end = (m+1)*n2
                    # locations_each[(0, i)] = start, end
                    # locations[(0, i)] = start, end
                    # m += 1
                # offset = m*n2

            # # k=3
            # if self.k3 is not None:
                # n3 = self.k3["grid"]["n"]
                # m = 0
                # for i in range(n_elem):
                    # for j in range(n_elem):
                        # for k in range(n_elem):
                            # if k >= i and (i == 0 or j == 0 or k == 0):
                                # start = m*n3
                                # end = (m+1)*n3
                                # locations_each[(i, j, k)] = start, end
                                # locations[(i, j, k)] = offset+start, offset+end
                                # m += 1

            # self._locations = locations
            # self._locations_each = locations_each
            # self.updated = False
            # if self.k2 is not None:
                # self.k2["grid"].nupdated = False
            # if self.k3 is not None:
                # self.k3["grid"].nupdated = False

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        """Used to check the validity of given atomic numbers and to initialize
        the C-memory layout for them.

        Args:
            value(iterable): Chemical species either as a list of atomic
                numbers or list of chemical symbols.
        """
        # The species are stored as atomic numbers for internal use.
        self._set_species(value)

        # The atomic number 0 is reserved for ghost atoms in this
        # implementation.
        if 0 in self._atomic_number_set:
            raise ValueError(
                "The atomic number 0 is reserved for the ghost atoms in this "
                "implementation."
            )
        self._atomic_number_set.add(0)
        indices = np.searchsorted(self._atomic_numbers, 0)
        self._atomic_numbers = np.insert(self._atomic_numbers, indices, 0)

        # Setup mappings between atom indices and types together with some
        # statistics
        self.atomic_number_to_index = {}
        self.index_to_atomic_number = {}
        for i_atom, atomic_number in enumerate(self._atomic_numbers):
            self.atomic_number_to_index[atomic_number] = i_atom
            self.index_to_atomic_number[i_atom] = atomic_number
        self.n_elements = len(self._atomic_numbers)
        self.max_atomic_number = max(self._atomic_numbers)
        self.min_atomic_number = min(self._atomic_numbers)

        # Recalculate locations
        self.updated = True
