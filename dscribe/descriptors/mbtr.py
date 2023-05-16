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
import sys
import math
import numpy as np

from ase import Atoms
import ase.data

from dscribe.core import System
from dscribe.descriptors.descriptorglobal import DescriptorGlobal
from dscribe.ext import MBTRWrapper
import dscribe.utils.geometry


k1_geometry_functions = set(["atomic_number"])
k2_geometry_functions = set(["distance", "inverse_distance"])
k3_geometry_functions = set(["angle", "cosine"])


def check_grid(grid: dict):
    """Used to ensure that the given grid settings are valid.

    Args:
        grid(dict): Dictionary containing the grid setup.
    """
    msg = "The grid information is missing the value for {}"
    val_names = ["min", "max", "sigma", "n"]
    for val_name in val_names:
        try:
            grid[val_name]
        except Exception:
            raise KeyError(msg.format(val_name))

    # Make the n into integer
    grid["n"] = int(grid["n"])
    if grid["min"] >= grid["max"]:
        raise ValueError("The min value should be smaller than the max value.")


def check_geometry(geometry: dict):
    """Used to ensure that the given geometry settings are valid.

    Args:
        geometry: Dictionary containing the geometry setup.
    """

    if "function" in geometry:
        function = geometry["function"]
        valid_functions = (
            k1_geometry_functions | k2_geometry_functions | k3_geometry_functions
        )
        if function not in valid_functions:
            raise ValueError(
                f"Unknown geometry function. Please use one of the following: {sorted(list(valid_functions))}"
            )
    else:
        raise ValueError("Please specify a geometry function.")


def check_weighting(k: int, weighting: dict, periodic: bool):
    """Used to ensure that the given weighting settings are valid.

    Args:
        k: The MBTR degree.
        weighting: Dictionary containing the weighting setup.
        periodic: Whether the descriptor is periodic or not.
    """
    if weighting is not None:
        if k == 1:
            valid_functions = set(["unity"])
        elif k == 2:
            valid_functions = set(["unity", "exp", "inverse_square"])
        elif k == 3:
            valid_functions = set(["unity", "exp", "smooth_cutoff"])
        function = weighting.get("function")
        if function not in valid_functions:
            raise ValueError(
                f"Unknown weighting function specified for k={k}. Please use one of the following: {sorted(list(valid_functions))}"
            )
        else:
            if function == "exp":
                if "threshold" not in weighting:
                    raise ValueError("Missing value for 'threshold' in the weighting.")
                if "scale" not in weighting and "r_cut" not in weighting:
                    raise ValueError(
                        "Provide either 'scale' or 'r_cut' in the weighting."
                    )
                if "scale" in weighting and "r_cut" in weighting:
                    raise ValueError(
                        "Provide either 'scale' or 'r_cut', not both in the weighting."
                    )
            elif function == "inverse_square":
                if "r_cut" not in weighting:
                    raise ValueError("Missing value for 'r_cut' in the weighting.")
            elif function == "smooth_cutoff":
                if "r_cut" not in weighting:
                    raise ValueError("Missing value for 'r_cut' in the weighting.")

    # Check that weighting function is specified for periodic systems
    if periodic and k > 1:
        valid = False
        if weighting is not None:
            function = weighting.get("function")
            if function is not None:
                if function != "unity":
                    valid = True
        if not valid:
            raise ValueError("Periodic systems need to have a weighting function.")


class MBTR(DescriptorGlobal):
    """Implementation of the Many-body tensor representation.

    You can use this descriptor for finite and periodic systems. When dealing
    with periodic systems or when using machine learning models that use the
    Euclidean norm to measure distance between vectors, it is advisable to use
    some form of normalization. This implementation does not support the use of
    a non-identity correlation matrix.
    """

    def __init__(
        self,
        geometry=None,
        grid=None,
        weighting=None,
        normalize_gaussians=True,
        normalization="none",
        species=None,
        periodic=False,
        sparse=False,
        dtype="float64",
    ):
        """
        Args:
            geometry (dict): Setup the geometry function.
                For example::

                "geometry": {"function": "atomic_number"}

                The geometry function determines the degree :math:`k` for MBTR.
                The order :math:`k` tells how many atoms are involved in the
                calculation and thus also heavily influence the computational
                time.

                The following geometry functions are available:

                * :math:`k=1`
                    * ``"atomic_number"``: The atomic number.
                * :math:`k=2`
                    * ``"distance"``: Pairwise distance in angstroms.
                    * ``"inverse_distance"``: Pairwise inverse distance in 1/angstrom.
                * :math:`k=3`
                    * ``"angle"``: Angle in degrees.
                    * ``"cosine"``: Cosine of the angle.

            grid (dict): Setup the discretization grid.
                For example::

                "grid": {"min": 0.1, "max": 2, "sigma": 0.1, "n": 50}

                In the grid setup *min* is the minimum value of the axis, *max*
                is the maximum value of the axis, *sigma* is the standard
                deviation of the gaussian broadening and *n* is the number of
                points sampled on the grid.

            weighting (dict): Setup the weighting function and its parameters.
                For example::

                "weighting" : {"function": "exp", "r_cut": 10, "threshold": 1e-3}

                The following weighting functions are available:

                * :math:`k=1`
                    * ``"unity"``: No weighting.
                * :math:`k=2`
                    * ``"unity"``: No weighting.
                    * ``"exp"``: Weighting of the form :math:`e^{-sx}`
                    * ``"inverse_square"``: Weighting of the form :math:`1/(x^2)`
                * :math:`k=3`
                    * ``"unity"``: No weighting.
                    * ``"exp"``: Weighting of the form :math:`e^{-sx}`
                    * ``"smooth_cutoff"``: Weighting of the form :math:`f_{ij}f_{ik}`,
                        where :math:`f = 1+y(x/r_{cut})^{y+1}-(y+1)(x/r_{cut})^{y}`

                The meaning of :math:`x` changes for different terms as follows:

                * For :math:`k=2`: :math:`x` = Distance between A->B
                * For :math:`k=3`: :math:`x` = Distance from A->B->C->A.

                The exponential weighting is motivated by the exponential decay
                of screened Coulombic interactions in solids. In the exponential
                weighting the parameters **threshold** determines the value of
                the weighting function after which the rest of the terms will be
                ignored. Either the parameter **scale** or **r_cut** can be used
                to determine the parameter :math:`s`: **scale** directly
                corresponds to this value whereas **r_cut** can be used to
                indirectly determine it through :math:`s=-\log()`:.

                The inverse square and smooth cutoff function weightings use a
                cutoff parameter **r_cut**, which is a radial distance after
                which the rest of the atoms will be ignored. For the smooth
                cutoff function, additional weighting key **sharpness** can be
                added, which changes the value of :math:`y`. If a value for it
                is not provided, it defaults to `2`.

            normalize_gaussians (bool): Determines whether the gaussians are
                normalized to an area of 1. Defaults to True. If False, the
                normalization factor is dropped and the gaussians have the form.
                :math:`e^{-(x-\mu)^2/2\sigma^2}`
            normalization (str): Determines the method for normalizing the
                output. The available options are:

                * ``"none"``: No normalization.
                * ``"l2"``: Normalize the Euclidean length of the output to unity.
                * ``"n_atoms"``: Normalize the output by dividing it with the number
                  of atoms in the system. If the system is periodic, the number
                  of atoms is determined from the given unit cell.
                * ``"valle_oganov"``: Use Valle-Oganov descriptor normalization, with
                  system cell volume and numbers of different atoms in the cell.
            species (iterable): The chemical species as a list of atomic
                numbers or as a list of chemical symbols. Notice that this is not
                the atomic numbers that are present for an individual system, but
                should contain all the elements that are ever going to be
                encountered when creating the descriptors for a set of systems.
                Keeping the number of chemical speices as low as possible is
                preferable.
            periodic (bool): Set to true if you want the descriptor output to
                respect the periodicity of the atomic systems (see the
                pbc-parameter in the constructor of ase.Atoms).
            sparse (bool): Whether the output should be a sparse matrix or a
                dense numpy array.
            dtype (str): The data type of the output. Valid options are:

                    * ``"float32"``: Single precision floating point numbers.
                    * ``"float64"``: Double precision floating point numbers.
        """
        super().__init__(periodic=periodic, sparse=sparse, dtype=dtype)
        self.system = None
        self.geometry = geometry
        self.grid = grid
        self.weighting = weighting
        self.species = species
        self.normalization = normalization
        self.normalize_gaussians = normalize_gaussians

        if self.normalization == "valle_oganov" and not periodic:
            raise ValueError(
                "Valle-Oganov normalization does not support non-periodic systems."
            )

        # Initializing .create() level variables
        self._interaction_limit = None

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        check_grid(value)
        self._grid = value

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        check_geometry(value)
        k_map = {
            "atomic_number": 1,
            "distance": 2,
            "inverse_distance": 2,
            "angle": 3,
            "cosine": 3,
        }
        self.k = k_map[value["function"]]
        self._geometry = value

    @property
    def weighting(self):
        return self._weighting

    @weighting.setter
    def weighting(self, value):
        check_weighting(self.k, value, self.periodic)
        self._weighting = value

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

    @property
    def normalization(self):
        return self._normalization

    @normalization.setter
    def normalization(self, value):
        """Checks that the given normalization is valid.

        Args:
            value(str): The normalization method to use.
        """
        norm_options = set(("none", "l2", "n_atoms", "valle_oganov"))
        if value not in norm_options:
            raise ValueError(
                "Unknown normalization option given. Please use one of the "
                "following: {}.".format(", ".join(sorted(list(norm_options))))
            )
        self._normalization = value

    def create(self, system, n_jobs=1, only_physical_cores=False, verbose=False):
        """Return MBTR output for the given systems.

        Args:
            system (:class:`ase.Atoms` or list of :class:`ase.Atoms`): One or many atomic structures.
            n_jobs (int): Number of parallel jobs to instantiate. Parallellizes
                the calculation across samples. Defaults to serial calculation
                with n_jobs=1. If a negative number is given, the used cpus
                will be calculated with, n_cpus + n_jobs, where n_cpus is the
                amount of CPUs as reported by the OS. With only_physical_cores
                you can control which types of CPUs are counted in n_cpus.
            only_physical_cores (bool): If a negative n_jobs is given,
                determines which types of CPUs are used in calculating the
                number of jobs. If set to False (default), also virtual CPUs
                are counted.  If set to True, only physical CPUs are counted.
            verbose(bool): Controls whether to print the progress of each job
                into to the console.

        Returns:
            np.ndarray | sparse.COO: MBTR for the given systems. The return type
            depends on the 'sparse' attribute.
        """
        # Combine input arguments
        system = [system] if isinstance(system, Atoms) else system
        inp = [(i_sys,) for i_sys in system]

        # Determine if the outputs have a fixed size
        static_size = [self.get_number_of_features()]

        # Create in parallel
        output = self.create_parallel(
            inp,
            self.create_single,
            n_jobs,
            static_size,
            only_physical_cores,
            verbose=verbose,
        )

        return output

    def create_single(self, system):
        """Return the many-body tensor representation for the given system.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.

        Returns:
            np.ndarray | sparse.COO: A single concatenated output vector is
            returned, either as a sparse or a dense vector.
        """
        # Ensuring variables are re-initialized when a new system is introduced
        self.system = system
        self._interaction_limit = len(system)

        # Check that the system does not have elements that are not in the list
        # of atomic numbers
        self.check_atomic_numbers(system.get_atomic_numbers())

        mbtr, _ = getattr(self, f"_get_k{self.k}")(system, True, False)

        # Handle normalization
        if self.normalization == "l2":
            mbtr /= np.linalg.norm(np.array(mbtr.data))
        elif self.normalization == "n_atoms":
            n_atoms = len(system)
            mbtr /= n_atoms

        return mbtr

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        n_elem = self.n_elements
        n_grid = self.grid["n"]

        if self.k == 1:
            n_features = n_elem * n_grid
        if self.k == 2:
            n_features = (n_elem * (n_elem + 1) / 2) * n_grid
        if self.k == 3:
            n_features = (n_elem * n_elem * (n_elem + 1) / 2) * n_grid

        return int(n_features)

    def get_location(self, species):
        """Can be used to query the location of a species combination in the
        the output.

        Args:
            species(tuple): A tuple containing a species combination as
                chemical symbols or atomic numbers. The tuple can be for example
                ("H"), ("H", "O") or ("H", "O", "H").

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
        if self.k is not k:
            raise ValueError(
                "Cannot retrieve the location for {}, as the term k={} has not "
                "been specified.".format(species, k)
            )

        # Change chemical elements into atomic numbers
        numbers = []
        for specie in species:
            if isinstance(specie, str):
                try:
                    specie = ase.data.atomic_numbers[specie]
                except KeyError:
                    raise ValueError("Invalid chemical species: {}".format(specie))
            numbers.append(specie)

        # Check that species exists
        self.check_atomic_numbers(numbers)

        # Change into internal indexing
        numbers = [self.atomic_number_to_index[x] for x in numbers]
        n_elem = self.n_elements

        n = self.grid["n"]
        if k == 1:
            i = numbers[0]
            m = i
            start = int(m * n)
            end = int((m + 1) * n)

        # k=2
        if k == 2:
            if numbers[0] > numbers[1]:
                numbers = list(reversed(numbers))

            i = numbers[0]
            j = numbers[1]

            # This is the index of the spectrum. It is given by enumerating the
            # elements of an upper triangular matrix from left to right and top
            # to bottom.
            m = j + i * n_elem - i * (i + 1) / 2
            start = int(m * n)
            end = int((m + 1) * n)

        # k=3
        if k == 3:
            if numbers[0] > numbers[2]:
                numbers = list(reversed(numbers))

            i = numbers[0]
            j = numbers[1]
            k = numbers[2]

            # This is the index of the spectrum. It is given by enumerating the
            # elements of a three-dimensional array where for valid elements
            # k>=i. The enumeration begins from [0, 0, 0], and ends at [n_elem,
            # n_elem, n_elem], looping the elements in the order k, i, j.
            m = j * n_elem * (n_elem + 1) / 2 + k + i * n_elem - i * (i + 1) / 2
            start = int(m * n)
            end = int((m + 1) * n)

        return slice(start, end)

    def _get_k1(self, system, return_descriptor, return_derivatives):
        """Calculates the first order term and/or its derivatives with
        regard to atomic positions.

        Returns:
            1D or 3D ndarray: K1 values. Returns a 1D array. If
                return_descriptor=False, returns an array of shape (0).
            3D ndarray: K1 derivatives. If return_derivatives=False, returns an
                array of shape (0,0,0).
        """
        start = self.grid["min"]
        stop = self.grid["max"]
        n = self.grid["n"]
        sigma = self.grid["sigma"]

        n_elem = self.n_elements
        n_features = n_elem * n

        if return_descriptor:
            # Determine the geometry function
            geom_func_name = self.geometry["function"]

            cmbtr = MBTRWrapper(
                self.atomic_number_to_index,
                self._interaction_limit,
                np.zeros((len(system), 3), dtype=int),
            )

            k1 = np.zeros((n_features), dtype=np.float64)
            cmbtr.get_k1(
                k1,
                system.get_atomic_numbers(),
                geom_func_name.encode(),
                b"unity",
                {},
                start,
                stop,
                sigma,
                n,
            )
        else:
            k1 = np.zeros((0), dtype=np.float64)

        if return_derivatives:
            k1_d = np.zeros((self._interaction_limit, 3, n_features), dtype=np.float64)
        else:
            k1_d = np.zeros((0, 0, 0), dtype=np.float64)

        # Denormalize if requested
        if not self.normalize_gaussians:
            max_val = 1 / (sigma * math.sqrt(2 * math.pi))
            k1 /= max_val
            k1_d /= max_val

        # Convert to the final output precision.
        if self.dtype == "float32":
            k1 = k1.astype(self.dtype)
            k1_d = k1_d.astype(self.dtype)

        return (k1, k1_d)

    def _get_k2(self, system, return_descriptor, return_derivatives):
        """Calculates the second order term and/or its derivatives with
        regard to atomic positions.
        Returns:
            1D ndarray:   K2 values. Returns a 1D array. If
                return_descriptor=False, returns an array of shape (0).
            3D ndarray: K2 derivatives. If return_derivatives=False, returns an
                array of shape (0,0,0).
        """
        start = self.grid["min"]
        stop = self.grid["max"]
        n = self.grid["n"]
        sigma = self.grid["sigma"]

        # Determine the weighting function and possible radial cutoff
        r_cut = None
        parameters = {}
        if self.weighting is not None:
            weighting_function = self.weighting["function"]
            if weighting_function == "exp":
                threshold = self.weighting["threshold"]
                r_cut = self.weighting.get("r_cut")
                scale = self.weighting.get("scale")
                if scale is not None and r_cut is None:
                    r_cut = -math.log(threshold) / scale
                elif scale is None and r_cut is not None:
                    scale = -math.log(threshold) / r_cut
                parameters = {b"scale": scale, b"threshold": threshold}
            elif weighting_function == "inverse_square":
                r_cut = self.weighting["r_cut"]
        else:
            weighting_function = "unity"

        # Determine the geometry function
        geom_func_name = self.geometry["function"]

        # If needed, create the extended system
        if self.periodic:
            centers = system.get_positions()
            ext_system, cell_indices = dscribe.utils.geometry.get_extended_system(
                system, r_cut, centers, return_cell_indices=True
            )
            ext_system = System.from_atoms(ext_system)
        else:
            ext_system = System.from_atoms(system)
            cell_indices = np.zeros((len(system), 3), dtype=int)

        cmbtr = MBTRWrapper(
            self.atomic_number_to_index, self._interaction_limit, cell_indices
        )

        # If radial cutoff is finite, use it to calculate the sparse
        # distance matrix to reduce computational complexity from O(n^2) to
        # O(n log(n))
        n_atoms = len(ext_system)
        if r_cut is not None:
            dmat = ext_system.get_distance_matrix_within_radius(r_cut)
            adj_list = dscribe.utils.geometry.get_adjacency_list(dmat)
            dmat_dense = np.full(
                (n_atoms, n_atoms), sys.float_info.max
            )  # The non-neighbor values are treated as "infinitely far".
            dmat_dense[dmat.row, dmat.col] = dmat.data
        # If no weighting is used, the full distance matrix is calculated
        else:
            dmat_dense = ext_system.get_distance_matrix()
            adj_list = np.tile(np.arange(n_atoms), (n_atoms, 1))

        n_elem = self.n_elements
        n_features = int((n_elem * (n_elem + 1) / 2) * n)

        if return_descriptor:
            k2 = np.zeros((n_features), dtype=np.float64)
        else:
            k2 = np.zeros((0), dtype=np.float64)

        if return_derivatives:
            k2_d = np.zeros((self._interaction_limit, 3, n_features), dtype=np.float64)
        else:
            k2_d = np.zeros((0, 0, 0), dtype=np.float64)

        # Generate derivatives for k=2 term
        cmbtr.get_k2(
            k2,
            k2_d,
            return_descriptor,
            return_derivatives,
            ext_system.get_atomic_numbers(),
            ext_system.get_positions(),
            dmat_dense,
            adj_list,
            geom_func_name.encode(),
            weighting_function.encode(),
            parameters,
            start,
            stop,
            sigma,
            n,
        )

        # Denormalize if requested
        if not self.normalize_gaussians:
            max_val = 1 / (sigma * math.sqrt(2 * math.pi))
            k2 /= max_val
            k2_d /= max_val

        # Valle-Oganov normalization is calculated separately for each pair.
        # Not implemented for derivatives.
        if self.normalization == "valle_oganov":
            volume = self.system.cell.volume
            # Calculate the amount of each element for N_A*N_B term
            values, counts = np.unique(
                self.system.get_atomic_numbers(), return_counts=True
            )
            counts = dict(zip(values, counts))
            for i_z in values:
                for j_z in values:
                    i = self.atomic_number_to_index[i_z]
                    j = self.atomic_number_to_index[j_z]
                    if j < i:
                        continue
                    if i == j:
                        count_product = 0.5 * counts[i_z] * counts[j_z]
                    else:
                        count_product = counts[i_z] * counts[j_z]

                    # This is the index of the spectrum. It is given by enumerating the
                    # elements of an upper triangular matrix from left to right and top
                    # to bottom.
                    m = int(j + i * n_elem - i * (i + 1) / 2)
                    start = m * n
                    end = (m + 1) * n
                    norm_factor = volume / (count_product * 4 * np.pi)

                    k2[start:end] *= norm_factor
                    k2_d[:, :, start:end] *= norm_factor

        # Convert to the final output precision.
        if self.dtype == "float32":
            k2 = k2.astype(self.dtype)
            k2_d = k2_d.astype(self.dtype)

        return (k2, k2_d)

    def _get_k3(self, system, return_descriptor, return_derivatives):
        """Calculates the third order term and/or its derivatives with
        regard to atomic positions.
        Returns:
            1D ndarray: K3 values. Returns a 1D array. If
                return_descriptor=False, returns an array of shape (0).
            3D ndarray: K3 derivatives. If return_derivatives=False, returns an
                array of shape (0,0,0).
        """
        start = self.grid["min"]
        stop = self.grid["max"]
        n = self.grid["n"]
        sigma = self.grid["sigma"]

        # Determine the weighting function and possible radial cutoff
        r_cut = None
        parameters = {}
        if self.weighting is not None:
            weighting_function = self.weighting["function"]
            if weighting_function == "exp":
                threshold = self.weighting["threshold"]
                r_cut = self.weighting.get("r_cut")
                scale = self.weighting.get("scale")
                # If we want to limit the triplets to a distance r_cut, we need
                # to allow x=2*r_cut in the case of k=3.
                if scale is not None and r_cut is None:
                    r_cut = -0.5 * math.log(threshold) / scale
                elif scale is None and r_cut is not None:
                    scale = -0.5 * math.log(threshold) / r_cut
                parameters = {b"scale": scale, b"threshold": threshold}
            if weighting_function == "smooth_cutoff":
                try:
                    sharpness = self.weighting["sharpness"]
                except Exception:
                    sharpness = 2
                parameters = {
                    b"sharpness": sharpness,
                    b"cutoff": self.weighting["r_cut"],
                }
                # Evaluating smooth-cutoff weighting values requires distances
                # between two neighbours of an atom, and the maximum distance
                # between them is twice the cutoff radius. To include the
                # neighbour-to-neighbour distances in the distance matrix, the
                # neighbour list is generated with double radius.
                r_cut = 2 * self.weighting["r_cut"]
        else:
            weighting_function = "unity"

        # Determine the geometry function
        geom_func_name = self.geometry["function"]

        # If needed, create the extended system
        if self.periodic:
            centers = system.get_positions()
            ext_system, cell_indices = dscribe.utils.geometry.get_extended_system(
                system, r_cut, centers, return_cell_indices=True
            )
            ext_system = System.from_atoms(ext_system)
        else:
            ext_system = System.from_atoms(system)
            cell_indices = np.zeros((len(system), 3), dtype=int)

        cmbtr = MBTRWrapper(
            self.atomic_number_to_index, self._interaction_limit, cell_indices
        )

        n_atoms = len(ext_system)
        if r_cut is not None:
            dmat = ext_system.get_distance_matrix_within_radius(r_cut)
            adj_list = dscribe.utils.geometry.get_adjacency_list(dmat)
            dmat_dense = np.full(
                (n_atoms, n_atoms), sys.float_info.max
            )  # The non-neighbor values are treated as "infinitely far".
            dmat_dense[dmat.col, dmat.row] = dmat.data
        # If no weighting is used, the full distance matrix is calculated
        else:
            dmat_dense = ext_system.get_distance_matrix()
            adj_list = np.tile(np.arange(n_atoms), (n_atoms, 1))

        n_elem = self.n_elements
        n_features = int((n_elem * n_elem * (n_elem + 1) / 2) * n)

        if return_descriptor:
            k3 = np.zeros((n_features), dtype=np.float64)
        else:
            k3 = np.zeros((0), dtype=np.float64)

        if return_derivatives:
            k3_d = np.zeros((self._interaction_limit, 3, n_features), dtype=np.float64)
        else:
            k3_d = np.zeros((0, 0, 0), dtype=np.float64)

        # Compute the k=3 term and its derivative
        cmbtr.get_k3(
            k3,
            k3_d,
            return_descriptor,
            return_derivatives,
            ext_system.get_atomic_numbers(),
            ext_system.get_positions(),
            dmat_dense,
            adj_list,
            geom_func_name.encode(),
            weighting_function.encode(),
            parameters,
            start,
            stop,
            sigma,
            n,
        )

        # Denormalize if requested
        if not self.normalize_gaussians:
            max_val = 1 / (sigma * math.sqrt(2 * math.pi))
            k3 /= max_val
            k3_d /= max_val

        # Valle-Oganov normalization is calculated separately for each triplet
        # Not implemented for derivatives.
        if self.normalization == "valle_oganov":
            volume = self.system.cell.volume
            # Calculate the amount of each element for N_A*N_B*N_C term
            values, counts = np.unique(
                self.system.get_atomic_numbers(), return_counts=True
            )
            counts = dict(zip(values, counts))
            for i_z in values:
                for j_z in values:
                    for k_z in values:
                        i = self.atomic_number_to_index[i_z]
                        j = self.atomic_number_to_index[j_z]
                        k = self.atomic_number_to_index[k_z]
                        if k < i:
                            continue
                        # This is the index of the spectrum. It is given by enumerating the
                        # elements of a three-dimensional array where for valid elements
                        # k>=i. The enumeration begins from [0, 0, 0], and ends at [n_elem,
                        # n_elem, n_elem], looping the elements in the order j, i, k.
                        m = int(
                            j * n_elem * (n_elem + 1) / 2
                            + k
                            + i * n_elem
                            - i * (i + 1) / 2
                        )
                        start = m * n
                        end = (m + 1) * n
                        count_product = counts[i_z] * counts[j_z] * counts[k_z]
                        norm_factor = volume / count_product

                        k3[start:end] *= norm_factor
                        k3_d[:, :, start:end] *= norm_factor

        # Convert to the final output precision.
        if self.dtype == "float32":
            k3 = k3.astype(self.dtype)
            k3_d = k3_d.astype(self.dtype)

        return (k3, k3_d)

    def validate_derivatives_method(self, method):
        """Used to validate and determine the final method for calculating the
        derivatives.
        """
        methods = {"numerical", "analytical", "auto"}
        if method not in methods:
            raise ValueError(
                "Invalid method specified. Please choose from: {}".format(methods)
            )

        if method == "numerical":
            return method

        # Check if analytical derivatives can be used
        try:
            supported_normalization = ["none", "n_atoms", "valle_oganov"]
            if self.normalization not in supported_normalization:
                raise ValueError(
                    "Analytical derivatives not implemented for normalization option '{}'. Please choose from: {}".format(
                        self.normalization, supported_normalization
                    )
                )
            # Derivatives are not currently implemented for all k3 options
            if self.k == 3:
                # "angle" function is not differentiable
                if self.geometry["function"] == "angle":
                    raise ValueError(
                        "Analytical derivatives not implemented for k3 geometry function 'angle'."
                    )
        except Exception as e:
            if method == "analytical":
                raise e
            elif method == "auto":
                method = "numerical"
        else:
            if method == "auto":
                method = "analytical"

        return method

    def derivatives_analytical(self, d, c, system, indices, return_descriptor):
        # Ensuring variables are re-initialized when a new system is introduced
        self.system = system
        self._interaction_limit = len(system)

        # Check that the system does not have elements that are not in the list
        # of atomic numbers
        self.check_atomic_numbers(system.get_atomic_numbers())

        mbtr, mbtr_d = getattr(self, f"_get_k{self.k}")(system, return_descriptor, True)

        # Handle normalization
        if self.normalization == "n_atoms":
            n_atoms = len(self.system)
            mbtr /= n_atoms
            mbtr_d /= n_atoms

        # For now, the derivatives are calculated with regard to all atomic
        # positions. The desired indices are extracted here at the end.
        i = 0
        for index in indices:
            d[i, :] = mbtr_d[index, :, :]
            i += 1

        if return_descriptor:
            np.copyto(c, mbtr)
