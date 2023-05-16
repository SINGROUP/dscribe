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
import scipy.spatial.distance
from sklearn.preprocessing import normalize
import sparse
from ase import Atoms
import ase.data

from dscribe.core import System
from dscribe.descriptors.mbtr import (
    check_geometry,
    check_weighting,
    check_grid,
    k1_geometry_functions,
)
from dscribe.descriptors.descriptorlocal import DescriptorLocal
from dscribe.ext import MBTRWrapper
import dscribe.utils.geometry


class LMBTR(DescriptorLocal):
    """
    Implementation of the Local Many-body tensor representation.

    You can use this descriptor for finite and periodic systems. When dealing
    with periodic systems or when using machine learning models that use the
    Euclidean norm to measure distance between vectors, it is advisable to use
    some form of normalization. This implementation does not support the use of
    a non-identity correlation matrix.

    Notice that the species of the central atom is not encoded in the output,
    but is instead represented by a chemical species X with atomic number 0.
    This allows LMBTR to be also used on general positions not corresponding to
    real atoms. The surrounding environment is encoded by the two- and
    three-body interactions with neighouring atoms. If there is a need to
    distinguish the central species, one can for example train a different model
    for each central species.
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

                "geometry": {"function": "distance"}

                The geometry function determines the degree :math:`k` for MBTR.
                The order :math:`k` tells how many atoms are involved in the
                calculation and thus also heavily influence the computational
                time.

                The following geometry functions are available:

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
        super().__init__(
            periodic=periodic,
            sparse=sparse,
            dtype=dtype,
        )
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
        if value["function"] in k1_geometry_functions:
            raise ValueError(
                "LMBTR does not support geometry functions for degree k=1."
            )
        k_map = {
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

    @property
    def normalization(self):
        return self._normalization

    @normalization.setter
    def normalization(self, value):
        """Checks that the given normalization is valid. Overrides the
        normalization check from original MBTR because the normalization with
        respect to number of atoms is not valid for a local descriptor.

        Args:
            value(str): The normalization method to use.
        """
        norm_options = set(("none", "l2"))
        if value not in norm_options:
            raise ValueError(
                "Unknown normalization option given. Please use one of the "
                "following: {}.".format(", ".join(sorted(list(norm_options))))
            )
        self._normalization = value

    def create(
        self, system, centers=None, n_jobs=1, only_physical_cores=False, verbose=False
    ):
        """Return the LMBTR output for the given systems and given centers.

        Args:
            system (:class:`ase.Atoms` or list of :class:`ase.Atoms`): One or
                many atomic structures.
            centers (list): Centers where to calculate LMBTR. Can be
                provided as cartesian positions or atomic indices. If no
                centers are defined, the LMBTR output will be created for all
                atoms in the system. When calculating LMBTR for multiple
                systems, provide the centers as a list for each system.
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
            np.ndarray | scipy.sparse.csr_matrix: The LMBTR output for the given
            systems and centers. The return type depends on the
            'sparse'-attribute. The first dimension is determined by the amount
            of centers and systems and the second dimension is determined by
            the get_number_of_features()-function.
        """
        # Combine input arguments
        if isinstance(system, Atoms):
            system = [system]
            centers = [centers]
        n_samples = len(system)
        if centers is None:
            inp = [(i_sys,) for i_sys in system]
        else:
            n_pos = len(centers)
            if n_pos != n_samples:
                raise ValueError(
                    "The given number of centers does not match the given"
                    "number of systems."
                )
            inp = list(zip(system, centers))

        # Determine if the outputs have a fixed size
        n_features = self.get_number_of_features()
        static_size = None
        if centers is None:
            n_centers = len(inp[0][0])
        else:
            first_sample, first_pos = inp[0]
            if first_pos is not None:
                n_centers = len(first_pos)
            else:
                n_centers = len(first_sample)

        def is_static():
            for i_job in inp:
                if centers is None:
                    if len(i_job[0]) != n_centers:
                        return False
                else:
                    if i_job[1] is not None:
                        if len(i_job[1]) != n_centers:
                            return False
                    else:
                        if len(i_job[0]) != n_centers:
                            return False
            return True

        if is_static():
            static_size = [n_centers, n_features]

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

    def create_single(
        self,
        system,
        centers=None,
    ):
        """Return the local many-body tensor representation for the given
        system and centers.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.
            centers (iterable): Centers for which LMBTR is created. Can be
                a list of integer numbers or a list of xyz-coordinates. If
                integers provided, the atoms at that index are used as centers.
                If cartesian positions are provided, new atoms are added at that
                position. If no centers are provided, all atoms in the system
                will be used as centers.

        Returns:
            1D ndarray: The local many-body tensor representations of given
            centers, for k terms, as an array. These are ordered as given in
            centers.
        """
        # Check that the system does not have elements that are not in the list
        # of atomic numbers
        atomic_number_set = set(system.get_atomic_numbers())
        self.check_atomic_numbers(atomic_number_set)
        self._interaction_limit = len(system)
        system_positions = system.get_positions()
        system_atomic_numbers = system.get_atomic_numbers()

        # Ensure that the atomic number 0 is not present in the system
        if 0 in atomic_number_set:
            raise ValueError(
                "Please do not use the atomic number 0 in local MBTR as it "
                "is reserved to mark the atoms use as analysis centers."
            )

        # Form a list of indices, centers and atomic numbers for the local
        # centers. k=3 and k=2 use a slightly different approach, so two
        # versions are built
        i_new = len(system)
        indices_k2 = []
        new_pos_k2 = []
        new_atomic_numbers_k2 = []
        indices_k3 = []
        new_pos_k3 = []
        new_atomic_numbers_k3 = []
        n_atoms = len(system)
        if centers is not None:
            # Check validity of centers definitions and create final cartesian
            # position list
            if len(centers) == 0:
                raise ValueError(
                    "The argument 'centers' should contain a non-empty set of"
                    " atomic indices or cartesian coordinates with x, y and z "
                    "components."
                )
            for i in centers:
                if np.issubdtype(type(i), np.integer):
                    i_len = len(system)
                    if i >= i_len or i < 0:
                        raise ValueError(
                            "The provided index {} is not valid for the system "
                            "with {} atoms.".format(i, i_len)
                        )
                    indices_k2.append(i)
                    indices_k3.append(i)
                    new_pos_k2.append(system_positions[i])
                    new_atomic_numbers_k2.append(system_atomic_numbers[i])
                elif isinstance(i, (list, tuple, np.ndarray)):
                    if len(i) != 3:
                        raise ValueError(
                            "The argument 'centers' should contain a "
                            "non-empty set of atomic indices or cartesian "
                            "coordinates with x, y and z components."
                        )
                    new_pos_k2.append(np.array(i))
                    new_pos_k3.append(np.array(i))
                    new_atomic_numbers_k2.append(0)
                    new_atomic_numbers_k3.append(0)
                    i_new += 1
                else:
                    raise ValueError(
                        "Create method requires the argument 'centers', a "
                        "list of atom indices and/or positions."
                    )
        # If centers are not supplied, it is assumed that each atom is used
        # as a center
        else:
            indices_k2 = np.arange(n_atoms)
            indices_k3 = np.arange(n_atoms)
            new_pos_k2 = system.get_positions()
            new_atomic_numbers_k2 = system.get_atomic_numbers()

        # Calculate the "raw" output
        if self.k == 2:
            new_system = System(
                symbols=new_atomic_numbers_k2,
                positions=new_pos_k2,
            )
            indices = indices_k2
        elif self.k == 3:
            new_system = System(
                symbols=new_atomic_numbers_k3,
                positions=new_pos_k3,
            )
            indices = indices_k3
        mbtr = getattr(self, f"_get_k{self.k}")(system, new_system, indices)

        # Handle normalization
        if self.normalization == "l2":
            normalize(mbtr.tocsr(), norm="l2", axis=1, copy=False)

        # Make into a dense array
        result = mbtr.todense()

        return result

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        The number of features for the LMBTR is calculated as follows:

        For the pair term (k=2), only pairs where at least one of the atom is
        the central atom (in periodic systems the central atom may connect to
        itself) are considered. This means that there are only as many
        combinations as there are different elements to pair the central atom
        with (n_elem). This number of combinations is the multiplied by the
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
        n_grid = self.grid["n"]

        if self.k == 2:
            n_features = (n_elem) * n_grid
        if self.k == 3:
            n_features = n_elem * (3 * n_elem - 1) * n_grid / 2

        return int(n_features)

    def _make_new_klist_local(self, kx_list):
        new_kx_list = []

        for item in kx_list:
            new_kx_map = {}
            item = dict(item)
            for key, value in item.items():
                new_key = tuple(int(x) for x in key.split(","))
                new_kx_map[new_key] = np.array(value, dtype=self.dtype)
            new_kx_list.append(new_kx_map)

        return new_kx_list

    def _get_k2(self, system, new_system, indices):
        """Calculates the second order terms where the scalar mapping is the
        inverse distance between atoms.

        Returns:
            1D ndarray: flattened K2 values.
        """
        start = self.grid["min"]
        stop = self.grid["max"]
        n = self.grid["n"]
        sigma = self.grid["sigma"]

        # Determine the weighting function and possible radial cutoff
        radial_cutoff = None
        parameters = {}
        if self.weighting is not None:
            weighting_function = self.weighting["function"]
            if weighting_function == "exponential" or weighting_function == "exp":
                scale = self.weighting["scale"]
                threshold = self.weighting["threshold"]
                if scale != 0:
                    radial_cutoff = -math.log(threshold) / scale
                parameters = {
                    b"scale": self.weighting["scale"],
                    b"threshold": self.weighting["threshold"],
                }
        else:
            weighting_function = "unity"

        # Determine the geometry function
        geom_func_name = self.geometry["function"]

        # Calculate extended system
        if self.periodic:
            centers = new_system.get_positions()
            ext_system, cell_indices = dscribe.utils.geometry.get_extended_system(
                system,
                radial_cutoff,
                centers,
                return_cell_indices=True,
            )
            ext_system = System.from_atoms(ext_system)
        else:
            ext_system = System.from_atoms(system)
            cell_indices = np.zeros((len(system), 3), dtype=int)

        cmbtr = MBTRWrapper(
            self.atomic_number_to_index, self._interaction_limit, cell_indices
        )

        # If radial cutoff is finite, use it to calculate the sparse distance
        # matrix to reduce computational complexity from O(n^2) to O(n log(n)).
        # If radial cutoff is not available, calculate full matrix.
        n_atoms_ext = len(ext_system)
        n_atoms_new = len(new_system)
        ext_pos = ext_system.get_positions()
        new_pos = new_system.get_positions()
        if radial_cutoff is not None:
            dmat = new_system.get_distance_matrix_within_radius(
                radial_cutoff, pos=ext_pos
            )
            adj_list = dscribe.utils.geometry.get_adjacency_list(dmat)
            dmat_dense = np.full(
                (n_atoms_new, n_atoms_ext), sys.float_info.max
            )  # The non-neighbor values are treated as "infinitely far".
            dmat_dense[dmat.row, dmat.col] = dmat.data
        else:
            dmat_dense = scipy.spatial.distance.cdist(new_pos, ext_pos)
            adj_list = np.tile(np.arange(n_atoms_ext), (n_atoms_new, 1))

        # Form new indices that include the existing atoms and the newly added
        # ones
        indices = np.array(
            np.append(
                indices, [n_atoms_ext + i for i in range(n_atoms_new - len(indices))]
            ),
            dtype=int,
        )

        k2_list = cmbtr.get_k2_local(
            indices,
            ext_system.get_atomic_numbers(),
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
        k2_list = self._make_new_klist_local(k2_list)
        n_elem = self.n_elements
        n_loc = len(indices)
        k2 = sparse.DOK((n_loc, n_elem * n), dtype=self.dtype)

        for i_loc, k2_map in enumerate(k2_list):
            for key, gaussian_sum in k2_map.items():
                i = key[1]
                m = i
                start = int(m * n)
                end = int((m + 1) * n)

                # Denormalize if requested
                if not self.normalize_gaussians:
                    max_val = 1 / (sigma * math.sqrt(2 * math.pi))
                    gaussian_sum /= max_val

                k2[i_loc, start:end] = gaussian_sum
        k2 = k2.to_coo()

        return k2

    def _get_k3(self, system, new_system, indices):
        """Calculates the second order terms where the scalar mapping is the
        inverse distance between atoms.

        Returns:
            1D ndarray: flattened K2 values.
        """
        start = self.grid["min"]
        stop = self.grid["max"]
        n = self.grid["n"]
        sigma = self.grid["sigma"]

        # Determine the weighting function and possible radial cutoff
        radial_cutoff = None
        parameters = {}
        if self.weighting is not None:
            weighting_function = self.weighting["function"]
            if weighting_function == "exponential" or weighting_function == "exp":
                scale = self.weighting["scale"]
                threshold = self.weighting["threshold"]
                if scale != 0:
                    radial_cutoff = -0.5 * math.log(threshold) / scale
                parameters = {b"scale": scale, b"threshold": threshold}
        else:
            weighting_function = "unity"

        # Determine the geometry function
        geom_func_name = self.geometry["function"]

        # Calculate extended system
        if self.periodic:
            centers_new = new_system.get_positions()
            centers_existing = system.get_positions()[indices]
            centers = np.concatenate((centers_new, centers_existing), axis=0)
            ext_system, cell_indices = dscribe.utils.geometry.get_extended_system(
                system,
                radial_cutoff,
                centers,
                return_cell_indices=True,
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
        fin_system = ext_system + new_system
        n_atoms_ext = len(ext_system)
        n_atoms_fin = len(fin_system)
        n_atoms_new = len(new_system)
        ext_pos = ext_system.get_positions()
        new_pos = new_system.get_positions()
        if radial_cutoff is not None:
            # Calculate distance within the extended system
            dmat_ext_to_ext = ext_system.get_distance_matrix_within_radius(
                radial_cutoff, pos=ext_pos
            )
            col = dmat_ext_to_ext.col
            row = dmat_ext_to_ext.row
            data = dmat_ext_to_ext.data
            dmat = scipy.sparse.coo_matrix(
                (data, (row, col)), shape=(n_atoms_fin, n_atoms_fin)
            )

            # Calculate the distances from the new positions to atoms in the
            # extended system using the cutoff
            if len(new_pos) != 0:
                dmat_ext_to_new = ext_system.get_distance_matrix_within_radius(
                    radial_cutoff, pos=new_pos
                )
                col = dmat_ext_to_new.col
                row = dmat_ext_to_new.row
                data = dmat_ext_to_new.data
                dmat.col = np.append(dmat.col, col + n_atoms_ext)
                dmat.row = np.append(dmat.row, row)
                dmat.data = np.append(dmat.data, data)
                dmat.col = np.append(dmat.col, row)
                dmat.row = np.append(dmat.row, col + n_atoms_ext)
                dmat.data = np.append(dmat.data, data)

            # Calculate adjacencies and transform to the dense matrix for
            # sending information to C++
            adj_list = dscribe.utils.geometry.get_adjacency_list(dmat)
            dmat_dense = np.full(
                (n_atoms_fin, n_atoms_fin), sys.float_info.max
            )  # The non-neighbor values are treated as "infinitely far".
            dmat_dense[dmat.row, dmat.col] = dmat.data

        # If no weighting is used, the full distance matrix is calculated
        else:
            dmat = scipy.sparse.lil_matrix((n_atoms_fin, n_atoms_fin))

            # Fill in block for extended system
            dmat_ext_to_ext = ext_system.get_distance_matrix()
            dmat[0:n_atoms_ext, 0:n_atoms_ext] = dmat_ext_to_ext

            # Fill in block for extended system to new system
            dmat_ext_to_new = scipy.spatial.distance.cdist(ext_pos, new_pos)
            dmat[
                0:n_atoms_ext, n_atoms_ext : n_atoms_ext + n_atoms_new
            ] = dmat_ext_to_new
            dmat[
                n_atoms_ext : n_atoms_ext + n_atoms_new, 0:n_atoms_ext
            ] = dmat_ext_to_new.T

            # Calculate adjacencies and the dense version
            dmat = dmat.tocoo()
            adj_list = dscribe.utils.geometry.get_adjacency_list(dmat)
            dmat_dense = np.full(
                (n_atoms_fin, n_atoms_fin), sys.float_info.max
            )  # The non-neighbor values are treated as "infinitely far".
            dmat_dense[dmat.row, dmat.col] = dmat.data

        # Form new indices that include the existing atoms and the newly added
        # ones
        indices = np.array(
            np.append(indices, [n_atoms_ext + i for i in range(n_atoms_new)]), dtype=int
        )

        k3_list = cmbtr.get_k3_local(
            indices,
            fin_system.get_atomic_numbers(),
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

        k3_list = self._make_new_klist_local(k3_list)

        n_elem = self.n_elements
        n_loc = len(indices)
        k3 = sparse.DOK(
            (n_loc, int((n_elem * (3 * n_elem - 1) * n / 2))), dtype=self.dtype
        )

        for i_loc, k3_map in enumerate(k3_list):
            for key, gaussian_sum in k3_map.items():
                i = key[0]
                j = key[1]
                k = key[2]

                # This is the index of the spectrum. It is given by enumerating the
                # elements of a three-dimensional array and only considering
                # elements for which k>=i and i || j == 0. The enumeration begins
                # from [0, 0, 0], and ends at [n_elem, n_elem, n_elem], looping the
                # elements in the order k, i, j.
                if j == 0:
                    m = k + i * n_elem - i * (i + 1) / 2
                else:
                    m = n_elem * (n_elem + 1) / 2 + (j - 1) * n_elem + k
                start = int(m * n)
                end = int((m + 1) * n)

                # Denormalize if requested
                if not self.normalize_gaussians:
                    max_val = 1 / (sigma * math.sqrt(2 * math.pi))
                    gaussian_sum /= max_val

                k3[i_loc, start:end] = gaussian_sum
        k3 = k3.to_coo()

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
        if k is not self.k:
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
                    raise ValueError("Invalid chemical species")
            numbers.append(specie)

        # Check that species exists and that X is included
        self.check_atomic_numbers(numbers)
        if 0 not in numbers:
            raise ValueError(
                "The central species X (atomic number 0) has to be one of the elements."
            )

        # Change into internal indexing
        numbers = [self.atomic_number_to_index[x] for x in numbers]
        n_elem = self.n_elements
        n = self.grid["n"]

        # k=2
        if k == 2:
            if numbers[0] > numbers[1]:
                numbers = list(reversed(numbers))

            j = numbers[1]
            m = j
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
            # elements of a three-dimensional array and only considering
            # elements for which k>=i and i || j == 0. The enumeration begins
            # from [0, 0, 0], and ends at [n_elem, n_elem, n_elem], looping the
            # elements in the order k, i, j.
            if j == 0:
                m = k + i * n_elem - i * (i + 1) / 2
            else:
                m = n_elem * (n_elem + 1) / 2 + (j - 1) * n_elem + k

            start = int(m * n)
            end = int((m + 1) * n)

        return slice(start, end)
