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

import sparse

from ase import Atoms
import ase.data

from dscribe.core import System
from dscribe.descriptors.descriptorglobal import DescriptorGlobal
from dscribe.ext import MBTRWrapper
import dscribe.utils.geometry


def check_k1(value):
    if value is not None:
        # Check that only valid keys are used in the setups
        for key in value.keys():
            valid_keys = set(("geometry", "grid", "weighting"))
            if key not in valid_keys:
                raise ValueError(
                    "The given setup contains the following invalid key: {}".format(key)
                )

        # Check the geometry function
        geom_func = value["geometry"].get("function")
        if geom_func is not None:
            valid_geom_func = set(("atomic_number",))
            if geom_func not in valid_geom_func:
                raise ValueError(
                    "Unknown geometry function specified for k=1. Please use one of"
                    " the following: {}".format(sorted(list(valid_geom_func)))
                )

        # Check the weighting function
        weighting = value.get("weighting")
        if weighting is not None:
            valid_weight_func = set(("unity",))
            weight_func = weighting.get("function")
            if weight_func not in valid_weight_func:
                raise ValueError(
                    "Unknown weighting function specified for k=1. Please use one of"
                    " the following: {}".format(sorted(list(valid_weight_func)))
                )

        # Check grid
        check_grid(value["grid"])


def check_k2(value):
    if value is not None:
        # Check that only valid keys are used in the setups
        for key in value.keys():
            valid_keys = set(("geometry", "grid", "weighting"))
            if key not in valid_keys:
                raise ValueError(
                    "The given setup contains the following invalid key: {}".format(key)
                )

        # Check the geometry function
        geom_func = value["geometry"].get("function")
        if geom_func is not None:
            valid_geom_func = set(("distance", "inverse_distance"))
            if geom_func not in valid_geom_func:
                raise ValueError(
                    "Unknown geometry function specified for k=2. Please use one of"
                    " the following: {}".format(sorted(list(valid_geom_func)))
                )

        # Check the weighting function
        weighting = value.get("weighting")
        if weighting is not None:
            valid_weight_func = set(("unity", "exp", "inverse_square"))
            weight_func = weighting.get("function")
            if weight_func not in valid_weight_func:
                raise ValueError(
                    "Unknown weighting function specified for k=2. Please use one of"
                    " the following: {}".format(sorted(list(valid_weight_func)))
                )
            else:
                if weight_func == "exp":
                    threshold = weighting.get("threshold")
                    if threshold is None:
                        raise ValueError(
                            "Missing value for 'threshold' in the k=2 weighting."
                        )
                    scale = weighting.get("scale")
                    r_cut = weighting.get("r_cut")
                    if scale is not None and r_cut is not None:
                        raise ValueError(
                            "Provide either 'scale' or 'r_cut', not both in the k=2 weighting."
                        )
                    if scale is None and r_cut is None:
                        raise ValueError(
                            "Provide either 'scale' or 'r_cut' in the k=2 weighting."
                        )
                elif weight_func == "inverse_square":
                    if weighting.get("r_cut") is None:
                        raise ValueError(
                            "Missing value for 'r_cut' in the k=2 weighting."
                        )

        # Check grid
        check_grid(value["grid"])


def check_k3(value):
    if value is not None:
        # Check that only valid keys are used in the setups
        for key in value.keys():
            valid_keys = set(("geometry", "grid", "weighting"))
            if key not in valid_keys:
                raise ValueError(
                    "The given setup contains the following invalid key: {}".format(key)
                )

        # Check the geometry function
        geom_func = value["geometry"].get("function")
        if geom_func is not None:
            valid_geom_func = set(("angle", "cosine"))
            if geom_func not in valid_geom_func:
                raise ValueError(
                    "Unknown geometry function specified for k=2. Please use one of"
                    " the following: {}".format(sorted(list(valid_geom_func)))
                )

        # Check the weighting function
        weighting = value.get("weighting")
        if weighting is not None:
            valid_weight_func = set(("unity", "exp", "smooth_cutoff"))
            weight_func = weighting.get("function")
            if weight_func not in valid_weight_func:
                raise ValueError(
                    "Unknown weighting function specified for k=3. Please use one of"
                    " the following: {}".format(sorted(list(valid_weight_func)))
                )
            else:
                if weight_func == "exp":
                    threshold = weighting.get("threshold")
                    if threshold is None:
                        raise ValueError(
                            "Missing value for 'threshold' in the k=3 weighting."
                        )
                    scale = weighting.get("scale")
                    r_cut = weighting.get("r_cut")
                    if scale is not None and r_cut is not None:
                        raise ValueError(
                            "Provide either 'scale' or 'r_cut', not both in the k=3 weighting."
                        )
                    if scale is None and r_cut is None:
                        raise ValueError(
                            "Provide either 'scale' or 'r_cut' in the k=3 weighting."
                        )
                elif weight_func == "smooth_cutoff":
                    if weighting.get("r_cut") is None:
                        raise ValueError(
                            "Missing value for 'r_cut' in the k=3 weighting."
                        )
        # Check grid
        check_grid(value["grid"])


def check_grid(grid):
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


def check_weighting(periodic, k):
    """Used to ensure that the given weighting settings are valid.

    Args:
        grid(dict): Dictionary containing the weighting setup.
    """
    # Check that weighting function is specified for periodic systems
    if periodic:
        if k is not None:
            valid = False
            weighting = k.get("weighting")
            if weighting is not None:
                function = weighting.get("function")
                if function is not None:
                    if function != "unity":
                        valid = True
            if not valid:
                raise ValueError("Periodic systems need to have a weighting function.")


class MBTR(DescriptorGlobal):
    """Implementation of the Many-body tensor representation up to :math:`k=3`.

    You can choose which terms to include by providing a dictionary in the
    k1, k2 or k3 arguments. This dictionary should contain information
    under three keys: "geometry", "grid" and "weighting". See the examples
    below for how to format these dictionaries.

    You can use this descriptor for finite and periodic systems. When dealing
    with periodic systems or when using machine learning models that use the
    Euclidean norm to measure distance between vectors, it is advisable to use
    some form of normalization.

    For the geometry functions the following choices are available:

    * :math:`k=1`:

       * "atomic_number": The atomic numbers.

    * :math:`k=2`:

       * "distance": Pairwise distance in angstroms.
       * "inverse_distance": Pairwise inverse distance in 1/angstrom.

    * :math:`k=3`:

       * "angle": Angle in degrees.
       * "cosine": Cosine of the angle.

    For the weighting the following functions are available:

    * :math:`k=1`:

       * "unity": No weighting.

    * :math:`k=2`:

       * "unity": No weighting.
       * "exp": Weighting of the form :math:`e^{-sx}`
       * "inverse_square": Weighting of the form :math:`1/(x^2)`

    * :math:`k=3`:

       * "unity": No weighting.
       * "exp": Weighting of the form :math:`e^{-sx}`
       * "smooth_cutoff": Weighting of the form :math:`f_{ij}f_{ik}`,
         where :math:`f = 1+y(x/r_{cut})^{y+1}-(y+1)(x/r_{cut})^{y}`

    The exponential weighting is motivated by the exponential decay of screened
    Coulombic interactions in solids. In the exponential weighting the
    parameters **threshold** determines the value of the weighting function after
    which the rest of the terms will be ignored. Either the parameter **scale**
    or **r_cut** can be used to determine the parameter :math:`s`: **scale**
    directly corresponds to this value whereas **r_cut** can be used to
    indirectly determine it through :math:`s=-\log()`:. The meaning of
    :math:`x` changes for different terms as follows:

    * :math:`k=2`: :math:`x` = Distance between A->B
    * :math:`k=3`: :math:`x` = Distance from A->B->C->A.

    The inverse square and smooth cutoff function weightings use a cutoff
    parameter **r_cut**, which is a radial distance after which the rest of
    the atoms will be ignored. For both, :math:`x` means the distance between
    A->B. For the smooth cutoff function, additional weighting key **sharpness**
    can be added, which changes the value of :math:`y`. If not, it defaults to `2`.

    In the grid setup *min* is the minimum value of the axis, *max* is the
    maximum value of the axis, *sigma* is the standard deviation of the
    gaussian broadening and *n* is the number of points sampled on the
    grid.

    A sparse.COO sparse matrix is returned. This sparse matrix is of size
    (n_features,), where n_features is given by get_number_of_features(). This
    vector is ordered so that the different k-terms are ordered in ascending
    order, and within each k-term the distributions at each entry (i, j, h) of
    the tensor are ordered in an ascending order by (i * n_elements) + (j *
    n_elements) + (h * n_elements).

    This implementation does not support the use of a non-identity correlation
    matrix.
    """

    def __init__(
        self,
        k1=None,
        k2=None,
        k3=None,
        normalize_gaussians=True,
        normalization="none",
        species=None,
        periodic=False,
        sparse=False,
        dtype="float64",
    ):
        """
        Args:
            k1 (dict): Setup for the k=1 term. For example::

                k1 = {
                    "geometry": {"function": "atomic_number"},
                    "grid": {"min": 1, "max": 10, "sigma": 0.1, "n": 50}
                }

            k2 (dict): Dictionary containing the setup for the k=2 term.
                Contains setup for the used geometry function, discretization and
                weighting function. For example::

                    k2 = {
                        "geometry": {"function": "inverse_distance"},
                        "grid": {"min": 0.1, "max": 2, "sigma": 0.1, "n": 50},
                        "weighting": {"function": "exp", "r_cut": 10, "threshold": 1e-2}
                    }

            k3 (dict): Dictionary containing the setup for the k=3 term.
                Contains setup for the used geometry function, discretization and
                weighting function. For example::

                    k3 = {
                        "geometry": {"function": "angle"},
                        "grid": {"min": 0, "max": 180, "sigma": 5, "n": 50},
                        "weighting" : {"function": "exp", "r_cut": 10, "threshold": 1e-3}
                    }

            normalize_gaussians (bool): Determines whether the gaussians are
                normalized to an area of 1. Defaults to True. If False, the
                normalization factor is dropped and the gaussians have the form.
                :math:`e^{-(x-\mu)^2/2\sigma^2}`
            normalization (str): Determines the method for normalizing the
                output. The available options are:

                * "none": No normalization.
                * "l2": Normalize the Euclidean length to unity.
                * "n_atoms": Normalize the output by dividing it with the number
                  of atoms in the system. If the system is periodic, the number
                  of atoms is determined from the given unit cell.
                * "l2_each": Normalize the Euclidean length of each k-term
                  individually to unity.
                * "valle_oganov": Use Valle-Oganov descriptor normalization, with
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
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.species = species
        self.normalization = normalization
        self.normalize_gaussians = normalize_gaussians

        if self.normalization == "valle_oganov" and not periodic:
            raise ValueError(
                "Valle-Oganov normalization does not support non-periodic systems."
            )

        # Initializing .create() level variables
        self._interaction_limit = None

        # Check that weighting function is specified for periodic systems
        check_weighting(self.periodic, self.k2)
        check_weighting(self.periodic, self.k3)

    @property
    def k1(self):
        return self._k1

    @k1.setter
    def k1(self, value):
        check_k1(value)
        self._k1 = value

    @property
    def k2(self):
        return self._k2

    @k2.setter
    def k2(self, value):
        check_k2(value)
        self._k2 = value

    @property
    def k3(self):
        return self._k3

    @k3.setter
    def k3(self, value):
        check_k3(value)
        self._k3 = value

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
        norm_options = set(("none", "l2", "n_atoms", "l2_each", "valle_oganov"))
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

        mbtr = {}
        if self.k1 is not None:
            mbtr["k1"], _ = self._get_k1(system, True, False)
        if self.k2 is not None:
            mbtr["k2"], _ = self._get_k2(system, True, False)
        if self.k3 is not None:
            mbtr["k3"], _ = self._get_k3(system, True, False)

        # Handle normalization
        if self.normalization == "l2_each":
            for key, value in mbtr.items():
                i_data = np.array(value.data)
                i_norm = np.linalg.norm(i_data)
                mbtr[key] = value / i_norm
        elif self.normalization == "l2":
            norm = 0
            for key, value in mbtr.items():
                i_data = np.array(value.data)
                norm += np.linalg.norm(i_data)
            for key, value in mbtr.items():
                mbtr[key] = value / norm
        elif self.normalization == "n_atoms":
            n_atoms = len(system)
            for key, value in mbtr.items():
                mbtr[key] = value / n_atoms

        keys = sorted(mbtr.keys())
        if len(keys) > 1:
            mbtr = np.concatenate([mbtr[key] for key in keys], axis=0)
        else:
            mbtr = mbtr[keys[0]]

        return mbtr

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        n_features = 0
        n_elem = self.n_elements

        if self.k1 is not None:
            n_k1_grid = self.k1["grid"]["n"]
            n_k1 = n_elem * n_k1_grid
            n_features += n_k1
        if self.k2 is not None:
            n_k2_grid = self.k2["grid"]["n"]
            n_k2 = (n_elem * (n_elem + 1) / 2) * n_k2_grid
            n_features += n_k2
        if self.k3 is not None:
            n_k3_grid = self.k3["grid"]["n"]
            n_k3 = (n_elem * n_elem * (n_elem + 1) / 2) * n_k3_grid
            n_features += n_k3

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
        term = getattr(self, "k{}".format(k))
        if term is None:
            raise ValueError(
                "Cannot retrieve the location for {}, as the term k{} has not "
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

        # k=1
        if len(numbers) == 1:
            n1 = self.k1["grid"]["n"]
            i = numbers[0]
            m = i
            start = int(m * n1)
            end = int((m + 1) * n1)

        # k=2
        if len(numbers) == 2:
            if numbers[0] > numbers[1]:
                numbers = list(reversed(numbers))

            n2 = self.k2["grid"]["n"]
            i = numbers[0]
            j = numbers[1]

            # This is the index of the spectrum. It is given by enumerating the
            # elements of an upper triangular matrix from left to right and top
            # to bottom.
            m = j + i * n_elem - i * (i + 1) / 2

            offset = 0
            if self.k1 is not None:
                n1 = self.k1["grid"]["n"]
                offset += n_elem * n1
            start = int(offset + m * n2)
            end = int(offset + (m + 1) * n2)

        # k=3
        if len(numbers) == 3:
            if numbers[0] > numbers[2]:
                numbers = list(reversed(numbers))

            n3 = self.k3["grid"]["n"]
            i = numbers[0]
            j = numbers[1]
            k = numbers[2]

            # This is the index of the spectrum. It is given by enumerating the
            # elements of a three-dimensional array where for valid elements
            # k>=i. The enumeration begins from [0, 0, 0], and ends at [n_elem,
            # n_elem, n_elem], looping the elements in the order k, i, j.
            m = j * n_elem * (n_elem + 1) / 2 + k + i * n_elem - i * (i + 1) / 2

            offset = 0
            if self.k1 is not None:
                n1 = self.k1["grid"]["n"]
                offset += n_elem * n1
            if self.k2 is not None:
                n2 = self.k2["grid"]["n"]
                offset += (n_elem * (n_elem + 1) / 2) * n2
            start = int(offset + m * n3)
            end = int(offset + (m + 1) * n3)

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
        grid = self.k1["grid"]
        start = grid["min"]
        stop = grid["max"]
        n = grid["n"]
        sigma = grid["sigma"]

        n_elem = self.n_elements
        n_features = n_elem * n

        if return_descriptor:
            # Determine the geometry function
            geom_func_name = self.k1["geometry"]["function"]

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
        grid = self.k2["grid"]
        start = grid["min"]
        stop = grid["max"]
        n = grid["n"]
        sigma = grid["sigma"]

        # Determine the weighting function and possible radial cutoff
        r_cut = None
        weighting = self.k2.get("weighting")
        parameters = {}
        if weighting is not None:
            weighting_function = weighting["function"]
            if weighting_function == "exp":
                threshold = weighting["threshold"]
                r_cut = weighting.get("r_cut")
                scale = weighting.get("scale")
                if scale is not None and r_cut is None:
                    r_cut = -math.log(threshold) / scale
                elif scale is None and r_cut is not None:
                    scale = -math.log(threshold) / r_cut
                parameters = {b"scale": scale, b"threshold": threshold}
            elif weighting_function == "inverse_square":
                r_cut = weighting["r_cut"]
        else:
            weighting_function = "unity"

        # Determine the geometry function
        geom_func_name = self.k2["geometry"]["function"]

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
                    y_normed = (k2[start:end] * volume) / (count_product * 4 * np.pi)

                    k2[start:end] = y_normed

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
        grid = self.k3["grid"]
        start = grid["min"]
        stop = grid["max"]
        n = grid["n"]
        sigma = grid["sigma"]

        # Determine the weighting function and possible radial cutoff
        r_cut = None
        weighting = self.k3.get("weighting")
        parameters = {}
        if weighting is not None:
            weighting_function = weighting["function"]
            if weighting_function == "exp":
                threshold = weighting["threshold"]
                r_cut = weighting.get("r_cut")
                scale = weighting.get("scale")
                # If we want to limit the triplets to a distance r_cut, we need
                # to allow x=2*r_cut in the case of k=3.
                if scale is not None and r_cut is None:
                    r_cut = -0.5 * math.log(threshold) / scale
                elif scale is None and r_cut is not None:
                    scale = -0.5 * math.log(threshold) / r_cut
                parameters = {b"scale": scale, b"threshold": threshold}
            if weighting_function == "smooth_cutoff":
                try:
                    sharpness = weighting["sharpness"]
                except Exception:
                    sharpness = 2
                parameters = {b"sharpness": sharpness, b"cutoff": weighting["r_cut"]}
                # Evaluating smooth-cutoff weighting values requires distances
                # between two neighbours of an atom, and the maximum distance
                # between them is twice the cutoff radius. To include the 
                # neighbour-to-neighbour distances in the distance matrix, the 
                # neighbour list is generated with the double radius.
                r_cut = 2 * weighting["r_cut"]
        else:
            weighting_function = "unity"

        # Determine the geometry function
        geom_func_name = self.k3["geometry"]["function"]

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
                        y_normed = (k3[start:end] * volume) / count_product
                        k3[start:end] = y_normed

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
            supported_normalization = ["none", "n_atoms"]
            if self.normalization not in supported_normalization:
                raise ValueError(
                    "Analytical derivatives not implemented for normalization option '{}'. Please choose from: {}".format(
                        self.normalization, supported_normalization
                    )
                )
            # Derivatives are not currently implemented for all k3 options
            if self.k3 is not None:
                # "angle" function is not differentiable
                if self.k3["geometry"]["function"] == "angle":
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

        mbtr = {}
        mbtr_d = {}
        if self.k1 is not None:
            k1, k1_d = self._get_k1(system, return_descriptor, True)
            mbtr["k1"] = k1
            mbtr_d["k1"] = k1_d
        if self.k2 is not None:
            k2, k2_d = self._get_k2(system, return_descriptor, True)
            mbtr["k2"] = k2
            mbtr_d["k2"] = k2_d
        if self.k3 is not None:
            k3, k3_d = self._get_k3(system, return_descriptor, True)
            mbtr["k3"] = k3
            mbtr_d["k3"] = k3_d

        # Handle normalization
        if self.normalization == "n_atoms":
            n_atoms = len(self.system)
            for key, value in mbtr.items():
                mbtr[key] = value / n_atoms
                mbtr_d[key] /= n_atoms

        keys = sorted(mbtr.keys())
        if len(keys) > 1:
            mbtr = np.concatenate([mbtr[key] for key in keys], axis=0)
            mbtr_d = np.concatenate([mbtr_d[key] for key in keys], axis=2)
        else:
            mbtr = mbtr[keys[0]]
            mbtr_d = mbtr_d[keys[0]]

        # For now, the derivatives are calculated with regard to all atomic
        # positions. The desired indices are extracted here at the end.
        # if len(indices) < len(self.system):
        #     mbtr_d = mbtr_d[indices]
        i = 0
        for index in indices:
            d[i, :] = mbtr_d[index, :, :]
            i += 1

        if return_descriptor:
            np.copyto(c, mbtr)
