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
import numpy as np

import sparse

from ase import Atoms
import ase.data

from dscribe.descriptors.descriptorglobal import DescriptorGlobal
import dscribe.utils.geometry
from dscribe.utils.species import get_atomic_numbers
import dscribe.ext


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
        flatten=True,
        species=None,
        periodic=False,
        sparse=False,
        dtype="float64",
    ):
        """
        Args:
            geometry (dict): Setup the geometry function. For example::

                "geometry": {"function": "atomic_number"}

                The following geometry functions are available:

                * "atomic_number": The atomic number.
                * "distance": Pairwise distance in angstroms.
                * "inverse_distance": Pairwise inverse distance in 1/angstrom.
                * "angle": Angle in degrees.
                * "cosine": Cosine of the angle.

            grid (dict): Setup the discretization grid. For example::

                "grid": {"min": 0.1, "max": 2, "sigma": 0.1, "n": 50}

                In the grid setup *min* is the minimum value of the axis, *max*
                is the maximum value of the axis, *sigma* is the standard
                deviation of the gaussian broadening and *n* is the number of
                points sampled on the grid.

            weighting (dict): Setup the weighting function and its parameters.
                For example::

                "weighting" : {"function": "exp", "r_cut": 10, "threshold": 1e-3}

                The following weighting functions are available:

                * "unity": No weighting.
                * "exp": Weighting of the form :math:`e^{-sx}`
                * "inverse_square": Weighting of the form :math:`1/(x^2)`
                * "smooth_cutoff": Weighting of the form :math:`f_{ij}f_{ik}`,
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

                * "none": No normalization.
                * "l2": Normalize the Euclidean length of the output to unity.
                * "n_atoms": Normalize the output by dividing it with the number
                  of atoms in the system. If the system is periodic, the number
                  of atoms is determined from the given unit cell.
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
        super().__init__(periodic=periodic, flatten=flatten, sparse=sparse, dtype=dtype)
        self.wrapper = dscribe.ext.MBTR(
            {} if geometry is None else geometry,
            {} if grid is None else grid,
            {} if weighting is None else weighting,
            normalize_gaussians,
            normalization,
            get_atomic_numbers(species),
            periodic,
        )

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
            np.ndarray | sparse.COO | list: MBTR for the
            given systems. The return type depends on the 'sparse' and
            'flatten'-attributes. For flattened output a single numpy array or
            sparse.COO matrix is returned. If the output is not flattened,
            dictionaries containing the MBTR tensors for each k-term are
            returned.
        """
        # Combine input arguments
        system = [system] if isinstance(system, Atoms) else system
        inp = [(i_sys,) for i_sys in system]

        # Determine if the outputs have a fixed size
        if self.flatten:
            static_size = [self.get_number_of_features()]
        else:
            static_size = None

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
            system (:class:`ase.Atoms`): Input system.

        Returns:
            dict | np.ndarray | sparse.COO: The return type is
            specified by the 'flatten' and 'sparse'-parameters. If the output
            is not flattened, a dictionary containing of MBTR outputs as numpy
            arrays is created. Each output is under a "kX" key. If the output
            is flattened, a single concatenated output vector is returned,
            either as a sparse or a dense vector.
        """
        # Calculate with extension
        output = np.zeros((self.get_number_of_features()), dtype=np.float64)
        system_cpp = dscribe.ext.System(
            system.get_positions(),
            system.get_atomic_numbers(),
            ase.geometry.cell.complete_cell(system.get_cell()),
            np.asarray(system.get_pbc(), dtype=bool),
        )
        self.wrapper.create(output, system_cpp)

        # Convert to the final output precision.
        if self.dtype != "float64":
            output = output.astype(self.dtype)

        # Make into a sparse array if requested
        if self._sparse:
            output = sparse.COO.from_numpy(output)

        return output

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        return self.wrapper.get_number_of_features()

    @property
    def geometry(self):
        return self.wrapper.geometry

    @geometry.setter
    def geometry(self, value):
        self.wrapper.geometry = value

    @property
    def grid(self):
        return self.wrapper.grid

    @grid.setter
    def grid(self, value):
        self.wrapper.grid = value

    @property
    def weighting(self):
        return self.wrapper.weighting

    @weighting.setter
    def weighting(self, value):
        self.wrapper.weighting = value

    @property
    def k(self):
        return self.wrapper.k

    @property
    def species(self):
        return self.wrapper.species

    @species.setter
    def species(self, value):
        self.wrapper.species = get_atomic_numbers(value)

    @property
    def normalization(self):
        return self.wrapper.normalization

    @normalization.setter
    def normalization(self, value):
        self.wrapper.normalization = value

    @property
    def normalize_gaussians(self):
        return self.wrapper.normalize_gaussians

    @normalize_gaussians.setter
    def normalize_gaussians(self, value):
        self.wrapper.normalize_gaussians = value

    def get_location(self, species):
        """Can be used to query the location of a species combination in the
        the flattened output.

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
        if k != self.k:
            species_string = ", ".join([f"'{x}'" for x in species])
            raise ValueError(
                f"Cannot retrieve the location for ({species_string}), as the used"
                + f" geometry function does not match the order k={k}."
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

        # k=1
        if len(numbers) == 1:
            loc = self.wrapper.get_location(numbers[0])
            start = loc[0]
            end = loc[1]
        # k=2
        if len(numbers) == 2:
            loc = self.wrapper.get_location(numbers[0], numbers[1])
            start = loc[0]
            end = loc[1]
        # k=3
        if len(numbers) == 3:
            loc = self.wrapper.get_location(numbers[0], numbers[1], numbers[2])
            start = loc[0]
            end = loc[1]

        return slice(start, end)

    def get_derivatives_method(self, method):
        methods = {"numerical", "auto"}
        if method not in methods:
            raise ValueError(
                "Invalid method specified. Please choose from: {}".format(methods)
            )
        if method == "auto":
            method = "numerical"
        return method