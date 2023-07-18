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

import numpy as np
import sparse as sp
from scipy.sparse import coo_matrix
from ase import Atoms

from dscribe.descriptors.descriptorlocal import DescriptorLocal
from dscribe.ext import ACSFWrapper
import dscribe.utils.geometry


class ACSF(DescriptorLocal):
    """Implementation of Atom-Centered Symmetry Functions.

    Notice that the species of the central atom is not encoded in the output,
    only the surrounding environment is encoded. In a typical application one
    can train a different model for each central species.

    For reference, see:
        "Atom-centered symmetry functions for constructing high-dimensional
        neural network potentials", Jörg Behler, The Journal of Chemical
        Physics, 134, 074106 (2011), https://doi.org/10.1063/1.3553717
    """

    def __init__(
        self,
        r_cut,
        g2_params=None,
        g3_params=None,
        g4_params=None,
        g5_params=None,
        species=None,
        periodic=False,
        sparse=False,
        dtype="float64",
    ):
        """
        Args:
            r_cut (float): The smooth cutoff value in angstroms. This cutoff
                value is used throughout the calculations for all symmetry
                functions.
            g2_params (n*2 np.ndarray): A list of pairs of :math:`\eta` and
                :math:`R_s` parameters for :math:`G^2` functions.
            g3_params (n*1 np.ndarray): A list of :math:`\kappa` parameters for
                :math:`G^3` functions.
            g4_params (n*3 np.ndarray): A list of triplets of :math:`\eta`,
                :math:`\zeta` and  :math:`\lambda` parameters for :math:`G^4` functions.
            g5_params (n*3 np.ndarray): A list of triplets of :math:`\eta`,
                :math:`\zeta` and  :math:`\lambda` parameters for :math:`G^5` functions.
            species (iterable): The chemical species as a list of atomic
                numbers or as a list of chemical symbols. Notice that this is not
                the atomic numbers that are present for an individual system, but
                should contain all the elements that are ever going to be
                encountered when creating the descriptors for a set of systems.
                Keeping the number of chemical species as low as possible is
                preferable.
            periodic (bool): Set to true if you want the descriptor output to
                respect the periodicity of the atomic systems (see the
                pbc-parameter in the constructor of ase.Atoms).
            sparse (bool): Whether the output should be a sparse matrix or a
                dense numpy array.
        """
        super().__init__(periodic=periodic, sparse=sparse, dtype=dtype)

        self.acsf_wrapper = ACSFWrapper()

        # Setup
        self.species = species
        self.g2_params = g2_params
        self.g3_params = g3_params
        self.g4_params = g4_params
        self.g5_params = g5_params
        self.r_cut = r_cut

    def create(
        self, system, centers=None, n_jobs=1, only_physical_cores=False, verbose=False
    ):
        """Return the ACSF output for the given systems and given centers.

        Args:
            system (:class:`ase.Atoms` or list of :class:`ase.Atoms`): One or
                many atomic structures.
            centers (list): Centers where to calculate ACSF. Can be
                provided as cartesian positions or atomic indices. If no
                centers are defined, the output will be created for all
                atoms in the system. When calculating output for multiple
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
            np.ndarray | sparse.COO: The ACSF output for the given
            systems and centers. The return type depends on the
            'sparse'-attribute. The first dimension is determined by the amount
            of centers and systems and the second dimension is determined by
            the get_number_of_features()-function. When multiple systems are
            provided the results are ordered by the input order of systems and
            their centers.
        """
        # Validate input / combine input arguments
        if isinstance(system, Atoms):
            system = [system]
            centers = [centers]
        if centers is None:
            inp = [(i_sys,) for i_sys in system]
        else:
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

    def create_single(self, system, centers=None):
        """Creates the descriptor for the given system.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.
            centers (iterable): Indices of the atoms around which the ACSF
                will be returned. If no centers defined, ACSF will be created
                for all atoms in the system.

        Returns:
            np.ndarray: The ACSF output for the given system and centers. The
            first dimension is given by the number of centers and the second
            dimension is determined by the get_number_of_features()-function.
        """
        # Check if there are types that have not been declared
        self.check_atomic_numbers(system.get_atomic_numbers())

        # Create C-compatible list of atomic indices for which the ACSF is
        # calculated
        calculate_all = False
        if centers is None:
            calculate_all = True
            indices = np.arange(len(system))
        else:
            indices = centers

        # If periodicity is not requested, and the output is requested for all
        # atoms, we skip all the intricate optimizations that will make things
        # actually slower for this case.
        if calculate_all and not self.periodic:
            n_atoms = len(system)
            all_pos = system.get_positions()
            dmat = dscribe.utils.geometry.get_adjacency_matrix(
                self.r_cut, all_pos, all_pos
            )
        # Otherwise the amount of pairwise distances that are calculated is
        # kept at minimum. Only distances for the given indices (and possibly
        # the secondary neighbours if G4 is specified) are calculated.
        else:
            # Create the extended system if periodicity is requested. For ACSF only
            # the distance from central atom needs to be considered in extending
            # the system.
            if self.periodic:
                system = dscribe.utils.geometry.get_extended_system(
                    system, self.r_cut, return_cell_indices=False
                )

            # First calculate distances from specified centers to all other
            # atoms. This is already enough for everything else except G4.
            n_atoms = len(system)
            all_pos = system.get_positions()
            central_pos = all_pos[indices]
            dmat_primary = dscribe.utils.geometry.get_adjacency_matrix(
                self.r_cut, central_pos, all_pos
            )

            # Create symmetric full matrix
            col = dmat_primary.col
            row = [
                indices[x] for x in dmat_primary.row
            ]  # Fix row numbering to refer to original system
            data = dmat_primary.data
            dmat = coo_matrix((data, (row, col)), shape=(n_atoms, n_atoms))
            dmat_lil = dmat.tolil()
            dmat_lil[col, row] = dmat_lil[row, col]

            # If G4 terms are requested, calculate also secondary neighbour distances
            if len(self.g4_params) != 0:
                neighbour_indices = np.unique(col)
                neigh_pos = all_pos[neighbour_indices]
                dmat_secondary = dscribe.utils.geometry.get_adjacency_matrix(
                    self.r_cut, neigh_pos, neigh_pos
                )
                col = [
                    neighbour_indices[x] for x in dmat_secondary.col
                ]  # Fix col numbering to refer to original system
                row = [
                    neighbour_indices[x] for x in dmat_secondary.row
                ]  # Fix row numbering to refer to original system
                dmat_lil[row, col] = np.array(dmat_secondary.data)

            dmat = dmat_lil.tocoo()

        # Get adjancency list and full dense adjancency matrix
        neighbours = dscribe.utils.geometry.get_adjacency_list(dmat)
        dmat_dense = np.full(
            (n_atoms, n_atoms), sys.float_info.max
        )  # The non-neighbor values are treated as "infinitely far".
        dmat_dense[dmat.col, dmat.row] = dmat.data

        # Calculate ACSF with C++
        output = np.array(
            self.acsf_wrapper.create(
                system.get_positions(),
                system.get_atomic_numbers(),
                dmat_dense,
                neighbours,
                indices,
            ),
            dtype=np.float64,
        )

        return output

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        wrapper = self.acsf_wrapper
        descsize = (1 + wrapper.n_g2 + wrapper.n_g3) * wrapper.n_types
        descsize += (wrapper.n_g4 + wrapper.n_g5) * wrapper.n_type_pairs

        return int(descsize)

    def validate_derivatives_method(self, method, attach):
        if not attach:
            raise ValueError(
                "ACSF derivatives can only be calculated with attach=True."
            )
        return super().validate_derivatives_method(method, attach)

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
        self.acsf_wrapper.atomic_numbers = self._atomic_numbers.tolist()

    @property
    def r_cut(self):
        return self.acsf_wrapper.r_cut

    @r_cut.setter
    def r_cut(self, value):
        """Used to check the validity of given radial cutoff.

        Args:
            value(float): Radial cutoff.
        """
        if value <= 0:
            raise ValueError("Cutoff radius should be positive.")
        self.acsf_wrapper.r_cut = value

    @property
    def g2_params(self):
        return self.acsf_wrapper.get_g2_params()

    @g2_params.setter
    def g2_params(self, value):
        """Used to check the validity of given G2 parameters.

        Args:
            value(n*3 array): List of G2 parameters.
        """
        # Disable case
        if value is None:
            value = np.array([])
        else:
            # Check dimensions
            value = np.array(value, dtype=np.float64)
            if value.ndim != 2:
                raise ValueError(
                    "g2_params should be a matrix with two columns (eta, Rs)."
                )
            if value.shape[1] != 2:
                raise ValueError(
                    "g2_params should be a matrix with two columns (eta, Rs)."
                )

            # Check that etas are positive
            if np.any(value[:, 0] <= 0) is True:
                raise ValueError("G2 eta parameters should be positive numbers.")

        self.acsf_wrapper.set_g2_params(value.tolist())

    @property
    def g3_params(self):
        return self.acsf_wrapper.g3_params

    @g3_params.setter
    def g3_params(self, value):
        """Used to check the validity of given G3 parameters and to
        initialize the C-memory layout for them.

        Args:
            value(array): List of G3 parameters.
        """
        # Handle the disable case
        if value is None:
            value = np.array([])
        else:
            # Check dimensions
            value = np.array(value, dtype=np.float64)
            if value.ndim != 1:
                raise ValueError("g3_params should be a vector.")

        self.acsf_wrapper.g3_params = value.tolist()

    @property
    def g4_params(self):
        return self.acsf_wrapper.g4_params

    @g4_params.setter
    def g4_params(self, value):
        """Used to check the validity of given G4 parameters and to
        initialize the C-memory layout for them.

        Args:
            value(n*3 array): List of G4 parameters.
        """
        # Handle the disable case
        if value is None:
            value = np.array([])
        else:
            # Check dimensions
            value = np.array(value, dtype=np.float64)
            if value.ndim != 2:
                raise ValueError(
                    "g4_params should be a matrix with three columns (eta, zeta, lambda)."
                )
            if value.shape[1] != 3:
                raise ValueError(
                    "g4_params should be a matrix with three columns (eta, zeta, lambda)."
                )

            # Check that etas are positive
            if np.any(value[:, 2] <= 0) is True:
                raise ValueError("3-body G4 eta parameters should be positive numbers.")

        self.acsf_wrapper.g4_params = value.tolist()

    @property
    def g5_params(self):
        return self.acsf_wrapper.g5_params

    @g5_params.setter
    def g5_params(self, value):
        """Used to check the validity of given G5 parameters and to
        initialize the C-memory layout for them.

        Args:
            value(n*3 array): List of G5 parameters.
        """
        # Handle the disable case
        if value is None:
            value = np.array([])
        else:
            # Check dimensions
            value = np.array(value, dtype=np.float64)
            if value.ndim != 2:
                raise ValueError(
                    "g5_params should be a matrix with three columns (eta, zeta, lambda)."
                )
            if value.shape[1] != 3:
                raise ValueError(
                    "g5_params should be a matrix with three columns (eta, zeta, lambda)."
                )

            # Check that etas are positive
            if np.any(value[:, 2] <= 0) is True:
                raise ValueError("3-body G5 eta parameters should be positive numbers.")

        self.acsf_wrapper.g5_params = value.tolist()
