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

from scipy.sparse import coo_matrix

from dscribe.descriptors.descriptor import Descriptor
from dscribe.core import System

from ase import Atoms

from dscribe.libacsf.acsfwrapper import ACSFWrapper
import dscribe.utils.geometry


class ACSF(Descriptor):
    """Implementation of Atom-Centered Symmetry Functions. Currently valid for
    finite systems only.

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
        rcut,
        g2_params=None,
        g3_params=None,
        g4_params=None,
        g5_params=None,
        species=None,
        periodic=False,
        sparse=False
    ):
        """
        Args:
            rcut (float): The smooth cutoff value in angstroms. This cutoff
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
            periodic (bool): Determines whether the system is considered to be
                periodic.
            sparse (bool): Whether the output should be a sparse matrix or a
                dense numpy array.
        """
        super().__init__(periodic=periodic, flatten=True, sparse=sparse)

        self.acsf_wrapper = ACSFWrapper()

        # Setup
        self.species = species
        self.g2_params = g2_params
        self.g3_params = g3_params
        self.g4_params = g4_params
        self.g5_params = g5_params
        self.rcut = rcut

    def create(self, system, positions=None, n_jobs=1, verbose=False):
        """Return the ACSF output for the given systems and given positions.

        Args:
            system (:class:`ase.Atoms` or list of :class:`ase.Atoms`): One or
                many atomic structures.
            positions (list): Positions where to calculate ACSF. Can be
                provided as cartesian positions or atomic indices. If no
                positions are defined, the SOAP output will be created for all
                atoms in the system. When calculating SOAP for multiple
                systems, provide the positions as a list for each system.
            n_jobs (int): Number of parallel jobs to instantiate. Parallellizes
                the calculation across samples. Defaults to serial calculation
                with n_jobs=1.
            verbose(bool): Controls whether to print the progress of each job
                into to the console.

        Returns:
            np.ndarray | scipy.sparse.csr_matrix: The ACSF output for the given
            systems and positions. The return type depends on the
            'sparse'-attribute. The first dimension is determined by the amount
            of positions and systems and the second dimension is determined by
            the get_number_of_features()-function. When multiple systems are
            provided the results are ordered by the input order of systems and
            their positions.
        """
        # If single system given, skip the parallelization
        if isinstance(system, (Atoms, System)):
            return self.create_single(system, positions)

        # Combine input arguments
        if positions is None:
            inp = [(i_sys,) for i_sys in system]
        else:
            inp = list(zip(system, positions))

        # For ACSF the output size for each job depends on the exact arguments.
        # Here we precalculate the size for each job to preallocate memory and
        # make the process faster.
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

        # Create in parallel
        output = self.create_parallel(inp, self.create_single, n_jobs, output_sizes, verbose=verbose)

        return output

    def create_single(self, system, positions=None):
        """Creates the descriptor for the given system.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.
            positions (iterable): Indices of the atoms around which the ACSF
                will be returned. If no positions defined, ACSF will be created
                for all atoms in the system.

        Returns:
            np.ndarray | scipy.sparse.coo_matrix: The ACSF output for the
            given system and positions. The return type depends on the
            'sparse'-attribute. The first dimension is given by the number of
            positions and the second dimension is determined by the
            get_number_of_features()-function.
        """
        # Transform the input system into the internal System-object
        system = self.get_system(system)

        # Create C-compatible list of atomic indices for which the ACSF is
        # calculated
        calculate_all = False
        if positions is None:
            calculate_all = True
            indices = np.arange(len(system))
        else:
            indices = positions

        # If periodicity is not requested, and the output is requested for all
        # atoms, we skip all the intricate optimizations that will make things
        # actually slower for this case.
        if calculate_all and not self.periodic:
            n_atoms = len(system)
            all_pos = system.get_positions()
            dmat = dscribe.utils.geometry.get_adjacency_matrix(self.rcut, all_pos, all_pos)
        # Otherwise the amount of pairwise distances that are calculated is
        # kept at minimum. Only distances for the given indices (and possibly
        # the secondary neighbours if G4 is specified) are calculated.
        else:
            # Create the extended system if periodicity is requested. For ACSF only
            # the distance from central atom needs to be considered in extending
            # the system.
            if self.periodic:
                system = dscribe.utils.geometry.get_extended_system(system, self.rcut, return_cell_indices=False)

            # First calculate distances from specified centers to all other
            # atoms. This is already enough for everything else except G4.
            n_atoms = len(system)
            all_pos = system.get_positions()
            central_pos = all_pos[indices]
            dmat_primary = dscribe.utils.geometry.get_adjacency_matrix(self.rcut, central_pos, all_pos)

            # Create symmetric full matrix
            col = dmat_primary.col
            row = [indices[x] for x in dmat_primary.row]  # Fix row numbering to refer to original system
            data = dmat_primary.data
            dmat = coo_matrix((data, (row, col)), shape=(n_atoms, n_atoms))
            dmat_lil = dmat.tolil()
            dmat_lil[col, row] = dmat_lil[row, col]

            # If G4 terms are requested, calculate also secondary neighbour distances
            if len(self.g4_params) != 0:
                neighbour_indices = np.unique(col)
                neigh_pos = all_pos[neighbour_indices]
                dmat_secondary = dscribe.utils.geometry.get_adjacency_matrix(self.rcut, neigh_pos, neigh_pos)
                col = [neighbour_indices[x] for x in dmat_secondary.col]  # Fix col numbering to refer to original system
                row = [neighbour_indices[x] for x in dmat_secondary.row]  # Fix row numbering to refer to original system
                dmat_lil[row, col] = np.array(dmat_secondary.data)

            dmat = dmat_lil.tocoo()

        # Get adjancency list and full dense adjancency matrix
        neighbours = dscribe.utils.geometry.get_adjacency_list(dmat)
        dmat_dense = np.full((n_atoms, n_atoms), sys.float_info.max)  # The non-neighbor values are treated as "infinitely far".
        dmat_dense[dmat.col, dmat.row] = dmat.data

        # Calculate ACSF with C++
        output = self.acsf_wrapper.create(
            system.get_positions(),
            system.get_atomic_numbers(),
            dmat_dense,
            neighbours,
            indices,
        )

        # Check if there are types that have not been declared
        self.check_atomic_numbers(system.get_atomic_numbers())

        # Return sparse matrix if requested
        if self._sparse:
            output = coo_matrix(output)

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
        self.acsf_wrapper.atomic_numbers = self._atomic_numbers

    @property
    def rcut(self):
        return self.acsf_wrapper.rcut

    @rcut.setter
    def rcut(self, value):
        """Used to check the validity of given radial cutoff.

        Args:
            value(float): Radial cutoff.
        """
        if value <= 0:
            raise ValueError("Cutoff radius should be positive.")
        self.acsf_wrapper.rcut = value

    @property
    def g2_params(self):
        return self.acsf_wrapper.g2_params

    @g2_params.setter
    def g2_params(self, value):
        """Used to check the validity of given G2 parameters.

        Args:
            value(n*3 array): List of G2 parameters.
        """
        # Disable case
        if value is None:
            value = []
        else:
            # Check dimensions
            value = np.array(value, dtype=np.float)
            if value.ndim != 2:
                raise ValueError("g2_params should be a matrix with two columns (eta, Rs).")
            if value.shape[1] != 2:
                raise ValueError("g2_params should be a matrix with two columns (eta, Rs).")

            # Check that etas are positive
            if np.any(value[:, 0] <= 0) is True:
                raise ValueError("G2 eta parameters should be positive numbers.")

        self.acsf_wrapper.g2_params = value

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
            value = []
        else:
            # Check dimensions
            value = np.array(value, dtype=np.float)
            if value.ndim != 1:
                raise ValueError("g3_params should be a vector.")

        self.acsf_wrapper.g3_params = value

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
            value = []
        else:
            # Check dimensions
            value = np.array(value, dtype=np.float)
            if value.ndim != 2:
                raise ValueError("g4_params should be a matrix with three columns (eta, zeta, lambda).")
            if value.shape[1] != 3:
                raise ValueError("g4_params should be a matrix with three columns (eta, zeta, lambda).")

            # Check that etas are positive
            if np.any(value[:, 2] <= 0) is True:
                raise ValueError("3-body G4 eta parameters should be positive numbers.")

        self.acsf_wrapper.g4_params = value

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
            value = []
        else:
            # Check dimensions
            value = np.array(value, dtype=np.float)
            if value.ndim != 2:
                raise ValueError("g5_params should be a matrix with three columns (eta, zeta, lambda).")
            if value.shape[1] != 3:
                raise ValueError("g5_params should be a matrix with three columns (eta, zeta, lambda).")

            # Check that etas are positive
            if np.any(value[:, 2] <= 0) is True:
                raise ValueError("3-body G5 eta parameters should be positive numbers.")

        self.acsf_wrapper.g5_params = value
