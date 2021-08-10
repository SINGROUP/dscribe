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

from ase import Atoms

import sparse

from dscribe.core import System
from dscribe.descriptors.matrixdescriptor import MatrixDescriptor

import dscribe.ext


class CoulombMatrix(MatrixDescriptor):
    """Calculates the zero padded Coulomb matrix.

    The Coulomb matrix is defined as:

        C_ij = 0.5 Zi**exponent, when i = j
             = (Zi*Zj)/(Ri-Rj), when i != j

    The matrix is padded with invisible atoms, which means that the matrix is
    padded with zeros until the maximum allowed size defined by n_max_atoms is
    reached.

    To reach invariance against permutation of atoms, specify a valid option
    for the permutation parameter.

    For reference, see:
        "Fast and Accurate Modeling of Molecular Atomization Energies with
        Machine Learning", Matthias Rupp, Alexandre Tkatchenko, Klaus-Robert
        Mueller, and O.  Anatole von Lilienfeld, Phys. Rev. Lett, (2012),
        https://doi.org/10.1103/PhysRevLett.108.058301
    and
        "Learning Invariant Representations of Molecules for Atomization Energy
        Prediction", Gregoire Montavon et. al, Advances in Neural Information
        Processing Systems 25 (NIPS 2012)
    """
    def __init__(
        self,
        n_atoms_max,
        permutation="sorted_l2",
        sigma=None,
        seed=None,
        sparse=False,
        flatten=True,
    ):
        super().__init__(
            n_atoms_max,
            permutation,
            sigma,
            seed,
            flatten,
            sparse,
        )

    def create(self, system, n_jobs=1, only_physical_cores=False, verbose=False):
        """Return the Coulomb matrix for the given systems.

        Args:
            system (:class:`ase.Atoms` or list of :class:`ase.Atoms`): One or
                many atomic structures.
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
            np.ndarray | sparse.COO: Coulomb matrix for the given systems. The
            return type depends on the 'sparse'-attribute. The first dimension
            is determined by the amount of systems.
        """
        if isinstance(system, Atoms):
            system = [system]

        # Check input validity
        for s in system:
            if len(s) > self.n_atoms_max:
                raise ValueError(
                    "One of the given systems has more atoms ({}) than allowed "
                    "by n_atoms_max ({}).".format(len(s), self.n_atoms_max)
                )

        # If single system given, skip the parallelization
        if len(system) == 1:
            output = self.create_single(system[0])
        else:
            # Combine input arguments
            inp = [(i_sys,) for i_sys in system]

            # Create in parallel
            output = self.create_parallel(
                inp,
                self.create_single,
                n_jobs,
                [self.get_number_of_features()],
                only_physical_cores,
                verbose=verbose,
            )

        # Unflatten output if so requested
        if not self.flatten and self.permutation != "eigenspectrum":
            output = self.unflatten(output, system)

        return output

    def unflatten(self, output, systems):
        n_systems = len(systems)
        if self.sparse:
            if n_systems != 1:
                full = sparse.zeros((n_systems, self.n_atoms_max, self.n_atoms_max), format="dok")
                for i_sys, system in enumerate(systems):
                    n_atoms = len(system)
                    full[i_sys, 0:n_atoms, 0:n_atoms] = output[i_sys, 0:n_atoms * n_atoms].reshape((n_atoms, n_atoms)).todense()
            else:
                full = sparse.zeros((self.n_atoms_max, self.n_atoms_max), format="dok")
                n_atoms = len(systems[0])
                full[0:n_atoms, 0:n_atoms] = output[0:n_atoms * n_atoms].reshape((n_atoms, n_atoms)).todense()
            full = full.to_coo()
        else:
            if n_systems != 1:
                full = np.zeros((n_systems, self.n_atoms_max, self.n_atoms_max))
                for i_sys, system in enumerate(systems):
                    n_atoms = len(system)
                    full[i_sys, 0:n_atoms, 0:n_atoms] = output[i_sys, 0:n_atoms * n_atoms].reshape((n_atoms, n_atoms))
            else:
                full = np.zeros((self.n_atoms_max, self.n_atoms_max))
                n_atoms = len(systems[0])
                full[0:n_atoms, 0:n_atoms] = output[0:n_atoms * n_atoms].reshape((n_atoms, n_atoms))
        return full

    def create_single(self, system):
        """
        Args:
            system (:class:`ase.Atoms`): Input system.

        Returns:
            ndarray: The zero padded matrix as a flattened 1D array.
        """
        # Initialize output array in dense format.
        out_des = np.zeros((self.get_number_of_features()), dtype=np.float64)

        # Calculate with C++ extension
        wrapper = dscribe.ext.CoulombMatrix(
            self.n_atoms_max,
            self.permutation,
            0 if self.sigma is None else self.sigma,
            0 if self.seed is None else self.seed,
        )
        wrapper.create(
            out_des,
            system.get_positions(),
            system.get_atomic_numbers(),
            system.get_cell(),
            system.get_pbc(),
        )

        # If a sparse matrix is requested, convert to sparse.COO
        if self._sparse:
            out_des = sparse.COO.from_numpy(out_des)

        return out_des
