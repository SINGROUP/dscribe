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

from dscribe.core import System
from dscribe.descriptors.matrixdescriptor import MatrixDescriptor


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
            return type depends on the 'sparse' and 'flatten'-attributes. For
            flattened output a single numpy array or sparse.COO is returned.
            The first dimension is determined by the amount of systems.
        """
        if isinstance(system, (Atoms, System)):
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
            return self.create_single(system[0])

        # Combine input arguments
        inp = [(i_sys,) for i_sys in system]

        # Determine if the outputs have a fixed size
        n_features = self.get_number_of_features()
        if self._flatten:
            static_size = [n_features]
        elif self.permutation == "eigenspectrum":
            static_size = [self.n_atoms_max]
        else:
            static_size = [self.n_atoms_max, self.n_atoms_max]

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

    def get_matrix(self, system):
        """Creates the Coulomb matrix for the given system.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.

        Returns:
            np.ndarray: Coulomb matrix as a 2D array.
        """
        # Make sure that the system is non-periodic
        system.set_pbc(False)

        # Calculate offdiagonals
        q = system.get_atomic_numbers()
        qiqj = q[None, :] * q[:, None]
        idmat = system.get_inverse_distance_matrix()
        np.fill_diagonal(idmat, 0)
        cmat = qiqj * idmat

        # Set diagonal
        np.fill_diagonal(cmat, 0.5 * q ** 2.4)

        return cmat
