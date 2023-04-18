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
from dscribe.descriptors.descriptormatrix import DescriptorMatrix


class SineMatrix(DescriptorMatrix):
    """Calculates the zero padded Sine matrix for different systems.

    The Sine matrix is defined as:

        Cij = 0.5 Zi**exponent      | i = j
            = (Zi*Zj)/phi(Ri, Rj)   | i != j

        where phi(r1, r2) = | B * sum(k = x,y,z)[ek * sin^2(pi * ek * B^-1
        (r2-r1))] | (B is the matrix of basis cell vectors, ek are the unit
        vectors)

    The matrix is padded with invisible atoms, which means that the matrix is
    padded with zeros until the maximum allowed size defined by n_max_atoms is
    reached.

    For reference, see:
        "Crystal Structure Representations for Machine Learning Models of
        Formation Energies", Felix Faber, Alexander Lindmaa, Anatole von
        Lilienfeld, and Rickard Armiento, International Journal of Quantum
        Chemistry, (2015),
        https://doi.org/10.1002/qua.24917
    """

    def create(self, system, n_jobs=1, only_physical_cores=False, verbose=False):
        """Return the Sine matrix for the given systems.

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
            np.ndarray | sparse.COO: Sine matrix for the given systems. The
            return type depends on the 'sparse'-attribute. The first dimension
            is determined by the amount of systems.
        """
        # Combine input arguments / check input validity
        system = [system] if isinstance(system, Atoms) else system
        for s in system:
            if len(s) > self.n_atoms_max:
                raise ValueError(
                    "One of the given systems has more atoms ({}) than allowed "
                    "by n_atoms_max ({}).".format(len(s), self.n_atoms_max)
                )
        inp = [(i_sys,) for i_sys in system]

        # Determine if the outputs have a fixed size
        n_features = self.get_number_of_features()
        static_size = [n_features]

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
        """Creates the Sine matrix for the given system.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.

        Returns:
            np.ndarray: Sine matrix as a 2D array.
        """
        # Force the use of periodic boundary conditions
        system = System.from_atoms(system)
        system.set_pbc(True)

        # Cell and inverse cell
        B = system.get_cell()
        try:
            B_inv = system.get_cell_inverse()
        except:
            raise ValueError(
                "The given system has a non-invertible cell matrix: {}.".format(B)
            )

        # Difference vectors as a 3D tensor
        diff_tensor = system.get_displacement_tensor()

        # Calculate phi
        arg_to_sin = np.pi * np.dot(diff_tensor, B_inv)
        phi = np.linalg.norm(np.dot(np.sin(arg_to_sin) ** 2, B), axis=2)

        with np.errstate(divide="ignore"):
            phi = np.reciprocal(phi)

        # Calculate Z_i*Z_j
        q = system.get_atomic_numbers()
        qiqj = q[None, :] * q[:, None]
        np.fill_diagonal(phi, 0)

        # Multiply by charges
        smat = qiqj * phi

        # Set diagonal
        np.fill_diagonal(smat, 0.5 * q**2.4)

        return smat
