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
import ase.geometry.cell

import sparse

from dscribe.descriptors.descriptormatrix import DescriptorMatrix
from dscribe.utils.dimensionality import is1d

import dscribe.ext


class CoulombMatrix(DescriptorMatrix):
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
    ):
        super().__init__(
            n_atoms_max,
            permutation,
            sigma,
            seed,
            sparse,
        )
        self.wrapper = dscribe.ext.CoulombMatrix(
            n_atoms_max,
            permutation,
            0 if sigma is None else sigma,
            0 if seed is None else seed,
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
        # Combine input arguments / check input validity
        system = [system] if isinstance(system, Atoms) else system
        for s in system:
            if len(s) > self.n_atoms_max:
                raise ValueError(
                    "One of the given systems has more atoms ({}) than allowed "
                    "by n_atoms_max ({}).".format(len(s), self.n_atoms_max)
                )
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

        return output

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
        self.wrapper.create(
            out_des,
            system.get_positions(),
            system.get_atomic_numbers(),
            system.get_cell(),
            system.get_pbc(),
        )

        return out_des

    def derivatives_numerical(
        self,
        d,
        c,
        system,
        indices,
        return_descriptor=True,
    ):
        """Return the numerical derivatives for the given system.
        Args:
            system (:class:`ase.Atoms`): Atomic structure.
            indices (list): Indices of atoms for which the derivatives will be
                computed for.
            return_descriptor (bool): Whether to also calculate the descriptor
                in the same function call. This is true by default as it
                typically is faster to calculate both in one go.
        Returns:
            If return_descriptor is True, returns a tuple, where the first item
            is the derivative array and the second is the descriptor array.
            Otherwise only returns the derivatives array. The derivatives array
            is a 3D numpy array. The dimensions are: [n_atoms, 3, n_features].
            The first dimension goes over the included atoms. The order is same
            as the order of atoms in the given system. The second dimension
            goes over the cartesian components, x, y and z. The last dimension
            goes over the features in the default order.
        """
        pos = system.get_positions()
        Z = system.get_atomic_numbers()
        cell = ase.geometry.cell.complete_cell(system.get_cell())
        pbc = np.asarray(system.get_pbc(), dtype=bool)

        self.wrapper.derivatives_numerical(
            d,
            c,
            pos,
            Z,
            cell,
            pbc,
            indices,
            return_descriptor,
        )
