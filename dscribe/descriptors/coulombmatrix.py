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

from dscribe.core import System
from dscribe.descriptors.matrixdescriptor import MatrixDescriptor
from dscribe.utils.dimensionality import is1d

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

        # Unflatten output if so requested
        if not self.flatten and self.permutation != "eigenspectrum":
            output = self.unflatten(output, system)

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

        # If a sparse matrix is requested, convert to sparse.COO
        if self._sparse:
            out_des = sparse.COO.from_numpy(out_des)

        return out_des

    def unflatten(self, output, systems):
        n_systems = len(systems)
        if self.sparse:
            if n_systems != 1:
                full = sparse.zeros(
                    (n_systems, self.n_atoms_max, self.n_atoms_max), format="dok"
                )
                for i_sys, system in enumerate(systems):
                    n_atoms = len(system)
                    full[i_sys] = (
                        output[i_sys]
                        .reshape((self.n_atoms_max, self.n_atoms_max))
                        .todense()
                    )
                full = full.to_coo()
            else:
                full = output.reshape((self.n_atoms_max, self.n_atoms_max))
        else:
            if n_systems != 1:
                full = np.zeros((n_systems, self.n_atoms_max, self.n_atoms_max))
                for i_sys, system in enumerate(systems):
                    n_atoms = len(system)
                    full[i_sys] = output[i_sys].reshape(
                        (self.n_atoms_max, self.n_atoms_max)
                    )
            else:
                full = output.reshape((self.n_atoms_max, self.n_atoms_max))
        return full

    def derivatives(
        self,
        system,
        include=None,
        exclude=None,
        method="auto",
        return_descriptor=True,
        n_jobs=1,
        only_physical_cores=False,
        verbose=False,
    ):
        """Return the descriptor derivatives for the given system(s).

        Args:
            system (:class:`ase.Atoms` or list of :class:`ase.Atoms`): One or
                many atomic structures.
            include (list): Indices of atoms to compute the derivatives on.
                When calculating descriptor for multiple systems, provide
                either a one-dimensional list that if applied to all systems or
                a two-dimensional list of indices. Cannot be provided together
                with 'exclude'.
            exclude (list): Indices of atoms not to compute the derivatives on.
                When calculating descriptor for multiple systems, provide
                either a one-dimensional list that if applied to all systems or
                a two-dimensional list of indices. Cannot be provided together
                with 'include'.
            method (str): The method for calculating the derivatives. Supports
                'numerical' and 'auto'. Defaults to using 'auto' which corresponds
                to the the most efficient available method.
            return_descriptor (bool): Whether to also calculate the descriptor
                in the same function call. Notice that it typically is faster
                to calculate both in one go.
            n_jobs (int): Number of parallel jobs to instantiate. Parallellizes
                the calculation across samples. Defaults to serial calculation
                with n_jobs=1. If a negative number is given, the number of jobs
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
            If return_descriptor is True, returns a tuple, where the first item
            is the derivative array and the second is the descriptor array.
            Otherwise only returns the derivatives array. The derivatives array
            is a either a 3D or 4D array, depending on whether you have
            provided a single or multiple systems. If the output shape for each
            system is the same, a single monolithic numpy array is returned.
            For variable sized output (e.g. differently sized systems,different
            number of included atoms), a regular python list is returned. The
            dimensions are: [(n_systems,) n_atoms, 3, n_features]. The first
            dimension goes over the different systems in case multiple were
            given. The second dimension goes over the included atoms. The order
            is same as the order of atoms in the given system. The third
            dimension goes over the cartesian components, x, y and z. The
            fourth dimension goes over the features in the default order.
        """
        # Validate/determine the appropriate calculation method.
        methods = {"numerical", "auto"}
        if method not in methods:
            raise ValueError(
                "Invalid method specified. Please choose from: {}".format(methods)
            )
        if method == "auto":
            method = "numerical"

        # Check input validity
        system = [system] if isinstance(system, Atoms) else system
        n_samples = len(system)
        if include is None:
            include = [None] * n_samples
        elif is1d(include, np.integer):
            include = [include] * n_samples
        if exclude is None:
            exclude = [None] * n_samples
        elif is1d(exclude, np.integer):
            exclude = [exclude] * n_samples
        n_inc = len(include)
        if n_inc != n_samples:
            raise ValueError(
                "The given number of includes does not match the given "
                "number of systems."
            )
        n_exc = len(exclude)
        if n_exc != n_samples:
            raise ValueError(
                "The given number of excludes does not match the given "
                "number of systems."
            )

        # Determine the atom indices that are displaced
        indices = []
        for sys, inc, exc in zip(system, include, exclude):
            n_atoms = len(sys)
            indices.append(self._get_indices(n_atoms, inc, exc))

        # Combine input arguments
        inp = list(
            zip(
                system,
                indices,
                [method] * n_samples,
                [return_descriptor] * n_samples,
            )
        )

        # Determine a fixed output size if possible
        n_features = self.get_number_of_features()

        def get_shapes(job):
            n_indices = len(job[1])
            return (n_indices, 3, n_features), (n_features,)

        derivatives_shape, descriptor_shape = get_shapes(inp[0])

        def is_variable(inp):
            for job in inp[1:]:
                i_der_shape, i_desc_shape = get_shapes(job)
                if i_der_shape != derivatives_shape or i_desc_shape != descriptor_shape:
                    return True
            return False

        if is_variable(inp):
            derivatives_shape = None
            descriptor_shape = None

        # Create in parallel
        output = self.derivatives_parallel(
            inp,
            self.derivatives_single,
            n_jobs,
            derivatives_shape,
            descriptor_shape,
            return_descriptor,
            only_physical_cores,
            verbose=verbose,
        )

        return output

    def derivatives_single(
        self,
        system,
        indices,
        method="numerical",
        return_descriptor=True,
    ):
        """Return the derivatives for the given system.

        Args:
            system (:class:`ase.Atoms`): Atomic structure.
            indices (list): Indices of atoms for which the derivatives will be
                computed for.
            method (str): The method for calculating the derivatives. Supports
                'numerical'.
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
        sorted_species = self._atomic_numbers
        n_indices = len(indices)

        # Initialize numpy arrays for storing the descriptor and derivatives.
        n_features = self.get_number_of_features()
        if return_descriptor:
            c = np.zeros(n_features)
        else:
            c = np.empty(0)
        d = np.zeros((n_indices, 3, n_features), dtype=np.float64)

        # Calculate numerically with extension
        if method == "numerical":
            self.wrapper.derivatives_numerical(
                d,
                c,
                pos,
                Z,
                ase.geometry.cell.complete_cell(system.get_cell()),
                np.asarray(system.get_pbc(), dtype=bool),
                indices,
                return_descriptor,
            )

        # Convert to the final output precision.
        if self.dtype == "float32":
            d = d.astype(self.dtype)
            c = c.astype(self.dtype)

        # Convert to sparse here. Currently everything up to this point is
        # calculated with dense matrices. This imposes some memory limitations.
        if self.sparse:
            d = sp.COO.from_numpy(d)
            c = sp.COO.from_numpy(c)

        if return_descriptor:
            return (d, c)
        return d
