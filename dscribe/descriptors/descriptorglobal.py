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
import sparse as sp
from ase import Atoms

from dscribe.descriptors.descriptor import Descriptor
from dscribe.utils.dimensionality import is1d


class DescriptorGlobal(Descriptor):
    """An abstract base class for all global descriptors."""

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
        method = self.validate_derivatives_method(method)

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
        n_indices = len(indices)

        # Initialize numpy arrays for storing the descriptor and derivatives.
        n_features = self.get_number_of_features()
        if return_descriptor:
            c = np.zeros(n_features, dtype=np.float64)
        else:
            c = np.empty(0)
        d = np.zeros((n_indices, 3, n_features), dtype=np.float64)

        if method == "numerical":
            self.derivatives_numerical(d, c, system, indices, return_descriptor)
        elif method == "analytical":
            self.derivatives_analytical(d, c, system, indices, return_descriptor)

        d = self.format_array(d)
        c = self.format_array(c)

        if return_descriptor:
            return (d, c)
        return d

    def derivatives_numerical(
        self,
        d,
        c,
        system,
        indices,
        return_descriptor=True,
    ):
        """Return the numerical derivatives for the given system. This is the
        default numerical implementation using python. You should overwrite this
        with a more optimized method whenever possible.

        Args:
            d (np.array): The derivatives array.
            c (np.array): The descriptor array.
            system (:class:`ase.Atoms`): Atomic structure.
            indices (list): Indices of atoms for which the derivatives will be
                computed for.
            return_descriptor (bool): Whether to also calculate the descriptor
                in the same function call. This is true by default as it
                typically is faster to calculate both in one go.
        """
        h = 0.0001
        n_atoms = len(system)
        n_features = self.get_number_of_features()

        # The maximum error depends on how big the system is. With a small system
        # the error is smaller for non-periodic systems than the corresponding
        # error when periodicity is turned on. The errors become equal (~1e-5) when
        # the size of the system is increased.
        coeffs = [-1.0 / 2.0, 1.0 / 2.0]
        deltas = [-1.0, 1.0]
        derivatives_python = np.zeros((n_atoms, 3, n_features))
        for i_atom in range(len(system)):
            for i_comp in range(3):
                for i_stencil in range(2):
                    system_disturbed = system.copy()
                    i_pos = system_disturbed.get_positions()
                    i_pos[i_atom, i_comp] += h * deltas[i_stencil]
                    system_disturbed.set_positions(i_pos)
                    d1 = self.create_single(system_disturbed)
                    derivatives_python[i_atom, i_comp, :] += coeffs[i_stencil] * d1 / h

        i = 0
        for index in indices:
            d[i, :] = derivatives_python[index, :, :]
            i += 1

        if return_descriptor:
            np.copyto(c, self.create_single(system))
