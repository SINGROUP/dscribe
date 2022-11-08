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
import ase.geometry.cell

import dscribe.ext
from dscribe.descriptors.descriptor import Descriptor
from dscribe.utils.dimensionality import is1d


class DescriptorGlobal(Descriptor):
    """An abstract base class for all global descriptors."""
    def __init__(self, periodic, flatten, sparse, dtype="float64"):
        super().__init__(periodic, flatten ,sparse, dtype)

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
        method = self.get_derivatives_method(method)

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
        n_indices = len(indices)

        # Initialize numpy arrays for storing the descriptor and derivatives.
        n_features = self.get_number_of_features()
        if return_descriptor:
            c = np.zeros(n_features)
        else:
            c = np.empty(0)
        d = np.zeros((n_indices, 3, n_features), dtype=np.float64)

        # Calculate numerically with extension
        cell = ase.geometry.cell.complete_cell(system.get_cell())
        pbc = np.asarray(system.get_pbc(), dtype=bool)
        if method == "numerical":
            system_cpp = dscribe.ext.System(
                pos,
                Z,
                ase.geometry.cell.complete_cell(system.get_cell()),
                np.asarray(system.get_pbc(), dtype=bool),
            )
            self.wrapper.derivatives_numerical(
                d,
                c,
                system_cpp,
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