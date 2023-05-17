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


class DescriptorLocal(Descriptor):
    """An abstract base class for all local descriptors."""

    def __init__(self, periodic, sparse, dtype="float64", average="off"):
        super().__init__(periodic=periodic, sparse=sparse, dtype=dtype)
        self.average = average

    def validate_derivatives_method(self, method, attach):
        """Used to validate and determine the final method for calculating the
        derivatives.
        """
        methods = {"numerical", "auto"}
        if method not in methods:
            raise ValueError(
                "Invalid method specified. Please choose from: {}".format(methods)
            )
        if method == "auto":
            method = "numerical"
        return method

    def derivatives(
        self,
        system,
        centers=None,
        include=None,
        exclude=None,
        method="auto",
        return_descriptor=True,
        attach=False,
        n_jobs=1,
        only_physical_cores=False,
        verbose=False,
    ):
        """Return the descriptor derivatives for the given systems and given centers.

        Args:
            system (:class:`ase.Atoms` or list of :class:`ase.Atoms`): One or
                many atomic structures.
            centers (list): Centers where to calculate the descriptor. Can be
                provided as cartesian positions or atomic indices. Also see the
                "attach"-argument that controls the interperation of centers
                given as atomic indices. If no centers are defined, the
                descriptor output will be created for all atoms in the system.
                When calculating descriptor for multiple systems, provide the
                centers as a list for each system.
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
            method (str): The method for calculating the derivatives. Provide
                either 'numerical', 'analytical' or 'auto'. If using 'auto',
                the most efficient available method is automatically chosen.
            attach (bool): Controls the behaviour of centers defined as
                atomic indices. If True, the centers tied to an atomic index will
                move together with the atoms with respect to which the derivatives
                are calculated against. If False, centers defined as atomic
                indices will be converted into cartesian locations that are
                completely independent of the atom location during derivative
                calculation.
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
            is a either a 4D or 5D array, depending on whether you have
            provided a single or multiple systems. If the output shape for each
            system is the same, a single monolithic numpy array is returned.
            For variable sized output (e.g. differently sized systems,
            different number of centers or different number of included atoms),
            a regular python list is returned. The dimensions are:
            [(n_systems,) n_centers, n_atoms, 3, n_features]. The first
            dimension goes over the different systems in case multiple were
            given. The second dimension goes over the descriptor centers in
            the same order as they were given in the argument. The third
            dimension goes over the included atoms. The order is same as the
            order of atoms in the given system. The fourth dimension goes over
            the cartesian components, x, y and z. The fifth dimension goes over
            the features in the default order.
        """
        method = self.validate_derivatives_method(method, attach)

        # If single system given, skip the parallelization
        if isinstance(system, Atoms):
            n_atoms = len(system)
            indices = self._get_indices(n_atoms, include, exclude)
            return self.derivatives_single(
                system,
                centers,
                indices,
                method=method,
                attach=attach,
                return_descriptor=return_descriptor,
            )

        # Check input validity
        n_samples = len(system)
        if centers is None:
            centers = [None] * n_samples
        if include is None:
            include = [None] * n_samples
        elif is1d(include, np.integer):
            include = [include] * n_samples
        if exclude is None:
            exclude = [None] * n_samples
        elif is1d(exclude, np.integer):
            exclude = [exclude] * n_samples
        n_pos = len(centers)
        if n_pos != n_samples:
            raise ValueError(
                "The given number of centers does not match the given "
                "number of systems."
            )
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
                centers,
                indices,
                [method] * n_samples,
                [attach] * n_samples,
                [return_descriptor] * n_samples,
            )
        )

        # For the descriptor, the output size for each job depends on the exact arguments.
        # Here we precalculate the size for each job to preallocate memory and
        # make the process faster.
        n_features = self.get_number_of_features()

        def get_shapes(job):
            centers = job[1]
            if centers is None:
                n_centers = len(job[0])
            else:
                n_centers = 1 if self.average != "off" else len(centers)
            n_indices = len(job[2])
            return (n_centers, n_indices, 3, n_features), (n_centers, n_features)

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

    def init_descriptor_array(self, n_centers):
        """Return a zero-initialized numpy array for the descriptor."""
        n_features = self.get_number_of_features()
        if self.average != "off":
            c = np.zeros((1, n_features), dtype=np.float64)
        else:
            c = np.zeros((n_centers, n_features), dtype=np.float64)
        return c

    def init_derivatives_array(self, n_centers, n_indices):
        """Return a zero-initialized numpy array for the derivatives."""
        n_features = self.get_number_of_features()
        if self.average != "off":
            return np.zeros((1, n_indices, 3, n_features), dtype=np.float64)
        else:
            return np.zeros((n_centers, n_indices, 3, n_features), dtype=np.float64)

    def derivatives_single(
        self,
        system,
        centers,
        indices,
        method="numerical",
        attach=False,
        return_descriptor=True,
    ):
        """Return the derivatives for the given system.
        Args:
            system (:class:`ase.Atoms`): Atomic structure.
            centers (list): Centers where to calculate SOAP. Can be
                provided as cartesian positions or atomic indices. If no
                centers are defined, the SOAP output will be created for all
                atoms in the system. When calculating SOAP for multiple
                systems, provide the centers as a list for each system.
            indices (list): Indices of atoms for which the derivatives will be
                computed for.
            method (str): The method for calculating the derivatives. Supports
                'numerical'.
            attach (bool): Controls the behaviour of centers defined as
                atomic indices. If True, the centers tied to an atomic index will
                move together with the atoms with respect to which the derivatives
                are calculated against. If False, centers defined as atomic
                indices will be converted into cartesian locations that are
                completely independent of the atom location during derivative
                calculation.
            return_descriptor (bool): Whether to also calculate the descriptor
                in the same function call. This is true by default as it
                typically is faster to calculate both in one go.
        Returns:
            If return_descriptor is True, returns a tuple, where the first item
            is the derivative array and the second is the descriptor array.
            Otherwise only returns the derivatives array. The derivatives array
            is a 4D numpy array. The dimensions are: [n_centers, n_atoms, 3,
            n_features]. The first dimension goes over the SOAP centers in the
            same order as they were given in the argument. The second dimension
            goes over the included atoms. The order is same as the order of
            atoms in the given system. The third dimension goes over the
            cartesian components, x, y and z. The last dimension goes over the
            features in the default order.
        """
        n_indices = len(indices)
        n_centers = len(system) if centers is None else len(centers)

        # Initialize numpy arrays for storing the descriptor and derivatives.
        if return_descriptor:
            c = self.init_descriptor_array(n_centers)
        else:
            c = np.empty(0)
        d = self.init_derivatives_array(n_centers, n_indices)

        # Calculate numerically with extension
        if method == "numerical":
            self.derivatives_numerical(
                d, c, system, centers, indices, attach, return_descriptor
            )
        elif method == "analytical":
            self.derivatives_analytical(
                d, c, system, centers, indices, attach, return_descriptor
            )

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
        centers,
        indices,
        attach=False,
        return_descriptor=True,
    ):
        """Return the numerical derivatives for the given system. This is the
        default numerical implementation with python. You should overwrite this
        with a more optimized method whenever possible.

        Args:
            d (np.array): The derivatives array.
            c (np.array): The descriptor array.
            system (:class:`ase.Atoms`): Atomic structure.
            centers (list): Centers where to calculate SOAP. Can be
                provided as cartesian positions or atomic indices. If no
                centers are defined, the SOAP output will be created for all
                atoms in the system. When calculating SOAP for multiple
                systems, provide the centers as a list for each system.
            indices (list): Indices of atoms for which the derivatives will be
                computed for.
            attach (bool): Controls the behaviour of centers defined as
                atomic indices. If True, the centers tied to an atomic index will
                move together with the atoms with respect to which the derivatives
                are calculated against. If False, centers defined as atomic
                indices will be converted into cartesian locations that are
                completely independent of the atom location during derivative
                calculation.
            return_descriptor (bool): Whether to also calculate the descriptor
                in the same function call. This is true by default as it
                typically is faster to calculate both in one go.
        """
        h = 0.0001
        coeffs = [-1.0 / 2.0, 1.0 / 2.0]
        deltas = [-1.0, 1.0]
        if centers is None:
            centers = range(len(system))
        if not attach and np.issubdtype(type(centers[0]), np.integer):
            centers = system.get_positions()[centers]

        for index, i_atom in enumerate(indices):
            for i_center, center in enumerate(centers):
                for i_comp in range(3):
                    for i_stencil in range(2):
                        system_disturbed = system.copy()
                        i_pos = system_disturbed.get_positions()
                        i_pos[i_atom, i_comp] += h * deltas[i_stencil]
                        system_disturbed.set_positions(i_pos)
                        d1 = self.create_single(system_disturbed, [center])
                        d[i_center, index, i_comp, :] += (
                            coeffs[i_stencil] * d1[0, :] / h
                        )
            index += 1

        if return_descriptor:
            d0 = self.create_single(system, centers)
            np.copyto(c, d0)
