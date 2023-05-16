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
from abc import ABC, abstractmethod

import numpy as np

import sparse as sp

from dscribe.utils.species import get_atomic_numbers

import joblib
from joblib import Parallel, delayed


class Descriptor(ABC):
    """An abstract base class for all descriptors."""

    def __init__(self, periodic, sparse, dtype="float64"):
        """
        Args:
            periodic (bool): Whether the descriptor should take PBC into account.
            sparse (bool): Whether the output should use a sparse format.
            dtype (str): The output data type.
        """
        supported_dtype = set(("float32", "float64"))
        if dtype not in supported_dtype:
            raise ValueError(
                "Invalid output data type '{}' given. Please use "
                "one of the following: {}".format(dtype, supported_dtype)
            )
        self.sparse = sparse
        self.periodic = periodic
        self.dtype = dtype
        self._atomic_numbers = None
        self._atomic_number_set = None
        self._species = None

    @abstractmethod
    def create(self, system, *args, **kwargs):
        """Creates the descriptor for the given systems.

        Args:
            system (ase.Atoms): The system for which to create the descriptor.
            args: Descriptor specific positional arguments.
            kwargs: Descriptor specific keyword arguments.

        Returns:
            np.array | scipy.sparse.coo_matrix: A descriptor for the system.
        """

    @abstractmethod
    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """

    def validate_derivatives_method(self, method):
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

    @property
    def sparse(self):
        return self._sparse

    @sparse.setter
    def sparse(self, value):
        """Sets whether the output should be sparse or not.

        Args:
            value(float): Should the output be in sparse format.
        """
        self._sparse = value

    @property
    def periodic(self):
        return self._periodic

    @periodic.setter
    def periodic(self, value):
        """Sets whether the inputs should be considered periodic or not.

        Args:
            value(float): Are the systems periodic.
        """
        self._periodic = value

    def _set_species(self, species):
        """Used to setup the species information for this descriptor. This
        information includes an ordered list of unique atomic numbers, a set
        of atomic numbers and the original variable contents.

        Args:
            species(iterable): Chemical species either as a list of atomic
                numbers or list of chemical symbols.
        """
        # The species are stored as atomic numbers for internal use.
        atomic_numbers = get_atomic_numbers(species)
        self._atomic_numbers = atomic_numbers
        self._atomic_number_set = set(self._atomic_numbers)
        self._species = species

    def check_atomic_numbers(self, atomic_numbers):
        """Used to check that the given atomic numbers have been defined for
        this descriptor.

        Args:
            species(iterable): Atomic numbers to check.

        Raises:
            ValueError: If the atomic numbers in the given system are not
            included in the species given to this descriptor.
        """
        # Check that the system does not have elements that are not in the list
        # of atomic numbers
        zs = set(atomic_numbers)
        if not zs.issubset(self._atomic_number_set):
            raise ValueError(
                "The following atomic numbers are not defined "
                "for this descriptor: {}".format(zs.difference(self._atomic_number_set))
            )

    def format_array(self, input):
        """Used to format a float64 numpy array in the final format that will be
        returned to the user.
        """
        if self.dtype != "float64":
            input = input.astype(self.dtype)
        if self.sparse:
            input = sp.COO.from_numpy(input)

        return input

    def create_parallel(
        self,
        inp,
        func,
        n_jobs,
        static_size=None,
        only_physical_cores=False,
        verbose=False,
        prefer="processes",
    ):
        """Used to parallelize the descriptor creation across multiple systems.

        Args:
            inp(list): Contains a tuple of input arguments for each processed
                system. These arguments are fed to the function specified by
                "func".
            func(function): Function that outputs the descriptor when given
                input arguments from "inp".
            n_jobs (int): Number of parallel jobs to instantiate. Parallellizes
                the calculation across samples. Defaults to serial calculation
                with n_jobs=1. If a negative number is given, the number of jobs
                will be calculated with, n_cpus + n_jobs, where n_cpus is the
                amount of CPUs as reported by the OS. With only_physical_cores
                you can control which types of CPUs are counted in n_cpus.
            output_sizes(list of ints): The size of the output for each job.
                Makes the creation faster by preallocating the correct amount of
                memory beforehand. If not specified, a dynamically created list of
                outputs is used.
            only_physical_cores (bool): If a negative n_jobs is given,
                determines which types of CPUs are used in calculating the
                number of jobs. If set to False (default), also virtual CPUs
                are counted.  If set to True, only physical CPUs are counted.
            verbose(bool): Controls whether to print the progress of each job
                into to the console.
            prefer (str): The parallelization method. Valid options are:

                - "processes": Parallelization based on processes. Uses the
                  "loky" backend in joblib to serialize the jobs and run them
                  in separate processes. Using separate processes has a bigger
                  memory and initialization overhead than threads, but may
                  provide better scalability if perfomance is limited by the
                  Global Interpreter Lock (GIL).

                - "threads": Parallelization based on threads. Has bery low
                  memory and initialization overhead. Performance is limited by
                  the amount of pure python code that needs to run. Ideal when
                  most of the calculation time is used by C/C++ extensions that
                  release the GIL.

        Returns:
            np.ndarray | sparse.COO | list: The descriptor output
            for each given input. The return type depends on the desciptor
            setup.
        """
        # If single system given, skip the parallelization overhead
        if len(inp) == 1:
            return self.format_array(func(*inp[0]))

        # Determine the number of jobs
        if n_jobs < 0:
            n_jobs = joblib.cpu_count(only_physical_cores) + n_jobs
        if n_jobs <= 0:
            raise ValueError("Invalid number of jobs specified.")

        # Split data into n_jobs (almost) equal jobs
        n_samples = len(inp)
        is_sparse = self._sparse
        k, m = divmod(n_samples, n_jobs)
        jobs = (
            inp[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n_jobs)
        )

        def create_multiple(arguments, func, is_sparse, index, verbose):
            """This is the function that is called by each job but with
            different parts of the data.
            """
            # Initialize output
            if static_size is None:
                i_shape = None
                results = []
            else:
                i_shape = [len(arguments)] + static_size
                if is_sparse:
                    data = []
                    coords = []
                else:
                    results = np.empty(i_shape, dtype=self.dtype)

            i_sample = 0
            old_percent = 0
            n_samples = len(arguments)

            for i_sample, i_arg in enumerate(arguments):
                i_out = func(*i_arg)
                i_out = self.format_array(i_out)

                # If the shape varies, just add result into a list
                if static_size is None:
                    results.append(i_out)
                else:
                    if is_sparse:
                        sample_index = np.full((1, i_out.data.size), i_sample)
                        data.append(i_out.data)
                        coords.append(np.vstack((sample_index, i_out.coords)))
                    else:
                        results[i_sample] = i_out

                if verbose:
                    current_percent = (i_sample + 1) / n_samples * 100
                    if current_percent >= old_percent + 1:
                        old_percent = current_percent
                        print("Process {0}: {1:.1f} %".format(index, current_percent))

            if static_size is not None and is_sparse:
                data = np.concatenate(data)
                coords = np.concatenate(coords, axis=1)
                results = sp.COO(coords, data, shape=i_shape)

            return (results, index)

        vec_lists = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(create_multiple)(i_args, func, is_sparse, index, verbose)
            for index, i_args in enumerate(jobs)
        )

        # Restore the calculation order. If using the threading backend, the
        # input order may have been lost.
        vec_lists.sort(key=lambda x: x[1])

        # Remove the job index
        vec_lists = [x[0] for x in vec_lists]

        # Create the final results by concatenating the results from each
        # process
        if static_size is not None:
            if self._sparse:
                sample_offset = 0
                data = []
                coords = []
                for i, i_res in enumerate(vec_lists):
                    i_n_desc = i_res.shape[0]
                    i_data = i_res.data
                    i_coords = i_res.coords
                    i_coords[0, :] += sample_offset

                    data.append(i_data)
                    coords.append(i_coords)

                    # Increase the sample offset
                    sample_offset += i_n_desc

                # Saves the descriptors as a sparse matrix
                data = np.concatenate(data)
                coords = np.concatenate(coords, axis=1)
                results = sp.COO(coords, data, shape=[n_samples] + static_size)
            else:
                results = np.concatenate(vec_lists, axis=0)
        else:
            results = []
            for part in vec_lists:
                results.extend(part)

        return results

    def _get_indices(self, n_atoms, include, exclude):
        """Given the number of atoms and which indices to include or exclude,
        returns a list of final indices that will be used. If not includes or
        excludes are defined, then by default all atoms will be included.

        Args:
            n_atoms(int): Number of atoms.
            include(list of ints or None): The atomic indices to include, or
                None if no specific includes made.
            exclude(list of ints or None): The atomic indices to exclude, or
                None if no specific excludes made.

        Returns:
            np.ndarray: List of atomic indices that will be included from the
            system.
        """
        if include is None and exclude is None:
            displaced_indices = np.arange(n_atoms)
        elif include is not None and exclude is None:
            include = np.asarray(include)
            if np.any(include > n_atoms - 1) or np.any(include < 0):
                raise ValueError(
                    "Invalid index provided in the list of included atoms."
                )
            displaced_indices = include
        elif exclude is not None and include is None:
            exclude = np.asarray(list(set(exclude)))
            if np.any(exclude > n_atoms - 1) or np.any(exclude < 0):
                raise ValueError(
                    "Invalid index provided in the list of excluded atoms."
                )
            displaced_indices = np.arange(n_atoms)
            if len(exclude) > 0:
                displaced_indices = np.delete(displaced_indices, exclude)
        else:
            raise ValueError("Provide either 'include' or 'exclude', not both.")
        n_displaced = len(displaced_indices)
        if n_displaced == 0:
            raise ValueError("Please include at least one atom.")

        return displaced_indices

    def derivatives_parallel(
        self,
        inp,
        func,
        n_jobs,
        derivatives_shape,
        descriptor_shape,
        return_descriptor,
        only_physical_cores=False,
        verbose=False,
        prefer="processes",
    ):
        """Used to parallelize the descriptor creation across multiple systems.

        Args:
            inp(list): Contains a tuple of input arguments for each processed
                system. These arguments are fed to the function specified by
                "func".
            func(function): Function that outputs the descriptor when given
                input arguments from "inp".
            n_jobs (int): Number of parallel jobs to instantiate. Parallellizes
                the calculation across samples. Defaults to serial calculation
                with n_jobs=1. If a negative number is given, the number of jobs
                will be calculated with, n_cpus + n_jobs, where n_cpus is the
                amount of CPUs as reported by the OS. With only_physical_cores
                you can control which types of CPUs are counted in n_cpus.
            derivatives_shape(list or None): If a fixed size output is produced from
                each job, this contains its shape. For variable size output
                this parameter is set to None
            derivatives_shape(list or None): If a fixed size output is produced from
                each job, this contains its shape. For variable size output
                this parameter is set to None
            only_physical_cores (bool): If a negative n_jobs is given,
                determines which types of CPUs are used in calculating the
                number of jobs. If set to False (default), also virtual CPUs
                are counted.  If set to True, only physical CPUs are counted.
            verbose(bool): Controls whether to print the progress of each job
                into to the console.
            prefer(str): The parallelization method. Valid options are:

                - "processes": Parallelization based on processes. Uses the
                  "loky" backend in joblib to serialize the jobs and run them
                  in separate processes. Using separate processes has a bigger
                  memory and initialization overhead than threads, but may
                  provide better scalability if perfomance is limited by the
                  Global Interpreter Lock (GIL).

                - "threads": Parallelization based on threads. Has bery low
                  memory and initialization overhead. Performance is limited by
                  the amount of pure python code that needs to run. Ideal when
                  most of the calculation time is used by C/C++ extensions that
                  release the GIL.

        Returns:
            np.ndarray | sparse.COO | list: The descriptor output
            for each given input. The return type depends on the desciptor
            setup.
        """
        # If single system given, skip the parallelization overhead
        if len(inp) == 1:
            return func(*inp[0])

        # Determine the number of jobs
        if n_jobs < 0:
            n_jobs = joblib.cpu_count(only_physical_cores) + n_jobs
        if n_jobs <= 0:
            raise ValueError("Invalid number of jobs specified.")

        # Split data into n_jobs (almost) equal jobs
        n_samples = len(inp)
        is_sparse = self._sparse
        k, m = divmod(n_samples, n_jobs)
        jobs = (
            inp[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n_jobs)
        )

        def create_multiple_with_descriptor(arguments, func, index, verbose):
            """This is the function that is called by each job but with
            different parts of the data.
            """
            # Initialize output. If a fixed size is given, a dense/sparse array
            # is initialized. For variable size output a regular list is
            # returned.
            n_samples = len(arguments)
            if derivatives_shape:
                shape_der = [n_samples]
                shape_der.extend(derivatives_shape)
                if is_sparse:
                    data_der = []
                    coords_der = []
                else:
                    derivatives = np.empty(shape_der, dtype=self.dtype)
            else:
                derivatives = []
            if descriptor_shape:
                shape_des = [n_samples]
                shape_des.extend(descriptor_shape)
                if is_sparse:
                    data_des = []
                    coords_des = []
                else:
                    descriptors = np.empty(shape_des, dtype=self.dtype)
            else:
                descriptors = []
            old_percent = 0

            # Loop through all samples assigned for this job
            for i_sample, i_arg in enumerate(arguments):
                i_der, i_des = func(*i_arg)
                if descriptor_shape:
                    if is_sparse:
                        sample_index = np.full((1, i_des.data.size), i_sample)
                        data_des.append(i_des.data)
                        coords_des.append(np.vstack((sample_index, i_des.coords)))
                    else:
                        descriptors[i_sample] = i_des
                else:
                    descriptors.append(i_des)

                if derivatives_shape:
                    if is_sparse:
                        sample_index = np.full((1, i_der.data.size), i_sample)
                        data_der.append(i_der.data)
                        coords_der.append(np.vstack((sample_index, i_der.coords)))
                    else:
                        derivatives[i_sample] = i_der
                else:
                    derivatives.append(i_der)

                if verbose:
                    current_percent = (i_sample + 1) / n_samples * 100
                    if current_percent >= old_percent + 1:
                        old_percent = current_percent
                        print("Process {0}: {1:.1f} %".format(index, current_percent))

            if is_sparse:
                if descriptor_shape is not None:
                    data_des = np.concatenate(data_des)
                    coords_des = np.concatenate(coords_des, axis=1)
                    descriptors = sp.COO(coords_des, data_des, shape=shape_des)
                if derivatives_shape is not None:
                    data_der = np.concatenate(data_der)
                    coords_der = np.concatenate(coords_der, axis=1)
                    derivatives = sp.COO(coords_der, data_der, shape=shape_der)

            return ((derivatives, descriptors), index)

        def create_multiple_without_descriptor(arguments, func, index, verbose):
            """This is the function that is called by each job but with
            different parts of the data.
            """
            # Initialize output
            n_samples = len(arguments)
            if derivatives_shape:
                shape_der = [n_samples]
                shape_der.extend(derivatives_shape)
                if is_sparse:
                    data_der = []
                    coords_der = []
                else:
                    derivatives = np.empty(shape_der, dtype=self.dtype)
            else:
                derivatives = []

            old_percent = 0

            # Loop through all samples assigned for this job
            for i_sample, i_arg in enumerate(arguments):
                i_der = func(*i_arg)
                if derivatives_shape:
                    if is_sparse:
                        sample_index = np.full((1, i_der.data.size), i_sample)
                        data_der.append(i_der.data)
                        coords_der.append(np.vstack((sample_index, i_der.coords)))
                    else:
                        derivatives[i_sample] = i_der
                else:
                    derivatives.append(i_der)

                if verbose:
                    current_percent = (i_sample + 1) / n_samples * 100
                    if current_percent >= old_percent + 1:
                        old_percent = current_percent
                        print("Process {0}: {1:.1f} %".format(index, current_percent))

            if is_sparse and derivatives_shape is not None:
                data_der = np.concatenate(data_der)
                coords_der = np.concatenate(coords_der, axis=1)
                derivatives = sp.COO(coords_der, data_der, shape=shape_der)
            return ((derivatives,), index)

        if return_descriptor:
            vec_lists = Parallel(n_jobs=n_jobs, prefer=prefer)(
                delayed(create_multiple_with_descriptor)(i_args, func, index, verbose)
                for index, i_args in enumerate(jobs)
            )
        else:
            vec_lists = Parallel(n_jobs=n_jobs, prefer=prefer)(
                delayed(create_multiple_without_descriptor)(
                    i_args, func, index, verbose
                )
                for index, i_args in enumerate(jobs)
            )

        # Restore the calculation order. If using the threading backend, the
        # input order may have been lost.
        vec_lists.sort(key=lambda x: x[1])

        # If the results are of the same length, we can simply concatenate them
        # into one numpy array. Otherwise we will return a regular python list.
        der_lists = [x[0][0] for x in vec_lists]
        if derivatives_shape:
            if is_sparse:
                derivatives = sp.concatenate(der_lists, axis=0)
            else:
                derivatives = np.concatenate(der_lists, axis=0)
        else:
            derivatives = []
            for part in der_lists:
                derivatives.extend(part)
        if return_descriptor:
            des_lists = [x[0][1] for x in vec_lists]
            if descriptor_shape:
                if is_sparse:
                    descriptors = sp.concatenate(des_lists, axis=0)
                else:
                    descriptors = np.concatenate(des_lists, axis=0)
            else:
                descriptors = []
                for part in des_lists:
                    descriptors.extend(part)
            return (derivatives, descriptors)

        return derivatives
