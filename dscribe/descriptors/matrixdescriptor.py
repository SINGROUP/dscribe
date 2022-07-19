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
from numpy.random import RandomState

import sparse

from dscribe.descriptors import Descriptor
from abc import abstractmethod


class MatrixDescriptor(Descriptor):
    """A common base class for two-body matrix-like descriptors."""

    def __init__(
        self,
        n_atoms_max,
        permutation="sorted_l2",
        sigma=None,
        seed=None,
        flatten=True,
        sparse=False,
    ):
        """
        Args:
            n_atoms_max (int): The maximum nuber of atoms that any of the
                samples can have. This controls how much zeros need to be
                padded to the final result.
            permutation (string): Defines the method for handling permutational
                invariance. Can be one of the following:
                    - none: The matrix is returned in the order defined by the
                      Atoms.
                    - sorted_l2: The rows and columns are sorted by the L2 norm.
                    - eigenspectrum: Only the eigenvalues are returned sorted
                      by their absolute value in descending order.
                    - random: The rows and columns are sorted by their L2 norm
                      after applying Gaussian noise to the norms. The standard
                      deviation of the noise is determined by the
                      sigma-parameter.
            sigma (float): Provide only when using the *random*-permutation
                option. Standard deviation of the gaussian distributed noise
                determining how much the rows and columns of the randomly
                sorted matrix are scrambled.
            seed (int): Provide only when using the *random*-permutation
                option. A seed to use for drawing samples from a normal
                distribution.
            flatten (bool): Whether the output of create() should be flattened
                to a 1D array.
            sparse (bool): Whether the output should be a sparse matrix or a
                dense numpy array.
        """
        super().__init__(periodic=False, flatten=flatten, sparse=sparse)

        # Check parameter validity
        if n_atoms_max <= 0:
            raise ValueError("The maximum number of atoms must be a positive number.")
        perm_options = set(
            ("sorted_l2", "none", "eigenspectrum", "eigenspectrum", "random")
        )
        if permutation not in perm_options:
            raise ValueError(
                "Unknown permutation option given. Please use one of the "
                "following: {}.".format(", ".join(perm_options))
            )

        if not sigma and permutation == "random":
            raise ValueError("Please specify sigma as a degree of random noise.")

        # Raise a value error if sigma specified, but random sorting not used
        if permutation != "random" and sigma is not None:
            raise ValueError(
                "Sigma value specified but the parameter 'permutation' not set "
                "as 'random'."
            )

        self.seed = seed
        self.random_state = RandomState(seed)
        self.n_atoms_max = n_atoms_max
        self.permutation = permutation
        self._norm_vector = None
        self.sigma = sigma

    def get_matrix(self, system):
        """Used to get the final matrix for this descriptor.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.

        Returns:
            np.ndarray: The final two-dimensional matrix for this descriptor.
        """

    def create_single(self, system):
        """
        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.

        Returns:
            ndarray: The zero padded matrix either as a 2D array or as
                a 1D array depending on the setting self._flatten.
        """
        # Remove the old norm vector for the new system
        self._norm_vector = None

        matrix = self.get_matrix(system)

        # Handle the permutation option
        if self.permutation == "none":
            pass
        elif self.permutation == "sorted_l2":
            matrix = self.sort(matrix)
        elif self.permutation == "eigenspectrum":
            matrix = self.get_eigenspectrum(matrix)
        elif self.permutation == "random":
            matrix = self.sort_randomly(matrix, self.sigma)

        # Add zero padding
        matrix = self.zero_pad(matrix)

        # Flatten
        if self.permutation == "eigenspectrum" or self._flatten:
            matrix = np.reshape(matrix, (matrix.size,))

        # If a sparse matrix is requested, convert to coo_matrix
        if self._sparse:
            matrix = sparse.COO.from_numpy(matrix)

        return matrix

    def sort(self, matrix):
        """Sorts the given matrix by using the L2 norm.

        Args:
            matrix(np.ndarray): The matrix to sort.

        Returns:
            np.ndarray: The sorted matrix.
        """
        # Sort the atoms such that the norms of the rows are in descending
        # order
        norms = np.linalg.norm(matrix, axis=1)
        sorted_indices = np.argsort(-norms, kind="stable", axis=0)
        sorted_matrix = matrix[sorted_indices]
        sorted_matrix = sorted_matrix[:, sorted_indices]

        return sorted_matrix

    def get_eigenspectrum(self, matrix):
        """Calculates the eigenvalues of the matrix and returns a list of them
        sorted by their descending absolute value.

        Args:
            matrix(np.ndarray): The matrix to sort.

        Returns:
            np.ndarray: A list of eigenvalues sorted by absolute value.
        """
        # Calculate eigenvalues
        eigenvalues, _ = np.linalg.eig(matrix)

        # Remove sign
        abs_values = np.absolute(eigenvalues)

        # Get ordering that sorts the values in descending order by absolute
        # value
        sorted_indices = np.argsort(abs_values)[::-1]
        eigenvalues = eigenvalues[sorted_indices]

        return eigenvalues

    def zero_pad(self, array):
        """Zero-pads the given matrix.

        Args:
            array (np.ndarray): The array to pad

        Returns:
            np.ndarray: The zero-padded array.
        """
        # Pad with zeros
        n_atoms = array.shape[0]
        n_dim = array.ndim
        padded = np.pad(array, [(0, self.n_atoms_max - n_atoms)] * n_dim, "constant")

        return padded

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        if self.permutation == "eigenspectrum":
            return int(self.n_atoms_max)
        else:
            return int(self.n_atoms_max**2)

    def sort_randomly(self, matrix, sigma):
        """
        Given a coulomb matrix, it adds random noise to the sorting defined by
        sigma. For sorting, L2-norm is used.

        Args:
            matrix(np.ndarray): The matrix to randomly sort.

        sigma:
            float: Width of gaussian distributed noise determining how much the
                rows and columns of the randomly sorted coulomb matrix are
                scrambled.

        Returns:
            np.ndarray: The randomly sorted matrix.
        """
        norm_vector = self._get_norm_vector(matrix)
        noise_norm_vector = self.random_state.normal(norm_vector, sigma)
        indexlist = np.argsort(noise_norm_vector)
        indexlist = indexlist[::-1]  # Order highest to lowest

        matrix = matrix[indexlist][:, indexlist]

        return matrix

    def _get_norm_vector(self, matrix):
        """
        Takes a coulomb matrix as input. Returns L2 norm of each row / column in a 1D-array.
        Args:
            matrix(np.ndarray): The matrix to sort.

        Returns:
            np.ndarray: L2 norm of each row / column.

        """
        if self._norm_vector is None:
            self._norm_vector = np.linalg.norm(matrix, axis=1)
        return self._norm_vector
