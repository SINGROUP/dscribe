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
from sklearn.metrics.pairwise import pairwise_kernels


class LocalSimilarityKernel(ABC):
    """An abstract base class for all kernels that use the similarity of local
    atomic environments to compute a global similarity measure.
    """
    def __init__(self, metric, gamma=None, degree=3, coef0=1, kernel_params=None, normalize_kernel=True):
        """
        Args:
            metric(string or callable): The pairwise metric used for
                calculating the local similarity. Accepts any of the sklearn
                pairwise metric strings (e.g. "linear", "rbf", "laplacian",
                "polynomial") or a custom callable. A callable should accept
                two arguments and the keyword arguments passed to this object
                as kernel_params, and should return a floating point number.
            gamma(float): Gamma parameter for the RBF, laplacian, polynomial,
                exponential chi2 and sigmoid kernels. Interpretation of the
                default value is left to the kernel; see the documentation for
                sklearn.metrics.pairwise. Ignored by other kernels.
            degree(float): Degree of the polynomial kernel. Ignored by other
                kernels.
            coef0(float): Zero coefficient for polynomial and sigmoid kernels.
                Ignored by other kernels.
            kernel_params(mapping of string to any): Additional parameters
                (keyword arguments) for kernel function passed as callable
                object.
            normalize_kernel(boolean): Whether to normalize the final global
                similarity kernel. The normalization is achieved by dividing each
                kernel element :math:`K_{ij}` with the factor
                :math:`\sqrt{K_{ii}K_{jj}}`
        """
        self.metric = metric
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.normalize_kernel = normalize_kernel

    def create(self, x, y=None):
        """Creates the kernel matrix based on the given lists of local
        features x and y.

        Args:
            x(iterable): A list of local feature arrays for each structure.
            y(iterable): An optional second list of features. If not specified
                it is assumed that y=x.

        Returns:
            The pairwise global similarity kernel K[i,j] between the given
            structures, in the same order as given in the input, i.e. the
            similarity of structures i and j is given by K[i,j], where features
            for structure i and j were in features[i] and features[j]
            respectively.
        """
        symmetric = False
        if y is None:
            y = x
            symmetric = True

        # First calculate the "raw" pairwise similarity of atomic environments
        n_x = len(x)
        n_y = len(y)

        C_ij_dict = {}
        for i in range(n_x):
            for j in range(n_y):

                # Skip lower triangular part for symmetric matrices
                if symmetric and j < i:
                    continue

                x_i = x[i]

                # Save time on symmetry
                if symmetric and j == i:
                    y_j = None
                else:
                    y_j = y[j]
                C_ij = self.get_pairwise_matrix(x_i, y_j)
                C_ij_dict[i, j] = C_ij

        # Calculate the global pairwise similarity between the entire
        # structures
        K_ij = np.zeros((n_x, n_y))
        for i in range(n_x):
            for j in range(n_y):

                # Skip lower triangular part for symmetric matrices
                if symmetric and j < i:
                    continue

                C_ij = C_ij_dict[i, j]
                k_ij = self.get_global_similarity(C_ij)
                K_ij[i, j] = k_ij

                # Save data also on lower triangular part for symmetric matrices
                if symmetric and j != i:
                    K_ij[j, i] = k_ij

        # Enforce kernel normalization if requested.
        if self.normalize_kernel:
            if symmetric:
                k_ii = np.diagonal(K_ij)
                x_k_ii_sqrt = np.sqrt(k_ii)
                y_k_ii_sqrt = x_k_ii_sqrt
            else:
                # Calculate self-similarity for X
                x_k_ii = np.empty(n_x)
                for i in range(n_x):
                    x_i = x[i]
                    C_ii = self.get_pairwise_matrix(x_i)
                    k_ii = self.get_global_similarity(C_ii)
                    x_k_ii[i] = k_ii
                x_k_ii_sqrt = np.sqrt(x_k_ii)

                # Calculate self-similarity for Y
                y_k_ii = np.empty(n_y)
                for i in range(n_y):
                    y_i = y[i]
                    C_ii = self.get_pairwise_matrix(y_i)
                    k_ii = self.get_global_similarity(C_ii)
                    y_k_ii[i] = k_ii
                y_k_ii_sqrt = np.sqrt(y_k_ii)

            K_ij /= np.outer(x_k_ii_sqrt, y_k_ii_sqrt)

        return K_ij

    def get_pairwise_matrix(self, X, Y=None):
        """Calculates the pairwise similarity of atomic environments with
        scikit-learn, and the pairwise metric configured in the constructor.

        Args:
            X(np.ndarray): Feature vector for the atoms in structure A
            Y(np.ndarray): Feature vector for the atoms in structure B

        Returns:
            np.ndarray: NxM matrix of local similarities between structures A
                and B, with N and M atoms respectively.

        """
        if callable(self.metric):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.metric,
                                filter_params=True, **params)

    @abstractmethod
    def get_global_similarity(self, localkernel):
        """
        Computes the global similarity between two structures A and B.

        Args:
            localkernel(np.ndarray): NxM matrix of local similarities between
                structures A and B, with N and M atoms respectively.
        Returns:
            float: Global similarity between the structures A and B.
        """
