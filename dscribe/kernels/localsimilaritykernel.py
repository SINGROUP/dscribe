# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
from abc import ABCMeta, abstractmethod
import numpy as np
from future.utils import with_metaclass
from sklearn.metrics.pairwise import pairwise_kernels


class LocalSimilarityKernel(with_metaclass(ABCMeta)):
    """An abstract base class for all kernels that use the similarity of local
    atomic environments to compute a global similarity measure.
    """
    def __init__(self, metric, gamma=None, degree=3, coef0=1, kernel_params=None):
        """
        """
        self.metric = metric
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

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
        self_sim = False
        if y is None:
            y = x
            self_sim = True

        # First calculate the "raw" pairwise similarity of atomic environments
        n_x = len(x)
        n_y = len(y)

        C_ij_dict = {}
        for i in range(n_x):
            for j in range(n_y):
                if j >= i:
                    x_i = x[i]
                    y_j = y[j]
                    C_ij = self.get_pairwise_matrix(x_i, y_j)
                    C_ij_dict[i, j] = C_ij

        # Calculate the global pairwise similarity between the entire
        # structures
        K_ij = np.zeros((n_x, n_y))
        for i in range(n_x):
            for j in range(n_y):
                if j >= i:
                    C_ij = C_ij_dict[i, j]
                    k_ij = self.get_global_similarity(
                        C_ij,
                    )
                    K_ij[i, j] = k_ij
                    if j != i:
                        K_ij[j, i] = k_ij

        # Normalize the values.
        if self_sim:
            k_ii = np.diagonal(K_ij)
            k_ii_sqrt = np.sqrt(k_ii)
        else:
            pass
        K_ij /= np.outer(k_ii_sqrt, k_ii_sqrt)

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
