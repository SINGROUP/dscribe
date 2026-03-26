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
import sparse
from dscribe.kernels.localsimilaritykernel import LocalSimilarityKernel


class AverageKernel(LocalSimilarityKernel):
    """Used to compute a global similarity of structures based on the average
    similarity of local atomic environments in the structure. More precisely,
    returns the similarity kernel K as:

    .. math::
        K(A, B) = \\frac{1}{N M}\sum_{ij} C_{ij}(A, B)

    where :math:`N` is the number of atoms in structure :math:`A`, :math:`M` is
    the number of atoms in structure :math:`B` and the similarity between local
    atomic environments :math:`C_{ij}` has been calculated with the pairwise
    metric (e.g. linear, gaussian) defined by the parameters given in the
    constructor.
    """

    def create(self, x, y=None):
        if self.metric != "linear":
            return super().create(x, y)
        
        x_mean = np.array([np.mean(a.todense(), axis=0) if isinstance(a, sparse.COO) else np.mean(a, axis=0) for a in x])
        if y is None:
            symmetric = True
            y_mean = x_mean
        else:
            symmetric = False
            y_mean = np.array([np.mean(a.todense(), axis=0) if isinstance(a, sparse.COO) else np.mean(a, axis=0) for a in y])
        
        K_ij = x_mean @ y_mean.T

        if not self.normalize_kernel:
            return K_ij

        if symmetric:
            x_ii_sqrt = np.sqrt(np.diagonal(K_ij))
            y_ii_sqrt = x_ii_sqrt
        else:
            x_ii_sqrt = np.linalg.norm(x_mean, axis=1)
            y_ii_sqrt = np.linalg.norm(y_mean, axis=1)

        K_ij /= np.outer(x_ii_sqrt, y_ii_sqrt)
        return K_ij
    

    def get_global_similarity(self, localkernel):
        """
        Computes the average global similarity between two structures A and B.

        Args:
            localkernel(np.ndarray): NxM matrix of local similarities between
                structures A and B, with N and M atoms respectively.
        Returns:
            float: Average similarity between the structures A and B.
        """
        K_ij = np.mean(localkernel)

        return K_ij
