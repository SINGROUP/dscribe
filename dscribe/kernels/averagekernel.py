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
