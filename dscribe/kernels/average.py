# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
import numpy as np
from dscribe.kernels.localsimilaritykernel import LocalSimilarityKernel


class AverageKernel(LocalSimilarityKernel):
    """Used to compute a global similarity of structures based on the average
    similarity of local atomic environments in the structure. More precisely,
    returns the similarity kernel K as:

    .. math::
        K(A, B) = \\frac{1}{2}\sum_{ij} C_{ij}(A, B)

    where the similarity between local atomic environments :math:`C_{ij}` has
    been calculated with the pairwise metric (e.g. linear, gaussian) defined by
    the parameters given in the constructor.
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
