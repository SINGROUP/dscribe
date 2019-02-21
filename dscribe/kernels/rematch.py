# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
import numpy as np
from dscribe.kernels.localsimilaritykernel import LocalSimilarityKernel


class REMatchKernel(LocalSimilarityKernel):
    """Used to compute a global similarity of structures based on the
    regularized-entropy match (REMatch) kernel of local atomic environments in
    the structure. More precisely, returns the similarity kernel K as:

    .. math::
        \DeclareMathOperator*{\\argmax}{argmax}
        K(A, B) &= \mathrm{Tr} \mathbf{P}^\\alpha \mathbf{C}(A, B)

        \mathbf{P}^\\alpha &= \\argmax_{\mathbf{P} \in \mathcal{U}(N, N)} \sum_{ij} P_{ij} (1-C_{ij} +\\alpha \ln P_{ij})

    where the similarity between local atomic environments :math:`C_{ij}` has
    been calculated with the pairwise metric (e.g. linear, gaussian) defined by
    the parameters given in the constructor.

    For reference, see:

    "Comparing molecules and solids across structural and alchemical
    space", Sandip De, Albert P. Bartók, Gábor Csányi and Michele Ceriotti,
    Phys.  Chem. Chem. Phys. 18, 13754 (2016),
    https://doi.org/10.1039/c6cp00415f
    """
    def __init__(self, alpha=0.1, threshold=1e-6, metric="linear", gamma=None, degree=3, coef0=1, kernel_params=None):
        """
        Args:
            alpha(float): Parameter controlling the entropic penalty. Values
                close to zero approach the best-match solution and values
                towards infinity approach the average kernel.
            threshold(float): Convergence threshold used in the
                Sinkhorn-algorithm.
            kernel(string or callable): Kernel mapping used internally. A
                callable should accept two arguments and the keyword arguments
                passed to this object as kernel_params, and should return a
                floating point number.
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
        """
        self.alpha = alpha
        self.threshold = threshold
        super().__init__(metric, gamma, degree, coef0, kernel_params)

    def get_global_similarity(self, localkernel):
        """
        Computes the REMatch similarity between two structures A and B.

        Args:
            localkernel(np.ndarray): NxM matrix of local similarities between
                structures A and B, with N and M atoms respectively.
        Returns:
            float: REMatch similarity between the structures A and B.
        """
        n, m = localkernel.shape
        K = np.exp(-(1 - localkernel) / self.alpha)

        # initialisation
        u = np.ones((n,)) / n
        v = np.ones((m,)) / m

        en = np.ones((n,)) / float(n)
        em = np.ones((m,)) / float(m)

        # converge balancing vectors u and v
        itercount = 0
        error = 1
        while (error > self.threshold):
            uprev = u
            vprev = v
            v = np.divide(em, np.dot(K.T, u))
            u = np.divide(en, np.dot(K, v))

            # determine error every now and then
            if itercount % 5:
                error = np.sum((u - uprev) ** 2) / np.sum((u) ** 2) + np.sum((v - vprev) ** 2) / np.sum((v) ** 2)
            itercount += 1

        # using Tr(X.T Y) = Sum[ij](Xij * Yij)
        # P.T * C
        # P_ij = u_i * v_j * K_ij
        pity = np.multiply( np.multiply(K, u.reshape((-1, 1))), v)

        glosim = np.sum( np.multiply( pity, localkernel))

        return glosim
