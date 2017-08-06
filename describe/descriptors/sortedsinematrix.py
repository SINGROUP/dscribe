import numpy as np
from describe.descriptors import SineMatrix


class SortedSineMatrix(SineMatrix):
    """Calculates the zero padded Sine matrix for different systems.

    The Sine matrix is defined as:

        Cij = 0.5 Zi**exponent      | i = j
            = (Zi*Zj)/phi(Ri, Rj)   | i != j

    where phi(r1, r2) = | B * sum(k = x,y,z)[ek * sin^2(pi * ek * B^-1
    (r2-r1))] | (B is the matrix of basis cell vectors, ek are the unit
    vectors)

    The rows of this matrix are further sorted by their euclidean norm, such
    that the matrix C satisfies ||C_i|| >= ||C_(i+1)|| \forall i, where C_i
    denotes the ith row of the Coulomb matrix.

    The matrix is also padded with invisible atoms, which means that the matrix is
    padded with zeros until the maximum allowed size defined by n_max_atoms is
    reached.
    """
    def sine_matrix(self, system):
        """Creates the Sine matrix for the given system.
        """
        smat = super().sine_matrix(system)

        # Sort the atoms such that the norms of the rows are in descending
        # order
        norms = np.linalg.norm(smat, axis=1)
        sorted_indices = np.argsort(norms, axis=0)[::-1]
        smat = smat[sorted_indices]
        smat = smat[:, sorted_indices]

        return smat
