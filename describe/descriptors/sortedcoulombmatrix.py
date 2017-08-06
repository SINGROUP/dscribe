import numpy as np
from describe.descriptors import CoulombMatrix


class SortedCoulombMatrix(CoulombMatrix):
    """Calculates the sorted and zero padded Coulomb matrix for different
    systems.

    The Coulomb matrix is defined as:

        C_ij = 0.5 Zi**exponent      | i = j
             = (Zi*Zj)/(Ri-Rj)	     | i != j

    The rows of this matrix are further sorted by their euclidean norm, such
    that the matrix C satisfies ||C_i|| >= ||C_(i+1)|| \forall i, where C_i
    denotes the ith row of the Coulomb matrix.

    The matrix is also padded with invisible atoms, which means that the matrix
    is padded with zeros until the maximum allowed size defined by n_max_atoms
    is reached.
    """

    def coulomb_matrix(self, system):
        """Creates the Coulomb matrix for the given system.
        """
        cmat = super().coulomb_matrix(system)

        # Sort the atoms such that the norms of the rows are in descending
        # order
        norms = np.linalg.norm(cmat, axis=1)
        sorted_indices = np.argsort(norms, axis=0)[::-1]
        cmat = cmat[sorted_indices]
        cmat = cmat[:, sorted_indices]

        return cmat
