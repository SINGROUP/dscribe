from __future__ import absolute_import, division, print_function
from builtins import super

import numpy as np

from describe.descriptors.matrixdescriptor import MatrixDescriptor


class CoulombMatrix(MatrixDescriptor):
    """Calculates the zero padded Coulomb matrix.

    The Coulomb matrix is defined as:

        C_ij = 0.5 Zi**exponent      | i = j
             = (Zi*Zj)/(Ri-Rj)	     | i != j

    The matrix is padded with invisible atoms, which means that the matrix is
    padded with zeros until the maximum allowed size defined by n_max_atoms is
    reached.

    To reach invariance against permutation of atoms, specify a valid option
    for the permutation parameter.

    For reference, see:
        "Fast and Accurate Modeling of Molecular Atomization Energies with
        Machine Learning", Matthias Rupp, Alexandre Tkatchenko, Klaus-Robert
        Müller, and O.  Anatole von Lilienfeld, Phys. Rev. Lett, (2012),
        https://doi.org/10.1103/PhysRevLett.108.058301
    and
        "Learning Invariant Representations of Molecules for Atomization Energy
        Prediction", Grégoire Montavon et. al, Advances in Neural Information
        Processing Systems 25 (NIPS 2012)
    """
    def __init__(self, n_atoms_max, permutation="sorted_l2", flatten=True):
        """
        Args:
            n_atoms_max (int): The maximum nuber of atoms that any of the
                samples can have. This controls how much zeros need to be
                padded to the final result.
            permutation (string): Defines the method for handling permutational
                invariance. Can be one of the following:
                    - none: The matrix is returned in the order defined by the Atoms.
                    - sorted_l2: The rows and columns are sorted by the L2 norm.
                    - eigenspectrum: Only the eigenvalues are returned sorted
                      by their absolute value in descending order.
                    - random: ?
            flatten (bool): Whether the output of create() should be flattened
                to a 1D array.
        """
        super().__init__(n_atoms_max, permutation, flatten)

    def describe(self, system):
        """
        Args:
            system (System): Input system.

        Returns:
            ndarray: The zero padded Coulomb matrix either as a 2D array or as
                a 1D array depending on the setting self.flatten.
        """
        cmat = self.coulomb_matrix(system)

        # Handle the permutation option
        if self.permutation == "none":
            pass
        elif self.permutation == "sorted_l2":
            cmat = self.sort(cmat)
        elif self.permutation == "eigenspectrum":
            cmat = self.get_eigenspectrum(cmat)
        else:
            raise ValueError("Invalid permutation method: {}".format(self.permutation))

        # Add zero padding
        cmat = self.zero_pad(cmat)

        # Flatten the matrix if requested
        if self.flatten:
            cmat = cmat.flatten()

        return cmat

    def coulomb_matrix(self, system):
        """Creates the Coulomb matrix for the given system.
        """
        # Calculate offdiagonals
        q = system.get_atomic_numbers()
        qiqj = q[None, :]*q[:, None]
        idmat = system.get_inverse_distance_matrix()
        np.fill_diagonal(idmat, 0)
        cmat = qiqj*idmat

        # Set diagonal
        np.fill_diagonal(cmat, 0.5 * q ** 2.4)

        return cmat
