import numpy as np
from describe.descriptors import Descriptor


class CoulombMatrix(Descriptor):
    """Calculates the zero padded Coulomb matrix for different systems.

    The Coulomb matrix is defined as:

        C_ij = 0.5 Zi**exponent      | i = j
             = (Zi*Zj)/(Ri-Rj)	     | i != j

    The matrix is padded with invisible atoms, which means that the matrix is
    padded with zeros until the maximum allowed size defined by n_max_atoms is
    reached.
    """
    def __init__(self, n_atoms_max, flatten=True):
        """
        Args:
            n_max_atoms (int): The maximum nuber of atoms that any of the
                samples can have. This controls how much zeros need to be
                padded to the final result.
            flatten (bool): Whether the output of create() should be flattened
                to a 1D array.
        """
        super().__init__(flatten)
        self.n_atoms_max = n_atoms_max

    def describe(self, system):
        """
        Args:
            system (System): Input system.

        Returns:
            ndarray: The zero padded Coulomb matrix either as a 2D array or as
                a 1D array depending on the setting self.flatten.
        """
        cmat = self.coulomb_matrix(system)
        cmat = self.zero_pad(cmat)
        if self.flatten:
            cmat = cmat.flatten()
        return cmat

    def coulomb_matrix(self, system):
        """Creates the Coulomb matrix for the given system.
        """
        # Calculate offdiagonals
        q = system.get_initial_charges()
        qiqj = q[None, :]*q[:, None]
        idmat = system.get_inverse_distance_matrix()
        np.fill_diagonal(idmat, 0)
        cmat = qiqj*idmat

        # Set diagonal
        np.fill_diagonal(cmat, 0.5 * q ** 2.4)

        return cmat

    def zero_pad(self, cmat):
        # Pad with zeros
        zeros = np.zeros((self.n_atoms_max, self.n_atoms_max))
        zeros[:cmat.shape[0], :cmat.shape[1]] = cmat
        cmat = zeros

        return cmat

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        return int(self.n_atoms_max**2)
