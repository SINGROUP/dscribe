import numpy as np
from describe.descriptors import Descriptor


class SineMatrix(Descriptor):
    """Calculates the zero padded Sine matrix for different systems.

    The Sine matrix is defined as:

        Cij = 0.5 Zi**exponent      | i = j
            = (Zi*Zj)/phi(Ri, Rj)   | i != j

        where phi(r1, r2) = | B * sum(k = x,y,z)[ek * sin^2(pi * ek * B^-1
        (r2-r1))] | (B is the matrix of basis cell vectors, ek are the unit
        vectors)

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
            ndarray: The zero padded Sine matrix either as a 2D array or as
                a 1D array depending on the setting self.flatten.
        """
        smat = self.sine_matrix(system)
        smat = self.zero_pad(smat)
        if self.flatten:
            smat = smat.flatten()
        return smat

    def sine_matrix(self, system):
        """Creates the Sine matrix for the given system.
        """
        # Cell and inverse cell
        B = system.get_cell()
        B_inv = system.get_cell_inverse()

        # Difference vectors in tensor 3D-tensor-form
        diff_tensor = system.get_displacement_tensor()

        # Calculate phi
        arg_to_sin = np.pi * np.dot(diff_tensor, B_inv)
        phi = np.linalg.norm(np.dot(np.sin(arg_to_sin)**2, B), axis=2)

        with np.errstate(divide='ignore'):
            phi = np.reciprocal(phi)

        # Calculate Z_i*Z_j
        q = system.get_initial_charges()
        qiqj = q[None, :]*q[:, None]
        np.fill_diagonal(phi, 0)

        # Multiply by charges
        smat = qiqj*phi

        # Set diagonal
        np.fill_diagonal(smat, 0.5 * q ** 2.4)

        return smat

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
