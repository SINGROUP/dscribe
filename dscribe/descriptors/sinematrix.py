from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range,
                      zip, round, input, int, pow, object)

import numpy as np

from dscribe.descriptors.matrixdescriptor import MatrixDescriptor


class SineMatrix(MatrixDescriptor):
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

    For reference, see:
        "Crystal Structure Representations for Machine Learning Models of
        Formation Energies", Felix Faber, Alexander Lindmaa, Anatole von
        Lilienfeld, and Rickard Armiento, International Journal of Quantum
        Chemistry, (2015),
        https://doi.org/10.1002/qua.24917
    """
    def get_matrix(self, system):
        """Creates the Sine matrix for the given system.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.

        Returns:
            np.ndarray: Sine matrix as a 2D array.
        """
        # Force the use of periodic boundary conditions
        system.set_pbc(True)

        # Cell and inverse cell
        B = system.get_cell()
        try:
            B_inv = system.get_cell_inverse()
        except:
            raise ValueError(
                "The given system has a non-invertible cell matrix: {}.".format(B)
            )

        # Difference vectors as a 3D tensor
        diff_tensor = system.get_displacement_tensor()

        # Calculate phi
        arg_to_sin = np.pi * np.dot(diff_tensor, B_inv)
        phi = np.linalg.norm(np.dot(np.sin(arg_to_sin)**2, B), axis=2)

        with np.errstate(divide='ignore'):
            phi = np.reciprocal(phi)

        # Calculate Z_i*Z_j
        q = system.get_atomic_numbers()
        qiqj = q[None, :]*q[:, None]
        np.fill_diagonal(phi, 0)

        # Multiply by charges
        smat = qiqj*phi

        # Set diagonal
        np.fill_diagonal(smat, 0.5 * q ** 2.4)

        return smat
