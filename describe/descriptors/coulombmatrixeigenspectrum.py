from __future__ import absolute_import, division, print_function
from builtins import super

import numpy as np

from describe.descriptors import CoulombMatrix


class CoulombMatrixEigenSpectrum(CoulombMatrix):
    """Calculates the zero-padded Eigenspectrum of a Coulomb matrix.

    The Coulomb matrix is defined as:

        C_ij = 0.5 Zi**exponent      | i = j
             = (Zi*Zj)/(Ri-Rj)	     | i != j

    The eigenspectrum is calculated as the list of eigenvalues of this matrix
    sorted in descending order by their absolute value and zero-padded at the
    end.

    For reference, see:
        "Fast and Accurate Modeling of Molecular Atomization Energies with
        Machine Learning", Matthias Rupp, Alexandre Tkatchenko, Klaus-Robert
        Müller, and O.  Anatole von Lilienfeld, Phys. Rev. Lett, (2012),
        https://doi.org/10.1103/PhysRevLett.108.058301
    """
    def __init__(self, n_atoms_max):
        """
        Args:
            n_atoms_max (int): The maximum nuber of atoms that any of the
                samples can have. This controls how much zeros need to be
                padded to the final result.
        """
        super().__init__(False)
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

        # Calculate eigenvalues
        eigenvalues, _ = np.linalg.eig(cmat)

        # Remove sign
        abs_values = np.absolute(eigenvalues)

        # Get ordering that sorts the values by absolute value
        sorted_indices = np.argsort(abs_values)[::-1]  # This sorts the list in descending order in place
        eigenvalues = eigenvalues[sorted_indices]

        # Add zero-pading
        n_atoms = len(system)
        eigenvalues = np.pad(eigenvalues, (0, self.n_atoms_max-n_atoms), 'constant')

        return eigenvalues

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        return int(self.n_atoms_max)
