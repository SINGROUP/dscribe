from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)
import math

import numpy as np

from scipy.special import erfc

from dscribe.descriptors.matrixdescriptor import MatrixDescriptor
from dscribe.core.lattice import Lattice


class EwaldMatrix(MatrixDescriptor):
    """
    Calculates an Ewald matrix for the a given system.

    Each entry M_ij of the Ewald matrix will contain the Coulomb energy between
    atoms i and j calculated with the Ewald summation method. In the Ewald
    method a constant neutralizing background charge has been added to
    counteract the positive net charge.

    The total electrostatic interaction energy in the system can calculated by
    summing the upper diagonal part of the matrix, including the diagonal
    itself.

    A screening parameter a controls the width of the Gaussian charge
    distributions in the Ewald summation, but the final matrix elements will be
    independent of the value of the screening parameter a that is used, as long
    as sufficient cutoff values are used.

    This implementation provides default values for

    For reference, see:
        "Crystal Structure Representations for Machine Learning Models of
        Formation Energies", Felix Faber, Alexander Lindmaa, Anatole von
        Lilienfeld, and Rickard Armiento, International Journal of Quantum
        Chemistry, (2015),
        https://doi.org/10.1002/qua.24917
    and
        "Ewald summation techniques in perspective: a survey", Abdulnour Y.
        Toukmaji, John A. Board Jr., Computer Physics Communications, (1996)
        https://doi.org/10.1016/0010-4655(96)00016-1
    and
        "R.A. Jackson and C.R.A. Catlow. Computer simulation studies of zeolite
        structure. Mol. Simul., 1:207-224, 1988,
        https://doi.org/10.1080/08927022.2013.840898
        "
    """
    def create(self, system, accuracy=1e-5, w=1, rcut=None, gcut=None, a=None):
        """
        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.
            accuracy (float): The accuracy to which the sum is converged to.
                Corresponds to the variable :math`A` in
                https://doi.org/10.1080/08927022.2013.840898. Used only if gcut,
                rcut and a have not been specified.
            w (float): Weight parameter that represents the relative
                computational expense of calculating a term in real and
                reciprocal space. This has little effect on the total energy,
                but may influence speed of computation in large systems. Note
                that this parameter is used only when the cutoffs and a are set
                to None.
            rcut (float): Real space cutoff radius dictating how
                many terms are used in the real space sum.
            gcut (float): Reciprocal space cutoff radius.
            a (float): The screening parameter that controls the width of the
                Gaussians. Corresponds to the standard deviation of the Gaussians
        """
        self.q = system.get_atomic_numbers()
        self.q_squared = self.q**2
        self.n_atoms = len(system)
        self.volume = system.get_volume()
        self.sqrt_pi = math.sqrt(np.pi)

        # If a is not provided, use a default value
        if a is None:
            a = (self.n_atoms * w / (self.volume ** 2)) ** (1 / 6) * self.sqrt_pi

        # If the real space cutoff, reciprocal space cutoff and a have not been
        # specified, use the accuracy and the weighting w to determine default
        # similarly as in https://doi.org/10.1080/08927022.2013.840898
        if rcut is None and gcut is None:
            f = np.sqrt(-np.log(accuracy))
            rcut = f / a
            gcut = 2 * a * f
        elif rcut is None or gcut is None:
            raise ValueError(
                "If you do not want to use the default cutoffs, please provide "
                "both cutoffs rcut and gcut."
            )

        self.a = a
        self.a_squared = self.a**2
        self.gcut = gcut
        self.rcut = rcut

        return super().create(system)

    def get_matrix(self, system):
        """
        The total energy matrix. Each matrix element (i, j) corresponds to the
        total interaction energy in a system with atoms i and j.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.

        Returns:
            np.ndarray: Ewald matrix.
        """
        # Force the use of periodic boundary conditions
        system.set_pbc(True)

        # Calculate the regular real and reciprocal space sums of the Ewald sum.
        ereal = self._calc_real(system)
        erecip = self._calc_recip(system)
        ezero = self._calc_zero()
        total = erecip + ereal + ezero

        return total

    def _calc_zero(self):
        """Calculates the constant part of the Ewald matrix.

        The constant part contains the correction for the self-interaction
        between the point charges and the Gaussian charge distribution added on
        top of them and the intearction between the point charges and a uniform
        neutralizing background charge.

        Returns:
            np.ndarray(): A 2D matrix containing the constant terms for each
            i,j pair.
        """
        # Calculate the self-interaction correction. The self term corresponds
        # to the interaction of the point charge with cocentric Gaussian cloud
        # introduced in the Ewald method. The correction is only applied to the
        # diagonal terms so that the correction is not counted multiple times
        # when calculating the total Ewald energy as the sum of diagonal
        # element + upper diagonal part.
        q = self.q
        matself = np.zeros((self.n_atoms, self.n_atoms))
        diag = q**2
        np.fill_diagonal(matself, diag)
        matself *= -self.a/self.sqrt_pi

        # Calculate the interaction energy between constant neutralizing
        # background charge. On the diagonal this is defined by
        matbg = 2*q[None, :]*q[:, None].astype(float)
        matbg *= -np.pi/(2*self.volume*self.a_squared)

        # The diagonal terms are divided by two
        diag = np.diag(matbg)/2
        np.fill_diagonal(matbg, diag)

        correction_matrix = matself + matbg

        return correction_matrix

    def _calc_real(self, system):
        """Used to calculate the Ewald real-space sum.

        Corresponds to equation (5) in
        https://doi.org/10.1016/0010-4655(96)00016-1

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.

        Returns:
            np.ndarray(): A 2D matrix containing the real space terms for each
            i,j pair.
        """
        fcoords = system.get_scaled_positions()
        coords = system.get_positions()
        n_atoms = len(system)
        ereal = np.zeros((n_atoms, n_atoms), dtype=np.float)
        lattice = Lattice(system.get_cell())

        # For each atom in the original cell, get the neighbours in the
        # infinite system within the real space cutoff and calculate the real
        # space portion of the Ewald sum.
        for i in range(n_atoms):

            # Get points that are within the real space cutoff
            nfcoords, rij, js = lattice.get_points_in_sphere(
                fcoords,
                coords[i],
                self.rcut,
                zip_results=False
            )
            # Remove the rii term, because a charge does not interact with
            # itself (but does interact with copies of itself).
            mask = rij > 1e-8
            js = js[mask]
            rij = rij[mask]
            nfcoords = nfcoords[mask]

            qi = self.q[i]
            qj = self.q[js]

            erfcval = erfc(self.a * rij)
            new_ereals = erfcval * qi * qj / rij

            # Insert new_ereals
            for k in range(n_atoms):
                ereal[k, i] = np.sum(new_ereals[js == k])

        # The diagonal terms are divided by two
        diag = np.diag(ereal)/2
        np.fill_diagonal(ereal, diag)

        return ereal

    def _calc_recip(self, system):
        """
        Perform the reciprocal space summation. Uses the fastest non mesh-based
        method described as given by equation (16) in
        https://doi.org/10.1016/0010-4655(96)00016-1

        The term G=0 is neglected, even if the system has nonzero charge.
        Physically this would mean that we are adding a constant background
        charge to make the cell charge neutral.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.

        Returns:
            np.ndarray(): A 2D matrix containing the real space terms for each
            i,j pair.
        """
        n_atoms = self.n_atoms
        erecip = np.zeros((n_atoms, n_atoms), dtype=np.float)
        coords = system.get_positions()

        # Get the reciprocal lattice points within the reciprocal space cutoff
        rcp_latt = 2*np.pi*system.get_reciprocal_cell()
        rcp_latt = Lattice(rcp_latt)
        recip_nn = rcp_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0],
                                                 self.gcut)

        # Ignore the terms with G=0.
        frac_coords = [fcoords for (fcoords, dist, i) in recip_nn if dist != 0]

        gs = rcp_latt.get_cartesian_coords(frac_coords)
        g2s = np.sum(gs ** 2, 1)
        expvals = np.exp(-g2s / (4 * self.a_squared))
        grs = np.sum(gs[:, None] * coords[None, :], 2)
        factors = np.divide(expvals, g2s)
        charges = self.q

        # Create array where q_2[i,j] is qi * qj
        qiqj = charges[None, :] * charges[:, None]

        for gr, factor in zip(grs, factors):

            # Uses the identity sin(x)+cos(x) = 2**0.5 sin(x + pi/4)
            m = (gr[None, :] + math.pi / 4) - gr[:, None]
            np.sin(m, m)
            m *= factor
            erecip += m

        erecip *= 4 * math.pi / self.volume * qiqj * 2 ** 0.5

        # The diagonal terms are divided by two
        diag = np.diag(erecip)/2
        np.fill_diagonal(erecip, diag)

        return erecip
