from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range,
                      zip, round, input, int, pow, object)
import math
import numpy as np
from scipy.special import erfc
from describe.descriptors.matrixdescriptor import MatrixDescriptor
from describe.core.lattice import Lattice
from describe import System
from ase import Atoms


class EwaldMatrix(MatrixDescriptor):
    """
    Calculates an 'Ewald matrix' for the a given system.

    Each entry x_ij of the Ewald matrix will contain the full Coulomb energy
    for a subsystem consisting of the atoms i and j in the unit cell (or just i
    on the diagonal) after a constant background charge has been added to
    counteract any possible net charge in that particular subsystem.

    The final matrix elements will not be dependent on the value of the
    screening parameter a that is used.

    The regular Ewald summation energy cannot be properly divided into parts
    for each ij pair of atoms in the unit cell, because the terms of the
    reciprocal and real space components depend on the value of the screening
    parameter a that is used. This dependency is countered with the scalar
    self-terms and the possible charge term, which make the sum a constant, but
    not the .

    For reference, see:
        "Crystal Structure Representations for Machine Learning Models of
        Formation Energies", Felix Faber, Alexander Lindmaa, Anatole von
        Lilienfeld, and Rickard Armiento, International Journal of Quantum
        Chemistry, (2015),
        https://doi.org/10.1002/qua.24917
    """
    def create(self, system, rcut, gcut, a=None):
        """
        Args:
            system (System): Input system.
            rcut (float): Real space cutoff radius dictating how
                many terms are used in the real space sum.
            gcut (float): Reciprocal space cutoff radius.
            a (float): The screening parameter that controls the width of the
                Gaussians. If not specified the default value of
        """
        # Ensure that we get a System
        if isinstance(system, Atoms):
            system = System.from_atoms(system)

        self.q = system.get_atomic_numbers()
        self.q_squared = self.q**2
        self.n_atoms = len(system)
        self.volume = system.get_volume()
        self.sqrt_pi = math.sqrt(np.pi)

        # If a is not specified, we provide a default. Notice that in
        # https://doi.org/10.1002/qua.24917 there is a mistake as the volume
        # should be squared.
        if a is None:
            a = self.sqrt_pi*np.power(0.01*self.n_atoms/self.volume**2, 1/6)

        self.a = a
        self.a_squared = self.a**2
        self.gcut = gcut
        self.rcut = rcut

        return super().describe(system)

    def get_matrix(self, system):
        """
        The total energy matrix. Each matrix element (i, j) corresponds to the
        total interaction energy in a system with atoms i and j.
        """
        # Calculate the regular real and reciprocal space sums of the Ewald sum.
        ereal = self._calc_real(system)
        erecip = self._calc_recip(system)
        total = erecip + ereal

        # Calculate the modification that makes each entry of the matrix to be
        # the full Ewald sum of the ij subsystem.
        total = self._calc_subsystem_energies(total)

        return total

    def _calc_self_term(self):
        """Calculate the self-term (constant term) of the Ewald sum.

        This term arises from the interaction between a point charge and the
        gaussian charge density that is centered on it.
        """
        values = -self.a/self.sqrt_pi*self.q_squared
        eself = np.sum(values)
        return eself

    def _calc_charge_correction(self):
        """Calculate the charge correction.

        Essentially through this correction we add a constant background charge
        to make the material charge neutral. Any material whose unit cell is
        not neutral will have infinite energy/volume (the G=0 term in the
        reciprocal term will be infinite), so we have to make this correction
        to make the system physical.
        """
        charge_correction = -np.pi/(2*self.volume*self.a_squared)*np.sum(self.q)**2
        return charge_correction

    def _calc_real(self, system):
        """Used to calculate the Ewald real-space sum.

        Corresponds to equation (5) in
        https://doi.org/10.1016/0010-4655(96)00016-1
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

        ereal *= 1/2
        return ereal

    def _calc_recip(self, system):
        """
        Perform the reciprocal space summation. Uses the fastest non mesh-based
        method described as given by equation (16) in
        https://doi.org/10.1016/0010-4655(96)00016-1

        The term G=0 is neglected, even if the system has nonzero charge.
        Physically this would mean that we are adding a constant background
        charge to make the cell charge neutral.
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

        erecip *= 2 * math.pi / self.volume * qiqj * 2 ** 0.5
        return erecip

    def _calc_subsystem_energies(self, ewald_matrix):
        """Modify the give matrix that consists of the real and reciprocal sums
        so that each entry x_ij is the full Ewald sum energy of a system
        consisting of atoms i and j.
        """
        q = self.q

        # Create the self-term array where q1[i,j] is qi**2 + qj**2, except for
        # the diagonal, where it is qi**2. The self term corresponds to the
        # interaction of the point charge with cocentric Gaussian cloud
        # introduced in the Ewald method.
        q1 = q[None, :]**2 + q[:, None]**2
        diag = np.diag(q1)/2
        np.fill_diagonal(q1, diag)
        q1_prefactor = -self.a/self.sqrt_pi

        # Create the charge correction array where q2[i,j] is (qi + qj)**2,
        # except for the diagonal where it is qi**2
        q2 = q[None, :] + q[:, None]
        q2 **= 2
        diag = np.diag(q2)/4
        np.fill_diagonal(q2, diag)
        q2_prefactor = -np.pi/(2*self.volume*self.a_squared)
        correction_matrix = q1_prefactor*q1 + q2_prefactor*q2

        # Add the terms coming from x_ii and x_jj to the off-diagonal along
        # with the corrections
        n_atoms = self.n_atoms
        final_matrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    final_matrix[i, j] = ewald_matrix[i, j]
                else:
                    pair_term = 2*ewald_matrix[i, j]
                    self_term_ii = ewald_matrix[i, i]
                    self_term_jj = ewald_matrix[j, j]
                    energy_total = pair_term + self_term_ii + self_term_jj
                    final_matrix[i, j] = energy_total
        final_matrix += correction_matrix

        return final_matrix
