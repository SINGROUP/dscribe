import math
import numpy as np
from scipy.special import erfc
import scipy.constants as constants
from describe.descriptors import Descriptor


class EwaldMatrix(Descriptor):
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
    """
    # Converts unit of q*q/r into eV
    CONV_FACT = 1e10 * constants.e / (4 * math.pi * constants.epsilon_0)

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

    def create(self, system, real_space_cut, recip_space_cut, a):
        """
        Initializes and calculates the Ewald sum. Default convergence
        parameters have been specified, but you can override them if you wish.

        Args:
            system (System): Input system.
            real_space_cut (float): Real space cutoff radius dictating how
                many terms are used in the real space sum.
            recip_space_cut (float): Reciprocal space cutoff radius.
            a (float): The screening parameter that controls the width of the
                Gaussians.
        """
        self.system = system
        self.q = system.charges
        self.q_squared = self.q**2
        self.n_atoms = len(system)
        self.volume = system.lattice.volume
        self.sqrt_pi = math.sqrt(np.pi)
        self.a = a
        self.a_squared = self.a**2
        self.gmax = recip_space_cut
        self.rmax = real_space_cut

        # Calculate the matrix
        emat = self.total_energy_matrix

        # Pad with zeros
        zeros = np.zeros((self.n_atoms_max, self.n_atoms_max))
        zeros[:emat.shape[0], :emat.shape[1]] = emat
        emat = zeros

        if self.flatten:
            emat = emat.flatten()
        return emat

    @property
    def total_energy_matrix(self):
        """
        The total energy matrix. Each matrix element (i, j) corresponds to the
        total interaction energy between site i and site j.
        """
        # Calculate the regular real and reciprocal space sums of the Ewald sum.
        ereal = self._calc_real_matrix()
        erecip = self._calc_recip()
        total = erecip + ereal

        # Calculate the modification that makes each entry of the matrix to be
        # the full Ewald sum of the ij subsystem.
        total = self._calc_subsystem_energies(total)
        total *= EwaldMatrix.CONV_FACT

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

    def _calc_subsystem_energies(self, ewald_matrix):
        """Modify the give matrix that consists of the eral and reciprocal sums
        so that each entry x_ij is the full Ewald sum energy of a system
        consisting of atoms i and j.
        """
        q = self.q

        # Create the self-term array where q1[i,j] is qi**2 + qj**2, except for
        # the diagonal, where it is qi**2
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

    def _calc_recip(self):
        """
        Perform the reciprocal space summation. Uses the fastest non mesh-based
        method described in "Ewald summation techniques in perspective: a
        survey"

        The term G=0 is neglected, even if the system has nonzero charge.
        Physically this would mean that we are adding a constant background
        charge to make the cell charge neutral.
        """
        n_atoms = self.n_atoms
        prefactor = 2 * math.pi / self.volume
        erecip = np.zeros((n_atoms, n_atoms), dtype=np.float)
        coords = self.system.cartesian_pos
        rcp_latt = self.system.lattice.reciprocal_lattice
        recip_nn = rcp_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0],
                                                 self.gmax)

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

        erecip *= prefactor * qiqj * 2 ** 0.5
        return erecip

    def _calc_real_matrix(self):
        """
        Determines the Ewald real-space sum.
        """
        fcoords = self.system.relative_pos
        coords = self.system.cartesian_pos
        n_atoms = len(self.system)
        prefactor = 0.5
        ereal = np.zeros((n_atoms, n_atoms), dtype=np.float)

        for i in range(n_atoms):

            # Get points that are within the real space cutoff
            nfcoords, rij, js = self.system.lattice.get_points_in_sphere(
                fcoords,
                coords[i],
                self.rmax,
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

        ereal *= prefactor
        return ereal
