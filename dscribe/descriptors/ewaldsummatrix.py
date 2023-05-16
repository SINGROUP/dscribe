# -*- coding: utf-8 -*-
"""Copyright 2019 DScribe developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import math

import numpy as np

from ase import Atoms

from scipy.special import erfc

from dscribe.descriptors.descriptormatrix import DescriptorMatrix
from dscribe.core.lattice import Lattice


class EwaldSumMatrix(DescriptorMatrix):
    """
    Calculates an Ewald sum matrix for the a given system.

    Each entry M_ij of the Ewald sum matrix will contain the Coulomb energy
    between atoms i and j calculated with the Ewald summation method. In the
    Ewald method a constant neutralizing background charge has been added to
    counteract the positive net charge.

    The total electrostatic interaction energy in the system can calculated by
    summing the upper diagonal part of the matrix, including the diagonal
    itself.

    A screening parameter a controls the width of the Gaussian charge
    distributions in the Ewald summation, but the final matrix elements will be
    independent of the value of the screening parameter a that is used, as long
    as sufficient cutoff values are used.

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

    def create(
        self,
        system,
        accuracy=1e-5,
        w=1,
        r_cut=None,
        g_cut=None,
        a=None,
        n_jobs=1,
        only_physical_cores=False,
        verbose=False,
    ):
        """Return the Ewald sum matrix for the given systems.

        Args:
            system (:class:`ase.Atoms` or list of :class:`ase.Atoms`): One or
                many atomic structures.
            accuracy (float): The accuracy to which the sum is converged to.
                Corresponds to the variable :math:`A` in
                https://doi.org/10.1080/08927022.2013.840898. Used only if
                g_cut, r_cut and a have not been specified. Provide either one
                value or a list of values for each system.
            w (float): Weight parameter that represents the relative
                computational expense of calculating a term in real and
                reciprocal space. This has little effect on the total energy,
                but may influence speed of computation in large systems. Note
                that this parameter is used only when the cutoffs and a are set
                to None. Provide either one value or a list of values for each
                system.
            r_cut (float): Real space cutoff radius dictating how many terms are
                used in the real space sum. Provide either one value or a list
                of values for each system.
            g_cut (float): Reciprocal space cutoff radius. Provide either one
                value or a list of values for each system.
            a (float): The screening parameter that controls the width of the
                Gaussians. If not provided, a default value of :math:`\\alpha =
                \\sqrt{\\pi}\\left(\\frac{N}{V^2}\\right)^{1/6}` is used.
                Corresponds to the standard deviation of the Gaussians. Provide
                either one value or a list of values for each system.
            n_jobs (int): Number of parallel jobs to instantiate. Parallellizes
                the calculation across samples. Defaults to serial calculation
                with n_jobs=1. If a negative number is given, the used cpus
                will be calculated with, n_cpus + n_jobs, where n_cpus is the
                amount of CPUs as reported by the OS. With only_physical_cores
                you can control which types of CPUs are counted in n_cpus.
            only_physical_cores (bool): If a negative n_jobs is given,
                determines which types of CPUs are used in calculating the
                number of jobs. If set to False (default), also virtual CPUs
                are counted.  If set to True, only physical CPUs are counted.
            verbose(bool): Controls whether to print the progress of each job
                into to the console.

        Returns:
            np.ndarray | sparse.COO: Ewald sum matrix for the given systems.
            The return type depends on the 'sparse'-attribute. The first
            dimension is determined by the amount of systems.
        """
        var_dict = {}
        for var_new in ["r_cut", "g_cut"]:
            loc = locals()
            var_old = "".join(var_new.split("_"))
            if loc.get(var_old) is not None:
                var_dict[var_new] = loc[var_old]
                if loc.get(var_new) is not None:
                    raise ValueError(
                        "Please provide only either {} or {}.".format(var_new, var_old)
                    )
            else:
                var_dict[var_new] = loc[var_new]
        r_cut = var_dict["r_cut"]
        g_cut = var_dict["g_cut"]

        # Combine input arguments / check input validity
        system = [system] if isinstance(system, Atoms) else system
        for s in system:
            if len(s) > self.n_atoms_max:
                raise ValueError(
                    "One of the given systems has more atoms ({}) than allowed "
                    "by n_atoms_max ({}).".format(len(s), self.n_atoms_max)
                )

        # Combine input arguments
        n_samples = len(system)
        if np.ndim(accuracy) == 0:
            accuracy = n_samples * [accuracy]
        if np.ndim(w) == 0:
            w = n_samples * [w]
        if np.ndim(r_cut) == 0:
            r_cut = n_samples * [r_cut]
        if np.ndim(g_cut) == 0:
            g_cut = n_samples * [g_cut]
        if np.ndim(a) == 0:
            a = n_samples * [a]
        inp = [
            (i_sys, i_accuracy, i_w, i_r_cut, i_g_cut, i_a)
            for i_sys, i_accuracy, i_w, i_r_cut, i_g_cut, i_a in zip(
                system, accuracy, w, r_cut, g_cut, a
            )
        ]

        # Determine if the outputs have a fixed size
        n_features = self.get_number_of_features()
        static_size = [n_features]

        # Create in parallel
        output = self.create_parallel(
            inp,
            self.create_single,
            n_jobs,
            static_size,
            only_physical_cores,
            verbose=verbose,
        )

        return output

    def create_single(self, system, accuracy=1e-5, w=1, r_cut=None, g_cut=None, a=None):
        """
        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.
            accuracy (float): The accuracy to which the sum is converged to.
                Corresponds to the variable :math:`A` in
                https://doi.org/10.1080/08927022.2013.840898. Used only if g_cut,
                r_cut and a have not been specified.
            w (float): Weight parameter that represents the relative
                computational expense of calculating a term in real and
                reciprocal space. This has little effect on the total energy,
                but may influence speed of computation in large systems. Note
                that this parameter is used only when the cutoffs and a are set
                to None.
            r_cut (float): Real space cutoff radius dictating how
                many terms are used in the real space sum.
            g_cut (float): Reciprocal space cutoff radius.
            a (float): The screening parameter that controls the width of the
                Gaussians. If not provided, a default value of :math:`\\alpha =
                \\sqrt{\\pi}\\left(\\frac{N}{V^2}\\right)^{1/6}` is used.
                Corresponds to the standard deviation of the Gaussians.
        """
        self.q = system.get_atomic_numbers()
        self.q_squared = self.q**2
        self.n_atoms = len(system)
        self.volume = system.get_volume()
        self.sqrt_pi = math.sqrt(np.pi)

        # If a is not provided, use a default value
        if a is None:
            a = (self.n_atoms * w / (self.volume**2)) ** (1 / 6) * self.sqrt_pi

        # If the real space cutoff, reciprocal space cutoff and a have not been
        # specified, use the accuracy and the weighting w to determine default
        # similarly as in https://doi.org/10.1080/08927022.2013.840898
        if r_cut is None and g_cut is None:
            f = np.sqrt(-np.log(accuracy))
            r_cut = f / a
            g_cut = 2 * a * f
        elif r_cut is None or g_cut is None:
            raise ValueError(
                "If you do not want to use the default cutoffs, please provide "
                "both cutoffs r_cut and g_cut."
            )

        self.a = a
        self.a_squared = self.a**2
        self.g_cut = g_cut
        self.r_cut = r_cut

        matrix = super().create_single(system)
        return matrix

    def get_matrix(self, system):
        """
        The total energy matrix. Each matrix element (i, j) corresponds to the
        total interaction energy in a system with atoms i and j.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.

        Returns:
            np.ndarray: Ewald matrix.
        """
        # Calculate the regular real and reciprocal space sums of the Ewald sum.
        ereal = self._calc_real(system)
        erecip = self._calc_recip(system)
        ezero = self._calc_zero()
        total = erecip + ereal + ezero

        return total

    def _calc_zero(self):
        """Calculates the constant part of the Ewald sum matrix.

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
        matself *= -self.a / self.sqrt_pi

        # Calculate the interaction energy between constant neutralizing
        # background charge. On the diagonal this is defined by
        matbg = 2 * q[None, :] * q[:, None].astype(float)
        matbg *= -np.pi / (2 * self.volume * self.a_squared)

        # The diagonal terms are divided by two
        diag = np.diag(matbg) / 2
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
        ereal = np.zeros((n_atoms, n_atoms), dtype=np.float64)
        lattice = Lattice(system.get_cell())

        # For each atom in the original cell, get the neighbours in the
        # infinite system within the real space cutoff and calculate the real
        # space portion of the Ewald sum.
        for i in range(n_atoms):
            # Get points that are within the real space cutoff
            nfcoords, rij, js = lattice.get_points_in_sphere(
                fcoords, coords[i], self.r_cut, zip_results=False
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
        diag = np.diag(ereal) / 2
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
        erecip = np.zeros((n_atoms, n_atoms), dtype=np.float64)
        coords = system.get_positions()

        # Get the reciprocal lattice points within the reciprocal space cutoff
        rcp_latt = 2 * np.pi * system.cell.reciprocal()
        rcp_latt = Lattice(rcp_latt)
        recip_nn = rcp_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], self.g_cut)

        # Ignore the terms with G=0.
        frac_coords = [fcoords for (fcoords, dist, i) in recip_nn if dist != 0]

        gs = rcp_latt.get_cartesian_coords(frac_coords)
        g2s = np.sum(gs**2, 1)
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

        erecip *= 4 * math.pi / self.volume * qiqj * 2**0.5

        # The diagonal terms are divided by two
        diag = np.diag(erecip) / 2
        np.fill_diagonal(erecip, diag)

        return erecip
