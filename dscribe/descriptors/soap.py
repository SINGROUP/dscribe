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
import numpy as np

from scipy.sparse import coo_matrix
from scipy.special import gamma
from scipy.linalg import sqrtm, inv

from ase import Atoms
import ase.data

from dscribe.descriptors import Descriptor
from dscribe.core import System
from dscribe.utils.geometry import get_extended_system
import dscribe.ext


class SOAP(Descriptor):
    """Class for generating a partial power spectrum from Smooth Overlap of
    Atomic Orbitals (SOAP). This implementation uses real (tesseral) spherical
    harmonics as the angular basis set and provides two orthonormalized
    alternatives for the radial basis functions: spherical primitive gaussian
    type orbitals ("gto") or the polynomial basis set ("polynomial").

    For reference, see:

    "On representing chemical environments, Albert P. Bartók, Risi Kondor, and
    Gábor Csányi, Phys. Rev. B 87, 184115, (2013),
    https://doi.org/10.1103/PhysRevB.87.184115

    "Comparing molecules and solids across structural and alchemical space",
    Sandip De, Albert P. Bartók, Gábor Csányi and Michele Ceriotti, Phys.
    Chem. Chem. Phys. 18, 13754 (2016), https://doi.org/10.1039/c6cp00415f

    "Machine learning hydrogen adsorption on nanoclusters through structural
    descriptors", Marc O. J. Jäger, Eiaki V. Morooka, Filippo Federici Canova,
    Lauri Himanen & Adam S. Foster, npj Comput. Mater., 4, 37 (2018),
    https://doi.org/10.1038/s41524-018-0096-5
    """
    def __init__(
            self,
            rcut,
            nmax,
            lmax,
            sigma=1.0,
            rbf="gto",
            species=None,
            periodic=False,
            crossover=True,
            average=False,
            sparse=False,
            ):
        """
        Args:
            rcut (float): A cutoff for local region in angstroms. Should be
                bigger than 1 angstrom.
            nmax (int): The number of radial basis functions.
            lmax (int): The maximum degree of spherical harmonics.
            species (iterable): The chemical species as a list of atomic
                numbers or as a list of chemical symbols. Notice that this is not
                the atomic numbers that are present for an individual system, but
                should contain all the elements that are ever going to be
                encountered when creating the descriptors for a set of systems.
                Keeping the number of chemical species as low as possible is
                preferable.
            sigma (float): The standard deviation of the gaussians used to expand the
                atomic density.
            rbf (str): The radial basis functions to use. The available options are:

                * "gto": Spherical gaussian type orbitals defined as :math:`g_{nl}(r) = \sum_{n'=1}^{n_\mathrm{max}}\,\\beta_{nn'l} r^l e^{-\\alpha_{n'l}r^2}`
                * "polynomial": Polynomial basis defined as :math:`g_{n}(r) = \sum_{n'=1}^{n_\mathrm{max}}\,\\beta_{nn'} (r-r_\mathrm{cut})^{n'+2}`

            periodic (bool): Determines whether the system is considered to be
                periodic.
            crossover (bool): Determines if crossover of atomic types should
                be included in the power spectrum. If enabled, the power
                spectrum is calculated over all unique species combinations Z
                and Z'. If disabled, the power spectrum does not contain
                cross-species information and is only run over each unique
                species Z. Turned on by default to correspond to the original
                definition
            average (bool): Whether to build an average output for all selected
                positions.
            sparse (bool): Whether the output should be a sparse matrix or a
                dense numpy array.
        """
        super().__init__(periodic=periodic, flatten=True, sparse=sparse)

        # Setup the involved chemical species
        self.species = species

        # Test that general settings are valid
        if (sigma <= 0):
            raise ValueError(
                "Only positive gaussian width parameters 'sigma' are allowed."
            )
        self._eta = 1/(2*sigma**2)
        self._sigma = sigma

        if lmax < 0:
            raise ValueError(
                "lmax cannot be negative. lmax={}".format(lmax)
            )
        supported_rbf = set(("gto", "polynomial"))
        if rbf not in supported_rbf:
            raise ValueError(
                "Invalid radial basis function of type '{}' given. Please use "
                "one of the following: {}".format(rbf, supported_rbf)
            )
        if nmax < 1:
            raise ValueError(
                "Must have at least one radial basis function."
                "nmax={}".format(nmax)
            )

        # Test that radial basis set specific settings are valid
        if rbf == "gto":
            if rcut <= 1:
                raise ValueError(
                    "When using the gaussian radial basis set (gto), the radial "
                    "cutoff should be bigger than 1 angstrom."
                )
            if lmax > 9:
                raise ValueError(
                    "When using the gaussian radial basis set (gto), lmax "
                    "cannot currently exceed 9. lmax={}".format(lmax)
                )
            # Precalculate the alpha and beta constants for the GTO basis
            self._alphas, self._betas = self.get_basis_gto(rcut, nmax)

        elif rbf == "polynomial":
            if lmax > 20:
                raise ValueError(
                    "When using the polynomial radial basis set, lmax "
                    "cannot currently exceed 20. lmax={}".format(lmax)
                )

        self._rcut = rcut
        self._nmax = nmax
        self._lmax = lmax
        self._rbf = rbf
        self.crossover = crossover
        self._average = average

    def create(self, system, positions=None, n_jobs=1, verbose=False):
        """Return the SOAP output for the given systems and given positions.

        Args:
            system (:class:`ase.Atoms` or list of :class:`ase.Atoms`): One or
                many atomic structures.
            positions (list): Positions where to calculate SOAP. Can be
                provided as cartesian positions or atomic indices. If no
                positions are defined, the SOAP output will be created for all
                atoms in the system. When calculating SOAP for multiple
                systems, provide the positions as a list for each system.
            n_jobs (int): Number of parallel jobs to instantiate. Parallellizes
                the calculation across samples. Defaults to serial calculation
                with n_jobs=1.
            verbose(bool): Controls whether to print the progress of each job
                into to the console.

        Returns:
            np.ndarray | scipy.sparse.csr_matrix: The SOAP output for the given
            systems and positions. The return type depends on the
            'sparse'-attribute. The first dimension is determined by the amount
            of positions and systems and the second dimension is determined by
            the get_number_of_features()-function. When multiple systems are
            provided the results are ordered by the input order of systems and
            their positions.
        """
        # If single system given, skip the parallelization
        if isinstance(system, (Atoms, System)):
            return self.create_single(system, positions)

        # Combine input arguments
        n_samples = len(system)
        if positions is None:
            inp = [(i_sys,) for i_sys in system]
        else:
            n_pos = len(positions)
            if n_pos != n_samples:
                raise ValueError(
                    "The given number of positions does not match the given"
                    "number os systems."
                )
            inp = list(zip(system, positions))

        # For SOAP the output size for each job depends on the exact arguments.
        # Here we precalculate the size for each job to preallocate memory and
        # make the process faster.
        k, m = divmod(n_samples, n_jobs)
        jobs = (inp[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_jobs))
        output_sizes = []
        for i_job in jobs:
            n_desc = 0
            if self._average:
                n_desc = len(i_job)
            elif positions is None:
                n_desc = 0
                for job in i_job:
                    n_desc += len(job[0])
            else:
                n_desc = 0
                for i_sample, i_pos in i_job:
                    if i_pos is not None:
                        n_desc += len(i_pos)
                    else:
                        n_desc += len(i_sample)
            output_sizes.append(n_desc)

        # Create in parallel
        output = self.create_parallel(inp, self.create_single, n_jobs, output_sizes, verbose=verbose)

        return output

    def create_single(self, system, positions=None):
        """Return the SOAP output for the given system and given positions.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.
            positions (list): Cartesian positions or atomic indices. If
                specified, the SOAP spectrum will be created for these points.
                If no positions are defined, the SOAP output will be created
                for all atoms in the system.

        Returns:
            np.ndarray | scipy.sparse.coo_matrix: The SOAP output for the
            given system and positions. The return type depends on the
            'sparse'-attribute. The first dimension is given by the number of
            positions and the second dimension is determined by the
            get_number_of_features()-function.
        """
        # Transform the input system into the internal System-object
        system = self.get_system(system)

        # Check that the system does not have elements that are not in the list
        # of atomic numbers
        self.check_atomic_numbers(system.get_atomic_numbers())
        sub_elements = np.array(list(set(system.get_atomic_numbers())))

        # Check if periodic is valid
        if self.periodic:
            cell = system.get_cell()
            if np.cross(cell[0], cell[1]).dot(cell[2]) == 0:
                raise ValueError(
                    "System doesn't have cell to justify periodicity."
                )

        # Setup the local positions
        if positions is None:
            list_positions = system.get_positions()
        else:
            # Check validity of position definitions and create final cartesian
            # position list
            list_positions = []
            if len(positions) == 0:
                raise ValueError(
                    "The argument 'positions' should contain a non-empty set of"
                    " atomic indices or cartesian coordinates with x, y and z "
                    "components."
                )
            for i in positions:
                if np.issubdtype(type(i), np.integer):
                    list_positions.append(system.get_positions()[i])
                elif isinstance(i, (list, tuple, np.ndarray)):
                    if len(i) != 3:
                        raise ValueError(
                            "The argument 'positions' should contain a "
                            "non-empty set of atomic indices or cartesian "
                            "coordinates with x, y and z components."
                        )
                    list_positions.append(i)
                else:
                    raise ValueError(
                        "Create method requires the argument 'positions', a "
                        "list of atom indices and/or positions."
                    )

        # The radial cutoff is extended by adding a padding that depends on
        # the used used sigma value. The padding is chosen so that the
        # gaussians decay to the specified threshold value at the cutoff
        # distance.
        threshold = 0.001
        cutoff_padding = self._sigma*np.sqrt(-2*np.log(threshold))

        # Create the extended system if periodicity is requested
        if self.periodic:
            system = get_extended_system(system, self._rcut+cutoff_padding, return_cell_indices=False)

        # Determine the SOAPLite function to call based on periodicity and
        # rbf
        if self._rbf == "gto":
            soap_mat = self.get_soap_locals_gto(
                system,
                list_positions,
                self._alphas,
                self._betas,
                rcut=self._rcut,
                cutoff_padding=cutoff_padding,
                nmax=self._nmax,
                lmax=self._lmax,
                eta=self._eta,
                crossover=self.crossover,
                atomic_numbers=None,
            )
        elif self._rbf == "polynomial":
            soap_mat = self.get_soap_locals_poly(
                system,
                list_positions,
                rcut=self._rcut,
                cutoff_padding=cutoff_padding,
                nmax=self._nmax,
                lmax=self._lmax,
                eta=self._eta,
                crossover=self.crossover,
                atomic_numbers=None,
            )

        # Map the output from subspace of elements to the full space of
        # elements
        soap_mat = self.get_full_space_output(
            soap_mat,
            sub_elements,
            self._atomic_numbers
        )

        # Create the averaged SOAP output if requested.
        if self._average:
            soap_mat = soap_mat.mean(axis=0)
            soap_mat = np.expand_dims(soap_mat, 0)

        # Make into a sparse array if requested
        if self._sparse:
            soap_mat = coo_matrix(soap_mat)

        return soap_mat

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        """Used to check the validity of given atomic numbers and to initialize
        the C-memory layout for them.

        Args:
            value(iterable): Chemical species either as a list of atomic
                numbers or list of chemical symbols.
        """
        # The species are stored as atomic numbers for internal use.
        self._set_species(value)

        # Setup mappings between atom indices and types
        self.atomic_number_to_index = {}
        self.index_to_atomic_number = {}
        for i_atom, atomic_number in enumerate(self._atomic_numbers):
            self.atomic_number_to_index[atomic_number] = i_atom
            self.index_to_atomic_number[i_atom] = atomic_number
        self.n_elements = len(self._atomic_numbers)

    def get_full_space_output(self, sub_output, sub_elements, full_elements_sorted):
        """Used to partition the SOAP output to different locations depending
        on the interacting elements. SOAPLite return the output partitioned by
        the elements present in the given system. This function correctly
        places those results within a bigger chemical space.

        Args:
            sub_output(np.ndarray): The output fron SOAPLite
            sub_elements(list): The atomic numbers present in the subspace
            full_elements_sorted(list): The atomic numbers present in the full
                space, sorted.

        Returns:
            np.ndarray: The given SOAP output mapped to the full chemical space.
        """
        # print(sub_output.shape)
        # Get mapping between elements in the subspace and alements in the full
        # space
        space_map = self.get_sub_to_full_map(sub_elements, full_elements_sorted)

        # Reserve space for a sparse matric containing the full output space
        n_features = self.get_number_of_features()
        n_elem_features = self.get_number_of_element_features()
        n_points = sub_output.shape[0]

        # Define the final output space as an array.
        output = np.zeros((n_points, n_features), dtype=np.float32)

        # When crossover is enabled, we need to store the contents for all
        # unique species combinations
        if self.crossover:
            n_elem_sub = len(sub_elements)
            n_elem_full = len(full_elements_sorted)
            for i_sub in range(n_elem_sub):
                for j_sub in range(n_elem_sub):
                    if j_sub >= i_sub:

                        # This is the index of the spectrum. It is given by enumerating the
                        # elements of an upper triangular matrix from left to right and top
                        # to bottom.
                        m = self.get_flattened_index(i_sub, j_sub, n_elem_sub)
                        start_sub = m*n_elem_features
                        end_sub = (m+1)*n_elem_features
                        sub_out = sub_output[:, start_sub:end_sub]

                        # Figure out position in the full element space
                        i_full = space_map[i_sub]
                        j_full = space_map[j_sub]
                        m_full = self.get_flattened_index(i_full, j_full, n_elem_full)

                        # Place output to full output vector
                        start_full = m_full*n_elem_features
                        end_full = (m_full+1)*n_elem_features
                        output[:, start_full:end_full] = sub_out
        # When crossover is disabled, we need to store only power spectrums
        # that contain for each species
        else:
            n_elem_sub = len(sub_elements)
            n_elem_full = len(full_elements_sorted)
            for m in range(n_elem_sub):
                # This is the index of the spectrum. It is given by enumerating the
                # elements of an upper triangular matrix from left to right and top
                # to bottom.
                start_sub = m*n_elem_features
                end_sub = (m+1)*n_elem_features
                sub_out = sub_output[:, start_sub:end_sub]
                # print(sub_out.shape)

                # Figure out position in the full element space
                m_full = space_map[m]

                # Place output to full output vector
                start_full = m_full*n_elem_features
                end_full = (m_full+1)*n_elem_features
                output[:, start_full:end_full] = sub_out

        return output

    def get_sub_to_full_map(self, sub_elements, full_elements):
        """Used to map an index in the sub-space of elements to the full
        element-space.
        """
        # Sort the elements according to atomic number
        sub_elements_sorted = np.sort(sub_elements)
        full_elements_sorted = np.sort(full_elements)

        mapping = {}
        for i_sub, z in enumerate(sub_elements_sorted):
            i_full = np.where(full_elements_sorted == z)[0][0]
            mapping[i_sub] = i_full

        return mapping

    def get_number_of_element_features(self):
        """Used to query the number of elements in the SOAP feature space for
        a single element pair.

        Returns:
            int: The number of features per element pair.
        """
        return int((self._lmax + 1) * self._nmax * (self._nmax + 1)/2)

    def get_flattened_index(self, i, j, n):
        """Returns the 1D index of an element in an upper diagonal matrix that
        has been flattened by iterating over the elements from left to right
        and top to bottom.
        """
        return int(j + i*n - i*(i+1)/2)

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        n_elems = len(self._atomic_numbers)
        if self.crossover:
            n_blocks = n_elems * (n_elems + 1)/2
        else:
            n_blocks = n_elems

        n_element_features = self.get_number_of_element_features()

        return int(n_element_features * n_blocks)

    def get_location(self, species):
        """Can be used to query the location of a species combination in the
        the flattened output.

        Args:
            species(tuple): A tuple containing a species combination as
                chemical symbols or atomic numbers. The tuple can be for
                example: ("H", "O") or (1, 6)

        Returns:
            slice: slice containing the location of the specified species
            combination. The location is given as a python slice-object, that
            can be directly used to target ranges in the output.

        Raises:
            ValueError: If the requested species combination is not in the
                output or if invalid species defined.
        """
        # Change chemical elements into atomic numbers
        numbers = []
        for specie in species:
            if isinstance(specie, str):
                try:
                    specie = ase.data.atomic_numbers[specie]
                except KeyError:
                    raise ValueError("Invalid chemical species: {}".format(specie))
            numbers.append(specie)

        # Check that the given atomic numbers are supported
        self.check_atomic_numbers(numbers)
        if not self.crossover and numbers[0] != numbers[1]:
            raise ValueError(
                "The output does not have pairwise terms as you have not "
                "enabled species crossover for SOAP. See the 'crossover' "
                "attribute."
            )

        # Change into internal indexing
        indices = [self.atomic_number_to_index[x] for x in numbers]
        n_elem = self.n_elements
        n_elem_features = self.get_number_of_element_features()

        # Makes sure that the upper diagonal part is accessed (idx2 >= idx1)
        idx1 = indices[0]
        idx2 = indices[1]
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        # With crossover
        if self.crossover:
            shift = self.get_flattened_index(idx1, idx2, n_elem)
            start = shift*n_elem_features
            end = (shift+1)*n_elem_features
        # Without crossover
        else:
            start = idx1*n_elem_features
            end = (idx1+1)*n_elem_features

        return slice(start, end)

    def flatten_positions(self, system, atomic_numbers=None):
        """Takes an ase Atoms object and returns flattened numpy arrays for the
        C-extension to use.

        Args:
            system (ase.atoms): The system to convert.
            atomic_numbers(): The atomic numbers to consider. Atoms that do not
                have these atomic numbers are ignored.

        Returns:
            (np.ndarray, list, int, np.ndarray): Returns the positions
            flattened and sorted by atomic number, atomic numbers flattened and
            sorted by atomic number, number of different species and the sorted
            set of atomic numbers.
        """
        Z = system.get_atomic_numbers()
        pos = system.get_positions()

        # Get a sorted list of atom types
        if atomic_numbers is not None:
            atomtype_set = set(atomic_numbers)
        else:
            atomtype_set = set(Z)
        atomic_numbers_sorted = np.sort(list(atomtype_set))

        # Form a flattened list of atomic positions, sorted by atomic type
        pos_lst = []
        z_lst = []
        for atomtype in atomic_numbers_sorted:
            condition = Z == atomtype
            pos_onetype = pos[condition]
            z_onetype = Z[condition]
            pos_lst.append(pos_onetype)
            z_lst.append(z_onetype)
        n_species = len(atomic_numbers_sorted)
        positions_sorted = np.concatenate(pos_lst, axis=0)
        atomic_numbers_sorted = np.concatenate(z_lst).ravel()

        return positions_sorted, atomic_numbers_sorted, n_species, atomic_numbers_sorted

    def flatten_positions_old(self, system, atomic_numbers=None):
        """ Takes an ase Atoms object and returns numpy arrays and integers
        which are read by the internal clusgeo. Apos is currently a flattened
        out numpy array

        Args:
            system (ase.atoms): The system to convert.
            atomic_numbers(): The atomic numbers to consider. Atoms that do not
                have these atomic numbers are ignored.

        Returns:
            (np.ndarray, list, int, np.ndarray): Returns the positions flattened
            and sorted by atomic number, numer of atoms per type, number of
            different species and the sorted atomic numbers.
        """
        Z = system.get_atomic_numbers()
        pos = system.get_positions()

        # Get a sorted list of atom types
        if atomic_numbers is not None:
            atomtype_set = set(atomic_numbers)
        else:
            atomtype_set = set(Z)
        atomic_numbers_sorted = np.sort(list(atomtype_set))

        # Form a flattened list of atomic positions, sorted by atomic type
        n_atoms_per_type = []
        pos_lst = []
        for atomtype in atomic_numbers_sorted:
            condition = Z == atomtype
            pos_onetype = pos[condition]
            n_onetype = pos_onetype.shape[0]
            pos_lst.append(pos_onetype)
            n_atoms_per_type.append(n_onetype)
        n_species = len(atomic_numbers_sorted)
        positions_sorted = np.concatenate(pos_lst).ravel()

        return positions_sorted, n_atoms_per_type, n_species, atomic_numbers_sorted

    def get_soap_locals_gto(self, system, centers, alphas, betas, rcut, cutoff_padding, nmax, lmax, eta, crossover, atomic_numbers=None):
        """Get the SOAP output for the given positions using the gto radial
        basis.

        Args:
            system (ase.Atoms): Atomic structure for which the SOAP output is
                calculated.
            centers (np.ndarray): Positions at which to calculate SOAP.
            alphas (np.ndarray): The alpha coeffients for the gto-basis.
            betas (np.ndarray): The beta coeffients for the gto-basis.
            rcut (float): Radial cutoff.
            cutoff_padding (float): The padding that is added for including
                atoms beyond the cutoff.
            nmax (int): Maximum number of radial basis functions.
            lmax (int): Maximum spherical harmonics degree.
            eta (float): The gaussian smearing width.
            crossover (bool): Whether to include species crossover in output.
            atomic_numbers (np.ndarray): Can be used to specify the species for
                which to calculate the output. If None, all species are included.
                If given the output is calculated only for the given species and is
                ordered by atomic number.

        Returns:
            np.ndarray: SOAP output with the gto radial basis for the given positions.
        """
        n_atoms = len(system)
        positions, Z_sorted, n_species, atomtype_lst = self.flatten_positions(system, atomic_numbers)
        centers = np.array(centers)
        n_centers = centers.shape[0]
        centers = centers.flatten()
        alphas = alphas.flatten()
        betas = betas.flatten()

        # Determine shape
        if crossover:
            c = np.zeros(int((nmax*(nmax+1))/2)*(lmax+1)*int((n_species*(n_species + 1))/2)*n_centers, dtype=np.float64)
            shape = (n_centers, int((nmax*(nmax+1))/2)*(lmax+1)*int((n_species*(n_species+1))/2))
        else:
            c = np.zeros(int((nmax*(nmax+1))/2)*(lmax+1)*int(n_species)*n_centers, dtype=np.float64)
            shape = (n_centers, int((nmax*(nmax+1))/2)*(lmax+1)*n_species)

        # Calculate with extension
        dscribe.ext.soap_gto(c, positions, centers, alphas, betas, Z_sorted, rcut, cutoff_padding, n_atoms, n_species, nmax, lmax, n_centers, eta, crossover)

        # Reshape from linear to 2D
        c = c.reshape(shape)

        return c

    def get_soap_locals_poly(self, system, centers, rcut, cutoff_padding, nmax, lmax, eta, crossover, atomic_numbers=None):
        """Get the SOAP output using polynomial radial basis for the given
        positions.
        Args:
            system(ase.Atoms): Atomic structure for which the SOAP output is
                calculated.
            centers(np.ndarray): Positions at which to calculate SOAP.
            alphas (np.ndarray): The alpha coeffients for the gto-basis.
            betas (np.ndarray): The beta coeffients for the gto-basis.
            rCut (float): Radial cutoff.
            cutoff_padding (float): The padding that is added for including
                atoms beyond the cutoff.
            nmax (int): Maximum number of radial basis functions.
            lmax (int): Maximum spherical harmonics degree.
            eta (float): The gaussian smearing width.
            crossover (bool): Whether to include species crossover in output.
            atomic_numbers (np.ndarray): Can be used to specify the species for
                which to calculate the output. If None, all species are included.
                If given the output is calculated only for the given species and is
                ordered by atomic number.
        Returns:
            np.ndarray: SOAP output with the polynomial radial basis for the
            given positions.
        """
        rx, gss = self.get_basis_poly(rcut, nmax)

        n_atoms = len(system)
        positions, Z_sorted, n_species, atomtype_lst = self.flatten_positions(system, atomic_numbers)
        centers = np.array(centers)
        n_centers = centers.shape[0]

        # Flatten arrays
        gss = gss.flatten()

        # Determine shape
        if crossover:
            c = np.zeros(int((nmax*(nmax+1))/2)*(lmax+1)*int((n_species*(n_species + 1))/2)*n_centers, dtype=np.float64)
            shape = (n_centers, int((nmax*(nmax+1))/2)*(lmax+1)*int((n_species*(n_species+1))/2))
        else:
            c = np.zeros(int((nmax*(nmax+1))/2)*(lmax+1)*int(n_species)*n_centers, dtype=np.float64)
            shape = (n_centers, int((nmax*(nmax+1))/2)*(lmax+1)*n_species)

        # Calculate with extension
        dscribe.ext.soap_general(c, positions, centers, Z_sorted, rcut, cutoff_padding, n_atoms, n_species, nmax, lmax, n_centers, eta, rx, gss, crossover)

        # Reshape from linear to 2D
        c = c.reshape(shape)

        return c

    def get_basis_gto(self, rcut, nmax):
        """Used to calculate the alpha and beta prefactors for the gto-radial
        basis.

        Args:
            rcut(float): Radial cutoff.
            nmax(int): Number of gto radial bases.

        Returns:
            (np.ndarray, np.ndarray): The alpha and beta prefactors for all bases
            up to a fixed size of l=10.
        """
        # These are the values for where the different basis functions should decay
        # to: evenly space between 1 angstrom and rcut.
        a = np.linspace(1, rcut, nmax)
        threshold = 1e-3  # This is the fixed gaussian decay threshold

        alphas_full = np.zeros((10, nmax))
        betas_full = np.zeros((10, nmax, nmax))

        for l in range(0, 10):
            # The alphas are calculated so that the GTOs will decay to the set
            # threshold value at their respective cutoffs
            alphas = -np.log(threshold/np.power(a, l))/a**2

            # Calculate the overlap matrix
            m = np.zeros((alphas.shape[0], alphas.shape[0]))
            m[:, :] = alphas
            m = m + m.transpose()
            S = 0.5*gamma(l + 3.0/2.0)*m**(-l-3.0/2.0)

            # Get the beta factors that orthonormalize the set with Löwdin
            # orthonormalization
            betas = sqrtm(inv(S))

            # If the result is complex, the calculation is currently halted.
            if (betas.dtype == np.complex128):
                raise ValueError(
                    "Could not calculate normalization factors for the radial "
                    "basis in the domain of real numbers. Lowering the number of "
                    "radial basis functions (nmax) or increasing the radial "
                    "cutoff (rcut) is advised."
                )

            alphas_full[l, :] = alphas
            betas_full[l, :, :] = betas

        return alphas_full, betas_full

    def get_basis_poly(self, rcut, nmax):
        """Used to calculate discrete vectors for the polynomial basis functions.

        Args:
            rcut(float): Radial cutoff.
            nmax(int): Number of polynomial radial bases.

        Returns:
            (np.ndarray, np.ndarray): Tuple containing the evaluation points in
            radial direction as the first item, and the corresponding
            orthonormalized polynomial radial basis set as the second item.
        """
        # Calculate the overlap of the different polynomial functions in a
        # matrix S. These overlaps defined through the dot product over the
        # radial coordinate are analytically calculable: Integrate[(rc - r)^(a
        # + 2) (rc - r)^(b + 2) r^2, {r, 0, rc}]. Then the weights B that make
        # the basis orthonormal are given by B=S^{-1/2}
        S = np.zeros((nmax, nmax), dtype=np.float64)
        for i in range(1, nmax+1):
            for j in range(1, nmax+1):
                S[i-1, j-1] = (2*(rcut)**(7+i+j))/((5+i+j)*(6+i+j)*(7+i+j))

        # Get the beta factors that orthonormalize the set with Löwdin
        # orthonormalization
        betas = sqrtm(np.linalg.inv(S))

        # If the result is complex, the calculation is currently halted.
        if (betas.dtype == np.complex128):
            raise ValueError(
                "Could not calculate normalization factors for the radial "
                "basis in the domain of real numbers. Lowering the number of "
                "radial basis functions (nmax) or increasing the radial "
                "cutoff (rcut) is advised."
            )

        # The radial basis is integrated in a very specific nonlinearly spaced
        # grid given by rx
        x = np.zeros(100)
        x[0] = -0.999713726773441234
        x[1] = -0.998491950639595818
        x[2] = -0.996295134733125149
        x[3] = -0.99312493703744346
        x[4] = -0.98898439524299175
        x[5] = -0.98387754070605702
        x[6] = -0.97780935848691829
        x[7] = -0.97078577576370633
        x[8] = -0.962813654255815527
        x[9] = -0.95390078292549174
        x[10] = -0.94405587013625598
        x[11] = -0.933288535043079546
        x[12] = -0.921609298145333953
        x[13] = -0.90902957098252969
        x[14] = -0.895561644970726987
        x[15] = -0.881218679385018416
        x[16] = -0.86601468849716462
        x[17] = -0.849964527879591284
        x[18] = -0.833083879888400824
        x[19] = -0.815389238339176254
        x[20] = -0.79689789239031448
        x[21] = -0.77762790964949548
        x[22] = -0.757598118519707176
        x[23] = -0.736828089802020706
        x[24] = -0.715338117573056447
        x[25] = -0.69314919935580197
        x[26] = -0.670283015603141016
        x[27] = -0.64676190851412928
        x[28] = -0.622608860203707772
        x[29] = -0.59784747024717872
        x[30] = -0.57250193262138119
        x[31] = -0.546597012065094168
        x[32] = -0.520158019881763057
        x[33] = -0.493210789208190934
        x[34] = -0.465781649773358042
        x[35] = -0.437897402172031513
        x[36] = -0.409585291678301543
        x[37] = -0.380872981624629957
        x[38] = -0.351788526372421721
        x[39] = -0.322360343900529152
        x[40] = -0.292617188038471965
        x[41] = -0.26258812037150348
        x[42] = -0.23230248184497397
        x[43] = -0.201789864095735997
        x[44] = -0.171080080538603275
        x[45] = -0.140203137236113973
        x[46] = -0.109189203580061115
        x[47] = -0.0780685828134366367
        x[48] = -0.046871682421591632
        x[49] = -0.015628984421543083
        x[50] = 0.0156289844215430829
        x[51] = 0.046871682421591632
        x[52] = 0.078068582813436637
        x[53] = 0.109189203580061115
        x[54] = 0.140203137236113973
        x[55] = 0.171080080538603275
        x[56] = 0.201789864095735997
        x[57] = 0.23230248184497397
        x[58] = 0.262588120371503479
        x[59] = 0.292617188038471965
        x[60] = 0.322360343900529152
        x[61] = 0.351788526372421721
        x[62] = 0.380872981624629957
        x[63] = 0.409585291678301543
        x[64] = 0.437897402172031513
        x[65] = 0.465781649773358042
        x[66] = 0.49321078920819093
        x[67] = 0.520158019881763057
        x[68] = 0.546597012065094168
        x[69] = 0.572501932621381191
        x[70] = 0.59784747024717872
        x[71] = 0.622608860203707772
        x[72] = 0.64676190851412928
        x[73] = 0.670283015603141016
        x[74] = 0.693149199355801966
        x[75] = 0.715338117573056447
        x[76] = 0.736828089802020706
        x[77] = 0.75759811851970718
        x[78] = 0.77762790964949548
        x[79] = 0.79689789239031448
        x[80] = 0.81538923833917625
        x[81] = 0.833083879888400824
        x[82] = 0.849964527879591284
        x[83] = 0.866014688497164623
        x[84] = 0.881218679385018416
        x[85] = 0.89556164497072699
        x[86] = 0.90902957098252969
        x[87] = 0.921609298145333953
        x[88] = 0.933288535043079546
        x[89] = 0.94405587013625598
        x[90] = 0.953900782925491743
        x[91] = 0.96281365425581553
        x[92] = 0.970785775763706332
        x[93] = 0.977809358486918289
        x[94] = 0.983877540706057016
        x[95] = 0.98898439524299175
        x[96] = 0.99312493703744346
        x[97] = 0.99629513473312515
        x[98] = 0.998491950639595818
        x[99] = 0.99971372677344123

        rx = rcut*0.5*(x + 1)

        # Calculate the value of the orthonormalized polynomial basis at the rx
        # values
        fs = np.zeros([nmax, len(x)])
        for n in range(1, nmax+1):
            fs[n-1, :] = (rcut-np.clip(rx, 0, rcut))**(n+2)

        gss = np.dot(betas, fs)

        return rx, gss
