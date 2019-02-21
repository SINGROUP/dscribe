# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from builtins import super
import numpy as np

from scipy.sparse import coo_matrix

from dscribe.descriptors import Descriptor
import soaplite


class SOAP(Descriptor):
    """Class for the Smooth Overlap of Atomic Orbitals (SOAP) descriptor. This
    implementation uses orthogonalized spherical primitive gaussian type
    orbitals as the radial basis set to reach a fast analytical solution.

    For reference, see:

    "On representing chemical environments, Albert P. Bartók, Risi Kondor, and
    Gábor Csányi, Phys. Rev. B 87, 184115, (2013),
    https://doi.org/10.1103/PhysRevB.87.184115

    "Comparing molecules and solids across structural and alchemical space",
    Sandip De, Albert P. Bartók, Gábor Csányi and Michele Ceriotti, Phys.
    Chem. Chem. Phys. 18, 13754 (2016), https://doi.org/10.1039/c6cp00415f
    """
    def __init__(
            self,
            atomic_numbers,
            rcut,
            nmax,
            lmax,
            sigma=1.0,
            rbf="gto",
            periodic=False,
            crossover=True,
            average=False,
            normalize=False,
            sparse=True
            ):
        """
        Args:
            atomic_numbers (iterable): A list of the atomic numbers that should
                be taken into account in the descriptor. Notice that this is
                not the atomic numbers that are present for an individual
                system, but should contain all the elements that are ever going
                to be encountered when creating the descriptors for a set of
                systems. Keeping the number of handled elements as low as
                possible is preferable.
            periodic (bool): Determines whether the system is considered to be
                periodic.
            rcut (float): A cutoff for local region in angstroms. Should be
                bigger than 1 angstrom.
            nmax (int): The number of basis functions to be used.
            lmax (int): The number of l's to be used. The computational time scales
            sigma (float): The standard deviation of the gaussians used to expand the
                atomic density.
            rbf (str): The radial basis functions to use. The available options are:

                * "gto": Spherical gaussian type orbitals defined as :math:`g_{nl}(r) = \sum_{n'=1}^{n_\mathrm{max}}\,\\beta_{nn'l} r^l e^{-\\alpha_{n'l}r^2}`
                * "polynomial": Polynomial basis defined as :math:`g_{n}(r) = \sum_{n'=1}^{n_\mathrm{max}}\,\\beta_{nn'} (r-r_\mathrm{cut})^{n'+2}`

            crossover (bool): Default True, if crossover of atomic types should
                be included in the power spectrum.
            average (bool): Whether to build an average output for all selected
                positions. Before averaging the outputs for individual atoms are
                normalized.
            normalize (bool): Whether to normalize the final output.
            sparse (bool): Whether the output should be a sparse matrix or a
                dense numpy array.
        """
        super().__init__(flatten=True, sparse=sparse)

        # Check that atomic numbers are valid
        self._atomic_number_set = set(atomic_numbers)
        self._atomic_numbers = np.sort(np.array(list(self._atomic_number_set)))
        if (self._atomic_numbers <= 0).any():
            raise ValueError(
                "Non-positive atomic numbers not allowed."
            )

        # Check that sigma is valid
        if (sigma <= 0):
            raise ValueError(
                "Only positive gaussian width parameters 'sigma' are allowed."
            )
        self._eta = 1/(2*sigma**2)

        # Check that rcut is valid
        if rbf == "gto" and rcut <= 1:
            raise ValueError(
                "When using the gaussian radial basis set (gto), the radial "
                "cutoff should be bigger than 1 angstrom."
            )

        supported_rbf = set(("gto", "polynomial"))
        if rbf not in supported_rbf:
            raise ValueError(
                "Invalid radial basis function of type '{}' given. Please use "
                "one of the following: {}".format(rbf, supported_rbf)
            )

        # Crossover cannot be disabled on poly rbf
        if not crossover and rbf == "polynomial":
            raise ValueError(
                "Disabling crossover is not currently supported when using "
                "polynomial radial basis function".format(rbf, supported_rbf)
            )

        self._rcut = rcut
        self._nmax = nmax
        self._lmax = lmax
        self._rbf = rbf
        self._periodic = periodic
        self._crossover = crossover
        self._average = average
        self._normalize = normalize

        if self._rbf == "gto":
            self._alphas, self._betas = soaplite.genBasis.getBasisFunc(self._rcut, self._nmax)

    def create(self, system, positions=None):
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
        zs = set(system.get_atomic_numbers())
        if not zs.issubset(self._atomic_number_set):
            raise ValueError(
                "The given system has the following atomic numbers not defined "
                "in the SOAP constructor: {}"
                .format(zs.difference(self._atomic_number_set))
            )

        sub_elements = np.array(list(set(system.get_atomic_numbers())))

        # Check if periodic is valid
        if self._periodic:
            cell = system.get_cell()
            if np.cross(cell[0], cell[1]).dot(cell[2]) == 0:
                raise ValueError(
                    "System doesn't have cell to justify periodicity."
                )

        # Positions specified, use them
        if positions is not None:

            # Check validity of position definitions and create final cartesian
            # position list
            list_positions = []
            if len(positions) == 0:
                raise ValueError(
                    "The argument 'positions' should contain a non-empty set of"
                    " atomic indices or cartesian coordinates"
                )
            for i in positions:
                if np.issubdtype(type(i), np.integer):
                    list_positions.append(system.get_positions()[i])
                elif isinstance(i, list) or isinstance(i, tuple):
                    list_positions.append(i)
                else:
                    raise ValueError(
                        "Create method requires the argument 'positions', a "
                        "list of atom indices and/or positions"
                    )

            # Determine the SOAPLite function to call based on periodicity and
            # rbf
            if self._rbf == "gto":
                if self._periodic:
                    soap_func = soaplite.get_periodic_soap_locals
                else:
                    soap_func = soaplite.get_soap_locals
                soap_mat = soap_func(
                    system,
                    list_positions,
                    self._alphas,
                    self._betas,
                    rCut=self._rcut,
                    nMax=self._nmax,
                    Lmax=self._lmax,
                    crossOver=self._crossover,
                    all_atomtypes=sub_elements.tolist(),
                    eta=self._eta
                )
            elif self._rbf == "polynomial":
                if self._periodic:
                    soap_func = soaplite.get_periodic_soap_locals_poly
                else:
                    soap_func = soaplite.get_soap_locals_poly
                soap_mat = soap_func(
                    system,
                    list_positions,
                    rCut=self._rcut,
                    nMax=self._nmax,
                    Lmax=self._lmax,
                    all_atomtypes=sub_elements.tolist(),
                    eta=self._eta
                )

        # No positions given, calculate SOAP for all atoms in the structure
        else:
            # Determine the SOAPLite function to call based on periodicity and
            # rbf
            if self._rbf == "gto":
                if self._periodic:
                    soap_func = soaplite.get_periodic_soap_structure
                else:
                    soap_func = soaplite.get_soap_structure
                soap_mat = soap_func(
                    system,
                    self._alphas,
                    self._betas,
                    rCut=self._rcut,
                    nMax=self._nmax,
                    Lmax=self._lmax,
                    crossOver=self._crossover,
                    all_atomtypes=sub_elements.tolist(),
                    eta=self._eta
                )
            elif self._rbf == "polynomial":
                if self._periodic:
                    soap_func = soaplite.get_periodic_soap_structure_poly
                else:
                    soap_func = soaplite.get_soap_structure_poly
                soap_mat = soap_func(
                    system,
                    rCut=self._rcut,
                    nMax=self._nmax,
                    Lmax=self._lmax,
                    all_atomtypes=sub_elements.tolist(),
                    eta=self._eta
                )

        # Map the output from subspace of elements to the full space of
        # elements
        soap_mat = self.get_full_space_output(
            soap_mat,
            sub_elements,
            self._atomic_numbers
        )

        # Create the averaged SOAP output if requested. The individual terms are
        # normalized first.
        if self._average:
            soap_mat = soap_mat / np.linalg.norm(soap_mat, axis=1)[:, None]
            soap_mat = soap_mat.mean(axis=0)
            soap_mat = np.expand_dims(soap_mat, 0)

        # Normalize if requested
        if self._normalize:
            soap_mat = soap_mat / np.linalg.norm(soap_mat, axis=1)[:, np.newaxis]

        # Make into a sparse array if requested
        if self._sparse:
            soap_mat = coo_matrix(soap_mat)

        return soap_mat

    def get_full_space_output(self, sub_output, sub_elements, full_elements_sorted):
        """Used to partition the SOAP output to different locations depending
        on the interacting elements. The SOAPLite implementation can currently
        only handle a limited amount of elements, but this wrapper enables the
        usage of more elements by partitioning the output.
        """
        # Get mapping between elements in the subspace and alements in the full
        # space
        space_map = self.get_sub_to_full_map(sub_elements, full_elements_sorted)

        # Reserve space for a sparse matric containing the full output space
        n_features = self.get_number_of_features()
        n_elem_features = self.get_number_of_element_features()
        n_points = sub_output.shape[0]

        # Define the final output space as an array.
        output = np.zeros((n_points, n_features), dtype=np.float32)

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
        if self._crossover:
            n_blocks = n_elems * (n_elems + 1)/2
        else:
            n_blocks = n_elems

        n_element_features = self.get_number_of_element_features()

        return int(n_element_features * n_blocks)
