from __future__ import absolute_import, division, print_function
from builtins import super
import numpy as np

from scipy.sparse import coo_matrix

from dscribe.descriptors import Descriptor
import soaplite


class SOAP(Descriptor):
    """Class for the Smooth Overlap of Atomic Orbitals (SOAP) descriptor.

    For reference, see:
        "On representing chemical environments Albert", Albert P. Bartók, Risi
        Kondor, and Gábor Csányi, Phys. Rev. B 87, 184115, (2013),
        https://doi.org/10.1103/PhysRevB.87.184115

        "Comparing molecules and solids across structural and alchemical
        space", Sandip De, Albert P. Bartók, Gábor Csányi and Michele Ceriotti,
        Phys.  Chem. Chem. Phys. 18, 13754 (2016),
        https://doi.org/10.1039/c6cp00415f
    """
    def __init__(
            self,
            atomic_numbers,
            rcut,
            nmax,
            lmax,
            periodic=False,
            crossover=True,
            average=False,
            normalize=False,
            sparse=True
            ):
        """
        Args
            atomic_numbers (iterable): A list of the atomic numbers that should
                be taken into account in the descriptor. Notice that this is
                not the atomic numbers that are present for an individual
                system, but should contain all the elements that are ever going
                to be encountered when creating the descriptors for a set of
                systems. Keeping the number of handled elements as low as
                possible is preferable.
            periodic (bool): Determines whether the system is considered to be
                periodic.
            rcut (float): A cutoff for local region.
            nmax (int): The number of basis to be used for each l.
            lmax (int): The number of l's to be used.
            crossover (bool): Default True, if crossover of atoms should be included.
            average (bool): Whether to build an average output for all selected
                positions. Before averaging the outputs for individual atoms are
                normalized.
            normalize (bool): Whether to normalize the final output.
            sparse (bool): Whether the output should be a sparse matrix or a
                dense numpy array.
        """
        super().__init__(flatten=True, sparse=sparse)

        # Check that atomic numbers are valid
        self.atomic_numbers = list(set(atomic_numbers))
        if (np.array(atomic_numbers) <= 0).any():
            raise ValueError(
                "Non-positive atomic numbers not allowed."
            )

        # Sort the elements according to atomic number
        self.atomic_numbers = np.sort(np.array(atomic_numbers))

        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self.periodic = periodic
        self.crossover = crossover
        self.average = average
        self.normalize = normalize
        self.update()

    def update(self):
        """Updates alphas and betas corresponding to change in rcut or nmax.
        """
        self.alphas, self.betas = soaplite.genBasis.getBasisFunc(self.rcut, self.nmax)

    def create(self, system, positions=None):
        """Return the SOAP spectrum for the given system.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.
            positions (list): Cartesian positions or atomic indices. If
                specified, the SOAP spectrum will be created for these points. If
                not positions defined, the SOAP spectrum will be created for all
                atoms in the system.

        Returns:
            np.ndarray | scipy.sparse.coo_matrix: The SOAP spectrum for the
            given system and positions. The return type depends on the
            'sparse'-attribute. The first dimension is given by the number of
            positions and the second dimension is determined by the
            get_number_of_features()-function.
        """
        # Transform the input system into the internal System-object
        system = self.get_system(system)

        # Ensuring self is updated
        self.update()
        sub_elements = np.array(list(set(system.get_atomic_numbers())))

        # Check if periodic is valid
        if self.periodic:
            cell = system.get_cell()
            if np.cross(cell[0], cell[1]).dot(cell[2]) == 0:
                raise ValueError(
                    "System doesn't have cell to justify periodicity."
                )

        # Positions specified, use them
        if positions is not None:

            # Change function if periodic
            if self.periodic:
                soap_func = soaplite.get_periodic_soap_locals
            else:
                soap_func = soaplite.get_soap_locals

            # Check validity of position definitions and create final cartesian
            # position list
            list_positions = []
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

            soap_mat = soap_func(
                system,
                list_positions,
                self.alphas,
                self.betas,
                rCut=self.rcut,
                NradBas=self.nmax,
                Lmax=self.lmax,
                crossOver=self.crossover,
                all_atomtypes=sub_elements.tolist()
            )
        # No positions given, calculate SOAP for all atoms in the structure
        else:

            # Change function if periodic
            if self.periodic:
                soap_func = soaplite.get_periodic_soap_structure
            else:
                soap_func = soaplite.get_soap_structure

            soap_mat = soap_func(
                system,
                self.alphas,
                self.betas,
                rCut=self.rcut,
                NradBas=self.nmax,
                Lmax=self.lmax,
                crossOver=self.crossover,
                all_atomtypes=sub_elements.tolist()
            )

        # Map the output from subspace of elements to the full space of
        # elements
        soap_mat = self.get_full_space_output(
            soap_mat,
            sub_elements,
            self.atomic_numbers
        )

        # Create the averaged SOAP output if requested. The individual terms are
        # normalized first.
        if self.average:
            soap_mat = soap_mat / np.linalg.norm(soap_mat, axis=1)[:, None]
            soap_mat = soap_mat.mean(axis=0)
            soap_mat = np.expand_dims(soap_mat, 0)

        # Normalize if requested
        if self.normalize:
            soap_mat = soap_mat / np.linalg.norm(soap_mat, axis=1)

        # Make into a sparse array if requested
        if self.sparse:
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

        # Define the final output space as a sparse matrix.
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
        return int((self.lmax + 1) * self.nmax * (self.nmax + 1)/2)

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
        n_elems = len(self.atomic_numbers)
        if self.crossover:
            n_blocks = n_elems * (n_elems + 1)/2
        else:
            n_blocks = n_elems

        n_element_features = self.get_number_of_element_features()

        return int(n_element_features * n_blocks)
