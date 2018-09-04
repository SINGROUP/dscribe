from __future__ import absolute_import, division, print_function
from builtins import super
import numpy as np

from describe.descriptors import Descriptor
import soaplite


class SOAP(Descriptor):
    """
    """
    def __init__(
            self,
            atomic_numbers,
            rcut,
            nmax,
            lmax,
            periodic=False,
            crossover=True,
            ):
        """
        Args
            atomic_numbers (iterable): A list of the atomic numbers that should
                be taken into account in the descriptor. Notice that this is
                not the atomic numbers that are present for an individual
                system, but should contain all the elements that are ever going
                to be encountered when creating the descriptors for a set of
                systems.  Keeping the number of handled elements as low as
                possible is preferable.
            periodic (bool): Boolean for if the system is periodic or none. If
                this is set to true, you should provide the primitive system as
                input.
            rcut (float):  A cutoff for local region.
            nmax (int): The number of basis to be used for each l.
            lmax (int): The number of l's to be used
            crossover (bool): Default True, if crossover of atoms should be included.
        """
        super().__init__(flatten=False)
        self.atomic_numbers = atomic_numbers
        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self.periodic = periodic
        self.crossover = crossover
        self.update()
        
    def update(self):
        '''
        Updates alphas and betas corresponding to change in rcut or nmax
        '''
        self.alphas, self.betas = soaplite.genBasis.getBasisFunc(self.rcut, self.nmax)

    def describe(self, system, positions=[]):
        """Return the SOAP spectrum for the given system.

        Args:
            system (System): The system for which the descriptor is
                             created.
            positions (list): positions or atom index of points, from
                                  which soap is created

        Returns:
            np.ndarray: The SOAP spectrum for the given positions.
        """
        # ensuring self is updated
        self.update()
        
        # Check if periodic is valid
        if self.periodic:
            cell = system.get_cell()
            if np.cross(cell[0], cell[1]).dot(cell[2]) == 0:
                raise ValueError("System doesn't have cell to justify periodicity.")
        
        # Change function if periodic
        if len(positions):
            if self.periodic:
                soap_func = soaplite.get_periodic_soap_locals
            else:
                soap_func = soaplite.get_soap_locals
        else:
            # No positions given
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
                all_atomtypes=self.atomic_numbers
                )  # OKAY
            return soap_mat
        
        list_positions = []
        
        for i in positions:
            if isinstance(i, int):  # gives index of atom (from zero)
                list_positions.append(system.get_positions()[i])
            elif isinstance(i, list) or isinstance(i, tuple):
                list_positions.append(i)
            else:
                raise ValueError("create method requires the argument 'positions',"
                                 " a list of atom indices and/or positions")
        
        soap_mat = soap_func(
            system,
            list_positions,
            self.alphas,
            self.betas,
            rCut=self.rcut,
            NradBas=self.nmax,
            Lmax=self.lmax,
            crossOver=self.crossover,
            all_atomtypes=self.atomic_numbers
            )  # OKAY

        return soap_mat

    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        if self.crossover:
            n_blocks = len(self.atomic_numbers) * (len(self.atomic_numbers) + 1) \
                / 2
        else:
            n_blocks = len(self.atomic_numbers)
        
        return (self.lmax + 1) * self.nmax * (self.nmax + 1) / 2 * n_blocks

