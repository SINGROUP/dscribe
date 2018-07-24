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
        """
        super().__init__(flatten=False)
        self.atomic_numbers = atomic_numbers
        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self.periodic = periodic
        self.crossover = crossover
        self.alphas, self.betas = soaplite.genBasis.getBasisFunc(rcut, nmax)

    def describe(self, system, positions=[]):
        """Return the SOAP spectrum for the given system.

        Args:
            system (System): The system for which the descriptor is
                             created.
            positions (iterable): positions or atom index of points, from
                                  which soap is created

        Returns:
            np.ndarray: The SOAP spectrum for the given positions.
        """
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
                raise ValueError("create method requires the argument positions,"
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
        return np.prod(self.get_shape())

    def get_shape(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
        if len(self.atomic_numbers) == 0:
            return None
        if self.envPos is None:
            if self.crossover is False:
                return np.array([(self.lmax+1)*(self.nmax*(self.nmax+1))/2*len(self.atomic_numbers), len(self.atomic_numbers)])
            elif self.crossover is True:
                return np.array([(self.lmax+1)*(self.nmax*(self.nmax+1))/2*(len(self.atomic_numbers)*(len(self.atomic_numbers) + 1))/2, len(self.atomic_numbers)])
        elif isinstance(self.envPos, int):  # gives index of atom (from zero)
            if self.crossover is False:
                return np.array([(self.lmax+1)*(self.nmax*(self.nmax+1))/2*len(self.atomic_numbers), 1])
            elif self.crossover is True:
                return np.array([(self.lmax+1)*(self.nmax*(self.nmax+1))/2*(len(self.atomic_numbers)*(len(self.atomic_numbers) + 1))/2, 1])
        else:
            if self.crossover is False:
                return np.array([(self.lmax+1)*(self.nmax*(self.nmax+1))/2*len(self.atomic_numbers), len(self.envPos)])
            elif self.crossover is True:
                return np.array([(self.lmax+1)*(self.nmax*(self.nmax+1))/2*(len(self.atomic_numbers)*(len(self.atomic_numbers) + 1))/2, len(self.envPos)])
