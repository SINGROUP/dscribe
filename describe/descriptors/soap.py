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
            periodic,
            envPos=None,
            crossover=True,
            all_atomtypes=[]
            ):
        """
        """
        super().__init__(flatten=False)
        self.atomic_numbers = atomic_numbers
        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self.periodic = periodic
        self.envPos = envPos
        self.crossover = crossover
        self.myAlphas, self.myBetas = soaplite.genBasis.getBasisFunc(rcut, nmax)
        self.all_atomtypes = all_atomtypes

    def describe(self, system):
        """Return the SOAP spectrum for the given system.

        Args:
            system (System): The system for which the descriptor is created.

        Returns:
            np.ndarray: The SOAP spectrum for the given system.
        """
        rcut = self.rcut
        nmax = self.nmax
        lmax = self.lmax
        periodic = self.periodic
        envPos = self.envPos
        crossover = self.crossover
        myAlphas, myBetas = self.myAlphas, self.myBetas
        all_atomtypes = self.all_atomtypes
        if envPos is None:
            if periodic is False:
                soap_mat = soaplite.get_soap_structure(system, myAlphas, myBetas, rCut=rcut, NradBas=nmax, Lmax=lmax, crossOver=crossover, all_atomtypes=all_atomtypes)  # OKAY
            else:
                soap_mat = soaplite.get_periodic_soap_structure(system, myAlphas, myBetas, rCut=rcut, NradBas=nmax, Lmax=lmax, crossOver=crossover, all_atomtypes=all_atomtypes)  # OKAY
        elif isinstance(envPos, int):  # gives index of atom (from zero)
            if periodic is False:
                soap_mat = soaplite.get_soap_locals(system, [system.get_positions()[envPos]], myAlphas, myBetas, rCut=rcut, NradBas=nmax, Lmax=lmax, crossOver=crossover, all_atomtypes=all_atomtypes)
            else:
                soap_mat = soaplite.get_periodic_soap_locals(system, [system.get_positions()[envPos]], myAlphas, myBetas, rCut=rcut, NradBas=nmax, Lmax=lmax, crossOver=crossover, all_atomtypes=all_atomtypes)
        else:
            if periodic is False:
                soap_mat = soaplite.get_soap_locals(system, envPos, myAlphas, myBetas, rCut=rcut, NradBas=nmax, Lmax=lmax, crossOver=crossover, all_atomtypes=all_atomtypes)  # OKAY
            else:
                soap_mat = soaplite.get_periodic_soap_locals(system, envPos, myAlphas, myBetas, rCut=rcut, NradBas=nmax, Lmax=lmax, crossOver=crossover, all_atomtypes=all_atomtypes)  # OKAY

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
