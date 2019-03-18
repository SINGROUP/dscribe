from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass

import numpy as np

from ase import Atoms
from dscribe.core.system import System
from dscribe.utils.species import get_atomic_numbers


class Descriptor(with_metaclass(ABCMeta)):
    """An abstract base class for all descriptors.
    """

    def __init__(self, flatten, sparse):
        """
        Args:
            flatten (bool): Whether the output of create() should be flattened
                to a 1D array.
        """
        self._sparse = sparse
        self._flatten = flatten
        self._atomic_numbers = None
        self._atomic_number_set = None
        self._species = None

    @abstractmethod
    def create(self, system, *args, **kwargs):
        """Creates the descriptor for the given systems.

        Args:
            system (ase.Atoms): The system for which to create the descriptor.
            args: Descriptor specific positional arguments.
            kwargs: Descriptor specific keyword arguments.

        Returns:
            np.array | scipy.sparse.coo_matrix: A descriptor for the system.
        """

    @abstractmethod
    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """

    def get_system(self, system):
        """Used to convert the given atomic system into a custom System-object
        that is used internally. The System class inherits from ase.Atoms, but
        includes built-in caching for geometric quantities that may be re-used
        by the descriptors.

        Args:
            system (:class:`ase.Atoms` | :class:`.System`): Input system.

        Returns:
            :class:`.System`: The given system transformed into a corresponding
                System-object.
        """
        if isinstance(system, Atoms):
            if type(system) == System:
                return system
            else:
                return System.from_atoms(system)
        else:
            raise ValueError(
                "Invalid system with type: '{}'.".format(type(system))
            )

    def _set_species(self, species):
        """Used to setup the species information for this descriptor. This
        information includes an ordered list of unique atomic numbers, a set
        of atomic numbers and the original variable contents.

        Args:
            species(iterable): Chemical species either as a list of atomic
                numbers or list of chemical symbols.
        """
        # The species are stored as atomic numbers for internal use.
        atomic_numbers = get_atomic_numbers(species)
        self._atomic_numbers = atomic_numbers
        self._atomic_number_set = set(self._atomic_numbers)
        self._species = species

    def get_species_definition(self, species, atomic_numbers):
        """A convenience function that is included to decide the correct source
        of species information.

        Args:
            species(iterable): Species information either as atomic numbers or
                chemical symbols.
            atomic_numbers(iterable): Species information as atomic numbers

        Returns:
            The correct variable for the species information.

        Raises:
            ValueError: If both or none of the species information is defined.
        """
        # First check that the chemical species are defined either as number or
        # symbols
        if atomic_numbers is None and species is None:
            raise ValueError(
                "Please provide the atomic species either as chemical symbols "
                "or as atomic numbers."
            )
        elif atomic_numbers is not None and species is not None:
            raise ValueError(
                "Both species and atomic numbers provided. Please only provide"
                "either one."
            )

        if atomic_numbers is not None:
            return atomic_numbers
        else:
            return species

    def check_atomic_numbers(self, atomic_numbers):
        """Used to check that the given atomic numbers have been defined for
        this descriptor.

        Args:
            species(iterable): Atomic numbers to check.

        Raises:
            ValueError: If the atomic numbers in the given system are not
            included in the species given to this descriptor.
        """
        # Check that the system does not have elements that are not in the list
        # of atomic numbers
        zs = set(atomic_numbers)
        if not zs.issubset(self._atomic_number_set):
            raise ValueError(
                "The given system has the following atomic numbers not defined "
                "for this descriptor: {}"
                .format(zs.difference(self._atomic_number_set))
            )
