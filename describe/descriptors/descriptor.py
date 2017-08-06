from abc import ABCMeta, abstractmethod
from ase import Atoms
from describe import System


class Descriptor(metaclass=ABCMeta):
    """An abstract base class for all descriptors.
    """

    def __init__(self, flatten=True):
        """
        Args:
            flatten (bool): Whether the output of create() should be flattened
                to a 1D array.
        """
        self.flatten = flatten

    def create(self, system):
        """Creates the descriptor for the given systems.

        Args:
            system (System): The system for which to create the
            descriptor.
        """
        # Ensure that we get a System
        if isinstance(system, Atoms):
            system = System.from_atoms(system)

        return self.describe(system)

    @abstractmethod
    def describe(self, system):
        """Creates the descriptor for the given systems.

        Args:
            system (System): The system for which to create the
            descriptor.

        Returns:
            A descriptor for the system in some numerical form.
        """

    @abstractmethod
    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """
