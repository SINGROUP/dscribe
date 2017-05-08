from abc import ABCMeta, abstractmethod


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

    @abstractmethod
    def create(system):
        """Creates the descriptor for the given systems.

        Args:
            system (System): The system for which to create the
            descriptor.
        """
