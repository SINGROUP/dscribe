from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass

from ase import Atoms
from dscribe.core.system import System


class Descriptor(with_metaclass(ABCMeta)):
    """An abstract base class for all descriptors.
    """

    def __init__(self, flatten, sparse):
        """
        Args:
            flatten (bool): Whether the output of create() should be flattened
                to a 1D array.
        """
        self.sparse = sparse
        self.flatten = flatten

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
        system_type = type(system)
        if system_type == Atoms:
            return System.from_atoms(system)
        elif system_type == System:
            return system
        else:
            raise ValueError(
                "Invalid system with type: '{}'.".format(system_type)
            )
