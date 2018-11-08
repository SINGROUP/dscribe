from ase import Atoms
from dscribe.core import System
import numpy as np


def system_stats(system_iterator):
    """
    Args:
        system_stats(iterable containing ASE.Atoms or System): The atomic
            systems for which to gather statistics.

    Returns:
        Dict: A dictionary of different statistics for the system. The
        dictionary will contain:

            n_atoms_max: The maximum number of atoms in a system.
            max_atomic_number: The highest atomic number
            min_atomic_number: The lowest atomic number
            atomic_numbers: List of present atomic numbers
            element_symbols: List of present atomic symbols
            min_distance: Minimum distance in the system
    """
    n_atoms_max = 0
    atomic_numbers = set()
    symbols = set()
    min_distance = None

    for system in system_iterator:
        n_atoms = len(system)

        # Make ASE.Atoms into a System object
        if isinstance(system, Atoms):
            system = System.from_atoms(system)

        i_atomic_numbers = set(system.get_atomic_numbers())
        i_symbols = set(system.get_chemical_symbols())
        distance_matrix = system.get_distance_matrix()

        # Gather atomic numbers and symbols
        atomic_numbers = atomic_numbers.union(i_atomic_numbers)
        symbols = symbols.union(i_symbols)

        # Gather maximum number of atoms
        if n_atoms > n_atoms_max:
            n_atoms_max = n_atoms

        # Gather min distance. For periodic systems we must also consider
        # distances from an atom to it's periodic copy, as given by
        # get_distance_matrix() on the diagonal.
        if np.any(system.get_pbc()):
            triu_indices = np.triu_indices(len(distance_matrix), k=0)
        else:
            triu_indices = np.triu_indices(len(distance_matrix), k=1)
        distances = distance_matrix[triu_indices]
        i_min_dist = distances.min()

        if min_distance is None or i_min_dist < min_distance:
            min_distance = i_min_dist

    return {
        "n_atoms_max": n_atoms_max,
        "max_atomic_number": max(list(atomic_numbers)),
        "min_atomic_number": min(list(atomic_numbers)),
        "atomic_numbers": list(atomic_numbers),
        "element_symbols": list(symbols),
        "min_distance": min_distance,
    }
