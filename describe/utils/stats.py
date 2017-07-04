from ase import Atoms
from describe.core import System


def system_stats(system_iterator):
    """
    Args:
        system_stats(iterable containing ASE.Atoms or System): The atomic
            systems for which to gather statistics.

    Returns:
        Dict: A dictionary of different statistics for the system. The
        dictionary will contain:

            n_atoms_max: The maximum number of atoms in a system.
            atomic_numbers: The set of atomic numbers that are present in the
                systems.
    """
    n_atoms_max = 0
    atomic_numbers = set()
    symbols = set()

    for atoms in system_iterator:
        n_atoms = len(atoms)
        if isinstance(atoms, Atoms):
            i_atomic_numbers = set(atoms.get_atomic_numbers())
            i_symbols = set(atoms.get_chemical_symbols())
        elif isinstance(atoms, System):
            i_atomic_numbers = set(atoms.numbers)
            i_symbols = set(atoms.symbols)
        else:
            print(atoms)
        atomic_numbers = atomic_numbers.union(i_atomic_numbers)
        symbols = symbols.union(i_symbols)
        if n_atoms > n_atoms_max:
            n_atoms_max = n_atoms

    return {
        "n_atoms_max": n_atoms_max,
        "atomic_numbers": list(atomic_numbers),
        "element_symbols": list(symbols),
    }
