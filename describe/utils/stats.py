def atoms_stats(atoms_iterator):
    """
    Args:
    Returns:
    """
    max_n_atoms = 0
    atomic_numbers = set()

    for atoms in atoms_iterator:
        n_atoms = len(atoms)
        i_atomic_numbers = set(atoms.get_atomic_numbers())
        atomic_numbers = atomic_numbers.union(i_atomic_numbers)
        if n_atoms > max_n_atoms:
            max_n_atoms = n_atoms

    return {
        "max_n_atoms": max_n_atoms,
        "atomic_numbers": list(atomic_numbers),
    }
