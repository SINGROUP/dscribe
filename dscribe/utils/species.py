import numpy as np

SYMBOL_TO_NUMBER_MAP = {
    'H': 1,  'He': 2, 'Li': 3, 'Be': 4,
    'B': 5,  'C': 6,  'N': 7,  'O': 8,  'F': 9,
    'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14,
    'P': 15,  'S': 16,  'Cl': 17, 'Ar': 18, 'K': 19,
    'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23,  'Cr': 24,
    'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
    'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39,
    'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44,
    'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49,
    'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53,  'Xe': 54,
    'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59,
    'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69,
    'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79,
    'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84,
    'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89,
    'Th': 90, 'Pa': 91, 'U': 92,  'Np': 93, 'Pu': 94,
    'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99,
    'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103,
}

NUMBER_TO_SYMBOL_MAP = [
    None, 'H',  'He', 'Li', 'Be',
    'B',  'C',  'N',  'O',  'F',
    'Ne', 'Na', 'Mg', 'Al', 'Si',
    'P',  'S',  'Cl', 'Ar', 'K',
    'Ca', 'Sc', 'Ti', 'V',  'Cr',
    'Mn', 'Fe', 'Co', 'Ni', 'Cu',
    'Zn', 'Ga', 'Ge', 'As', 'Se',
    'Br', 'Kr', 'Rb', 'Sr', 'Y',
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
    'Rh', 'Pd', 'Ag', 'Cd', 'In',
    'Sn', 'Sb', 'Te', 'I',  'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr',
    'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm',
    'Yb', 'Lu', 'Hf', 'Ta', 'W',
    'Re', 'Os', 'Ir', 'Pt', 'Au',
    'Hg', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn', 'Fr', 'Ra', 'Ac',
    'Th', 'Pa', 'U',  'Np', 'Pu',
    'Am', 'Cm', 'Bk', 'Cf', 'Es',
    'Fm', 'Md', 'No', 'Lr'
]


def symbols_to_numbers(symbols):
    """Transforms a set of chemical symbols into the corresponding atomic
    numbers.

    Args:
        symbols(iterable): List of chemical symbols.

    Returns:
        np.ndarray: Atomic numbers as an array of integers.
    """
    numbers = []

    for symbol in symbols:
        number = SYMBOL_TO_NUMBER_MAP.get(symbol)
        if number is None:
            raise ValueError(
                "Given chemical symbol {} is invalid and doesn't have an atomic"
                "number associated with it.".format(symbol)
            )
        numbers.append(number)

    return np.array(numbers, dtype=int)


def get_atomic_numbers(species):
    """Given a list of chemical species either as a atomic numbers or chemcal
    symbols, return the correponding list of ordered atomic numbers with
    duplicates removed.

    Args:
        species(iterable of ints or strings):

    Returns:
        np.ndarray: list of atomic numbers as an integer array.
    """
    # Check that an iterable is given
    is_iterable = hasattr(species, '__iter__')
    is_string = isinstance(species, str)
    if not is_iterable or is_string:
        raise ValueError(
            "Please provide the species as an iterable, e.g. a list."
        )

    # Determine if the given species are atomic numbers or chemical symbols
    if all(isinstance(x, int) for x in species):
        atomic_numbers = species
    elif all(isinstance(x, str) for x in species):
        atomic_numbers = symbols_to_numbers(species)
    else:
        raise ValueError(
            "The given list of species does not seem to contain strictly "
            "chemical symbols or atomic numbers, but a mixture. Please use only"
            " either one."
        )

    # Return species as atomic numbers with possible duplicates removed
    new_atomic_numbers = sorted(list(set(atomic_numbers)))

    return new_atomic_numbers

    # First check that the chemical species are defined either as number or
    # symbols, not both
    # if atomic_numbers is None and species is None:
        # raise ValueError(
            # "Please provide the atomic species either as chemical symbols "
            # "or as atomic numbers."
        # )
    # elif atomic_numbers is not None and species is not None:
        # raise ValueError(
            # "Both species and atomic numbers provided. Please only provide"
            # "either one."
        # )

    # Handle atomic numbers
    # if atomic_numbers is not None:

        # # Check that an iterable is given
        # is_iterable = hasattr(value, '__iter__')
        # if not is_iterable:
            # raise ValueError(
                # "Please provide the atomic numbers as an iterable, e.g. a list."
            # )

    # # Handle chemical symbols
    # else:
        # # Check that an iterable is given
        # is_iterable = hasattr(value, '__iter__')
        # if not is_iterable:
            # raise ValueError(
                # "Please provide the atomic numbers as an iterable, e.g. a list."
            # )
        # # If species given, determine if they are atomic numbers or chemical
        # # symbols
        # if all(isinstance(x, int) for x in species):
            # atomic_numbers = species
        # elif all(isinstance(x, str) for x in species):
            # atomic_numbers = symbols_to_numbers(species)

    # # Save species as atomic numbers internally
    # new_atomic_numbers = list(set(atomic_numbers))
    # if (np.array(new_atomic_numbers) <= 0).any():
        # raise ValueError(
            # "Non-positive atomic numbers not allowed."
        # )
