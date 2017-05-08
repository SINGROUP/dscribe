"""Used to store data related to atomic elements. Also has some conversion
utilities related to the data.
"""
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

"""List of atomic names in order."""
NUMBER_TO_NAME_MAP = [
    None,   'Hydrogen',  'Helium', 'Lithium', 'Beryllium',
    'Boron',  'Carbon',  'Nitrogen',  'Oxygen',  'Fluorine',
    'Neon', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon',
    'Phosphorus',  'Sulfur',  'Chlorine', 'Argon', 'Potassium',
    'Calcium', 'Scandium', 'Titanium', 'Vanadium',  'Chromium',
    'Manganese', 'Iron', 'Cobalt', 'Nickel', 'Copper',
    'Zinc', 'Gallium', 'Germanium', 'Arsenic', 'Selenium',
    'Bromine', 'Krypton', 'Rubidium', 'Strontium', 'Yttrium',
    'Zirconium', 'Niobium', 'Molybdenum', 'Technetium', 'Ruthenium',
    'Rhodium', 'Palladium', 'Silver', 'Cadmium', 'Indium',
    'Tin', 'Antimony', 'Tellurium', 'Iodine',  'Xenon',
    'Cesium', 'Barium', 'Lanthanum', 'Cerium', 'Praseodymium',
    'Neodymium', 'Promethium', 'Samarium', 'Europium', 'Gadolinium',
    'Terbium', 'Dysprosium', 'Holmium', 'Erbium', 'Thulium',
    'Ytterbium', 'Lutetium', 'Hafnium', 'Tantalum', 'Tungsten',
    'Rhenium', 'Osmium', 'Iridium', 'Platinum', 'Gold',
    'Mercury', 'Thallium', 'Lead', 'Bismuth', 'Polonium',
    'Astatine', 'Radon', 'Francium', 'Radium', 'Actinium',
    'Thorium', 'Protactinium', 'Uranium',  'Neptunium', 'Plutonium',
    'Americium', 'Curium', 'Berkelium', 'Californium', 'Einsteinium',
    'Fermium', 'Mendelevium', 'Nobelium', 'Lawrencium'
]


def numbers_to_symbols(numbers):
    """Given atomic number(s), returns the symbol(s).

    Args:
        numbers (int or list of int): Atomic numbers(s) (number of protons).

    Returns:
        ndarray: Atomic symbol(s).

    Raises:
        ValueError: If a given atomic number is invalid and doesn't have a
        corresponding symbol.
    """
    error_msg = (
        "Given atomic number {} is invalid and doesn't have a symbol "
        "associated with it."
    )
    single_value = False
    if isinstance(numbers, (int, np.int32, np.int64)):
        numbers = [numbers]
        single_value = True
    symbols = []
    for number in numbers:
        try:
            symbol = NUMBER_TO_SYMBOL_MAP[number]
        except IndexError:
            raise ValueError(error_msg.format(number))
        if symbol is None:
            raise ValueError(error_msg.format(number))
        symbols.append(symbol)
    return symbols[0] if single_value else np.array(symbols)


def symbols_to_numbers(symbols):
    """Given element symbol(s), return the atomic number(s) (number of protons).

    Args:
        symbols (str or list of str): Atomic symbol(s).

    Returns:
        ndarray: Atomic number(s) (number of protons).

    Raises:
        ValueError: If a given atomic symbol is invalid and doesn't have a
        corresponding number.
    """
    single_value = False
    if isinstance(symbols, (str, np.str_)):
        symbols = [symbols]
        single_value = True
    numbers = []
    for symbol in symbols:
        number = SYMBOL_TO_NUMBER_MAP.get(symbol)
        if number is None:
            raise ValueError(
                "Given atomic symbol {} is invalid and doesn't have a number "
                "associated with it.".format(symbol)
            )
        numbers.append(number)
    return numbers[0] if single_value else np.array(numbers)
