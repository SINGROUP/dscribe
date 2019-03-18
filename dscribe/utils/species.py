# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)

import numpy as np
import ase.data


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
        number = ase.data.atomic_numbers.get(symbol)
        if number is None:
            raise ValueError(
                "Given chemical symbol {} is invalid and doesn't have an atomic"
                " number associated with it.".format(symbol)
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
    if all(isinstance(x, (int, np.integer)) for x in species):
        atomic_numbers = species
    elif all(isinstance(x, (str, np.str)) for x in species):
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
