"""
Demonstrates the use of the ASE library to load and create atomic
configurations that can be used in the dscribe package.
"""
import ase.io

from dscribe.descriptors import MBTR
from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import SineMatrix

from dscribe.utils.stats import system_stats

# Load configuration from an XYZ file with ASE. See
# "https://wiki.fysik.dtu.dk/ase/ase/io/io.html" for a list of supported file
# formats.
atoms = ase.io.read("nacl.xyz")
atoms.set_cell([5.640200, 5.640200, 5.640200])
atoms.set_initial_charges(atoms.get_atomic_numbers())

# There are utilities for automatically detecting statistics for ASE Atoms
# objects. Typically some statistics are needed for the descriptors in order to
# e.g. define a proper zero-padding
stats = system_stats([atoms])
n_atoms_max = stats["n_atoms_max"]
atomic_numbers = stats["atomic_numbers"]

print(stats)

# Create descriptors for this system directly from the ASE atoms
cm = CoulombMatrix(n_atoms_max, permutation="sorted_l2").create(atoms)
sm = SineMatrix(n_atoms_max, permutation="sorted_l2").create(atoms)
mbtr = MBTR(
    species=atomic_numbers,
    geometry={"function": "inverse_distance"},
    grid={"min": 0.1, "max": 2, "sigma": 0.1, "n": 50},
    weighting={"function": "exp", "scale": 0.75, "threshold": 1e-2},
    periodic=True,
).create(atoms)
