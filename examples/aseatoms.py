"""
Demonstrates the use of the ASE library to load and create atomic
configurations that can be used in the describe package.
"""
import ase.io

from describe.descriptors import MBTR
from describe.descriptors import CoulombMatrix
from describe.descriptors import SineMatrix

from describe.utils import atoms_stats

# Load configuration from an XYZ file with ASE. See
# "https://wiki.fysik.dtu.dk/ase/ase/io/io.html" for a list of supported file
# formats.
atoms = ase.io.read("nacl.xyz")
atoms.set_cell([5.640200, 5.640200, 5.640200])
atoms.set_initial_charges(atoms.get_atomic_numbers())

# There are utilities for automatically detecting statistics for ASE Atoms
# objects. Typically some statistics are needed for the descriptors in order to
# e.g. define a proper zero-padding
stats = atoms_stats([atoms])
max_n_atoms = stats["max_n_atoms"]
atomic_numbers = stats["atomic_numbers"]

# Create descriptors for this system directly from the ASE atoms
cm = CoulombMatrix(max_n_atoms).create(atoms)
sm = SineMatrix(max_n_atoms).create(atoms)
mbtr = MBTR(atomic_numbers, k=3, periodic=True, weighting="exponential").create(atoms)
