"""
Demonstrates the use of the ASE library to load and create atomic
configurations that can be used in the describe package.
"""
import ase.io

from describe.descriptors import MBTR
from describe.descriptors import CoulombMatrix
from describe.descriptors import SineMatrix

from describe.utils import system_stats

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

# Create descriptors for this system directly from the ASE atoms
cm = CoulombMatrix(n_atoms_max, permutation="sorted_l2").create(atoms)
sm = SineMatrix(n_atoms_max, permutation="sorted_l2").create(atoms)
mbtr = MBTR(atomic_numbers, k=[1, 2, 3], periodic=True, weighting="exponential").create(atoms)
