from describe.descriptors import MBTR
from describe.descriptors import CoulombMatrix
from describe.descriptors import SineMatrix
import describe.utils

import ase.io

#===============================================================================
# 1. DEFINING AN ATOMISTIC SYSTEM
#===============================================================================
# Load configuration from an XYZ file with ASE. See
atoms = ase.io.read("nacl.xyz")
atoms.set_cell([5.640200, 5.640200, 5.640200])
atoms.set_initial_charges(atoms.get_atomic_numbers())

#===============================================================================
# 2. CREATING DESCRIPTORS FOR THE SYSTEM
#===============================================================================
# Getting some basic statistics from the processed systems. This information is
# used by the different descriptors for e.g. zero padding.
stats = describe.utils.system_stats([atoms])
n_atoms_max = stats["n_atoms_max"]
atomic_numbers = stats["atomic_numbers"]

# Defining the properties of the descriptors
cm_desc = CoulombMatrix(n_atoms_max=n_atoms_max)
sm_desc = SineMatrix(n_atoms_max=n_atoms_max)
mbtr_desc = MBTR(
    atomic_numbers=atomic_numbers,
    k=[1, 2],
    periodic=True,
    weighting="exponential")

# Creating the descriptors
cm = cm_desc.create(atoms)
sm = sm_desc.create(atoms)
mbtr = mbtr_desc.create(atoms)

# When dealing with multiple systems, create the descriptors in a loop. This
# allows you to control the final output format and also allows you to create
# multiple descriptors from the same system, while using cached intermediate
# results to speed up calculation.
ase_atoms = ase.io.iread("multiple.extxyz", format="extxyz")
for atoms in ase_atoms:
    atoms.set_initial_charges(atoms.get_atomic_numbers())
    cm = cm_desc.create(atoms)
    sm = sm_desc.create(atoms)
    mbtr = mbtr_desc.create(atoms)

#===============================================================================
# 3. USING DESCRIPTORS IN MACHINE LEARNING
#===============================================================================
# The result of the .create() function is a (possibly sparse) 1D vector that
# can now be directly used in various machine-learning libraries.
