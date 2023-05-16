import math
import scipy

from dscribe.descriptors import EwaldSumMatrix

# Setting up the Ewald sum matrix descriptor
esm = EwaldSumMatrix(n_atoms_max=6)

# Creation
from ase.build import bulk

# NaCl crystal created as an ASE.Atoms
nacl = bulk("NaCl", "rocksalt", a=5.64)

# Create output for the system
nacl_ewald = esm.create(nacl)

# Create output for multiple system
al = bulk("Al", "fcc", a=4.046)
fe = bulk("Fe", "bcc", a=2.856)
samples = [nacl, al, fe]
ewald_matrices = esm.create(samples)            # Serial
ewald_matrices = esm.create(samples, n_jobs=2)  # Parallel

# Accuracy

# Easiest way to control accuracy is to use the "accuracy" parameter. Lower
# values mean better accuracy.
ewald_1 = esm.create(nacl, accuracy=1e-3)
ewald_2 = esm.create(nacl, accuracy=1e-5)

# Another option is to directly use the real- and reciprocal space cutoffs.
ewald_3 = esm.create(nacl, r_cut=10, g_cut=10)

# Energy

# Ewald summation parameters
r_cut = 40
g_cut = 40
a = 3

# Calculate Ewald sum matrix with DScribe
ems = EwaldSumMatrix(n_atoms_max=3, permutation="none")
ems_out = ems.create(al, a=a, r_cut=r_cut, g_cut=g_cut)
ems_out = ems.unflatten(ems_out)

# Calculate the total electrostatic energy of the crystal
total_energy = ems_out[0, 0] + ems_out[1, 1] + ems_out[0, 1]

# Converts unit of q*q/r into eV
conversion = 1e10 * scipy.constants.e / (4 * math.pi * scipy.constants.epsilon_0)
total_energy *= conversion
print(total_energy)
