import math
import scipy

from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core.structure import Structure

from dscribe.descriptors import EwaldSumMatrix

atomic_numbers = [1, 8]
rcut = 6.0
nmax = 8
lmax = 6

# Setting up the Ewald sum matrix descriptor
esm = EwaldSumMatrix(
    n_atoms_max=6,
)

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
ewald_3 = esm.create(nacl, rcut=10, gcut=10)

# Energy

# Ewald summation parameters
rcut = 40
gcut = 40
a = 3

# Calculate Ewald sum matrix with DScribe
ems = EwaldSumMatrix(
    n_atoms_max=3,
    permutation="none",
    flatten=False
)
ems_out = ems.create(al, a=a, rcut=rcut, gcut=gcut)

# Calculate the total electrostatic energy of the crystal
total_energy = ems_out[0, 0] + ems_out[1, 1] + ems_out[0, 1]

# Converts unit of q*q/r into eV
conversion = 1e10 * scipy.constants.e / (4 * math.pi * scipy.constants.epsilon_0)
total_energy *= conversion
print(total_energy)

# Test against implementation in pymatgen
structure = Structure(
    lattice=al.get_cell(),
    species=al.get_atomic_numbers(),
    coords=al.get_scaled_positions(),
)
structure.add_oxidation_state_by_site(al.get_atomic_numbers())
ewald = EwaldSummation(structure, eta=a, real_space_cut=rcut, recip_space_cut=gcut)
energy = ewald.total_energy
print(energy)
