from dscribe.descriptors import CoulombMatrix

# Setting up the CM descriptor
cm = CoulombMatrix(n_atoms_max=6)

# Creation
from ase.build import molecule

# Molecule created as an ASE.Atoms
methanol = molecule("CH3OH")

# Create CM output for the system
cm_methanol = cm.create(methanol)

print(cm_methanol)

# Create output for multiple system
samples = [molecule("H2O"), molecule("NO2"), molecule("CO2")]
coulomb_matrices = cm.create(samples)            # Serial
coulomb_matrices = cm.create(samples, n_jobs=2)  # Parallel

# Zero-padding
cm = CoulombMatrix(n_atoms_max=10)
cm_methanol = cm.create(methanol)

print("zero-padded", cm_methanol)
print(cm_methanol.shape)

# Not meant for periodic systems
methanol.set_pbc([True, True, True])
methanol.set_cell([[10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0],
    [0.0, 0.0, 10.0],
    ])

cm = CoulombMatrix(n_atoms_max=6)

cm_methanol = cm.create(methanol)
print("with pbc", cm_methanol)

# Invariance
cm = CoulombMatrix(
    n_atoms_max=6,
    permutation="sorted_l2"
)

# Translation
methanol.translate((5, 7, 9))
cm_methanol = cm.create(methanol)
print(cm_methanol)

# Rotation
methanol.rotate(90, 'z', center=(0, 0, 0))
cm_methanol = cm.create(methanol)
print(cm_methanol)

# Permutation
upside_down_methanol = methanol[::-1]
cm_methanol = cm.create(upside_down_methanol)
print(cm_methanol)

# Options for permutation

# No sorting
cm = CoulombMatrix(n_atoms_max=6, permutation='none')

cm_methanol = cm.create(methanol)
print(methanol.get_chemical_symbols())
print("in order of appearance", cm_methanol)

# Sort by Euclidean (L2) norm.
cm = CoulombMatrix(n_atoms_max=6, permutation='sorted_l2')

cm_methanol = cm.create(methanol)
print("default: sorted by L2-norm", cm_methanol)

# Random
cm = CoulombMatrix(
    n_atoms_max=6,
    permutation='random',
    sigma=70,
    seed=None
)

cm_methanol = cm.create(methanol)
print("randomly sorted", cm_methanol)

# Eigenspectrum
cm = CoulombMatrix(
    n_atoms_max=6,
    permutation='eigenspectrum'
)

cm_methanol = cm.create(methanol)
print("eigenvalues", cm_methanol)
