from dscribe.descriptors import CoulombMatrix

atomic_numbers = [1, 8]
rcut = 6.0
nmax = 8
lmax = 6

# Setting up the CM descriptor
cm = CoulombMatrix(
    n_atoms_max=6,
)

# Creating an atomic system as an ase.Atoms-object
from ase.build import molecule
methanol = molecule("CH3OH")
print(methanol)

# Create CM output for the system
cm_methanol = cm.create(methanol)

print(cm_methanol)
print("flattened", cm_methanol.shape)

# No flattening
cm = CoulombMatrix(
    n_atoms_max=6, flatten = False
)
cm_methanol = cm.create(methanol)

print(cm_methanol)
print("not flattened", cm_methanol.shape)


# Introduce zero-padding
cm = CoulombMatrix(
    n_atoms_max=10, flatten = False
)
cm_methanol = cm.create(methanol)

print("zero-padded", cm_methanol)
print(cm_methanol.shape)

# Not meant for periodic systems
methanol.set_pbc([True, True, True])
methanol.set_cell([[10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0],
    [0.0, 0.0, 10.0],
    ])

cm = CoulombMatrix(
    n_atoms_max=6, flatten = False
)

cm_methanol = cm.create(methanol)
print("with pbc", cm_methanol)

## Sparse output
# no example

# Translation, rotation and permutation
cm = CoulombMatrix(
    n_atoms_max=6, flatten = False
)
print("invariance with respect to translation, rotation and permutation")
# translate
methanol.translate((5, 7, 9))
cm_methanol = cm.create(methanol)
print(cm_methanol)

#rotate
methanol.rotate(90, 'z', center=(0, 0, 0))
cm_methanol = cm.create(methanol)
print(cm_methanol)

#permutate
upside_down_methanol = methanol[::-1]
cm_methanol = cm.create(upside_down_methanol)
print(cm_methanol)

# Options for permutation

# No Sorting
cm = CoulombMatrix(
    n_atoms_max=6, flatten = False, 
    permutation='none')

cm_methanol = cm.create(methanol)
print(methanol.get_chemical_symbols())
print("in order of appearance", cm_methanol)

# Sort by L2-norm from high to low
cm = CoulombMatrix(
    n_atoms_max=6, flatten = False, 
    permutation='sorted_l2')

cm_methanol = cm.create(methanol)
print("default: sorted by L2-norm", cm_methanol)

# Smoothening by randomly sorting
cm = CoulombMatrix(
    n_atoms_max=6, flatten = False,
    permutation='random', 
    sigma=70, 
    seed = None)

cm_methanol = cm.create(methanol)
print("randomly sorted", cm_methanol)

# Denser descriptor Eigenvector
cm = CoulombMatrix(
    n_atoms_max=6, flatten = False, 
    permutation='eigenspectrum')

cm_methanol = cm.create(methanol)
print("eigenvalues", cm_methanol)


# Creating multiple descriptors in parallel
from dscribe.utils import batch_create
water = molecule("H2O")

molecule_lst = [water, methanol]

cm = CoulombMatrix(
    n_atoms_max=6,
)
batch = batch_create(cm, molecule_lst, n_proc=1)

print(batch.shape)
