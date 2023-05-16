from dscribe.descriptors import ACSF

# Setting up the ACSF descriptor
acsf = ACSF(
    species=["H", "O"],
    r_cut=6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)

# Creating an atomic system as an ase.Atoms-object
from ase.build import molecule

water = molecule("H2O")

# Create MBTR output for the hydrogen atom at index 1
acsf_water = acsf.create(water, centers=[1])

print(acsf_water)
print(acsf_water.shape)
