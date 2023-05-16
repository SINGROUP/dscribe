import numpy as np
from dscribe.descriptors import MBTR

# Setup
mbtr = MBTR(
    species=["H", "O"],
    geometry={"function": "inverse_distance"},
    grid={"min": 0, "max": 1, "n": 100, "sigma": 0.1},
    weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
    periodic=False,
    normalization="l2",
)

# Create
from ase.build import molecule

water = molecule("H2O")

# Create MBTR output for the system
mbtr_water = mbtr.create(water)

print(mbtr_water)
print(mbtr_water.shape)

# Locations
# The locations of specific element combinations can be retrieved like this.
ho_loc = mbtr.get_location(("H", "O"))

# These locations can be directly used to slice the corresponding part from an
# MBTR output for e.g. plotting.
mbtr_water[ho_loc]

# Visualization
from ase.build import bulk
import matplotlib.pyplot as mpl

nacl = bulk("NaCl", "rocksalt", a=5.64)
decay = 0.5
mbtr = MBTR(
    species=["Na", "Cl"],
    geometry={"function": "inverse_distance"},
    grid={"min": 0, "max": 0.5, "sigma": 0.01, "n": 200},
    weighting={"function": "exp", "scale": decay, "threshold": 1e-3},
    periodic=True,
    sparse=False
)
mbtr_output = mbtr.create(nacl)

# Create the mapping between an index in the output and the corresponding
# chemical symbol
n_elements = len(mbtr.species)
x = np.linspace(0, 0.5, 200)

# Plot k=2
fig, ax = mpl.subplots()
for i in range(n_elements):
    for j in range(n_elements):
        if j >= i:
            i_species = mbtr.species[i]
            j_species = mbtr.species[j]
            loc = mbtr.get_location((i_species, j_species))
            mpl.plot(x, mbtr_output[loc], label="{}-{}".format(i_species, j_species))
ax.set_xlabel("Inverse distance (1/angstrom)")
ax.legend()
mpl.show()

# Finite
desc = MBTR(
    species=["C"],
    geometry={"function": "distance"},
    grid={"min": 0.4, "max": 8, "sigma": 0.1, "n": 200},
    periodic=False,
    sparse=False,
    normalization="l2",
)

system = molecule("C60")

# No weighting
output_no_weight = desc.create(system)

# Exponential weighting
desc.weighting = {"function": "exp", "scale": 1.1, "threshold": 1e-2}
output_weight = desc.create(system)

fig, ax = mpl.subplots()
x = np.linspace(0.3, 10, 200)
ax.set_xlabel("Distance angstrom")
ax.plot(x, output_no_weight, label="No weighting")
ax.plot(x, output_weight, label="Exponential weighting")
ax.legend()
mpl.show()

# Geometry functions

# Periodic
desc = MBTR(
    species=["H"],
    periodic=True,
    geometry={"function": "inverse_distance"},
    grid={"min": 0, "max": 1.0, "sigma": 0.02, "n": 200},
    weighting={"function": "exp", "scale": 1.0, "threshold": 1e-3},
    sparse=False,
)

a1 = bulk('H', 'fcc', a=2.0)
output = desc.create(a1)

# Supercells
a1 = bulk('H', 'fcc', a=2.0)                     # Primitive
a2 = a1*[2, 2, 2]                                # Supercell
a3 = bulk('H', 'fcc', a=2.0, orthorhombic=True)  # Orthorhombic
a4 = bulk('H', 'fcc', a=2.0, cubic=True)         # Conventional cubic

# Without normalization the shape of the output is the same, but the intensity
# is not
desc.normalization = "none"

output = desc.create([a1, a2, a3, a4])
fig, ax = mpl.subplots()
ax.plot(output[0, :], label="Primitive cell")
ax.plot(output[1, :], label="2x2 supercell")
ax.plot(output[2, :], label="Orthorhombic cell")
ax.plot(output[3, :], label="Conventional cubic cell")
ax.legend()
mpl.show()

# With normalization to unit euclidean length the outputs become identical
desc.normalization = "l2"
output = desc.create([a1, a2, a3, a4])
fig, ax = mpl.subplots()
ax.plot(output[0, :], label="Primitive cell")
ax.plot(output[1, :], label="2x2 supercell")
ax.plot(output[2, :], label="Orthorhombic cell")
ax.plot(output[3, :], label="Conventional cubic cell")
ax.legend()
mpl.show()
