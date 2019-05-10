import numpy as np
from dscribe.descriptors import MBTR

# Setup
mbtr = MBTR(
    species=atomic_numbers,
    k2={
        "geometry": {"function": "inverse_distances", "min": 0, "max": 1, "n": n, "sigma": 0.1},
        "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
    },
    periodic=False,
)

# Create
from ase.build import molecule

water = molecule("H2O")

# Create MBTR output for the system
mbtr_water = mbtr.create(water)

print(mbtr_water)
print(mbtr_water.shape)

# Visualization
import ase.data
from ase.build import bulk
import matplotlib.pyplot as mpl

# The MBTR-object is configured with flatten=False so that we can easily
# visualize the different terms.
nacl = bulk("NaCl", "rocksalt", a=5.64)
decay = 0.5
mbtr = MBTR(
    species=["Na", "Cl"],
    k=[2],
    periodic=True,
    grid={
        "k2": {"min": 0, "max": 0.5, "sigma": 0.01, "n": 200},
    },
    weighting={
        "k2": {"function": "exponential", "scale": decay, "cutoff": 1e-3},
    },
    flatten=False,
    sparse=False
)
mbtr_output = mbtr.create(nacl)

# Create the mapping between an index in the output and the corresponding
# chemical symbol
n_elements = len(mbtr.species)
imap = mbtr.index_to_atomic_number
x = np.linspace(0, 0.5, 200)
smap = {index: ase.data.chemical_symbols[number] for index, number in imap.items()}

# Plot k=2
fig, ax = mpl.subplots()
for i in range(n_elements):
    for j in range(n_elements):
        if j >= i:
            mpl.plot(x, mbtr_output["k2"][i, j, :], label="{}-{}".format(smap[i], smap[j]))
ax.set_xlabel("Inverse distance (1/angstrom)")
ax.legend()
mpl.show()

# Finite
from ase.visualize import view

desc = MBTR(
    species=["C"],
    k=[2],
    periodic=False,
    grid={
        "k2": {"min": 0, "max": 1.0, "sigma": 0.02, "n": 200},
    },
    flatten=True,
    sparse=False,
    normalization="l2_each",
)

system = molecule("C60")
view(system)

# No weighting
output_no_weight = desc.create(system)

# Exponential weighting
desc.weighting = {
    "k2": {"function": "exponential", "scale": 1.1, "cutoff": 1e-2},
}
output_weight = desc.create(system)

fig, ax = mpl.subplots()
x = np.linspace(0, 1, 200)
ax.set_xlabel("Inverse distance (1/angstrom)")
ax.plot(x, output_no_weight[0, :], label="No weighting")
ax.plot(x, output_weight[0, :], label="Exponential weighting")
ax.legend()
mpl.show()

# Periodic
desc = MBTR(
    species=["H"],
    k=[1, 2, 3],
    periodic=True,
    grid={
        "k1": {"min": 0, "max": 2, "sigma": 0.1, "n": 100},
        "k2": {"min": 0, "max": 1.0, "sigma": 0.02, "n": 200},
        "k3": {"min": -1.0, "max": 1.0, "sigma": 0.02, "n": 200},
    },
    weighting={
        "k2": {"function": "exponential", "scale": 1, "cutoff": 1e-3},
        "k3": {"function": "exponential", "scale": 1, "cutoff": 1e-3},
    },
    flatten=True,
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
desc.normalization = "l2_each"
output = desc.create([a1, a2, a3, a4])
fig, ax = mpl.subplots()
ax.plot(output[0, :], label="Primitive cell")
ax.plot(output[1, :], label="2x2 supercell")
ax.plot(output[2, :], label="Orthorhombic cell")
ax.plot(output[3, :], label="Conventional cubic cell")
ax.legend()
mpl.show()
