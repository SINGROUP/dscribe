import numpy as np
from dscribe.descriptors import ValleOganov

# Setup
vo = ValleOganov(
    species=["H", "O"],
    k2={
        "sigma": 10**(-0.5),
        "n": 100,
        "r_cut": 5
    },
    k3={
        "sigma": 10**(-0.5),
        "n": 100,
        "r_cut": 5
    },
)

# Create
from ase import Atoms
import math
water = Atoms(
    cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
    positions=[
        [0, 0, 0],
        [0.95, 0, 0],
        [
            0.95 * (1 + math.cos(76 / 180 * math.pi)),
            0.95 * math.sin(76 / 180 * math.pi),
            0.0,
        ],
    ],
    symbols=["H", "O", "H"],
)

# Create ValleOganov output for the system
vo_water = vo.create(water)

print(vo_water)
print(vo_water.shape)

# Visualization
import matplotlib.pyplot as plt
import ase.data
from ase.build import bulk

# The ValleOganov-object is configured with flatten=False so that we can easily
# visualize the different terms.
nacl = bulk("NaCl", "rocksalt", a=5.64)
vo = ValleOganov(
    species=["Na", "Cl"],
    k2={
        "n": 200,
        "sigma": 10**(-0.625),
        "r_cut": 10
    },
    flatten=False,
    sparse=False
)
vo_nacl = vo.create(nacl)

# Create the mapping between an index in the output and the corresponding
# chemical symbol
n_elements = len(vo.species)
imap = vo.index_to_atomic_number
x = np.linspace(0, 10, 200)
smap = {index: ase.data.chemical_symbols[number] for index, number in imap.items()}

# Plot k=2
fig, ax = plt.subplots()
legend = []
for i in range(n_elements):
    for j in range(n_elements):
        if j >= i:
            plt.plot(x, vo_nacl["k2"][i, j, :])
            legend.append(f'{smap[i]}-{smap[j]}')
ax.set_xlabel("Distance (angstrom)")
plt.legend(legend)
plt.show()

# MBTR setup for the same structure and descriptor
from dscribe.descriptors import MBTR
mbtr = MBTR(
    species=["Na", "Cl"],
    k2={
        "geometry": {"function": "distance"},
        "grid": {"min": 0, "max": 10, "sigma": 10**(-0.625), "n": 200},
        "weighting": {"function": "inverse_square", "r_cut": 10},
    },
    normalize_gaussians=True,
    normalization="valle_oganov",
    periodic=True,
    flatten=False,
    sparse=False
)

mbtr_nacl = mbtr.create(nacl)

# Comparing to MBTR output
nacl = bulk("NaCl", "rocksalt", a=5.64)

decay = 0.5
mbtr = MBTR(
    species=["Na", "Cl"],
    k2={
        "geometry": {"function": "distance"},
        "grid": {"min": 0, "max": 20, "sigma": 10**(-0.625), "n": 200},
        "weighting": {"function": "exp", "scale": decay, "threshold": 1e-3},
    },
    periodic=True,
    flatten=False,
    sparse=False
)

vo = ValleOganov(
    species=["Na", "Cl"],
    k2={
        "n": 200,
        "sigma": 10**(-0.625),
        "r_cut": 20
    },
    flatten=False,
    sparse=False
)

mbtr_output = mbtr.create(nacl)
vo_output = vo.create(nacl)

n_elements = len(vo.species)
imap = vo.index_to_atomic_number
x = np.linspace(0, 20, 200)
smap = {index: ase.data.chemical_symbols[number] for index, number in imap.items()}

fig, ax = plt.subplots()
legend = []
for key, output in {"MBTR": mbtr_output, "Valle-Oganov": vo_output}.items():
    plt.plot(x, output["k2"][0, 1, :])
    legend.append(f'{key}: {smap[0]}-{smap[1]}')
ax.set_xlabel("Distance (angstrom)")
plt.legend(legend)
plt.show()
