from dscribe.descriptors import MBTR

atomic_numbers = [1, 8]
n = 100

# Setting up the MBTR descriptor
mbtr = MBTR(
    species=atomic_numbers,
    settings={
        "k2": {
            "geometry": {"function": "inverse_distances", "min": 0, "max": 1, "n": n, "sigma": 0.1},
            "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
        },
    },
    periodic=False,
)

# Creating an atomic system as an ase.Atoms-object
from ase.build import molecule
import ase.data

water = molecule("H2O")

# Create MBTR output for the system
mbtr_water = mbtr.create(water)

print(mbtr_water)
print(mbtr_water.shape)

from ase.build import bulk
nacl = bulk("NaCl", "rocksalt", a=5.64)

# Optionally we can preserve the tensorial nature of the data by specifying
# "Flatten" as False. In this case the output will be a list of
# multidimensional numpy arrays for each k-term. This form is easier to
# visualize as done in the following.
import matplotlib.pyplot as plt

decay = 0.5
mbtr = MBTR(
    species=[11, 17],
    k=[1, 2, 3],
    periodic=True,
    grid={
        "k1": {"min": 10, "max": 18, "sigma": 0.1, "n": 200},
        "k2": {"min": 0, "max": 0.7, "sigma": 0.01, "n": 200},
        "k3": {"min": -1.0, "max": 1.0, "sigma": 0.05, "n": 200}
    },
    weighting={
        "k2": {"function": "exponential", "scale": decay, "cutoff": 1e-3},
        "k3": {"function": "exponential", "scale": decay, "cutoff": 1e-3},
    },
    flatten=False,
    sparse=False
)
mbtr_output = mbtr.create(nacl)

# Create the mapping between an index in the output and the corresponding
# chemical symbol
n_elements = len(set(nacl.get_atomic_numbers()))
imap = mbtr.index_to_atomic_number
smap = {}
for index, number in imap.items():
    smap[index] = ase.data.chemical_symbols[number]

# Plot K2
x2 = mbtr._axis_k2
for i in range(n_elements):
    for j in range(n_elements):
        if j >= i:
            plt.plot(x2, mbtr_output["k2"][i, j, :], label="{}-{}".format(smap[i], smap[j]))
plt.ylabel("$\phi$ (a.u.)", size=20)
plt.xlabel("Inverse distance (1/angstrom)", size=20)
plt.legend()
plt.show()
