from dscribe.descriptors import MBTR

atomic_numbers = [1, 8]
n = 100

# Setting up the MBTR descriptor
# mbtr = MBTR(
    # species=atomic_numbers,
    # k=2,
    # periodic=False,
    # grid={
        # "k2": {"min": 0, "max": 1, "n": n, "sigma": 0.1}
    # },
    # weighting=None
# )

# # Creating an atomic system as an ase.Atoms-object
# from ase.build import molecule
# import ase.data

# water = molecule("H2O")

# # Create MBTR output for the system
# mbtr_water = mbtr.create(water)

# # print(mbtr_water)
# # print(mbtr_water.shape)

from ase.build import bulk
# nacl = bulk("NaCl", "rocksalt", a=5.64)

# # Optionally we can preserve the tensorial nature of the data by specifying
# # "Flatten" as False. In this case the output will be a list of
# # multidimensional numpy arrays for each k-term. This form is easier to
# # visualize as done in the following.
import matplotlib.pyplot as mpl

# decay = 0.5
# mbtr = MBTR(
    # species=[11, 17],
    # k=[1, 2, 3],
    # periodic=True,
    # grid={
        # "k1": {"min": 10, "max": 18, "sigma": 0.1, "n": 200},
        # "k2": {"min": 0, "max": 0.7, "sigma": 0.01, "n": 200},
        # "k3": {"min": -1.0, "max": 1.0, "sigma": 0.05, "n": 200}
    # },
    # weighting={
        # "k2": {"function": "exponential", "scale": decay, "cutoff": 1e-3},
        # "k3": {"function": "exponential", "scale": decay, "cutoff": 1e-3},
    # },
    # flatten=False,
    # sparse=False
# )
# mbtr_output = mbtr.create(nacl)

# Create the mapping between an index in the output and the corresponding
# chemical symbol
# n_elements = len(set(nacl.get_atomic_numbers()))
# imap = mbtr.index_to_atomic_number
# smap = {}
# for index, number in imap.items():
    # smap[index] = ase.data.chemical_symbols[number]

# # Plot K2
# x2 = mbtr._axis_k2
# for i in range(n_elements):
    # for j in range(n_elements):
        # if j >= i:
            # plt.plot(x2, mbtr_output["k2"][i, j, :], label="{}-{}".format(smap[i], smap[j]))
# plt.ylabel("$\phi$ (a.u.)", size=20)
# plt.xlabel("Inverse distance (1/angstrom)", size=20)
# plt.legend()
# plt.show()

# Periodic
import numpy as np

decay = 1
desc = MBTR(
    species=[1],
    k=[3],
    periodic=True,
    grid={
        "k1": {
            "min": 0,
            "max": 2,
            "sigma": 0.1,
            "n": 100,
        },
        "k2": {
            "min": 0,
            "max": 1.0,
            "sigma": 0.02,
            "n": 200,
        },
        "k3": {
            "min": -1.0,
            "max": 1.0,
            "sigma": 0.02,
            "n": 200,
        },
    },
    weighting={
        "k2": {
            "function": "exponential",
            "scale": decay,
            "cutoff": 1e-3
        },
        "k3": {
            "function": "exponential",
            "scale": decay,
            "cutoff": 1e-3
        },
    },
    flatten=True,
    sparse=False,
)

# Testing that the same crystal, but different unit cells will have a
# similar spectrum after normalization.
# a1 = bulk('H', 'fcc', a=2.0)
from ase.visualize import view
from ase import Atoms

a = Atoms(
    cell=[1, 1, 1],
    scaled_positions=[[0.5, 0.5, 0.5]],
    symbols=["H"],
)
# view(a)
# a1 = a*[2, 1, 1]
# a2 = a*[1, 2, 1]
# a3 = a*[1, 1, 2]

a1 = bulk('H', 'fcc', a=2.0)
a2 = bulk('H', 'fcc', a=2.0, orthorhombic=True)
a3 = bulk('H', 'fcc', a=2.0, cubic=True)

# a2 = a1*[2,2,1]
# a3 = a1*[3,3,1]
# a4 = a1*[4,4,1]

# view(a1)
# view(a2)
# view(a3)

# o1 = desc.create(a1)
# print(len(desc._k3_geoms[(0, 0, 0)]))
# geoms1 = sorted(desc._k3_geoms.values())
# o2 = desc.create(a2)
# print(len(desc._k3_geoms[(0, 0, 0)]))
# geoms2 = sorted(desc._k2_geoms.values())
# o3 = desc.create(a3)
# print(len(desc._k3_geoms[(0, 0, 0)]))
# geoms3 = sorted(desc._k2_geoms.values())

# print(np.allclose(geoms1, geoms2, atol=1e-2))
# print(np.allclose(geoms1, geoms3))

output = desc.create([a1, a2, a3])
# output = desc.create([a1, a2, a3])
# output = desc.create([a1])

# Plot the MBTR output for the different cells before normalization
n_feat = desc.get_number_of_features()
# mpl.plot(desc._axis_k3, output[0, :], label="x")
# mpl.plot(desc._axis_k3, output[1, :], label="y")
# mpl.plot(desc._axis_k3, output[3, :], label="z")
# mpl.legend()
# mpl.show()

# Normalize the output to unit euclidean length
output /= np.linalg.norm(output, axis=1)[:, np.newaxis]

# Plot the MBTR output for the different cells after normalization
mpl.plot(range(n_feat), output[0, :], label="Triclinic cell")
mpl.plot(range(n_feat), output[1, :], label="Orthorhombic cell")
mpl.plot(range(n_feat), output[2, :], label="Cubic cell")
# mpl.plot(range(n_feat), output[3, :], label="Cubic cell")
mpl.show()
