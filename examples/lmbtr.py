from describe.descriptors import LMBTR
from describe.core import System
from describe.data.element_data import numbers_to_symbols
import numpy as np
import matplotlib.pyplot as mpl
from ase.visualize import view

# Define the system under study: NaCl in a conventional cell.
NaCl_conv = System(
    cell=[
        [5.6402, 0.0, 0.0],
        [0.0, 5.6402, 0.0],
        [0.0, 0.0, 5.6402]
    ],
    scaled_positions=[
        [0.0, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.5]
    ],
    symbols=["Na", "Cl", "Na", "Cl", "Na", "Cl", "Na", "Cl"],
)
view(NaCl_conv)

# Create a local MBTR desciptor around the atomic index 6 corresponding to a Na
# atom
decay_factor = 0.5
mbtr = LMBTR(
    atomic_numbers=[11, 17],
    k=[2, 3],
    periodic=True,
    grid={
        "k1": {
            "min": 10,
            "max": 18,
            "sigma": 0.1,
            "n": 200,
        },
        "k2": {
            "min": 0,
            "max": 0.7,
            "sigma": 0.01,
            "n": 200,
        },
        "k3": {
            "min": -1.0,
            "max": 1.0,
            "sigma": 0.05,
            "n": 200,
        }
    },
    weighting={
        "k2": {
            "function": lambda x: np.exp(-decay_factor*x),
            "threshold": 1e-3
        },
        "k3": {
            "function": lambda x: np.exp(-decay_factor*x),
            "threshold": 1e-3
        },
    },
    flatten=False)

# Two equivalent ways
desc = mbtr.create(NaCl_conv, positions=[3]) # 1
# desc = mbtr.create(NaCl_conv, positions=[[0, 0, 0]]) # 2
    

# Plot the results for the angle distribution
x1 = mbtr._axis_k1
x2 = mbtr._axis_k2
x3 = mbtr._axis_k3
imap = mbtr.index_to_atomic_number
smap = {}
for index, number in imap.items():
    smap[index] = numbers_to_symbols(number)

# Plot K2
mpl.plot(x2, desc[0][0][ 1, :], label="Cl", color="orange")
mpl.plot(x2, desc[0][0][ 0, :], label="Na", color="green")
mpl.ylabel("$\phi$ (arbitrary units)", size=20)
mpl.xlabel("Inverse distance (1/angstrom)", size=20)
mpl.title("The exponentially weighted inverse distance distribution in NaCl crystal.", size=20)
mpl.legend()
mpl.show()

# Plot K3
mpl.plot(x3, desc[0][1][0, 0, :], label="Na, Na".format(smap[0], smap[0], smap[0]), color="blue")
mpl.plot(x3, desc[0][1][0, 1, :], label="Na, Cl".format(smap[0], smap[0], smap[1]), color="orange")
mpl.plot(x3, desc[0][1][1, 1, :], label="Cl, Cl".format(smap[1], smap[0], smap[1]), color="green")
mpl.ylabel("$\phi$ (arbitrary units)", size=20)
mpl.xlabel("cos(angle)", size=20)
mpl.title("The exponentially weighted angle distribution in NaCl crystal.", size=20)
mpl.legend()
mpl.show()
