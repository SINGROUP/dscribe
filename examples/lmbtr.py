from dscribe.descriptors import LMBTR
from dscribe.core import System

import matplotlib.pyplot as mpl

import ase.data

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
# view(NaCl_conv)

# Create a local MBTR desciptor around the atomic index 6 corresponding to a Na
# atom
decay_factor = 0.5
mbtr = LMBTR(
    species=[11, 17],
    k=[2, 3],
    periodic=True,
    virtual_positions=False,
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
            "function": "exponential",
            "scale": decay_factor,
            "cutoff": 1e-3
        },
        "k3": {
            "function": "exponential",
            "scale": decay_factor,
            "cutoff": 1e-3
        },
    },
    sparse=False,
    flatten=False
)

# The output of local MBTR is a list of atomic environments for each given
# position
desc = mbtr.create(NaCl_conv, positions=[3])

# Create variable contains the mapping between an index in the output and the
# corresponding atomic number
elements = NaCl_conv.get_atomic_numbers().tolist()
elements.append(0)  # We add the ghost atom element
n_elements = len(set(elements))
imap = mbtr.index_to_atomic_number
smap = {}
for index, number in imap.items():
    smap[index] = ase.data.chemical_symbols[number]

# Plot K2. Only the combinations i-j where j >= 1 are included in the output.
# The combination where i > i would contain the same information due to
# distance symmetry with respect to changing the indices.
x2 = mbtr._axis_k2
for i in range(n_elements):
    for j in range(n_elements):
        i_elem = smap[i]
        j_elem = smap[j]
        if j >= i and (i_elem == "X" or j_elem == "X"):
            mpl.plot(x2, desc[0]["k2"][i, j, :], label="{}-{}".format(smap[i], smap[j]))
mpl.ylabel("$\phi$ (arbitrary units)", size=20)
mpl.xlabel("Inverse distance (1/angstrom)", size=20)
mpl.title("The exponentially weighted inverse distance distribution in NaCl crystal.", size=20)
mpl.legend()
mpl.show()

# Plot K3. Only the combinations i-j-k where k >= i are included in the output.
# The combination where i > k would contain the same information due to angle
# symmetry with respect to permuting i and k.
x3 = mbtr._axis_k3
for i in range(n_elements):
    for j in range(n_elements):
        for k in range(n_elements):
            i_elem = smap[i]
            j_elem = smap[j]
            k_elem = smap[k]
            if k >= i and (i_elem == "X" or j_elem == "X" or k_elem == "X"):
                mpl.plot(x3, desc[0]["k3"][i, j, k, :], label="{}-{}-{}".format(smap[i], smap[j], smap[k]))
mpl.ylabel("$\phi$ (arbitrary units)", size=20)
mpl.xlabel("cos(angle)", size=20)
mpl.title("The exponentially weighted angle distribution in NaCl crystal.", size=20)
mpl.legend()
mpl.show()
