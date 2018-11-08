from dscribe.descriptors import MBTR
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

# Setup the MBTR desciptor for the system
decay_factor = 0.5
mbtr = MBTR(
    atomic_numbers=[11, 17],
    k=[1, 2, 3],
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
    flatten=True
)

# Lets first create a flattened output vector that is directly suitable for
# machine learning applications. When the "flatten" parameter is set to True,
# the create function will return a flattened sparse COO-matrix.
desc = mbtr.create(NaCl_conv)

# The corresponding dense numpy array can be create like this:
desc_dense = desc.toarray()

# Optionally we can preserve the tensorial nature of the data by specifying
# "Flatten" as False. In this case the output will be a list of
# multidimensional numpy arrays for each k-term. This form is easier to
# visualize as done in the following.
mbtr.flatten = False
desc = mbtr.create(NaCl_conv)

# Create variable contains the mapping between an index in the output and the
# corresponding atomic number
n_elements = len(set(NaCl_conv.get_atomic_numbers()))
imap = mbtr.index_to_atomic_number
smap = {}
for index, number in imap.items():
    smap[index] = ase.data.chemical_symbols[number]

# Plot K1
x1 = mbtr._axis_k1
for i in range(n_elements):
    mpl.plot(x1, desc["k1"][i, :], label="{}".format(smap[i]))
mpl.ylabel("$\phi$ (arbitrary units)", size=20)
mpl.xlabel("Atomic number", size=20)
mpl.title("The element count in NaCl crystal.", size=20)
mpl.legend()
mpl.show()

# Plot K2. Only the combinations i-j where j >= 1 are included in the output.
# The combination where i > i would contain the same information due to
# distance symmetry with respect to changing the indices.
x2 = mbtr._axis_k2
for i in range(n_elements):
    for j in range(n_elements):
        if j >= i:
            mpl.plot(x2, desc["k2"][i, j, :], label="{}-{}".format(smap[i], smap[j]))
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
            if k >= i:
                mpl.plot(x3, desc["k3"][i, j, k, :], label="{}-{}-{}".format(smap[i], smap[j], smap[k]))
mpl.ylabel("$\phi$ (arbitrary units)", size=20)
mpl.xlabel("cos(angle)", size=20)
mpl.title("The exponentially weighted angle distribution in NaCl crystal.", size=20)
mpl.legend()
mpl.show()
