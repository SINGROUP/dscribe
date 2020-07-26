import numpy as np
from dscribe.descriptors import LMBTR
from ase.build import bulk
import matplotlib.pyplot as mpl

# Setup
lmbtr = LMBTR(
    species=["H", "O"],
    k2={
        "geometry": {"function": "distance"},
        "grid": {"min": 0, "max": 5, "n": 100, "sigma": 0.1},
        "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
    },
    k3={
        "geometry": {"function": "angle"},
        "grid": {"min": 0, "max": 180, "n": 100, "sigma": 0.1},
        "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
    },
    periodic=False,
    normalization="l2_each",
)

# Create
from ase.build import molecule

water = molecule("H2O")

# Create MBTR output for the system
mbtr_water = lmbtr.create(water, positions=[0])

print(mbtr_water)
print(mbtr_water.shape)

# Surface sites
# Build a surface and extract different adsorption positions
from ase.build import fcc111, add_adsorbate
slab_pure = fcc111('Al', size=(2, 2, 3), vacuum=10.0)
slab_ads = slab_pure.copy()
add_adsorbate(slab_ads, 'H', 1.5, 'ontop')
ontop_pos = slab_ads.get_positions()[-1]
add_adsorbate(slab_ads, 'H', 1.5, 'bridge')
bridge_pos = slab_ads.get_positions()[-1]
add_adsorbate(slab_ads, 'H', 1.5, 'hcp')
hcp_pos = slab_ads.get_positions()[-1]
add_adsorbate(slab_ads, 'H', 1.5, 'fcc')
fcc_pos = slab_ads.get_positions()[-1]

# LMBTR Setup
lmbtr = LMBTR(
    species=["Al"],
    k2={
        "geometry": {"function": "distance"},
        "grid": {"min": 1, "max": 5, "n": 200, "sigma": 0.05},
        "weighting": {"function": "exponential", "scale": 1, "cutoff": 1e-2},
    },
    periodic=True,
    normalization="none",
)

# Create output for each site
sites = lmbtr.create(
    slab_pure,
    positions=[ontop_pos, bridge_pos, hcp_pos, fcc_pos],
)

# Plot the site-aluminum distributions for each site
al_slice = lmbtr.get_location(("X", "Al"))
x = lmbtr.get_k2_axis()
mpl.plot(x, sites[0, al_slice], label="top")
mpl.plot(x, sites[1, al_slice], label="bridge")
mpl.plot(x, sites[2, al_slice], label="hcp")
mpl.plot(x, sites[3, al_slice], label="fcc")
mpl.xlabel("Site-Al distance (Å)")
mpl.legend()
mpl.show()
