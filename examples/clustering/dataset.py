import numpy as np
import ase
import ase.build
from ase import Atoms
from ase.visualize import view
from dscribe.descriptors import SOAP


# Lets create an FCC(111) surface
a=3.597
system = ase.build.fcc111(
    "Cu",
    (4,4,4),
    a=3.597,
    vacuum=10,
    periodic=True
)

# Setting up the SOAP descriptor
soap = SOAP(
    sigma=0.1,
    n_max=12,
    l_max=12,
    weighting={"function": "poly", "r0": 12, "m": 2, "c": 1, "d": 1},
    species=["Cu"],
    periodic=True,
)

# Scan the surface in a 2D grid 1 Å above the top-most atom
n = 100
cell = system.get_cell()
top_z_scaled = (system.get_positions()[:, 2].max() + 1) / cell[2, 2]
range_xy = np.linspace(0, 1, n)
x, y, z = np.meshgrid(range_xy, range_xy, [top_z_scaled])
positions_scaled = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
positions_cart = cell.cartesian_positions(positions_scaled)

# Create the SOAP desciptors for all atoms in the sample.
D = soap.create(system, positions_cart)

# Save to disk for later training
np.save("r.npy", positions_cart)
np.save("D.npy", D)
