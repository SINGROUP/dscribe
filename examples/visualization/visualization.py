import numpy as np
import ase.io
from ase.build import bulk
from dscribe.descriptors import LMBTR

# Lets create iron in BCC phase
n_z = 8
n_xy_bcc = 10
a_bcc = 2.866
bcc = bulk("Fe", "bcc", a=a_bcc, cubic=True) * [n_xy_bcc, n_xy_bcc, n_z]

# Lets create iron in FCC phase
a_fcc = 3.5825
n_xy_fcc = 8
fcc = bulk("Fe", "fcc", a=a_fcc, cubic=True) * [n_xy_fcc, n_xy_fcc, n_z]

# Setting up the descriptor
descriptor = LMBTR(
    grid={"min": 0, "max": 12, "sigma": 0.1, "n": 200},
    geometry={"function": "distance"},
    weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
    species=["Fe"],
    periodic=True,
)

# Calculate feature references
bcc_features = descriptor.create(bcc, [0])
fcc_features = descriptor.create(fcc, [0])

# Combine into one large grain boundary system with some added noise
bcc.translate([0, 0, n_z * a_fcc])
combined = bcc + fcc
combined.set_cell([
    n_xy_bcc * a_bcc,
    n_xy_bcc * a_bcc,
    n_z * (a_bcc + a_fcc)
])
combined.rattle(0.1, seed=7)

# Create a measure of of how similar the atoms are to the reference. Euclidean
# distance is used and the values are scaled between 0 and 1 from least similar
# to identical with reference.
def metric(values, reference):
    dist = np.linalg.norm(values - reference, axis=1)
    dist_max = np.max(dist)
    return  1 - (dist / dist_max)

# Create the features and metrics for all atoms in the sample.
combined_features = descriptor.create(combined)
fcc_metric = metric(combined_features, fcc_features)
bcc_metric = metric(combined_features, bcc_features)

# Write an image with the atoms coloured corresponding to their similarity with
# BCC and FCC: BCC = blue, FCC = red
n_atoms = len(combined)
colors = np.zeros((n_atoms, 3))
for i in range(n_atoms):
    colors[i] = [fcc_metric[i], 0, bcc_metric[i]]
ase.io.write(
    f'coloured.png',
    combined,
    rotation='90x,20y,20x',
    colors=colors,
    show_unit_cell=1,
    maxwidth=2000,
)

# Write the original system image
ase.io.write(
    f'original.png',
    combined,
    rotation='90x,20y,20x',
    show_unit_cell=1,
    maxwidth=2000,
)
