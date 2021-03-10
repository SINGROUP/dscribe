import numpy as np
import ase

from ase import units
from ase.io import read, write
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nvtberendsen import NVTBerendsen
from ase.constraints import FixAtoms

import matplotlib.pyplot as plt

from dscribe.descriptors import SOAP

np.random.seed(42)

# Generate dataset of Lennard-Jones energies and forces. We use a periodic
# system with two atoms, both randomly positioned within the cell.
n_samples = 10
n_atoms = 2
energies = np.zeros(n_samples)
forces = np.zeros((n_samples, n_atoms, 3))

# Let's set up a simple LJ force field with a finite cutoff
cutoff = 5
calculator = LennardJones(
    epsilon=1.0 ,
    sigma=1/3*cutoff,
    ro=2/3*cutoff,
    cutoff=cutoff,
    smooth=True
)

# Two carbon atoms in a small periodic cell
atoms = ase.Atoms(
    'CC',
    scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
    cell=[3, 3, 3],
    pbc=False
)
atoms.set_calculator(calculator)

# Set the initial velocities to T=300K
MaxwellBoltzmannDistribution(atoms, temp=300)

c = FixAtoms(indices=[1])
atoms.set_constraint(c)

i = 0
def snapshot():
    global i
    copy = atoms.copy()
    # copy.wrap()
    traj.append(copy)
    energies[i] = atoms.get_potential_energy()
    forces[i] = atoms.get_forces()
    i += 1

# NPT ensemble
dyn = VelocityVerlet(atoms, 0.2 * units.fs)
#dyn = NVTBerendsen(atoms, 0.2 * units.fs, 300, taut=0.5*1000*units.fs)
traj = []
dyn.attach(snapshot, interval=1)
dyn.run(n_samples-1)
write('trajectory.xyz', traj)
traj = read("trajectory.xyz", index=":")

# Create the SOAP desciptors and their derivatives for all samples. We use both
# of the atoms as centers.
soap = SOAP(
    species=["C"],
    periodic=False,
    rcut=cutoff,
    sigma=0.5,
    nmax=4,
    lmax=3,
)
# print(soap.get_number_of_features())
derivatives, descriptors1 = soap.derivatives(traj)
descriptors2 = soap.create(traj)
print(np.max(np.abs(descriptors1-descriptors2)))

# Save to disk for later training
# np.save("E.npy", energies)
# np.save("D.npy", descriptors)
# np.save("dD_dr.npy", derivatives)
# np.save("F.npy", forces)
