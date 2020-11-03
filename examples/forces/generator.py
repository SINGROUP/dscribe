import ase
from ase.visualize import view
from ase.calculators.lj import LennardJones
import numpy as np
from dscribe.descriptors import SOAP
import matplotlib.pyplot as plt

species = ["H"]
rcut = 5.0
nmax = 3
lmax = 0
sigma=0.5

# Setting up the SOAP descriptor
soap = SOAP(
    species=species,
    periodic=False,
    rcut=rcut,
    sigma=sigma,
    nmax=nmax,
    lmax=lmax,
)

# center is origin
CENTER = [0,0,0]

# one example of h2 molecule
d = 5
atoms = ase.Atoms('HH', positions = [[-0.5 * d, 0, 0], [0.5 * d, 0, 0]])
#view(atoms)
atoms.set_calculator(LennardJones(epsilon = 1e-10 , sigma = 5.0, rc = 10.0))
forces = atoms.get_forces()
total_energy = atoms.get_total_energy()

# Generate dataset of LJ energies and forces
traj = []
traj_forces = []
traj_energies = []
r = np.linspace(2.5, 5.0, 200) # bond lengths of h2
for d in r:
    a = ase.Atoms('HH', positions = [[-0.5 * d, 0, 0], [0.5 * d, 0, 0]])
    a.set_calculator(LennardJones(epsilon=1.0 , sigma=2.9))
    traj_forces.append(a.get_forces())
    traj.append(a)
    traj_energies.append(a.get_total_energy())
	
y = traj_energies
fig, ax = plt.subplots()
line, = ax.plot(r, y, color='blue', lw=2)
plt.show()
traj_energies = np.array(traj_energies)
traj_forces = np.array(traj_forces)

# Create SOAP output for the system
descriptors = []
derivatives = []
for t in traj:
    i_derivative, i_descriptor = soap.derivatives_single(t, positions=[CENTER], method="numerical")
    descriptors.append(i_descriptor[0])
    derivatives.append(i_derivative[0])
descriptors = np.array(descriptors)
derivatives = np.array(derivatives)

# Save to disk for later training
np.save("r.npy", r)
np.save("E.npy", traj_energies)
np.save("D.npy", descriptors)
np.save("dD_dr.npy", derivatives)
np.save("F.npy", traj_forces)
