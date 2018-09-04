"""Demonstrates the use of the utility RematchKernel
In this example global structures are compared based on the 
local descriptors of each atom within the structures"""
from describe.descriptors import SOAP, ACSF
import ase
import numpy as np
from describe.utils import RematchKernel
from ase.build import molecule
from ase.collections import g2

# Choose descriptor
descriptor = "SOAP"

# Compute local descriptors
all_atomtypes = [1,6,7,8]
if descriptor == "SOAP":
    desc = SOAP(all_atomtypes, 
        10.0, 2, 0, periodic=False,crossover=True)
elif descriptor == "ACSF":
    desc = ACSF(n_atoms_max=15, types=[1,6,7,8],bond_params=[[1,2,], [4,5,]], 
        bond_cos_params=[1,2,3,4], 
        ang4_params=[[1,2,3],[3,1,4], [4,5,6], [7,8,9]], 
        ang5_params=[[1,2,3],[3,1,4], [4,5,6], [7,8,9]], flatten=False)
else:
    print("Add your local descriptor here")
    exit(0)

re = RematchKernel()
desc_list = []
# choose a few molecules from ase database
for molname in g2.names:
    atoms = molecule(molname)
    atomic_numbers = atoms.get_atomic_numbers()
    leftover = set(atomic_numbers) - set(all_atomtypes)
    if len(leftover) > 0:
        continue
    local_a = desc.create(atoms)
    if len(local_a) == 0:
        continue
    else:
        desc_list.append(local_a)

# Compute environment kernels for all structure pairs
envkernel_dict = re.get_all_envkernels(desc_list)

# Compute global similarity matrix
global_matrix = re.get_global_kernel(envkernel_dict, 
    gamma = 0.01, threshold= 1e-6)

print(global_matrix.shape)
