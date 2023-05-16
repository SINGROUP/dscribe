import numpy as np
from ase.build import molecule
from dscribe.descriptors import SOAP
from dscribe.descriptors import CoulombMatrix

# Define atomic structures
samples = [molecule("H2O"), molecule("NO2"), molecule("CO2")]

# Setup descriptors
cm_desc = CoulombMatrix(n_atoms_max=3, permutation="sorted_l2")
soap_desc = SOAP(species=["C", "H", "O", "N"], r_cut=5, n_max=8, l_max=6, crossover=True)

# Create descriptors as numpy arrays or sparse arrays
water = samples[0]
coulomb_matrix = cm_desc.create(water)
soap = soap_desc.create(water, centers=[0])

# Easy to use also on multiple systems, can be parallelized across processes
coulomb_matrices = cm_desc.create(samples)
coulomb_matrices = cm_desc.create(samples, n_jobs=3)
oxygen_indices = [np.where(x.get_atomic_numbers() == 8)[0] for x in samples]
oxygen_soap = soap_desc.create(samples, oxygen_indices, n_jobs=3)

# Descriptors also allow calculating derivatives with respect to atomic
# positions
der, des = soap_desc.derivatives(samples, return_descriptor=True)
