from dscribe.descriptors import SOAP
from dscribe.descriptors import CoulombMatrix
from ase.build import molecule

# Define geometry
mol = molecule("H2O")

# Setup descriptors
cm_desc = CoulombMatrix(n_atoms_max=3, permutation="sorted_l2")
soap_desc = SOAP(species=[1, 8], rcut=5, nmax=8, lmax=6, crossover=True)

# Create descriptors as numpy arrays or scipy sparse matrices
input_cm = cm_desc.create(mol)
input_soap = soap_desc.create(mol, positions=[0])
