from ase.io import read
from ase.build import molecule
from ase import Atoms

# Let's use ASE to create atomic structures as ase.Atoms objects.
structure1 = read("water.xyz")
structure2 = molecule("H2O")
structure3 = Atoms(symbols=["C", "O"], positions=[[0, 0, 0], [1.128, 0, 0]])

# Let's create a list of structures and gather the chemical elements that are
# in all the structures.
structures = [structure1, structure2, structure3]
species = set()
for structure in structures:
    species.update(structure.get_chemical_symbols())

# Let's configure the SOAP descriptor.
from dscribe.descriptors import SOAP

soap = SOAP(
    species=species,
    periodic=False,
    r_cut=5,
    n_max=8,
    l_max=8,
    average="outer",
    sparse=False
)

# Let's create SOAP feature vectors for each structure
feature_vectors = soap.create(structures, n_jobs=1)

# Let's create derivatives and feature vectors for each structure
derivatives, feature_vectors = soap.derivatives(
    structures,
    return_descriptor=True,
    n_jobs=1
)