import sparse
from ase.build import molecule
from dscribe.descriptors import SOAP

# Let's create SOAP feature vectors for two structures and all positions. If
# the output sizes are the same for each structure, a single 3D array is
# created.
soap = SOAP(
    species=["C", "H", "O"],
    periodic=False,
    r_cut=5,
    n_max=8,
    l_max=8,
    average="off",
    sparse=True
)
soap_features = soap.create([molecule("H2O"), molecule("CO2")])

# Save the output to disk and load it back.
sparse.save_npz("soap.npz", soap_features)
soap_features = sparse.load_npz("soap.npz")

# Convert to numpy/scipy formats:
dense = soap_features.todense()
csr = soap_features[0, :, :].tocsr()
csc = soap_features[0, :, :].tocsc()
