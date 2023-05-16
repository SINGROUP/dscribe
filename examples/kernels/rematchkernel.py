"""Demostrates how global similarity kernels can be built from local atomic
environments.
"""
from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel

from ase.build import molecule

from sklearn.preprocessing import normalize

# We will compare two similar molecules
a = molecule("H2O")
b = molecule("H2O2")

# First we will have to create the features for atomic environments. Lets
# use SOAP.
desc = SOAP(species=["H", "O"], r_cut=5.0, n_max=2, l_max=2, sigma=0.2, periodic=False, crossover=True, sparse=False)
a_features = desc.create(a)
b_features = desc.create(b)

# Before passing the features we normalize them. Depending on the metric, the
# REMatch kernel can become numerically unstable if some kind of normalization
# is not done.
a_features = normalize(a_features)
b_features = normalize(b_features)

# Calculates the similarity with the REMatch kernel and a linear metric. The
# result will be a full similarity matrix.
re = REMatchKernel(metric="linear", alpha=1, threshold=1e-6)
re_kernel = re.create([a_features, b_features])

# Any metric supported by scikit-learn will work: e.g. a Gaussian.
re = REMatchKernel(metric="rbf", gamma=1, alpha=1, threshold=1e-6)
re_kernel = re.create([a_features, b_features])
